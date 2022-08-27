# Copyright 2022 TranNhiem.

# Code base Inherence from https://github.com/facebookresearch/dino/

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

"""
ViT from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
from functools import partial
import torch
from torch import nn, einsum
from utils import trunc_normal_
from einops import rearrange, repeat


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

''' Building cross attention model with the input query, key and value are from attention layer output.'''
class CrossAttention(nn.Module):
    def __init__(self, query_embedim, num_heads=1, qkv_bias=False, qk_scale=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = query_embedim.size(-1) // num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        #q, k, v = map(lambda t: rearrange(t, 'b d-> b d'), (query, key, value))
        q, k, v= query, key, value
        sim = einsum('b d, j d ->  j b', q, k) * self.scale
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h =self.num_heads)
            sim.masked_fill_(~mask, max_neg_value)

        # attention output based on the similarity matrix
        attn = sim.softmax(dim=-1)
        out= einsum('j b, i d ->  i b', attn, v)
        #out= rearrange(out, 'b n (h d) -> (b h) n d', h = self.num_heads)
        
        return out, attn
    
class Block(nn.Module): 
    '''
    Object of the Transformer Encoder --> Stacking multiple layers to build deeper
    '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                qk_scale=None, drop=0., attn_drop=0.,drop_path=0.,
                 act_layer=nn.GELU, norm_layer= nn.LayerNorm
        ): 

        super().__init__()
        self.norm1= norm_layer(dim)
        self.attn= Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False): 
        y, attn= self.attn(self.norm1(x))
        if return_attention: 
            return attn 
        x= x+ self.drop_path(y)
        x= x+ self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module): 
    """
    Dividing image into patches 
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=786): 
        super().__init__()
        num_patches= (img_size // patch_size) * (img_size // patch_size)
        self.img_size= img_size 
        self.patch_size= patch_size
        self.num_patches= num_patches
        ## --- Hybrid architecture --> Patches flatten from Conv2D spatial map----
        self.proj= nn.Conv2d(in_chans, embed_dim, kernel_size= patch_size, stride= patch_size)

    def forward(self, x):
        B, C, H, W= x.shape
        x=self.proj(x).flatten(2).transpose(1, 2)
        return x 

class VisionTransformer(nn.Module): 
    """
    Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, 
        embed_dim= 786, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs
        ): 
        super().__init__()
        self.num_features= self.embed_dim= embed_dim
        self.patch_embed= PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token= nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed= nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop= nn.Dropout(p=drop_rate)

        dpr= [x.item() for x in torch.linspace(0, drop_path_rate, depth)] #stochastic depth decay rule 
        self.blocks= nn.ModuleList([
            Block(dim=embed_dim,num_heads= num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
             drop=drop_rate, attn_drop= attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm= norm_layer(embed_dim)

        # Classifier Head 
        self.head= nn.Linear(embed_dim, num_classes) if num_classes >0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m): 
        if isinstance(m, nn.Linear): 
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): 
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h): 
        npatch= x.shape[1] -1 
        N= self.pos_embed.shape[1]- 1
        if npatch==N and w==h: 
            return self.pos_embed 
        class_pos_embed=self.pos_embed[:, 0]
        patch_pos_embed= self.pos_embed[:, 1:]
        dim= x.shape[-1]
        w0= w// self.patch_embed.patch_size 
        h0= h// self.patch_embed.patch_size 

        # We add a small number to avoid floating point error in the interpolation 
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 +0.1, h0+0.1
        patch_pos_embed= nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0,3,1,2), 
            scale_factor=(w0/math.sqrt(N), h0/math.sqrt(N)), mode='bicubic', 
        )
        assert int(w0)== patch_pos_embed.shape[-2] and int(h0)== patch_pos_embed.shape[-1]
        patch_pos_embed= patch_pos_embed.permute(0, 2, 3, 1).view(1, -1,dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x): 
        x= self.prepare_tokens(x)
        for blk in self.blocks: 
            x= blk(x)
        x= self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self,x): 
        x= self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks): 
            if i < len(self.blocks) -1: 
                x= blk(x)
            else: 
                # return attention of the last block 
                return blk(x, return_attention=True)
    
    def get_intermediate_layers(self, x, n=1): 
        x= self.prepare_tokens(x)
        # we return the output tokens from the 'n' last blocks
        output= [] 
        for i, blk in enumerate(self.blocks): 
            x= blk(x)
            if len(self.blocks) -1 <= n: 
                output.append(self.norm(x))
        return output

def vit_tiny(patch_size=16, **kwargs): 
    model=VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, 
        num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs): 
    model= VisionTransformer(
        patch_size= patch_size, embed_dim=384, depth=12, num_heads=6, 
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 

def vit_base(patch_size=16, **kwargs): 
    model=VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 

def vit_base_ibot_16(patch_size=16, **kwargs): 
    model=VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 

def vit_L_16_ibot(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

"""Computing Patch similarity"""
class patch_head(nn.Module):
    '''
    This class returns the similarity between patches 
    '''

    def __init__(self, in_dim, k_num):
        
        super().__init__()
        self.k_num = k_num
        self.k_size = 3
        self.loc224 = self.get_local_index(196, self.k_size)
        self.loc96 = self.get_local_index(36, self.k_size)
        self.embed_dim = in_dim

    def forward(self, x, ):
        ## X is Batched image tensor with shape [B, patch*patch, embedding_dim]
        k_size = self.k_size
        if x.shape[1] == 196:
            local_idx = self.loc224
        elif x.shape[1] == 36:
            if self.k_size == 14:
                k_size = 6
            local_idx = self.loc96
        else:
            print(x.shape)
            assert (False)

        x_norm = nn.functional.normalize(x, dim=-1)
        # Compute Cosine Similarity Matrix
        sim_matrix = x_norm[:,local_idx] @ x_norm.unsqueeze(2).transpose(-2, -1)
        top_idx = sim_matrix.squeeze().topk(k=self.k_num, dim=-1)[1].view(-1, self.k_num, 1)
        x_loc = x[:, local_idx].view(-1, k_size**2-1, self.embed_dim)
        ## For multi-gpus we need to use gather
        x_loc = torch.gather(x_loc, 1, top_idx.expand(-1, -1, self.embed_dim))
        
        return top_idx, x_loc

    @staticmethod
    def get_local_index(N_patches, k_size):
        loc_weight = []
        w = torch.LongTensor(list(range(int(math.sqrt(N_patches)))))
        # Why we need to iterate through all patches
        for i in range(N_patches):
            ix, iy = i // len(w), i % len(w)
            wx = torch.zeros(int(math.sqrt(N_patches)))
            wy = torch.zeros(int(math.sqrt(N_patches)))
            wx[ix] = 1
            wy[iy] = 1
            # Iteration through all N patches of Single Images?
            for j in range(1, int(k_size//2)+1):
                wx[(ix+j) % len(wx)] = 1
                wx[(ix-j) % len(wx)] = 1
                wy[(iy+j) % len(wy)] = 1
                wy[(iy-j) % len(wy)] = 1

            weight = (wy.unsqueeze(0) * wx.unsqueeze(1)).view(-1)
            weight[i] = 0
            loc_weight.append(weight.nonzero().squeeze())

        return torch.stack(loc_weight)
