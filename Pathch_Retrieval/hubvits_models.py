# Copyright 2022 TranNhiem SSL Team.
# Code base Inherence from https://github.com/facebookresearch/dino/blob/main/hubconf.py
import os 
import sys
import vision_transformer as vits 
import torch

def dino_vits16(pretrained=True, **kwargs): 
    """
    ViT-small /16x16 pre-trained with DINO 
    """
    model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
    if pretrained: 
        state_dict= torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)  
    return model

def dino_vits8(pretrained=True, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    """
    model = vits.__dict__["vit_small"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb16(pretrained=True, **kwargs):
    """
    ViT-Base/16x16 pre-trained with DINO.
    """
    model = vits.__dict__["vit_base"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb8(pretrained=True, **kwargs):
    """
    ViT-Base/8x8 pre-trained with DINO.
    """
    model = vits.__dict__["vit_base"](patch_size=8,  num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model