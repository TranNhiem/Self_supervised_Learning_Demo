
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

import os
import torch
import torchvision
import sys
from utils import load_pretrained_weights, normal_dataloader, attention_retrieving, attention_map_color
import torch.nn as nn
import vision_transformer as vits
import argparse
from tqdm.auto import tqdm
from torchvision import models as torchvision_models
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms as pth_transforms
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Any, Callable, List, Tuple, Optional
from torchvision.transforms.functional import to_pil_image
from torchvision import io
# Get the torch hub model
from hubvits_models import dino_vitb8, dino_vitb16, dino_vits8, dino_vits16
from helper_functions import attention_heatmap

# Visualization and Plot the Image
# from torchvision.utils import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

# ******************************************************
# Arguments needed to load model checkpoint
# ******************************************************

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():

    parser = argparse.ArgumentParser('Patch-Match Retrieval', add_help=False)
    # ********************************************************
    # Model parameters
    # ********************************************************
    parser.add_argument('--arch', default='vit_L_16_ibot', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base',
                                 'vit_base_ibot_16', 'vit_L_16_ibot']
                        + torchvision_archs +
                        torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--drop_path_rate', type=float,
                        default=0.1, help="stochastic depth rate")

    # ********************************************************
    # Setting the Saving Experiments Result
    # ********************************************************
    parser.add_argument('--image_path', default="/home/rick/offline_finetune/Pets/images/", type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--dataloader_patches', default=False, type=bool,
                        help='Decided loading dataloader with or without Patches image')
    parser.add_argument('--subset_data', default=0.1, type=float,
                        help='How many percentage of your Demo Dataset you want to Use.')
    parser.add_argument('--single_img_path', default="/home/rick/offline_finetune/Pets/images/american_bulldog_72.jpg", type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Image_size resizes standard for all input images.')

    parser.add_argument('--output_dir', default="/home/rick/pretrained_weight/DINO_Weight/",
                        type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--save_dir', default="/home/rick/Visualization/DINO/",
                        type=str, help='Path to save Attention map out.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    return parser


def load_model(args):
    '''
    Following arguments using 
    args.arch --> this will loading the ViT architecture
    args.patch_size --> This helps specified input_shape
    args.
    '''
    # if the network is Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,)  # stochastic depth)
        embed_dim = model.embed_dim
        print(f"model embedding shape {embed_dim}")
        model = model.cuda()
        pretrained_model = load_pretrained_weights(
            model, pretrained_weights="None", checkpoint_key=None, model_name=args.arch, patch_size=args.patch_size)

    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        pretrained_model = load_pretrained_weights(
            model, pretrained_weights="None", checkpoint_key=None, model_name=args.arch, patch_size=args.patch_size)

    else:
        print(f"Unknow architecture: {args.arch}")

    return pretrained_model

# ******************************************************
# Loading Image --> Preprocessing Steps
# ******************************************************

# Get Batch of all images from Demo dataset

def batch_images(args):
    '''
    Args
    1: Image_path ->  dataset folder structure is Containing all images inside.
    2: image_format -> The format of image (because of using *glob* function to get all image Name) Default ("*.jpg")
    3: img_resize ->  resize all image the same size
    4: Subset of Demo Dataset -> Using only Proportion of Dataset
    5: transform_ImageNet --> Normalize the image with ImageNet (mean, std)

    '''
    val_dataset = normal_dataloader(image_path=args.image_path, img_size=args.image_size,
                                    image_format="*.jpg", subset_data=args.subset_data, transform_ImageNet=True)

    if args.dataloader_patches:
        print("Loading Patches dataloader")
        chanels = 3  # RGB image
        val_dl = val_dataset.val_dataloader_patches(args.patch_size, chanels)
    else:
        print("Loading Normal dataloader without Patching Images")
        val_dl = val_dataset.val_dataloader()

    return val_dl, val_dataset

# Get a Single image

def get_image(args, image_example_path=None):
    '''
    Args:  
    args.image_path: This provides the image in your machine or other source
    args.image_resize: This resizes image to expected size
    '''
    if args.single_img_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, Using Default Pets Images ")
        default_image = '/home/rick/offline_finetune/birdsnap__/dataset/train/Canvasback/68362.jpg'
        with open(default_image, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    if image_example_path is None:
        if os.path.isfile(args.single_img_path):
            with open(args.single_img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            print(f"Provided image path {args.image_path} is non valid.")
            sys.exit(1)
    else:
        if os.path.isfile(image_example_path):
            print("using image_example_path")
            with open(image_example_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            print(f"Provided image path {args.image_path} is non valid.")
            sys.exit(1)

    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.image_size, args.image_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img)

    return img_tensor


def patches_similarity(anchor_embedding, reference_embedding, normalize=True):
    '''
    args: 
    anchor_embedding: This is the embedding of the anchor image size of ()
    reference_embedding: This is the embedding of the reference image
    normalize: This is to normalize the embedding
    return: 
    similarity of each patches between anchor and reference image [patches, patches]
    '''

    if normalize:
        anchor_embedding = nn.functional.normalize(
            anchor_embedding, p=2, dim=1)
        reference_embedding = nn.functional.normalize(
            reference_embedding, p=2, dim=1)

    #similarity= ancher_patch.cpu().numpy() @ other_patches.cpu().numpy().T
    similarity = torch.bmm(
        anchor_embedding, reference_embedding.permute(0, 2, 1))
    score, value = similarity.max(-1)
    print(f'this is score shape {score.shape}')
    print(f'this is value shape {value.shape}')
    return score, value

def image_retrieval_topk(embedding: torch.FloatTensor, i: int, topk=4, normalize=True):
    if normalize:
        #embedding = embedding.norm(dim=-1, keepdim=True)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        print("All image embedding normalized shape {}".format(embedding.shape))

    # print(f"This is all images embedding shape: {embedding.shape}")
    print(f"This is image I embedding position shape: {embedding[i,:].shape}")
    similarity = (embedding @ embedding[i, :].T)  # .softmax(dim=-1)
    print(f"Similarity method 1: {similarity.shape}")
    # similarity= torch.mm(embedding , embedding[i, :].T)
    # print(f"Similarity method 2: {similarity.shape}")
    scores, idx = similarity.cpu().topk(topk,)  # dim=-1
    print(scores[0])
    print(idx[0])
    # test=idx.argmax(dim=0)
    # print(test.shape)
    print("this is idx value:", idx.shape)
    print("This is score value", scores.shape)

    return scores.numpy(), idx.numpy()  # .numpy()

def image_retrieval_topk_extra(source_data_embedding: torch.FloatTensor,
                               anchor_image_embedding: torch.FloatTensor,
                               topk=4, normalize=True):
    if normalize:
        source_data_embedding = nn.functional.normalize(
            source_data_embedding, p=2, dim=1)
        anchor_image_embedding = nn.functional.normalize(
            anchor_image_embedding, p=2, dim=1)
        print("All image embedding normalized shape {}".format(
            source_data_embedding.shape))

    # print(f"This is all images embedding shape: {source_data_embedding.shape}")
    # print(f"This is anchor image shape: {anchor_image_embedding.shape}")
    anchor_image_embedding= anchor_image_embedding.flatten()
    print(anchor_image_embedding.shape)
    source_data_embedding=source_data_embedding.flatten( start_dim=1)
    print(source_data_embedding.shape)
    #similarity =   source_data_embedding@anchor_image_embedding.T
    # source_embedding=source_data_embedding.permute(2, 0, 1)
    # print(f"source_embedding shape: {source_embedding.shape}")
    # anchor_embedding=anchor_image_embedding.permute(2, 0, 1)
    # print(f"anchor_embedding shape: {anchor_embedding.shape}")
    #similarity= torch.bmm(source_embedding,anchor_embedding )
    #print(f"similarity use method 1: {similarity.shape}")

    #similarity = torch.mm(source_data_embedding @ anchor_image_embedding.T)
    #print(f"similarity use method 2: {similarity.shape}")
    #scores, idx = similarity.topk(topk, dim=0)
    similarity= ( source_data_embedding @ anchor_image_embedding.T)
    print(f"similarity use method 3: {similarity.shape}")
    scores, idx = similarity.cpu().topk(topk,sorted=True, dim=-1)

    print("this is idx value:", idx.shape)
    print("This is score value", scores.shape)
    return scores.cpu().numpy(), idx.cpu().numpy()

def plotting_image_retrieval(args, source_data_embedding, ancher_embedding, i, val_dat,
                             topk, retrieval_from_exmaple=False, image_path: Optional[str] = None, ):

    if retrieval_from_exmaple:
        print("you must provide 2 argugment 1: image_path argument, 2:dataloader_patches == False ")

        if image_path is None:
            raise ValueError("You are not providing image_path argument")
        else:
            resize = pth_transforms.Resize((args.image_size, args.image_size,))
            image = to_pil_image(resize(io.read_image(image_path)))
            plt.imshow(image)
            scores, idx = image_retrieval_topk_extra(
                source_data_embedding, ancher_embedding, topk=topk)
            print(scores.shape)
            print(idx.shape)
            print(idx)
            print(scores)
            ig, axs = plt.subplots(1, len(idx), figsize=(12, 5))
            for i_, score, ax in zip(idx, scores, axs):
                img = to_pil_image(resize(io.read_image(val_dat.image_files[i_])))
                ax.imshow(img)
                ax.set_title(f"{score:.4f}")
            plt.show()
    else:
        resize = pth_transforms.Resize((args.image_size, args.image_size,))
        print("image_path", val_dat.image_files[i])
        image = to_pil_image(resize(io.read_image(val_dat.image_files[i])))
        plt.imshow(image)
        scores, idx = image_retrieval_topk(source_data_embedding, i, topk=topk)
        idx = idx
        scores = scores
    
        ig, axs = plt.subplots(1, 5, figsize=(12, 5))
        j = 0
        for i_, score, ax in zip(idx, scores, axs):
            print(i_.shape)
            print("This is idx image", i_)
            img = to_pil_image(
                resize(io.read_image(val_dat.image_files[i_[j][0]])))
            ax.imshow(img)
            ax.set_title(f"{score[j][0]:.4f}")
            j += 1
        plt.show()


def plot_similarty_matrix(embedding,  val_data):
    '''
    Args: 
    embeding: Dimension should be single *Embedding TENSOR*

    '''
    original_images = []
    for filename in val_data.image_files:
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        original_images.append(image)

    # embedding /= embedding.norm(dim=0, keepdim=True)
    # similarity = embedding.cpu().numpy() @ embedding[1, :].cpu().numpy().T
    similarity = embedding[0, :] @ embedding[1, :].T
    similarity = similarity.cpu().numpy()

    print(f"This is the shape of similiarity matrix: {similarity.shape}")
    count = 8
    plt.figure(figsize=(20, 14))
    # plt.colorbar() # plt.yticks(range(count), anchor_image, fontsize=18)
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    texts = ['test', "test", 'test', "test", 'test', "test", 'test', "test"]
    plt.yticks(range(count), texts, fontsize=18)
    # for i, image in enumerate(original_images):
    #     plt.imshow(image, extent=(
    #         i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(
            i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[0]):
        for y in range(similarity.shape[1]):
            plt.text(x, y, f"{similarity[y, x]:.2f}",
                     ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    plt.title("Cosine similarity between text and image features", size=20)

    plt.show()


if __name__ == '__main__':

    # ******************************************************
    # Unit Test Code
    # ******************************************************

    # -----------------------------------
    # 1---- Get all Input Argments
    # -----------------------------------
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -----------------------------------
    # 2--- loading Model
    # -----------------------------------
    model = load_model(args)
    # model = dino_vitb16() #dino_vitb8(), dino_vits8(), dino_vits16()
    model = model.to(device)
    model_seq = nn.Sequential(*model.blocks)
    model_seq = model_seq.eval().to(device)
    print("Successfull loading model")

    # -----------------------------------
    # 3--- Loading dataset
    # -----------------------------------
    # Single Image
    # img=get_image(args)
    # image_=img[None, :]
    # patches_img=model.patch_embed(image_)
    # #print(patches_img.shape)
    # print(f"Succeed loading and convert: image_shape {patches_img.shape}")
    # out_ = (model_seq(patches_img.to(device)))
    # print(out_.shape)

    # Loading Batches of Images
    val_dl, data_path = batch_images(args)
    #print(f"Succeed loading inference dataloader: data len {len(val_dl)}")
    out_embedding = []
    with torch.no_grad():
        # #If setting args.dataloader_patches ==True
        # for img_ in tqdm(val_dl):
        #     #print(f"Patch image shape: {img_.shape}")
        #     out_ = (model_seq(img_.to(device)))
        #     #out_embedding.append(out_)

        # If setting args.dataloader_patches ==False
        for img_ in tqdm(val_dl):
            patches_img = model.patch_embed(img_.to(device))
            #print(f"Patch image shape: {patches_img.shape}")
            # attentions = model.get_last_selfattention(img_.to(device))
            # print(f"Attention Map shape: {attentions.shape}")
            out_ = (model_seq(patches_img.to(device)))
            #print("This is one image output shape: {out_.shape}")
            out_embedding.append(out_)
            # break

    out_embedding = torch.cat(out_embedding, dim=0)
    print("Casting all images embedding", out_embedding.shape)

    # -----------------------------------
    # 4 Visualization Attention Map
    # -----------------------------------
    # # ## Get Image and Transform to tensor
    # img = get_image(args)
    # image_ = img.view([1, 3, args.image_size, args.image_size]).to(device)
    # print(f"This is image shape: {image_.shape}")

    # # Forward pass through model get (Ouput representation)
    # with torch.no_grad():
    #     attentions_out = model.get_last_selfattention(image_.to(device))
    #     #attentions_out = model.get_intermediate_layers(image_.to(device), n=4)
    # print(f"The attention  shape: {attentions_out[-1].shape}")
    # nh = attentions_out.shape[1]  # number of head
    # print(f"Number of attention head: {nh}")

    # ******************************************************
    # 4.1 Visualization Attention Heat Map for each head
    # ******************************************************
    # attention_map = attention_heatmap(args, attentions_out, image_)
    # n_row = int(nh / 4)
    # ncolm=int(nh/n_row)
    # fig, axes = plt.subplots(nrows=n_row, ncols=ncolm, figsize=(15, 15))
    # idx = 0
    # resize = pth_transforms.Resize((args.image_size, args.image_size,))
    # image = to_pil_image(resize(io.read_image(args.single_img_path)))
    # for i in range(n_row):
    #     for j in range(ncolm):
    #         if idx < nh:
    #             axes[i, j].imshow(image)
    #             axes[i, j].imshow(attention_map[idx,... ],cmap="inferno", alpha=0.5)
    #             axes[i, j].title.set_text(f"Attention head: {idx}")
    #             axes[i, j].axis("off")
    #             idx += 1
    # plt.show()

    # ******************************************************
    # 4.2 Visualization Attention Heat Map and Colors attention Map for all heads Saving to file
    # ******************************************************
    # threshold = 0.5  # If threshold is None, all attention map will be saved in Heatmap image only
    # attentions, th_attn, img_, attns = attention_retrieving(args, img, threshold, attentions_out,
    #                                                         args.save_dir, blur=False, contour=False, alpha=0.5, visualize_each_head=True)
    # print(f"image shape: {img.shape}")
    # img = img.permute(1, 2, 0)
    # attention_map_color_ = attention_map_color(
    #     args, img, th_attn, attentions, args.save_dir, contour=True, alpha=1)

    # final_pic = Image.new(
    #     'RGB', (img_.size[1] * 2 + attns.size[0], img_.size[1]))
    # final_pic.paste(img_, (0, 0))
    # final_pic.paste(attention_map_color_, (img_.size[1], 0))
    # final_pic.paste(attns, (img_.size[1] * 2, 0))
    # final_pic.save(args.save_dir + "concate_image_attention_map.png")
    # display(final_pic)
    # plt.imshow(final_pic)

    # -----------------------------------------------------------------
    # 5 --- Computing Cosine Similarity (Anchor Images and batch Random Images)
    # -----------------------------------------------------------------

    # i = 2
    # topk = 5
    # anchor_i = 1
    # # data_path.image_files[anchor_i]
    # anchor_img = get_image(args, data_path.image_files[anchor_i])
    # image_ = anchor_img.view(
    #     [1, 3, args.image_size, args.image_size]).to(device)
    # patches_img = model.patch_embed(image_)
    # with torch.no_grad():
    #     anchor_embedding = model_seq(patches_img.to(device))
        
    #     anchor_embedding= anchor_embedding.view([anchor_embedding.size(1), anchor_embedding.size(2)])
    #     print(f"anchor_embedding shape: {anchor_embedding.shape}")
    # #anchor_embedding = None
    # plotting_image_retrieval(args, out_embedding, anchor_embedding, i, data_path, topk,
    #                          retrieval_from_exmaple=True, image_path=args.single_img_path, )
    # # Viusalization of similarity matrix.
    #plot_similarty_matrix(out_embedding, data_path)

    # ******************************************************************************
    # 5.2 Sparse Correspondence between two embedding (based on cosine similarity)
    # ******************************************************************************
    # anchor_i = 1
    # # data_path.image_files[anchor_i]
    # anchor_img = get_image(args, data_path.image_files[anchor_i])
    # image_ = anchor_img.view(
    #     [1, 3, args.image_size, args.image_size]).to(device)
    # patches_img = model.patch_embed(image_)
    # with torch.no_grad():
    #     anchor_embedding = model_seq(patches_img.to(device))
    #     #anchor_embedding= anchor_embedding.view([784, 768])
    # print(f"Single image embedding shape: {anchor_embedding.shape}")
    # # reference image embedding get from Demo dataset embedding
    # i = 2
    # reference_image = get_image(args, image_example_path=data_path.image_files[i])
    # reference_image = reference_image.view(
    #     [1, 3, args.image_size, args.image_size]).to(device)
    # #reference_embedding = out_embedding[i, :].unsqueeze(0)
    # patches_ref_img = model.patch_embed(image_)
    # with torch.no_grad():
    #     reference_embedding = model_seq(patches_ref_img.to(device))
    # print(f"Reference image embedding shape: {reference_embedding.shape}")
    # score, index = patches_similarity(anchor_embedding, reference_embedding)

   

    # # Plotting sparse correspondence between anchor image and reference image base on Top K


    # data_path.image_files[anchor_i]
    anchor_img = get_image(args,)
    image_ = anchor_img.view(
        [1, 3, args.image_size, args.image_size]).to(device)
    patches_img = model.patch_embed(image_)
    with torch.no_grad():
        anchor_embedding = model_seq(patches_img.to(device))
        #anchor_embedding= anchor_embedding.view([anchor_embedding.size(1), anchor_embedding.size(2)])
        print(f"anchor_embedding shape: {anchor_embedding.shape}")

    # Retrieving the topk most attention points of the anchor image.
    with torch.no_grad():
        attentions_out = model.get_last_selfattention(image_.to(device))
    print(f"The attention  shape: {attentions_out.shape}")
    nh = attentions_out.shape[1]  # number of head
    print(f"Number of attention head: {nh}")
    # Removing CLS token from the attention map
    attentions_out = attentions_out[:, :, 0, 1:].view(anchor_embedding.size(
        0), -1, args.image_size // args.patch_size, args.image_size // args.patch_size,)
    print(f"The attention  shape: {attentions_out.shape}")
    attentions_1 = nn.functional.interpolate(
        attentions_out, scale_factor=0.5, mode="nearest")
    print(f"The attention_1  shape: {attentions_1.shape}")
    ## Average all attention maps
    attentions_out=nn.functional.normalize(attentions_out, p=2, dim=1)
    attentions_1=nn.functional.normalize(attentions_1, p=2, dim=1)
    attentions = attentions_out.mean(1, keepdim=True)
    attentions1 = attentions_1.mean(1, keepdim=True)
    ## Keeping each attention map and normlaize


    # attentions=attentions_out.norm(dim=0, keepdim=True)
    # attentions1=attentions_1.norm(dim=0, keepdim=True)

    # the attention map USE directly without normalization
    # attentions = attentions_out
    # attentions1 = attentions_1
    anchor_embedding_= anchor_embedding.view([anchor_embedding.size(1), anchor_embedding.size(2)])
    print(f"The attention  shape: {attentions.shape, attentions1.shape}")
    score, image_id= image_retrieval_topk_extra(anchor_image_embedding=anchor_embedding_, source_data_embedding=out_embedding,topk=5)

    for _, id in enumerate(image_id): 
        reference_image=get_image(args, image_example_path=data_path.image_files[id])

        reference_image = reference_image.view(
            [1, 3, args.image_size, args.image_size]).to(device)
        #reference_embedding = out_embedding[i, :].unsqueeze(0)
        patches_ref_img = model.patch_embed(reference_image)
        with torch.no_grad():
            reference_embedding = model_seq(patches_ref_img.to(device))
        print(f"Reference image embedding shape: {reference_embedding.shape}")

        score, index = patches_similarity(anchor_embedding, reference_embedding)
        num_head=attentions.size(1)
        topk_ = 5
        for id, im1, im2, attention, attention1, in zip(index, image_, reference_image, attentions, attentions1):
            for nh in range(num_head):
                # Using the original attention map
                #attention = attention.view([6, 60, 60])
                attn = attention[nh].flatten()  # Original Attention Map
                # Using the attention map after interpolation
                #attention1 = attention1.view([6, 30, 30])
                attn1 = attention1[nh].flatten()  # Downsampled Attention Map
                st = attn1.topk(topk_)[1]

                point1 = st
                point2 = id[st]
                i1 = torchvision.utils.make_grid(
                    im1, normalize=True, scale_each=True)
                i1 = Image.fromarray(i1.mul(255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                i2 = torchvision.utils.make_grid(
                    im2, normalize=True, scale_each=True)
                i2 = Image.fromarray(i2.mul(255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                full_img = Image.new('RGB', (args.image_size * 2, args.image_size))
                draw = ImageDraw.Draw(full_img)
                unit = args.image_size // args.patch_size
                full_img.paste(i1, (0, 0))
                full_img.paste(i2, (args.image_size, 0))
                full_img.save(args.save_dir + "Image Similarity"+ "TopK"+ str(_)+"_.png")
                for p1, p2 in zip(point1, point2):
                    # p1y, p1x = p1 // unit + 0.5, p1 % unit + 0.5
                    # p2y, p2x = p2 // unit + 0.5, p2 % unit + 0.5
                    p1y, p1x = torch.div(
                        p1, unit, rounding_mode='trunc') + 0.5, p1 % unit + 0.5
                    p2y, p2x = torch.div(
                        p2, unit, rounding_mode='trunc') + 0.5, p2 % unit + 0.5

                    draw.line((p1x * args.patch_size,
                            p1y * args.patch_size,
                            p2x * args.patch_size + args.image_size,
                            p2y * args.patch_size), width=2, fill='red')
                    full_img.save(args.save_dir + "Image_Id"+ str(id[-5]) + "attention head_" +
                                str(nh) + "patch_sparse_.png")
