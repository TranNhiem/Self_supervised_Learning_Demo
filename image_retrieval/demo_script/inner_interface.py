import clip
import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch
import torchvision
import pickle

def retriv_by_phase(qurey_phases, key_images, clear_images, model, top_k=5):

    def _phase_chk(qurey_phases):
        if len(qurey_phases) > model.context_length:
            raise ValueError(f'The query list is too long, it should be less then {model.context_length} phase.')

        for phase in qurey_phases:
            if len(phase) > model.vocab_size:
                raise ValueError('The content is too long of the phase')

    phases_token = clip.tokenize(qurey_phases).cuda()
    norm_vec = lambda feat : feat / feat.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        query_features = norm_vec( model.encode_text(phases_token).float() )
        query_features = query_features.type(key_images.dtype)

        similarity = query_features @ key_images.T
        _, indices = similarity.topk(top_k)
    
        topk_ims = clear_images[indices]

    # post-proc for torch2numpy 
    topk_ims = [ topk_im.permute(0, 2, 3, 1).cpu().numpy() for topk_im in topk_ims]
    return topk_ims[0] 


def retriv_by_img(query_img, key_features, clear_images, model, preprocess, top_k=5):
    norm_vec = lambda feat : feat / feat.norm(dim=-1, keepdim=True)
    
    # place cuda device :
    query_img = query_img.cuda()
    key_features = key_features.cuda()

    with torch.no_grad():
        query_features = norm_vec( model.encode_image(query_img).float() )
        query_features = query_features.type(key_features.dtype)

        similarity = query_features @ key_features.T
        _, indices = similarity.topk(top_k)
        breakpoint()
        topk_ims = clear_images[indices]
        
    topk_ims = [ topk_im.permute(0, 2, 3, 1).cpu().numpy() for topk_im in topk_ims]
    
    return topk_ims[0]  # return the matched image of first phase..
    

def select_backbone(backbone="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(backbone, device=device)
    return model, preprocess


def __get_mod_info(model):
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters : ", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution : ", input_resolution)
    print("Context length : ", context_length)
    print("Vocab size : ", vocab_size)


if __name__ == "__main__":
    model, preprocess = select_backbone(backbone="ViT-L/14")
    breakpoint()
    
    #topk_im = retriv_by_img(key_images[0].unsqueeze(0).cuda(), key_images.cuda(), model, preprocess)
    #print(topk_im[0].shape)


    #transform=preprocess, 
    #stl_ds = torchvision.datasets.STL10(root = "../data/", transform=torchvision.transforms.ToTensor(), download = True)
    #imgnet = torchvision.datasets.ImageFolder("/data/train", transform=torchvision.transforms.ToTensor())
    #dataloader = torch.utils.data.DataLoader(dataset=imgnet, batch_size=12, pin_memory=True, shuffle=True)  # len(stl_ds)
    
    #encode_imgs(image, model)

    #key_images = image.cuda()
    #query_images, _ = next(iter(dataloader))
    #query_image = query_images[0]

    #qurey_phases = [f"This is a photo of a {label}" for label in stl_ds.classes]
    
    #topk_im = retriv_by_phase(qurey_phases, key_images, model)
    #topk_im = retriv_by_img(query_image.unsqueeze(0).cuda(), key_images, model)
    #print(topk_im[0].shape)