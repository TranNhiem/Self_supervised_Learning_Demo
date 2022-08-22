import clip
import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

from collections import OrderedDict
import torch
import torchvision
from torchvision.utils import save_image
import pickle
from barbar import Bar


def select_backbone(backbone="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(backbone, device=device)
    return model, preprocess


def encode_imgs(ld, model):
    norm_vec = lambda feat : feat / feat.norm(dim=-1, keepdim=True)
    emb_lst = []
    print("encoding..")
    for im, _ in ld:
        with torch.no_grad():
            #breakpoint()
            embs = norm_vec( model.encode_image(im.to('cuda:0')) )  # 200 ok!
        emb_lst.append(embs.cpu())
    
    embs = torch.cat(emb_lst)

    breakpoint()
    with open('./stl_emb_ViTxx.pickle', 'wb') as f:
        pickle.dump(embs.cpu(), f)


def save_preproc_im(dataset):
    print("saving preprocessing images..\n")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=400, pin_memory=True, num_workers=12)
    tnsr_lst = []

    for im, _ in Bar(dataloader):
        tnsr_lst.append(im)
    
    breakpoint()
    image = torch.cat(tnsr_lst)
    with open('./imgnet_clnIms.pickle', 'wb') as f:
        pickle.dump(image.cpu(), f)


def save_ims_from_ds(dataset):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, pin_memory=True, shuffle=True, num_workers=12)
    tnsr_lst = []
    ims, _ = next(iter(dataloader))
    
    for idx, im in enumerate(ims):
        save_image(im, f"stl_{idx}.jpg")
    

if __name__ == "__main__":
    model, preprocess = select_backbone(backbone="ViT-L/14")  # 

    stl_ds = torchvision.datasets.STL10(root = "../data/", transform=torchvision.transforms.ToTensor())
    #preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize([224, 224]), torchvision.transforms.ToTensor()])

    #imnt_ds = torchvision.datasets.ImageFolder("/data/val", transform=preprocess)
    #save_preproc_im(imnt_ds)

    save_ims_from_ds(stl_ds)

    #dataloader = torch.utils.data.DataLoader(dataset=stl_ds, batch_size=len(stl_ds), pin_memory=True)
    #dataloader = torch.utils.data.DataLoader(dataset=imnt_ds, batch_size=100, pin_memory=True)
    #encode_imgs(dataloader, model)
