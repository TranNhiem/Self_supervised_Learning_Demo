import os 
import sys 
import time 
import math 
from collections import defaultdict, deque
import numpy as np 
import torch 

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size): 
    
    '''
    model: ViT architecutre design with random itit Weight  
    pretrained_weights: The path of pretrained weights from your local machine 
    checkpoint_key:  If specific layer neeed loading check point ?? 
    model_name: provide name to loading checkpoint from MetaAI hub checkpoint if pretrained_weights is not provided
    patch_size: this argument provide the patch_size of pretrained model need to load

    '''
    
    if os.path.isfile(pretrained_weights): 
        state_dict= torch.load(pretrained_weights, map_location='cpu')
        if checkpoint_key is not None and checkpoint_key in state_dict: 
            print(f'take key {checkpoint_key} in provided checkpoint dict')
            state_dict= state_key[checkpoint_key]
        ## remove 'Module' prefix 
        state_dict= {k.replace("module.", ""): v for k,v in state_dict.items()}
        # remove 'backbone' prefix induced by multicrop wrapper 
        state_dict= {k.replace("backbone.", ""): v for k,v in state_dict.items()}
        msg= model.load_state_dict(state_dict, strict=False)
        print("Pretrained weight found at {}".format(pretrained_weights, msg))

    else: 
        print("please use the '--pretrained weights', argument to indicate the path of the checkpoint to evaluate.")
        url= None 

        if model_name =="vit_small" and patch_size==16: 
            url="dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None: 
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

def load_pretrained_linear_weights(linear_classifier, model_name, patch_size): 
    url= None 
    if model_name == "vit_small" and patch_size==16: 
         url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")
