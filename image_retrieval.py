
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

import torch 
import torch.nn as nn
# Get the torch hub model
from hubvits_models import dino_vitb8



'''Image Retrieval for the ImageNet Dataset'''
def image_retrieval(image_path, model, num_patches=100, patch_size=32, stride=32, device='cpu'):
    '''
    image_path: path to the image
    model: the model to use for inference
    num_patches: number of patches to extract
    patch_size: size of the patches
    stride: stride of the patches
    device: the device to use for inference
    '''
    # Load the image
    image = Image.open(image_path)
    # Resize the image
    image = image.resize((image.size[0] // stride, image.size[1] // stride))
    # Get the patches
    patches = image_utils.extract_patches(image, patch_size, stride)
    # Convert the patches to tensors
    patches = torch.from_numpy(patches).to(device)
    # Get the predictions
    with torch.no_grad():
        predictions = model(patches)
    # Get the top-k predictions
    top_k = predictions.topk(num_patches, dim=1)[1]
    # Get the top-k patches
    top_k_patches = patches[top_k]
    # Convert the top-k patches to numpy
    top_k_patches = top_k_patches.cpu().numpy()
    # Resize the top-k patches
    top_k_patches = np.reshape(top_k_patches, (top_k_patches.shape[0], top_k_patches.shape[1], top_k_patches.shape[2] * top_k_patches.shape[3]))
    # Resize the top-k patches
    top_k_patches = np.reshape(top_k_patches, (top_k_patches.shape[0], top_k_patches.shape[1], top_k_patches.shape[2] // stride, top_k_patches.shape[3] // stride))
    # Resize the top-k patches
    top_k_patches =_