
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
 