
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

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from torchvision import io
from torchvision import transforms as pth_transforms
from torchvision.transforms.functional import to_pil_image


# **********************************************
# Helper function for image Retrieval
# **********************************************
'''Compute the average precision (AP) of a ranked list of images'''
def compute_ap(ranks, positive_sample):
    '''
    args: 
    ranks: zerro-based ranks of positive images
    positive_sample: number of positive images
    return: average precision
    '''
    # Compute the average precision (AP) from the ranks
    ap = 0.0
    # number of images ranked by the system
    nimgranks = len(ranks)

    recall_step = 1. / positive_sample
    for i in np.arange(nimgranks):
        rank = ranks[i]
        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(i) / rank

        precision_1 = float(i+1) / (rank+1)
        ap += (precision_0+precision_1)*recall_step/2.
    return ap
'''
The function computes the mAP for given set of returned results
    Usage: 
    map = compute_map (ranks, gnd)
    computes mean average precsion (map) only
    map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
'''
def compute_map(ranks, gnd, kappas=[]):
    '''
    args: 
        ranks: is the GPUs  
        gnd: ground truth
        kappas: If there are no positive images for some query, that query is excluded from the evaluation
    return: 
        map: mean average precision
        aps: average precision at each query
        pr: mean precision at kappas
        prs: precision at kappas at each query
    '''
    # Compute the average precision (AP) from the ranks
    nqueries = len(gnd)  # number of queries
    aps = np.zeros(nqueries)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nqueries, len(kappas)))
    nempty = 0
    map = 0.0
    for i in np.arange(nqueries):
        qgnd = np.array(gnd[i]['ok'])
        # No positive images, skip this query
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndi = np.array(gnd[i]['junk'])
        except:

            qgndi = np.empty(0)
        # Sort positions of positive and junk images (0-based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndi)]
        k = 0
        ij = 0
        if len(junk):
            # Decrease positions of positive based on the number of junk images appearing before them
            ip = 0
            while ip < len(pos):
                while (ij < len(junk) and junk[ij] < pos[ip]):
                    ij += 1
                    k += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # Compute precision @ k
        pos += 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nqueries - nempty)
    pr = pr / (nqueries - nempty)

    return map, aps, pr, prs



'''Function helps to plot the similarity matrix'''
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


def plotting_image_retrieval(args,anchor_image, score,idx, val_dat ):
    
    resize = pth_transforms.Resize((args.image_size, args.image_size,))
    plt.imshow(anchor_image)
    ig, axs = plt.subplots(1, len(idx), figsize=(12, 5))
    for i_, score, ax in zip(idx, score, axs):
        img = to_pil_image(
            resize(io.read_image(val_dat.image_files[i_])))
        ax.imshow(img)
        ax.set_title(f"{score:.4f}")
    plt.show()



