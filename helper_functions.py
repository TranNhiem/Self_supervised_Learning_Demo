
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


# **********************************************
# Helper Visualization Attention Map and Images
# **********************************************


def attention_heatmap(args,  attention_input, img,):
    '''
    Args: 
    image: the input image tensor (3, h, w)
    patch_size: the image will patches into multiple patches (patch_size, patch_size)
    threshold: to a certain percentage of the mass 
    attention_input: the attention output from VIT model (Usually from the last attention block of the ViT architecture)

    '''
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - \
        img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    print(f"image after patching shape : {img.shape}")

    w_featmap = int(img.shape[-2] // args.patch_size)
    h_featmap = int(img.shape[-1] // args.patch_size)
    print(f"w_featmap size of : {w_featmap, h_featmap}")
    # Number of head
    nh = int(attention_input.shape[1])
    # We only keep the output Patch attention -> #Removing CLS token
    attentions = attention_input[0, :, 0, 1:].reshape(nh, -1)
    print(f"attentions shape : {attentions.shape}")
    # Reshape the attention score to resmeble mini patches
    attentions = attentions.reshape(nh, w_featmap, h_featmap).float()
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    print(f"attention shape after interpolation : {attentions.shape}")
    return attentions
    #  ploting multiple attention maps from different attention heads.
    # n_row= int(nh / 2)
    # fig, axes = plt.subplots(nrows=n_row, ncols=2, figsize=(10, 10))
    # idx = 0
    # for i in range(n_row):
    #     for j in range(2):
    #         if idx < nh:
    #             axes[i, j].imshow(img[0])
    #             axes[i, j].imshow(attentions[..., idx],
    #                               cmap="inferno", alpha=0.6)
    #             axes[i, j].title.set_text(f"Attention head: {idx}")
    #             axes[i, j].axis("off")
    #             idx += 1
