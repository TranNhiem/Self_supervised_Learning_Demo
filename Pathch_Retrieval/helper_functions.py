
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

from array import array
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import torch.nn as nn
from PIL import Image
from torchvision.utils import ImageDraw
from torchvision import io, transforms
from torchvision import transforms as pth_transforms
from torchvision.transforms.functional import to_pil_image
import seaborn as sns
import cv2
import torchvision
from IPython.display import display

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


def plotting_image_retrieval(args, anchor_image, score, idx, val_dat):

    resize = pth_transforms.Resize((args.image_size, args.image_size,))
    plt.imshow(anchor_image)
    ig, axs = plt.subplots(1, len(idx), figsize=(12, 5))
    for i_, score, ax in zip(idx, score, axs):
        img = to_pil_image(
            resize(io.read_image(val_dat.image_files[i_])))
        ax.imshow(img)
        ax.set_title(f"{score:.4f}")
    plt.show()


def visualization_patches_image(args, query_image: torch.Tensor, figure_size: tuple = (4, 6), my_dpi: float = 300., axis_font_size: int = 5, patch_number_font_size: int = 6, show_image: bool = False):
    '''
    Args:
        query_image: is the query image
        my_dpi: is the dpi of the image
        axis_font_size: is the font size of the axis
    Return: 
        patch_ids and its coordinates    
    '''
    query_image = query_image.permute(1, 2, 0).cpu().numpy()
    # Set up figure
    plt.rc('xtick', labelsize=axis_font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=axis_font_size)
    # fig=plt.figure(figsize=(float(image.size[0])/my_dpi,float(image.size[1])/my_dpi),dpi=my_dpi)
    fig = plt.figure(figsize=figure_size, dpi=my_dpi)
    ax = fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Set the gridding interval: here we use the major tick interval
    myInterval = args.patch_size
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-',)

    # Add the image
    ax.imshow(query_image)

    # Find number of gridsquares in x and y direction
    nx = abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(myInterval)))
    ny = abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(myInterval)))
    patch_dic = {}
    for j in range(ny):
        y = myInterval+j*myInterval
        #print(f"this is Y axis value {x}")
        for i in range(nx):
            x = myInterval+float(i)*myInterval
            #print(f"this is X axis value {x}")
            key = i+j*nx
            if key not in patch_dic:
                patch_dic.update({key: (x, y)})

    # Add some labels to the gridsquares
    for j in range(ny):
        y = myInterval/2+j*myInterval
        for i in range(nx):
            x = myInterval/2.+float(i)*myInterval
            ax.text(x, y, '{:d}'.format(i+j*nx), color='w',
                    ha='center', va='center',  size=patch_number_font_size)
    # Save the figure
    fig.savefig(args.save_dir + 'my_patches_image.jpg', dpi=my_dpi)
    if show_image:
        plt.show()
    else:
        plt.close()
    return patch_dic


def seanborn_heatmap_color(heat_map_color_array: tuple = (255, 255), show_color_map: bool = False, cmap: str = "bwr"):
    color = np.linspace(
        0, heat_map_color_array[0], heat_map_color_array[1], endpoint=True)
    data = np.array([color, color])
    ax = sns.heatmap(data, center=0, cmap=cmap, robust=False)
    im = ax.collections[0]
    if show_color_map:
        #plt.imshow(data, cmap=cmap)
        plt.show()
    else:
        plt.close()  # uncomment to disable display Image

    rgba_values = im.cmap(im.norm(im.get_array()))
    rgba_values = np.flip(rgba_values)
    print(f"The color heatmap generated in range {rgba_values.shape}")
    return rgba_values


def plotting_patch_level_retrieval(args, query_img: torch.Tensor, reference_image_id: str, ref_patches_coordinate: dict,
                                   user_patch_id: list, score: list, idx: list,
                                   mask_color: array, val_dat: list, alpha: float = 0.6, show_image: bool = False, save_name: str = "final_patch_level_matching"):
    '''
    Args: 
        query_img: is the query image in tensor format (1, channel, height, width) # 
        ref_patches_coordinate: dictionary contains all patches coordinates
        user_patch_id: list of user selected patch id
        idx: list of topk patch_ids similar to user selected patch_id
        score: list of topk patch_ids similarity with user selected patch_id score 
        mask_color: color of the mask
        val_dat: path for the query and reference image plot
        alpha: alpha value for the mask intensity appearance

    '''

    # For each list idx of single image reference
    if 0 < len(user_patch_id) <= 1:
        print("User only provide single patch-id")
        image_path = val_dat.image_files[reference_image_id]
        ref_image = cv2.imread(image_path)
        ref_image = cv2.resize(
            ref_image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        shapes = np.zeros_like(ref_image, np.uint8)
        alpha = alpha
        j = 0
        for patch_id in idx:
            x_y_coord = ref_patches_coordinate[patch_id]
            x0, y0 = int(x_y_coord[0]), int(x_y_coord[1])
            x1, y1 = int(x0-args.patch_size), int(y0-args.patch_size)
            #shape_=((x0, y0), (x1, y1))
            cv2.rectangle(shapes, (x0, y0), (x1, y1), tuple(
                [int(i*255) for i in mask_color[j]]), cv2.FILLED)
            mask = shapes.astype(bool)
            out = ref_image.copy()
            out[mask] = cv2.addWeighted(
                ref_image, alpha, shapes, 1-alpha, 0)[mask]
            if len(idx) <= 10:
                j += 8
            elif 10 < len(idx) < 50:
                j += 10
            elif 100 >= len(idx) >= 50:
                j += 4
            else:
                j += 1
        cv2.imwrite(args.save_dir + save_name + 'refer_img_only.jpg', out)

    else:
        print("User  provide multiple patches-id")
        for i_, patches_id in enumerate(idx):
            if i_ == 0:
                path = val_dat.image_files[reference_image_id]
            else:
                path = new_path

            ref_image = cv2.imread(path)
            ref_image = cv2.resize(
                ref_image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
            shapes = np.zeros_like(ref_image, np.uint8)
            alpha = alpha

            j = 0
            for patch_id in patches_id:
                x_y_coord = ref_patches_coordinate[patch_id]
                x0, y0 = int(x_y_coord[0]), int(x_y_coord[1])
                x1, y1 = int(x0-args.patch_size), int(y0-args.patch_size)
                #shape_=((x0, y0), (x1, y1))

                cv2.rectangle(shapes, (x0, y0), (x1, y1), tuple(
                    [int(i*255) for i in mask_color[j]]), cv2.FILLED)
                mask = shapes.astype(bool)
                out = ref_image.copy()
                out[mask] = cv2.addWeighted(
                    ref_image, alpha, shapes, 1-alpha, 0)[mask]
                if len(patches_id) <= 10:
                    j += 8
                elif 10 < len(patches_id) < 50:
                    j += 10
                elif 100 >= len(patches_id) >= 50:
                    j += 4
                else:
                    j += 1
            cv2.imwrite(args.save_dir + save_name + 'refer_img_only.jpg', out)
            new_path = args.save_dir + save_name + 'refer_img_only.jpg'

    image_result = args.save_dir + save_name + 'refer_img_only.jpg'
    with open(image_result, 'rb') as f:
        img_result = Image.open(f)
        img_result = img_result.convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(size=(args.image_size, args.image_size)),
        pth_transforms.ToTensor(),
        #pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_result = transform(img_result)
    img_result = img_result.unsqueeze(0)
    result = torchvision.utils.make_grid(
        img_result, normalize=True, scale_each=True)
    result = Image.fromarray(result.mul(255).add_(0.5).clamp_(
        0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    full_img = Image.new('RGB', (args.image_size * 2, args.image_size))
    draw = ImageDraw.Draw(full_img)
    i1 = torchvision.utils.make_grid(
        query_img, normalize=True, scale_each=True)
    i1 = Image.fromarray(i1.mul(255).add_(0.5).clamp_(
        0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    draw_img1 = ImageDraw.Draw(i1, "RGBA")

    for patch_id in user_patch_id:  # User input Patch ID
        x_y_coord = ref_patches_coordinate[patch_id]
        x0, y0 = x_y_coord[0], x_y_coord[1]
        x1, y1 = (x0-16.0), (y0-16.0)
        shape = ((x0, y0), (x1, y1))
        draw_img1.rectangle(shape, fill=(200, 100, 0, 127))
        full_img.paste(i1, (0, 0))

    full_img.paste(result, (args.image_size, 0))
    full_img.save(args.save_dir + save_name + '.jpg')
    # display(full_img)
    if show_image:
        final_image = args.save_dir + save_name + '.jpg'
        fig = plt.figure(figsize=(5, 8))
        plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=5)
        with open(final_image, 'rb') as f:
            final_result = Image.open(f)
            final_result = final_result.convert('RGB')
        plt.imshow(final_result)
        plt.show()
