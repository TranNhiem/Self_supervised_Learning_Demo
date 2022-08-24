# For Visual Attention Map
import os 
import colorsys
import random
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import cv2
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import skimage.io
import numpy as np 
from PIL import Image

# ******************************************************
# Visualization Attention Map Functions
# ******************************************************

company_colors = [
    (0,160,215), # blue
    (220,55,60), # red
    (245,180,0), # yellow
    (10,120,190), # navy
    (40,150,100), # green
    (135,75,145), # purple
]

company_colors = [(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0) for c in company_colors]

# Create the transparence mask
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * \
            (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

# Creat Apply Mask 2 
def apply_mask2(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    t= 0.2
    mi = np.min(mask)
    ma = np.max(mask)
    mask = (mask - mi) / (ma - mi)
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * np.sqrt(mask) * (mask>t))+ alpha * np.sqrt(mask) * (mask>t) * color[c] * 255
    
    return image

# Create the random Color for the mask
def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname='test', figsize=(5, 5), blur=False, contour=True, alpha=0.5, visualize_each_head=False):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()
    if visualize_each_head:
        N = 1
        mask = mask[None, :, :]
    else: 
        # Change this number corresponding the number of attention heads
        N=6
        #mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)
    # Show area outside image boudaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]

        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # substract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor='none', edgecolor=color)
                ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8), cmap="inferno", aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

def attention_retrieving(args, img, threshold, attention_input, save_dir, blur=False, contour=True, alpha=0.5, visualize_each_head=True):
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
    image = img
    print(f"image after patching shape : {image.shape}")

    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size
    print(f"w_featmap size of : {w_featmap}")
    # Number of head
    nh = attention_input.shape[1]

    # We only keep the output Patch attention#Removing CLS token
    attentions = attention_input[0, :, 0, 1:].reshape(nh, -1)

    print(f"This is the Shape attentions using CLS Token : {attentions.shape}")
    th_attn=None
    if threshold is not None:
        # Keeping only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1-threshold)

        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]

        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # Interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(
            0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # Saving attention heatmaps
    os.makedirs(save_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(
        image, normalize=True, scale_each=True), os.path.join(save_dir, "attention_visual_.png"))
    
    attns = Image.new('RGB', (attentions.shape[2] * nh, attentions.shape[1]))
    img_= Image.open(os.path.join(save_dir, "attention_visual_.png"))
    for j in range(nh):
        fname = os.path.join(save_dir, "attn-head_" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f'{fname} saved.')
        attns.paste(Image.open(fname),(j*attentions.shape[2], 0))

    if threshold is not None:
        image = skimage.io.imread(os.path.join(
            save_dir, "attention_visual_.png"))
        if visualize_each_head:
            for j in range(nh):
                display_instances(image, th_attn[j], fname=os.path.join(
                    save_dir, "mask_th" + str(threshold) + "_head" + str(j) + '.png'), blur=blur, contour=contour, alpha=alpha, visualize_each_head=visualize_each_head)
        else: 
            display_instances(image, th_attn, fname=os.path.join(
                save_dir, "mask_th" + str(threshold) + "all_head" + '.png'), blur=blur, contour=contour, alpha=alpha, visualize_each_head=visualize_each_head)
    
    return attentions, th_attn, img_, attns

def attention_map_color(args, image, th_attn, attention_image, save_dir, blur=False, contour=False, alpha=0.5): 
    M= image.max()
    m= image.min() 
    
    span=64 
    image =  ((image-m)/(M-m))*span + (256 -span)
    image = image.mean(axis=2)
    image= np.repeat(image[:, :, np.newaxis], 3, axis=2)
    print(f"this is image shape: {image.shape}")

    att_head= attention_image.shape[0]

    for j in range(att_head):
        m = attention_image[j]
        m *= th_attn[j]
        attention_image[j]= m 
    mask = np.stack([attention_image[j] for j in range(att_head)])
    print(f"this is mask shape : {mask.shape}")
    
    figsize = tuple([i / 100 for i in (args.image_size, args.image_size,)])
    fig = plt.figure(figsize=figsize, frameon=False, dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    if len(mask.shape) == 3:
        N = mask.shape[0]
        print(f"this is N : {N}")
    else:
        N = 1
        mask = mask[None, :, :]

    for i in range(N):
        mask[i] = mask[i] * ( mask[i] == np.amax(mask, axis=0))
    a = np.cumsum(mask, axis=0)
    for i in range(N):
        mask[i] = mask[i] * (mask[i] == a[i])
    if N > 6:  
        N=6 
    colors = company_colors[:N]

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    image=image.numpy()
    masked_image = 0.1*image.astype(np.uint32).copy()
    print(f"this is masked image shape : {masked_image.shape}")
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask2(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros(
                (_mask.shape[0] + 2, _mask.shape[1] + 2))#, dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    ax.axis('image')
    #fname = os.path.join(output_dir, 'bnw-{:04d}'.format(imid))
    fname = os.path.join(save_dir, "attn_color.png")
    fig.savefig(fname)
    attn_color = Image.open(fname)

    return attn_color
