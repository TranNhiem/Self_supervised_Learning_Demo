import os
import math
import torch
import argparse
import numpy as np
from PIL import ImageFile, Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
import torch.backends.cudnn as cudnn

from torchvision import models as torchvision_models
from torchvision import transforms
import vision_transformer as vits
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from demo_dataloader import all_images_in_1_folder_dataloader, ImageFolderInstance
from patch_retrieval import load_model, batch_images
ImageFile.LOAD_TRUNCATED_IMAGES = True

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

torch.distributed.init_process_group(
    backend="nccl", init_method="tcp://localhost:23456", rank=0, world_size=1)


def concate_all_gather(tensor):
    '''
    gather all tensor from all gpus and concate them 
    '''
    tensor_list = [torch.empty_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list, tensor, async_op=False)

    tensor = torch.cat(tensor_list, dim=0)
    return tensor


def get_args_parser():

    parser = argparse.ArgumentParser(
        'Image & Patch-level Retrieval', add_help=False)

    # ********************************************************
    # Model parameters
    # ********************************************************
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_base_ibot_16', 'vit_L_16_ibot']
                        + torchvision_archs +
                        torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). """)
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--drop_path_rate', type=float,
                        default=0.1, help="stochastic depth rate")

    # ********************************************************
    # Cluster Parameters
    # ********************************************************

    parser.add_argument("--chunks", type=int, default=16, help="""Number of counting chunks. Set this larger (e.g., 128
        for DINO w/ 65536 out dim) when the model output dimension is large to avoid memory overflow.""")
    parser.add_argument('--num_pic_show', type=int,
                        default=10, help="Top-K images in the cluster for showing")
    parser.add_argument('--patch_window', type=int,
                        default=5, help="Top-K patches for each image")
    parser.add_argument('--batch_size', type=int,
                        default=16, help="Batch size for training")
    parser.add_argument('--topk', type=int,
                        default=196, help="Top-K patches for each image")
    parser.add_argument('--type', default="patch", type=str, choices=['cls', 'patch'], help="""wether to visualize
        patterns on patch level or cls level.""")

    # ********************************************************
    # Setting the Saving Experiments Result
    # ********************************************************
    parser.add_argument('--image_path', default="/data/downstream_datasets/coco/birdsnap__/dataset/train/", type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--dataloader_patches', default=False, type=bool,
                        help='Decided loading dataloader with or without Patches image')
    parser.add_argument('--subset_data', default=0.1, type=float,
                        help='How many percentage of your Demo Dataset you want to Use.')
    parser.add_argument('--single_img_path', default="/home/rick/offline_finetune/Pets/images/american_bulldog_72.jpg", type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Image_size resizes standard for all input images.')

    parser.add_argument('--output_dir', default="/data/downstream_tasks/DINO_weights/",
                        type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--save_dir', default="/data/downstream_tasks/visualization/Dino/",
                        type=str, help='Path to save Attention map out.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")

    return parser


parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
args = parser.parse_args()


def main(args, ref_patches_coordinate=None, user_select_patch_id=None):
    ## Making directory for saving results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Data loader for all images in one folder
    # val_data, data_path= batch_images(args)
    # n_train_points  = len(data_path.image_files)
    # Only execute 1 Time
    # if args.local_rank == 0:
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="tcp://localhost:23456", rank=0, world_size=1)
    # else:

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 2. Data loader for multiple folders
    cudnn.benchmark = True
    transform_ImageNet = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_dataset = ImageFolderInstance(
        args.image_path, transform=transform_ImageNet)

    n_train_points = len(val_dataset)
    train_data = list(range(n_train_points))
    test_set, train_set = train_test_split(
        train_data, test_size=args.subset_data, random_state=42)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=False,)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                sampler=val_sampler, pin_memory=True, num_workers=args.num_workers)

    # 3. ViTs Pre-trained Model
    model = load_model(args)
    model=model.to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model, )# device_ids=[args.local_rank]
    # ********************************************************
    # Create memory bank for all images in the dataset
    # ********************************************************
    try:
        data = torch.load(os.path.join(
            args.save_dir, f"memory_{args.type}.pth"))
        memory_bank = data['memory_bank']
        num_per_cluster = data.get('num_per_cluster', None)

    except:
        memory_bank = None
        num_per_cluster = None
        model.eval()
        val_sampler.set_epoch(0)

        for data in tqdm(val_dataloader,):  # desc="Creating Memory Bank"

            img, idx, _ = data
            idx = idx.to(device)
            imgs = img.to(device)
            print(f"Batch image shape {imgs.shape}")
            print(f"Batch idx shape {idx.shape}")
            print(f"remove {_.shape[0]} images")
            # for runing inference with torch.no_grad()
            # images_patches=model.patch_embed(imgs)
            embedding = model(imgs)[1].contiguous(
            ) if args.type == 'patch' else model(imgs)[0].contiguous()
            all_embedding = concate_all_gather(embedding).detach().cpu()
            print(f"all_embedding.shape {all_embedding.shape}")
            idx = concate_all_gather(idx)

            if memory_bank is None:
                print("Initializing memory_bank {} points.".format(n_train_points))

                memory_bank = torch.zeros(n_train_points, all_embedding.size(
                    0), 2) if args.type == 'patch' else torch.zeros(n_train_points, 2)
                memory_bank = memory_bank.to("cpu").detach()

            with torch.no_grad():
                memory_bank[idx] = torch.stack(all_embedding.max(-1), dim=-1)
        torch.save({'memory_bank': memory_bank, 'num_per_cluster': num_per_cluster},
                   os.path.join(args.save_dir, f"memory_{args.type}.pth"))

    if num_per_cluster is None and args.local_rank == 0:

        num_per_cluster = torch.Tensor([])
        all_dim = torch.arange(args.out_dim).chunk(args.chunks)
        for i in tqdm(all_dim):
            mask = memory_bank[..., 1, None] == i.view(1, 1, -1)
            # consider dim=0 and dim=1
            num_per_cluster = torch.cat(
                (num_per_cluster, mask.sum((0, 1))), dim=0)
        torch.save(
            {'memory_bank': memory_bank,
             'num_per_cluster': num_per_cluster},
            os.path.join(args.save_dir, f'memory_{args.type}.pth'),
        )

    if args.local_rank == 0:
        patterns = {}
        for i in num_per_cluster.topk(args.num_pic_show)[1]:
            mask = memory_bank[..., 1] == i
            if args.type == 'patch':
                # Retrieving the image-level embedding
                values, spatial_id = (memory_bank[..., 0]*mask).max(dim=-1)

                # Retrieving topk most simmilar Patches for each image
                values, instance_id = torch.topk(
                    values, k=args.topk*2, )  # dim=-1
                spatial_id = spatial_id[instance_id]
                npatch = args.image_size//args.patch_size
                height_id = torch.div(spatial_id, npatch, rounding_mode='trunc') #spatial_id // npatch
                width_id = spatial_id % npatch
                indices = torch.stack(
                    (instance_id, height_id, width_id), dim=-1)
            else:
                values, indices = torch.topk(
                    (memory_bank[..., 0] * mask), args.topk)

            patterns[i.item()] = indices

        transforms_image = transforms.Compose([
            transforms.Resize(args.image_size // 7 * 8),
            transforms.CenterCrop(args.image_size),
        ])

        train_dataset = ImageFolderInstance(
            args.image_path, transform=transforms_image)

        for nrank, (cluster, idxs) in enumerate(patterns.items()):
            size = math.ceil(args.topk**0.5)  # 6
            unit = args.patch_size if args.type == 'patch' else args.img_size  # 16 /224
            # how many patches for visualization
            # Play with Visualization Unit for full resolution of image
            # Image size = patch_size * patch_size = unit* args.patch_window(patch_size)
            # 80 /224
            vis_unit = (
                unit*args.patch_window) if args.type == 'patch' else unit

            img = Image.new('RGB', (size*vis_unit, size*vis_unit))

            i = 0
            for idx in idxs.numpy():
                if args.type == 'patch':

                    raw, _, _ = train_dataset[idx[0]]
                    
                    #print(f"raw shape {raw.shape}")

                    data = raw.crop((
                        (idx[2] - args.patch_window // 2) * unit,
                        (idx[1] - args.patch_window // 2) * unit,
                        (idx[2] + args.patch_window // 2 + 1) * unit,
                        (idx[1] + args.patch_window // 2 + 1) * unit))

                    # filter too dark patch for visualization
                    hsv = np.array(data.convert('HSV'))
                    if hsv[..., -1].mean() <= 40:
                        continue

                    if ref_patches_coordinate is not None and user_select_patch_id is not None:
                        draw = ImageDraw.Draw(data, "RGBA")
                        for patch_id in user_select_patch_id:
                            x_y_coord = ref_patches_coordinate[patch_id]
                            x0, y0 = x_y_coord[0], x_y_coord[1]
                            x1, y1 = (x0-args.patch_size), (y0-args.patch_size)
                            shape = ((x0, y0), (x1, y1))
                            draw.rectangle(shape, fill=(200, 100, 0, 127))
                    # draw highlight region with Given Patch_Window
                    else:
                        if args.patch_window > 1:
                            draw = ImageDraw.Draw(data, "RGBA")
                            draw.rectangle((
                                args.patch_window // 2 * unit,
                                args.patch_window // 2 * unit,
                                (args.patch_window // 2 + 1) * unit,
                                (args.patch_window // 2 + 1) * unit),
                                fill=(200, 100, 0, 127))
                else:
                    _, data, _ = train_dataset[idx]

                img.paste(data, (i % size * vis_unit, i // size * vis_unit))

                i += 1
                if i >= args.topk:
                    break
            
            img.save(os.path.join(args.save_dir, 'c{}_crank{}_cid{}_top{}.jpg'.format(
                args.type, nrank, cluster, args.topk)))
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
