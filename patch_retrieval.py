import os 
import torch
from utils import load_pretrained_weights
import vision_transformer as vits
import argparse



#******************************************************
# Arguments needed to load model checkpoint 
#******************************************************

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser(): 
    parser= argparse.ArgumentParser('Patch-Match Retrieval', add_help=False)
    #********************************************************
    # Model parameters
    #********************************************************
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base',] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
        
    #********************************************************
    ## Setting the Saving Experiments Result 
    #********************************************************
    parser.add_argument('--data_path', default='/data1/1K_New/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/data1/solo_MASSL_ckpt/mvar/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    return parser
    
parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
args = parser.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

def load_model(args): 

    #if the network is Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys(): 
        model = vits.__dict__[args.arch](
            patch_size= args.patch_size , 
            drop_path_rate=args.drop_path_rate, # stochastic depth
            )
        embed_dim= model.embed_dim
    elif "torch_hub" in args.arch: 
        
    elif args.arch in torchvision_models.__dict__.keys(): 
        model= torchvision_models.__dict__[args.arch]() 
    else: 
        print(f"Unknow architecture: {args.arch}")