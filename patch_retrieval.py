import os 
import torch
from utils import load_pretrained_weights, normal_dataloader
import torch.nn as nn
import vision_transformer as vits
import argparse
import tqdm
from torchvision import models as torchvision_models 
from pathlib import Path
from PIL import Image
from torchvision import transforms as pth_transforms
import matplotlib.pyplot as plt

### Get the torch hub model 
from hubvits_models import dino_vitb8

### Visualization and Plot the Image 
# from torchvision.utils import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

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
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    #********************************************************
    ## Setting the Saving Experiments Result 
    #********************************************************
    parser.add_argument('--image_path', default="/img_data/Patch_level_matching_pets_dataset/images/", type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--subset_data', default=0.1, type=float,
        help='How many percentage of your Demo Dataset you want to Use.')
    parser.add_argument('--single_img_path', default=None, type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--image_size', default=224, type=int,
        help='Image_size resizes standard for all input images.')
            
    parser.add_argument('--output_dir', default="/data1/solo_MASSL_ckpt/mvar/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    return parser
    

def load_model(args): 
    '''
    Following arguments using 
    args.arch --> this will loading the ViT architecture
    args.patch_size --> This helps specified input_shape
    args.
    '''

    #if the network is Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys(): 
        model = vits.__dict__[args.arch](
            patch_size= args.patch_size , 
            drop_path_rate=args.drop_path_rate,) # stochastic depth)
        embed_dim= model.embed_dim
        print(f"model embedding shape {embed_dim}")
        model= model.cuda()
        pretrained_model =load_pretrained_weights(model, pretrained_weights="None", checkpoint_key=None, model_name=args.arch, patch_size=args.patch_size)
    
    elif args.arch in torchvision_models.__dict__.keys(): 
        model= torchvision_models.__dict__[args.arch]() 
        pretrained_model =load_pretrained_weights(model, pretrained_weights="None", checkpoint_key=None, model_name=args.arch, patch_size=args.patch_size)

    else: 
        print(f"Unknow architecture: {args.arch}") 
    
    return pretrained_model 


#******************************************************
# Loading Image --> Preprocessing Steps
#******************************************************


# Get Batch of all images from Demo dataset
def batch_images(args): 
    
    '''
    Args
    1: Image_path ->  dataset folder structure is Containing all images inside.
    2: image_format -> The format of image (because of using *glob* function to get all image Name) Default ("*.jpg")
    3: img_resize ->  resize all image the same size
    4: Subset of Demo Dataset -> Using only Proportion of Dataset
    5: transform_ImageNet --> Normalize the image with ImageNet (mean, std)

    '''
    val_dataset= normal_dataloader(image_path=args.image_path,img_size=args.image_size, 
                                    image_format= "*.jpg", subset_data= args.subset_data, transform_ImageNet=True )
    
    val_dl=val_dataset.val_dataloader()
    
    return val_dl

# Get a Single image 
def get_image(args):
    '''
    Args:  
    args.image_path: This provides the image in your machine or other source
    args.image_resize: This resizes image to expected size
    '''
    if args.single_img_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, Using Default Pets Images ")
        default_image='/img_data/Patch_level_matching_pets_dataset/images/miniature_pinscher_153.jpg'
        with open(default_image, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    elif os.path.isfile(args.single_img_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)

    transform = pth_transforms.Compose([
            pth_transforms.Resize((args.image_size, args.image_size)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    img_tensor = transform(img)

    return img_tensor


def top_k_cos_similarity(ancher_image_patch, other_patches, k=4, normalize=True ): 
    if normalize: 
        ancher_patch=ancher_patch_embeddings.norm(dim=-1, keepdim=True)
        other_patches= other_patches.norm(dim=-1, keepdim=True)
    #similarity= ancher_patch.cpu().numpy() @ other_patches.cpu().numpy().T
    similarity= (ancher_patch @ other_patches.T)
    similarity_top_k= similarity.topk(k, dim=-1)

def image_retrieval_topk(embedding: torch.FloatTensor, i: int, topk=4): 
    similarity = embedding @ embedding[i, :].T
    scores, idx= similarity.topk(topk)
    return scores.cpu().numpy(), idx.cpu().numpy() 

def image_retrieval(embedding, i, files)

if __name__ == '__main__':
    
    #******************************************************
    # Unit Test Code
    #******************************************************
    # Get all Input Argments
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # loading Model
    # model=load_model(args)
    model =dino_vitb8()
    model_seq= nn.Sequential(*model.blocks)

    model_seq= model_seq.eval().to(device)
    print("Successfull loading model")

    # Loading dataset
    ## Single Image 
    # img=get_image(args)
    # image_=img[None, :]
    # patches_img=model.patch_embed(image_)
    # print(patches_img.shape)
    #print(f"Succeed loading and convert: image_shape {image.shape}")

    ## Loading Batches of Images 
    val_dl= batch_images(args)
    #print(f"Succeed loading inference dataloader: data len {len(val_dl)}")

    out_embedding=[]
    with torch.no_grad():
        #for img_ in tqdm(val_dl):
        for idx, img_ in enumerate(val_dl):
            patches_img=model.patch_embed(img_)
            out_= (model_seq(patches_img.to(device)))
            out_embedding.append(out_)
    embedding= torch.cat(out_embedding, dim=0)
    print(embedding.shape)
    # ## Testing Model Shape Output

    