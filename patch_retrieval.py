import os 
import torch
from utils import load_pretrained_weights, normal_dataloader
import torch.nn as nn
import vision_transformer as vits
import argparse
from tqdm.auto import tqdm
from torchvision import models as torchvision_models 
from pathlib import Path
from PIL import Image
from torchvision import transforms as pth_transforms
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Tuple, Optional
from torchvision.transforms.functional import to_pil_image
from torchvision import io
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
    parser.add_argument('--dataloader_patches', default=True, type=bool,
        help='Decided loading dataloader with or without Patches image')
    parser.add_argument('--subset_data', default=0.1, type=float,
        help='How many percentage of your Demo Dataset you want to Use.')
    parser.add_argument('--single_img_path', default="/img_data/Patch_level_matching_pets_dataset/images/miniature_pinscher_153.jpg", type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--image_size', default=256, type=int,
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
    
    
    if args.dataloader_patches:
        print("Loading Patches dataloader") 
        chanels=3 # RGB image
        val_dl=val_dataset.val_dataloader_patches(args.patch_size, chanels)
    else: 
        print("Loading Normal dataloader without Patching Images")
        val_dl=val_dataset.val_dataloader()


    return val_dl , val_dataset

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
        with open(args.single_img_path, 'rb') as f:
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

def image_retrieval_topk(embedding: torch.FloatTensor, i: int, topk=4, normalize=True): 
    if normalize: 
        ancher_patch=embedding.norm(dim=-1, keepdim=True)

    similarity = (embedding @ embedding[i, :].T).softmax(dim=-1)
    scores, idx= similarity.cpu().topk(topk, )#dim=-1
    print("this is idx value:", idx.shape)
    print("This is score value", scores.shape)
    
    return scores, idx #.numpy() 

def image_retrieval_topk_extra(source_data_embedding: torch.FloatTensor, 
                    ancher_image_embedding: torch.FloatTensor, 
                    topk= 4
                    ): 
    similarity = source_data_embedding @ ancher_image_embedding.T
    scores, idx= similarity.topk(topk)
    return scores.cpu().numpy(), idx.cpu().numpy() 

def plotting_image_retrieval(args, source_data_embedding,ancher_embedding, i, val_dat, 
                             topk, retrieval_from_exmaple=False, image_path : Optional[str] = None, ):
    if retrieval_from_exmaple: 
        print("you must provide 2 argugment 1: image_path argument, 2:dataloader_patches == False ")

        if image_path is None: 
            raise ValueError("You are not providing image_path argument")
        else:
            resize= pth_transforms.Resize((args.image_size, args.image_size,))
            image= to_pil_image(resize(io.read_image(image_path)))
            plt.imshow(image)
            scores, idx= image_retrieval_topk_extra(source_data_embedding, ancher_embedding, topk=topk)
            ig, axs = plt.subplots(1, len(idx), figsize=(12, 5))
            for i_, score, ax in zip(idx, scores, axs):
                img= to_pil_image(resize(io.read_image(val_dat.image_files[i_])))
                ax.imshow(img)
                ax.set_title(f"{score:.4f}")
            plt.show()
    else: 
        resize= pth_transforms.Resize((args.image_size, args.image_size,))
        print("image_path", val_dat.image_files[i])
        image= to_pil_image(resize(io.read_image(val_dat.image_files[i])))
        plt.imshow(image)
        scores, idx= image_retrieval_topk(source_data_embedding, i, topk=topk)
    
        ig, axs = plt.subplots(1, len(idx), figsize=(12, 5))
        for i_, score, ax in zip(idx[0, :], scores[0,:], axs):
            print("This is idx image", i_)
            for _, j in enumerate(i_): 
                print("image index", j)
                img= to_pil_image(resize(io.read_image(val_dat.image_files[j])))
                ax.imshow(img)
                ax.set_title(f"{score[_]:.4f}")
        plt.show()
        

if __name__ == '__main__':
    
    #******************************************************
    # Unit Test Code
    #******************************************************
    ## 1---- Get all Input Argments
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## 2--- loading Model
    # model=load_model(args)
    model =dino_vitb8()
    model_seq= nn.Sequential(*model.blocks)
    model_seq= model_seq.eval().to(device)
    print("Successfull loading model")

    ## 3--- Loading dataset
    ## Single Image 
    # img=get_image(args)
    # image_=img[None, :]
    # patches_img=model.patch_embed(image_)
    # print(patches_img.shape)
    #print(f"Succeed loading and convert: image_shape {image.shape}")

    ## Loading Batches of Images 
    val_dl, data_path= batch_images(args)
    #print(f"Succeed loading inference dataloader: data len {len(val_dl)}")
    out_embedding=[]
    with torch.no_grad():
        ## If setting args.dataloader_patches ==True
        for img_ in tqdm(val_dl):
            #print(f"Patch image shape: {img_.shape}")
            out_= (model_seq(img_.to(device)))
            out_embedding.append(out_)
         
        
        ## If setting args.dataloader_patches ==False
        # for img_ in tqdm(val_dl):
        #     patches_img=model.patch_embed(img_)
        #     #print(f"Patch image shape: {patches_img.shape}")
        #     out_= (model_seq(patches_img.to(device)))
        #     out_embedding.append(out_)

        out_embedding= torch.cat(out_embedding, dim=0)
    print("Catting all images embedding", out_embedding.shape)
    # ## Testing Model Shape Output

    ## 4 --- Computing Cosine Similarity 
    anchor_embedding=None
    img=get_image(args)
    image_=img.view([1,3,args.image_size,args.image_size])
    patches_img=model.patch_embed(image_)
    anchor_embedding=model_seq(patches_img.to(device))
    print(f"Single image embedding shape: {anchor_embedding.shape}")
    i=60
    topk=5
    plotting_image_retrieval(args, out_embedding, anchor_embedding,i, data_path,topk, 
                            retrieval_from_exmaple=False, image_path =args.single_img_path, )