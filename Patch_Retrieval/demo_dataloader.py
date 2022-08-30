from random import shuffle
import torch 
from typing import Any, Callable, List, Tuple

# For Dataloader Inference
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.datasets import ImageFolder

# ******************************************************
# Inference DataLoader
# ******************************************************

class collateFn_patches:

    def __init__(self, image_size, patch_size, chanels):
        self.patch_size = patch_size
        self.chanels = chanels
        self.num_patches = (image_size//patch_size)**2

    def reshape(self, batch):
        patches = torch.stack(batch) \
            .unfold(2, self.patch_size, self.patch_size)\
            .unfold(3, self.patch_size, self.patch_size)

        num_images = len(patches)
        patches = patches.reshape(
            num_images,
            self.chanels,
            self.num_patches,
            self.patch_size,
            self.patch_size,)

        patches.transpose_(1, 2)
        return patches.reshape(num_images, self.num_patches, -1) / 255.0 - 0.5

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.FloatTensor:
        return self.reshape(batch)

class collatesingle_img:

    def __call__(self, batch: List[torch.Tensor]) -> torch.FloatTensor:
        return batch

class ImageOriginalData(Dataset):
    def __init__(self, files: List[str], img_size: int, transform_ImageNet=False):
        self.files = files
        self.resize = transforms.Resize((img_size, img_size))
        self.transform_ImageNet = transform_ImageNet
        if self.transform_ImageNet:
            print("Using imageNet normalization")
        self.transform_normal = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    # Iterative through all images in dataste

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        #img = io.read_image(self.files[i])
        # Checking the Image Channel
        # if img.shape[0] == 1:
        #     img= torch.cat([img]*3)
        if self.transform_ImageNet:
            return self.transform_normal(img)
        else:
            return self.resize(img)

class all_images_in_1_folder_dataloader:
    '''
    This normal dataloader supports for dataset with all images containing in *ONLY One Folder* 

    '''

    def __init__(self, image_path, image_format="*.jpg", img_size=224, batch_size=4, subset_data=0.2, transform_ImageNet=False):
        image_files_ = [str(file) for file in Path(image_path).glob("*.jpg")]
        _,  self.image_files = train_test_split(
            image_files_, test_size=subset_data, random_state=42)

        self.img_size = img_size
        self.batch_size = batch_size
        self.transform_ImageNet = transform_ImageNet

    def val_dataloader(self):
        val_data = ImageOriginalData(
            self.image_files, self.img_size, self.transform_ImageNet)
        print(f" total images in Demo Dataset: {len(val_data)}")
        val_dl = DataLoader(
            val_data,
            self.batch_size*2,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            #collate_fn= collatesingle_img()
        )
        return val_dl

    def val_dataloader_patches(self, patch_size, chanels):
        val_data = ImageOriginalData(
            self.image_files, self.img_size, self.transform_ImageNet)
        print(f" total images in Demo Dataset: {len(val_data)}")
        val_dl = DataLoader(
            val_data,
            self.batch_size*2,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collateFn_patches(
                image_size=self.img_size, patch_size=patch_size, chanels=chanels)
        )
        return val_dl


class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class multiple_folders_dataloader(ImageFolder): 
    def __init__(self, args, transform_ImageNet=None):
        super(multiple_folders_dataloader, self).__init__(args.img_path, transform_ImageNet)
    
        self.img_size = args.img_size
        self.img_path=args.img_path
        self.batch_size = args.batch_size
        if transform_ImageNet is not None:
            self.transform_ImageNet = transform_ImageNet
        else: 
            self.transform_ImageNet = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    
    def val_dataloader(self):
        val_data = ImageFolderInstance(self.img_path, self.transform_ImageNet)
        val_dl = DataLoader(
            val_data,
            self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            #collate_fn= collatesingle_img()
        )
        return val_dl
