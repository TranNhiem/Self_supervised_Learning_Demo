import os, shutil
from pathlib import Path 
import pytorch_lightning as pl 
import torch.nn.functional as F 
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import autoaugment as auto_aug
from torchvision import transforms

## ----------------------------------------------------
## Function to Create the Subset of ImageNet
## ----------------------------------------------------

def split_imagenet_subset(one_per_txt='/home/harry/ssl_downstream_task/Self_supervised_Learning_Demo/linear_semi_supervised_INET/1percent.txt', 
                         ten_per_txt='/home/harry/ssl_downstream_task/Self_supervised_Learning_Demo/linear_semi_supervised_INET/10percent.txt'
                            , train_path='/data1/1K_New/train/', one_per_path= '/data1/1K_New/one_per/dataset/train', 
                                ten_per_path= '/data1/1K_New/ten_per/dataset/train',):
    ## 
    '''
    one_per.txt, ten_per.txt can download from here https://github.com/google-research/simclr/tree/master/imagenet_subsets
    
    Args:
        one_per_txt: Path to the text file containing the list of classes to be used for 1% of ImageNet
        ten_per_txt: Path to the text file containing the list of classes to be used for 10% of ImageNet
        train_path: Path to the ImageNet training set
        one_per_path: Path to SAVE the 1% of ImageNet training set
        ten_per_path: Path to SAVE the 10% of ImageNet training set
    '''
    
    class_names = [x for x in os.listdir(train_path) if '.tar' not in x]
    #print(len(class_names))
    for class_name in class_names:
        Path(os.path.join(one_per_path, class_name)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(ten_per_path, class_name)).mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(os.path.join(one_per_path, class_name)):
        #     os.mkdir(os.path.join(one_per_path, class_name))

        # if not os.path.exists(os.path.join(ten_per_path, class_name)):
        #     os.mkdir(os.path.join(ten_per_path, class_name))

    with open(one_per_txt, 'r') as f:
        one_per_images = f.readlines()
        for image in one_per_images:
            image = image[:-1]
            label, _ = image.split('_')
            src_path = os.path.join(train_path, label, image)
            dest_path = os.path.join(one_per_path, label, image)
            if not os.path.exists(dest_path) and os.path.exists(src_path):
                shutil.copyfile(src_path, dest_path)

    with open(ten_per_txt, 'r') as f:
        ten_per_images = f.readlines()
        for image in ten_per_images:
            image = image[:-1]
            label, _ = image.split('_')
            src_path = os.path.join(train_path, label, image)
            dest_path = os.path.join(ten_per_path, label, image)
            if not os.path.exists(dest_path) and os.path.exists(src_path):
                shutil.copyfile(src_path, dest_path)

## ----------------------------------------------------
## Dataloader Module for ImageNet dataset 
## ----------------------------------------------------
class DownstreamDataloader(pl.LightningDataModule):
    def __init__(self, 
            task: str,
            batch_size: int,
            num_workers: int,
            root_dir: str ,
            imgNet_valpath: str,
            RandAug: bool= False, num_transfs: int= 2, magni_transfs: int =7, 
            **kwargs):


        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.RandAug=RandAug
        self.num_ops= num_transfs
        self.magnitude= magni_transfs
        self.imgNet_valpath=imgNet_valpath
  
        self.dataset_transforms = {
            "linear_eval": {
                "train": self.linear_eval_train_transforms,
                "val": self.linear_eval_val_transforms,
                "test": self.linear_eval_val_transforms,
            },
            
            "finetune": {
                "train": self.finetune_train_transforms,
                "val": self.finetune_val_transforms,
                "test": self.finetune_val_transforms,
            }, 
             "ImageNet_linear": {
                "train": self.finetune_train_transforms,
                "val": self.finetune_val_transforms,
                "test": self.finetune_val_transforms,
            }

        }


    def __dataloader(self, task: str, mode: str):
        is_train = True if mode == 'train' else False
        
        if mode=="val" or mode=='test': 
            print("Validation set is the same as Test Set")    
            dataset = self.create_dataset(self.imgNet_valpath, self.dataset_transforms[task][mode])
            return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
        
        elif mode== "train":
            dataset = self.create_dataset(self.root_dir, self.dataset_transforms[task][mode])
            return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)

    def create_dataset(self, root_path, transform):
        return ImageFolder(root_path, transform)

    def train_dataloader(self):
        return self.__dataloader(task=self.task, mode='train')

    def val_dataloader(self):
        return self.__dataloader(task=self.task, mode='val')

    def test_dataloader(self):
        return self.__dataloader(task=self.task, mode='test')

    @property
    def data_path(self):
        # if self.dataset_name == "ImageNet":
        #     return Path(self.root_dir)
        # else:
        #     return Path(self.root_dir).joinpath("dataset")
        return Path(self.root_dir)

    @property
    def linear_eval_train_transforms(self):
        if self.RandAug: 
            return transforms.Compose(
            [
                transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                auto_aug.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            ]
        )
        else: 
            return transforms.Compose(
                [
                    transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
                ]
            )

    @property
    def linear_eval_val_transforms(self):
   
        return transforms.Compose(
                [
                    transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
                ]
            )

    @property
    def finetune_train_transforms(self):
        
        if self.RandAug: 
        
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    auto_aug.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
                ]
            )

    @property
    def finetune_val_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            ]
        )