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

def split_imagenet_subset(one_per_txt='/code_spec/downstream_tasks/one_per.txt', 
                         ten_per_txt='/code_spec/downstream_tasks/ten_per.txt'
                            , train_path='/img_data/train', one_per_path= '/img_data/one_per/dataset/train', 
                                ten_per_path= '/img_data/ten_per/dataset/train',):
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

    for class_name in class_names:
        if not os.path.exists(os.path.join(one_per_path, class_name)):
            os.mkdir(os.path.join(one_per_path, class_name))
        if not os.path.exists(os.path.join(ten_per_path, class_name)):
            os.mkdir(os.path.join(ten_per_path, class_name))

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
            
            dataset_name: str, download: bool, 
            task: str,
            batch_size: int,
            num_workers: int,
            root_dir: str ,
            RandAug: bool= False, num_transfs: int= 2, magni_transfs: int =7, 

        
            ):

        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.download = download
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.RandAug=RandAug
        self.num_ops= num_transfs
        self.magnitude= magni_transfs
        self.dataset_urls = {
            'food101': 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',

        }
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

    def prepare_data(self):
        if self.download:
            download_and_extract_archive(self.dataset_urls[self.dataset_name], self.root_dir)

    def __dataloader(self, task: str, mode: str):
        
        if self.task=="ImageNet_linear":
            dataset = self.create_dataset(self.data_path.joinpath(mode), self.dataset_transforms[task][mode])
        else: 
            if mode == 'val' or mode == 'test':
                dataset = self.create_dataset(self.data_path.joinpath(mode, 'val'), self.dataset_transforms[task][mode])
            
            else:
                dataset = self.create_dataset(self.data_path.joinpath(mode), self.dataset_transforms[task][mode])
            
        is_train = True if mode == 'train' else False
        return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train,prefetch_factor=10 )
    
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
        if self.dataset_name == "ImageNet":
            return Path(self.root_dir)
        else:
            return Path(self.root_dir).joinpath("dataset")
        #return Path(self.root_dir)

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