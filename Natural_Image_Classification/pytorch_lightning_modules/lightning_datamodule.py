# Whole pipeline for Natural Image Classification Dataloader :
# Download -> Extract -> Split -> Dataset -----> Transform -> dataloader
## 1. Download 
# Download the dataset from the internet --> Checkout the Download_dataset_scripts folder
## 2. Extract -->> Done by yourself
## 3. Split -->> (Structure Dataset folder) --> [train, val, test] 
## 4. Transform --> Using Standard Validation, Test transform of ImageNet dataste + Extra adding RandAugment
## 5. Dataloader --> Using Pytorch Lightning Dataloader (LightningDataModule)

from tkinter import Image
from sklearn.utils import shuffle
import pytorch_lightning as pl
from pathlib import Path
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.utils import download_and_extract_archive, download_url
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torchvision import  transforms as transform
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torchvision.transforms import autoaugment as auto_aug
from torch.utils.data import ConcatDataset

## --------------------------------------_-------------------
## --------Support Concatenate Train,Val loader together---

class DownstreamDataloader(pl.LightningDataModule):
    def __init__(self, 
                    root_dir: str ,
                    dataset_name: str, 
                    task: str,
                    batch_size: int, 
                    num_workers: int, 
                RandAug:bool, num_transfs: int, magni_transfs: int, 
                concate_dataloader: bool= False, 
                **kwargs):
            
        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)
        self.task = task
        self.RandAug = RandAug
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.concate_dataloader= concate_dataloader
        self.num_ops= num_transfs
        self.magnitude= magni_transfs

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
            }
        }


    def __dataloader(self, task: str, mode: str ):
        
        is_train = True if mode == 'train' else False
        
        ## Train_Loader & Val_Loader for training, Test_Loader for Evaluate and Report
        ## This is Configure from BYOL, SimCLR
        
        print("ConcatDataset Train and Val")
        if self.dataset_name == "CIFAR10": 
            print("Using Cifar Train and Test without Val set")
            datapath=self.data_path.joinpath('dataset') 
            # load the dataset
            train_dataset = datasets.CIFAR10(root=datapath.joinpath(mode), train=True,
                download=False, transform=self.dataset_transforms[self.task][mode],)
    
            dataset = datasets.CIFAR10(datapath.joinpath(mode), train=False,download=False, transform=self.dataset_transforms[self.task][mode],)
            
            if mode == "train": 
                return DataLoader(
                    train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            elif mode =="val" or "test": 
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train, num_workers=self.num_workers, )
            else:
                raise ValueError(f"mode {mode} is not supported")

        if self.dataset_name == "CIFAR100" : 
            print("Using Cifar100 Train and Test without Val set")
                
            datapath=self.data_path.joinpath('dataset') 
            # load the dataset
            train_dataset = datasets.CIFAR100(
                root=datapath.joinpath(mode), train=True,
                download=False, transform=self.dataset_transforms[self.task][mode],
            )
    
            dataset = datasets.CIFAR100(datapath.joinpath(mode), train=False,
                download=False, transform=self.dataset_transforms[self.task][mode],)
        
            if mode == "train": 
                return DataLoader(
                    train_dataset, batch_size=self.batch_size,num_workers=self.num_workers,)
            elif mode =="val" or "test": 
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
            else: 
                raise ValueError(f"mode {mode} is not supported")
        
        else: 
            if self.concate_dataloader:
                if mode=="train":
                    print("Preparing ConcateDataset Loader")
                    dataset = ConcatDataset([self.create_dataset(self.root_dir.joinpath("train/"), self.dataset_transforms[task]["train"]), 
                                            self.create_dataset(self.root_dir.joinpath("val/"), self.dataset_transforms[task]["val"])])
                    return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
                    
            else: 
                dataset = self.create_dataset(self.root_dir.joinpath(mode), self.dataset_transforms[task][mode])
                return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)

            if mode=="val" or mode=='test': 
                print("Validation set is the same as Test Set")
                dataset = self.create_dataset(self.root_dir.joinpath("test/"), self.dataset_transforms[task][mode])
                return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
            else: 
                raise ValueError ("dataset only [train,val,test] Set")
       
    
            # num_train = len(train_dataset)
            # indices = list(range(num_train))
            # np.random.seed(100)
            # np.random.shuffle(indices)
            # train_idx, valid_idx = indices[split:], indices[:split]
            # # train_sampler = SubsetRandomSampler(train_idx)
            # # valid_sampler = SubsetRandomSampler(valid_idx)
            # train_sampler = DistributedSampler(train_idx)
            # valid_sampler = DistributedSampler(valid_idx)

            # if mode == "train": 
            #     return DataLoader(
            #         train_dataset, batch_size=self.batch_size, sampler=train_sampler,   
            #         num_workers=self.num_workers, shuffle=(train_sampler is None))
            # elif mode =="val": 
            #     return DataLoader(
            #         valid_dataset, batch_size=self.batch_size, sampler=valid_sampler, 
            #         num_workers=self.num_workers, shuffle=(valid_sampler is None))
            # elif mode=="test":      
            #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
            
    
    def create_dataset(self, root_path, transform):
        return ImageFolder(root_path, transform)    

    def train_dataloader(self):

        return self.__dataloader(task=self.task, mode="train")

    def val_dataloader(self):

        return self.__dataloader(task=self.task, mode='val')

    def test_dataloader(self):
        return self.__dataloader(task=self.task, mode='test')
      

    @property
    def linear_eval_train_transforms(self):
        if self.RandAug:
            print("RandAugment Implement") 
        
            return transforms.Compose(
                [
                    transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                    auto_aug.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
                ]
            )
        else: 
            return transforms.Compose(
                [
                    transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                    #auto_aug.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
                ]
            )

    @property
    def linear_eval_val_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                #auto_aug.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            ]
        )

    @property
    def finetune_train_transforms(self):
        if self.RandAug:
            print("RandAugment Implement") 
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
                #auto_aug.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
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
    
        
# --- Functions to prepare every individual dataset ---

