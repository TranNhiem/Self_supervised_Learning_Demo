# Whole pipeline:
# Download -> Extract -> Split -> Dataset -----> Transform -> dataloader
# 

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
import patoolib
from torchvision.transforms import autoaugment as auto_aug
from torch.utils.data import ConcatDataset

DATASET_NUM_CLASSES = {
    'food101': 101,
    'flowers102': 102,
    'DTD': 47,
    'Cars': 196,
    'CIFAR10': 10, 
    'CIFAR100': 100,
    'Pets': 37,
    'SUN397': 397,
    'Aircrafts': 100,
    'Caltech101': 101,
    'birdsnap': 500, 

}

class DownstreamDataloader(pl.LightningDataModule):
    def __init__(self, dataset_name: str, download: bool, task: str, replica_batch_size: int, num_workers: int, 
                concate_dataloader: bool, datafolder_combine: bool, RandAug:bool, num_transfs: int, magni_transfs: int):
        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = Path('/media/rick/2TB_1/ImageNet_dataset').joinpath(dataset_name) if 'per' in dataset_name else Path('/home/rick/offline_finetune').joinpath(dataset_name)
        self.download = download
        self.task = task
        self.RandAug = RandAug
        self.batch_size = replica_batch_size
        self.num_workers = num_workers
        self.concate_dataloader= concate_dataloader
        self.datafolder_combine=datafolder_combine
        self.num_ops= num_transfs
        self.magnitude= magni_transfs
        self.dataset_urls = {
            'food101': 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',
            'flowers102': '',
            'dtd': '',
            'cars': '',
            'cifar10': '', 
            'cifar100': '',
            'pets': '',
            'sun397': '',
            'aircrafts': '',
            # adding complete dataset
            'Caltech101': 'https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12855005/Caltech101ImageDataset.rar', 
            'birdsnap': '', 
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
            }
        }

    def prepare_data(self):
        if self.download:
            download_and_extract_archive(self.dataset_urls[self.dataset_name], self.root_dir)
   
        ## Download dataset for the first time
        # if self.dataset_name== "Caltech101": 
        #     print('download Caltech101')
        #     download_url(self.dataset_urls[self.dataset_name], self.root_dir)
        #     # extract 
        #     path_=os.path.join(self.root_dir, "Caltech101ImageDataset.rar")
        #     patoolib.extract_archive(path_, outdir=self.root_dir)
    

    def __dataloader(self, task: str, mode: str ):
        
        is_train = True if mode == 'train' else False
        
        ## Train_Loader & Val_Loader for training, Test_Loader for Evaluate and Report
        ## This is Configure from BYOL, SimCLR
        if self.concate_dataloader:
            
            if self.dataset_name == "CIFAR10": 
                print("Using Cifar Train and Test without Val set")
                datapath=self.data_path.joinpath('dataset') 
                # load the dataset
                train_dataset = datasets.CIFAR10(
                    root=datapath.joinpath(mode), train=True,
                    download=False, transform=self.dataset_transforms[self.task][mode],
                )
        
                dataset = datasets.CIFAR10(datapath.joinpath(mode), train=False,
                    download=False, transform=self.dataset_transforms[self.task][mode],)
                
                if mode == "train": 
                    return DataLoader(
                        train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
                elif mode =="val": 
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                elif mode=="test":      
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                    
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
                elif mode =="val": 
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                elif mode=="test":      
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
            else: 
                if mode=="train_val":
                    datapath=self.data_path.joinpath("dataset")
                    print("Preparing ConcateDataset Loader")
                    concate_dataset = ConcatDataset([self.create_dataset(datapath.joinpath("train"), self.dataset_transforms[task]["train"]), 
                                            self.create_dataset(datapath.joinpath("val"), self.dataset_transforms[task]["val"])])
                    return DataLoader(dataset=concate_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
                
                elif mode=="val" or mode=='test': 
                    print("Validation set is the same as Test Set")
                    datapath=self.data_path.joinpath("dataset")
                    dataset = self.create_dataset(datapath.joinpath('test'), self.dataset_transforms[task]['test'])
                    return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
                else: 
                    raise ValueError ("dataset only [train,val,test] Set")
        ## Train_Loader, Val_Loader, Test_Loader
        if self.datafolder_combine: 
       
            if self.dataset_name == "CIFAR10": 
                print("Using Cifar Train and Test without Val set")
                datapath=self.data_path.joinpath('dataset') 
                # load the dataset
                train_dataset = datasets.CIFAR10(
                    root=datapath.joinpath(mode), train=True,
                    download=False, transform=self.dataset_transforms[self.task][mode],
                )
        
                dataset = datasets.CIFAR10(datapath.joinpath(mode), train=False,
                    download=False, transform=self.dataset_transforms[self.task][mode],)
                
                if mode == "train": 
                    return DataLoader(
                        train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
                elif mode =="val": 
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                elif mode=="test":      
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                    
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
                elif mode =="val": 
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                elif mode=="test":      
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                
            else: 
                if mode=="train": 
                    print("Enable CombineFolder")
                    datapath=self.data_path.joinpath("dataset_concate_train_val")
                    dataset = self.create_dataset(datapath.joinpath(mode), self.dataset_transforms[task][mode])
                    return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
                    
                elif mode=="val" or mode=='test': 
                    print("Validation set is the same as Test Set")
                    datapath=self.data_path.joinpath("dataset")
                    dataset = self.create_dataset(datapath.joinpath('test'), self.dataset_transforms[task]['test'])
                    return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
                else: 
                    raise ValueError ("dataset only [train,val,test] Set")
        else: 

            if self.dataset_name == "CIFAR10" or self.dataset_name =="CIFAR100" : 
                if self.dataset_name== "CIFAR10": 
                    # load the dataset

                    train_dataset = datasets.CIFAR10(
                        root=self.data_path.joinpath(mode), train=True,
                        download=False, transform=self.dataset_transforms[self.task][mode],
                    )

                    valid_dataset = datasets.CIFAR10(
                        root=self.data_path.joinpath(mode), train=True,
                        download=False, transform=self.dataset_transforms[self.task][mode],
                    )
                    split = 5000
                    
                    dataset = datasets.CIFAR10( self.data_path.joinpath(mode), train=False,
                        download=True, transform=self.dataset_transforms[self.task][mode],)
                else: 
                    # load the dataset
                    train_dataset = datasets.CIFAR100(
                        root=self.data_path.joinpath(mode), train=True,
                        download=False, transform=self.dataset_transforms[self.task][mode],
                    )

                    valid_dataset = datasets.CIFAR100(
                        root=self.data_path.joinpath(mode), train=True,
                        download=False, transform=self.dataset_transforms[self.task][mode],
                    )  
                    split = 5067
                    dataset = datasets.CIFAR100( self.data_path.joinpath(mode), train=False,
                        download=False, transform=self.dataset_transforms[self.task][mode],)
                
                num_train = len(train_dataset)
                indices = list(range(num_train))
                
                np.random.seed(100)
                np.random.shuffle(indices)
                train_idx, valid_idx = indices[split:], indices[:split]
                # train_sampler = SubsetRandomSampler(train_idx)
                # valid_sampler = SubsetRandomSampler(valid_idx)
                train_sampler = DistributedSampler(train_idx)
                valid_sampler = DistributedSampler(valid_idx)

                if mode == "train": 
                    return DataLoader(
                        train_dataset, batch_size=self.batch_size, sampler=train_sampler,   
                        num_workers=self.num_workers, shuffle=(train_sampler is None))
                elif mode =="val": 
                    return DataLoader(
                        valid_dataset, batch_size=self.batch_size, sampler=valid_sampler, 
                        num_workers=self.num_workers, shuffle=(valid_sampler is None))

                elif mode=="test":      
                    return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,num_workers=self.num_workers, )
                
            else:
                datapath=self.data_path.joinpath('dataset') 
                dataset = self.create_dataset(datapath.joinpath(mode), self.dataset_transforms[task][mode])
                return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
            
    def create_dataset(self, root_path, transform):
        return ImageFolder(root_path, transform)    

    def train_dataloader(self):
        # if self.dataset_name=="Place365" or "Caltech101" or "Birdsnap":
        #     return DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        # else: 
        if self.concate_dataloader: 
            print("Enable ConcateDataloader")
            return self.__dataloader(task=self.task, mode="train_val")
        else: 
            return self.__dataloader(task=self.task, mode='train')

    def val_dataloader(self):

        return self.__dataloader(task=self.task, mode='val')

    def test_dataloader(self):

        return self.__dataloader(task=self.task, mode='test')
      
    @property
    def data_path(self):
        return Path(self.root_dir)

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

