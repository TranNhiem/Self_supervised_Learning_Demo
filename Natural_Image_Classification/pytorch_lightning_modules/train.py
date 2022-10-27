import argparse
from operator import truediv 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning_models import DownstreamLinearModule
from lightning_datamodule import DownstreamDataloader
import os
parser = argparse.ArgumentParser()

parser.add_argument("-da", "--dataset", type=str, default="Cars",  help="SSL Backbone pretrained weight")
parser.add_argument("-m", "--method", type=str, default="HAPiCLR_SimCLR", help="SSL Backbone pretrained weight")
parser.add_argument("-r", "--root_dir", type=str, default="/data/downstream_datasets/classification/Cars/dataset/", help="SSL Backbone pretrained weight")
parser.add_argument("-w", "--weight_path", type=str, default='/code_spec/SSL_Downstream_tasks/moco_v2_200ep_pretrain.pth.tar', help="SSL Backbone pretrained weight")
parser.add_argument("-b", "--batch_size",type=int, default=256,  help="batch_size for evaluation")
parser.add_argument("-d", "--metric",type=str, default="accuracy_1_5_torchmetric", choices=["accuracy_1_5_torchmetric",  "accuracy_1_5", "Mean_average_per_cls"], help="Which metric to use")
parser.add_argument("-t", "--task",required=True, choices=["linear_eval", "finetune"], help="linear_eval or finetune")
parser.add_argument("-ep", "--epochs", type= int, default =140, help="number of iterations")
parser.add_argument("-wed", "--weight_decay", type=float, default=5e-7, help="The amount of weight decay to use")
parser.add_argument("-ra", "--RandAug", type=bool, default=False, help="linear_eval or finetune")
parser.add_argument("-lr", "--Init_lr", type=float, default=1e-2, help="The initial learning rate")
parser.add_argument("-lr_sch", "--lr_scheduler", type=str, default='step', help="Scheduler lr value during evaluation")
parser.add_argument("-optim", "--optimizier", type=str, default='sgd',choices=['sgd', 'adamw'], help="Scheduler lr value during evaluation")

args = parser.parse_args()
path= "/clf_ds/"
root_dir= os.path.join(path, args.dataset, "data/")
num_gpu={
    "MoCo_V2_re":  [0, 1],
    "MoCo_V2": [0, 1],
    "SimCLR": [2, 3],
    "HAPiCLR_m":  [3, 4],
    "HAPiCLR_S":  [4, 5],
    "DenseCLR":  [6, 7],
    "PixelPro":  [2, 3],

}
weight_path={
    #"MoCo_V2": "/code_spec/SSL_Downstream_tasks/moco_v2_200ep_pretrain.pth.tar",
    "MoCo_V2_re": "/code_spec/SSL_Downstream_tasks/MOCO_v2_200epochs_baseline.pt",
    "MoCo_V2" : "/code_spec/SSL_Downstream_tasks/moco_v2_200ep_pretrain.pth.tar",
    "SimCLR":"/code_spec/SSL_Downstream_tasks/simclr-200ep-imagenet-3n5pcokg-ep=199.ckpt", 
    "HAPiCLR_m": "/code_spec/SSL_Downstream_tasks/hapiclr-org_sum_mulitGPU_4096-mocov2+-200ep-imagenet-315kge9v-ep=199.ckpt", 
    "HAPiCLR_S": "/code_spec/SSL_Downstream_tasks/mscrl-imagenet-simclr+pixel_level_contrastive_background-dim1024-batch1024-200ep-paper-ndv2f2wz-ep=199.ckpt", 
    "DenseCLR": "/code_spec/SSL_Downstream_tasks/densecl_r50_imagenet_200ep.pth",
    "PixelPro": "/code_spec/SSL_Downstream_tasks/pixpro_base_r50_100ep_md5_91059202.pth",
}
ckpt_type= {
    "MoCo_V2_re": "solo_learn",
    "MoCo_V2": "moco",
    "SimCLR": "solo_learn",
    "HAPiCLR_m": "solo_learn",
    "HAPiCLR_S": "solo_learn",
    "DenseCLR": "DenseCL",
    "PixelPro": "pixelpro",

}


DATASET_NUM_CLASSES = {
    'food-101': 101,
    'oxford_flowers102': 102,
    "Caltech-101": 101, 
    'DTD': 47,
    'cars196': 196,
    'cifar10': 10, 
    'cifar100': 100,
    'Pets': 37,
    'SUN397': 397,
    'aircraft': 100,
    'caltech101': 101,
    'birdsnap': 500, 

}
## Some Parameters setting depend Machine training
kwargs = {
    "num_classes": DATASET_NUM_CLASSES[args.dataset], 
    "precision": 32,
    "lars": False,
    "auto_lr_find": False,# auto
    "exclude_bias_n_norm": False,
    "gpus": 2,    # Number of GPUs
    "lr_decay_steps": [30, 75, 100], #[30, 6, 75],
    "num_workers": 20,
    "num_transfs": 2,  
    "magni_transfs": 5,
    "dataset_name": args.dataset, 
    "concate_dataloader": False, 
    "batch_size": args.batch_size,
    "metric": args.metric,
    "task": args.task,
    "backbone_weights": weight_path[args.method], 
    "RandAug": args.RandAug, 
    "epochs": args.epochs,
    "lr": args.Init_lr,
    "weight_decay": args.weight_decay,
    "scheduler": args.lr_scheduler,
    "optimizier": args.optimizier,
    "root_dir": root_dir,
    "ckpt_type": ckpt_type[args.method], 
}

model = DownstreamLinearModule(**kwargs)

dataloader = DownstreamDataloader(**kwargs )

wandb_logger = WandbLogger(
    # name of the experiment
    #name=f'{METHOD}_semi-supervised_{DATASET}_lr={kwargs["lr"]}_lr_schedu={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}',
    name=f'{args.method}_{args.task}_{args.dataset}_lr={kwargs["lr"]}_lr_sched={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}_batch={kwargs["batch_size"]}_optim_{args.optimizier}',
    project="HAPiCLR_Downstream_Tasks",  # name of the wandb project
    entity='tranrick',
    group=args.dataset,
    #job_type='semi_supervised',
    job_type=args.task,
    offline=False,)

sync_batch=True if kwargs["gpus"] > 1 else False
wandb_logger.watch(model, log="all", log_freq=50)
trainer = Trainer(accelerator='gpu', auto_select_gpus=False, gpus=kwargs["gpus"],
                   logger=wandb_logger, max_epochs=kwargs["epochs"], auto_lr_find=kwargs["auto_lr_find"], strategy="ddp", sync_batchnorm=sync_batch)

if __name__ == '__main__':
    print("start training")
    print(f"Weights : {args.weight_path}")
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    print("end training")

