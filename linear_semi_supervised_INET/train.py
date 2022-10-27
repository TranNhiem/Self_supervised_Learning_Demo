import argparse 
from distutils.command.config import config
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet50
from lightning_models import DownstreamLinearModule
from prepare_imagenet_subset import DownstreamDataloader

parser = argparse.ArgumentParser()
parser.add_argument("-da", "--DATASET", type=str, default="one_per", help="SSL Backbone pretrained weight")
parser.add_argument("-m", "--METHOD", type=str, default="HAPiCLR_SimCLR", help="SSL Backbone pretrained weight")
parser.add_argument("-r", "--root_dir", type=str, default="/img_data/one_per", help="SSL Backbone pretrained weight")
parser.add_argument("-w", "--weight_path", type=str, default='/data/downstream_tasks/HAPiCLR/Classification/mscrl-imagenet-simclr+pixel_level_contrastive_background-dim1024-paperep=99.ckpt', help="SSL Backbone pretrained weight")
parser.add_argument("-b", "--batch_size",type=int, default=256,  help="batch_size for evaluation")
parser.add_argument("-d", "--metric",type=str, default="accuracy_1_5_torchmetric", choices=["accuracy_1_5_torchmetric",  "accuracy_1_5", "Mean_average_per_cls"], help="Which metric to use")
parser.add_argument("-t", "--task", required=True, choices=["ImageNet_linear", "linear_eval", "finetune"], help="linear_eval or finetune")
parser.add_argument("-ep", "--epochs", type= int, default =60, help="number of iterations")
parser.add_argument( "-gpus", "--gpus", type= list, default =[0,1], help="number of iterations")
parser.add_argument("-wed", "--weight_decay", type=float, default=5e-7, help="The amount of weight decay to use")
parser.add_argument("-ra", "--RandAug", type=bool, default=False, help="linear_eval or finetune")
parser.add_argument("--Init_lr", type=float, default=1e-2, help="The initial learning rate")
parser.add_argument("-lr_sch", "--lr_scheduler", type=str, default='step', help="Scheduler lr value during evaluation")
parser.add_argument("-optim", "--optimizier", type=str, default='sgd',choices=['sgd', 'adamw'], help="Scheduler lr value during evaluation")

args = parser.parse_args()
gpus_args={
    "HAPiCLR_m":[0, 1],
    "HAPiCLR_s":[0, 1],
    "DenseCLR": [2,3],
    "PixelPro": [2,3],
    "SimCLR": [4,5],
    "MoCo_V2": [4,5]
}

weight_path={
    #"MoCo_V2": "/code_spec/SSL_Downstream_tasks/moco_v2_200ep_pretrain.pth.tar",
    "MoCo_V2_re": "/code_spec/SSL_Downstream_tasks/MOCO_v2_200epochs_baseline.pt",
    "MoCo_V2" : "/home/harry/ssl_downstream_task/moco_v2_200ep_pretrain.pth.tar",
    "SimCLR":"/home/harry/ssl_downstream_task/simclr-4096-200ep-imagenet-onfm82pd-ep=199.ckpt", 
    "HAPiCLR_m": "/home/harry/ssl_downstream_task/hapiclr-org_sum_mulitGPU_4096-mocov2+-200ep-imagenet-315kge9v-ep=199.ckpt", 
    "HAPiCLR_s": "/home/harry/ssl_downstream_task/mscrl-imagenet-simclr+pixel_level_contrastive_background-dim1024-batch1024-200ep-paper-ndv2f2wz-ep=199.ckpt", 
    "DenseCLR": "/home/harry/ssl_downstream_task/densecl_r50_imagenet_200ep.pth",
    "PixelPro": "/home/harry/ssl_downstream_task/pixpro_base_r50_100ep_md5_91059202.pth",
}

ckpt_type= {
    "MoCo_V2_re": "solo_learn",
    "MoCo_V2": "moco",
    "SimCLR": "solo_learn",
    "HAPiCLR_m": "solo_learn",
    "HAPiCLR_s": "solo_learn",
    "DenseCLR": "DenseCL",
    "PixelPro": "pixelpro",

}

root_dir={
    "one_per": "/data1/1K_New/one_per/dataset/train/",
    "ten_per": "/data1/1K_New/ten_per/dataset/train/",
    "imagenet" : "/data1/1K_New/train/",
    "val_path": "/data1/1K_New/val/",
}
lr_decay_steps={
    "one_per": [15, 35, 45],
    "ten_per": [15, 35, 45],
    "imagenet" : [30, 55, 75], 
}
## Some Parameters setting depend Machine training
kwargs = {
    "num_classes": 1000,
    "precision": 32,
    "lars": False,
    "auto_lr_find": False,# auto
    "exclude_bias_n_norm": False,
    "gpus": gpus_args[args.METHOD],    # Number of GPUs
    "lr_decay_steps": lr_decay_steps[args.DATASET],  #[30, 55, 75], #[30, 6, 75],
    "num_workers": 20,
    "num_transfs": 2,  
    "magni_transfs": 5,
    "batch_size": args.batch_size,
    "num_workers": 20,
    "metric": args.metric,
    "task": args.task,
    "backbone_weights": weight_path[args.METHOD], 
    "RandAug": args.RandAug, 
    "num_transfs":2,  
    "magni_transfs": 5,
    "epochs": args.epochs,
    "lr": args.Init_lr,
    "weight_decay": args.weight_decay,
    "scheduler": args.lr_scheduler,
    "optimizier": args.optimizier,
    "root_dir": root_dir[args.DATASET],
    "ckpt_type": ckpt_type[args.METHOD],
    "imgNet_valpath":  root_dir["val_path"], 
}


model = DownstreamLinearModule(**kwargs)

dataloader = DownstreamDataloader(**kwargs)

wandb_logger = WandbLogger(
    # name of the experiment
    #name=f'{METHOD}_semi-supervised_{DATASET}_lr={kwargs["lr"]}_lr_schedu={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}',
    name=f'{args.METHOD}_{args.task}_{args.DATASET}_lr={kwargs["lr"]}_lr_sched={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}_batch={kwargs["batch_size"]}_optim_{args.optimizier}',
    project="HAPiCLR_Downstream_Tasks",  # name of the wandb project
    entity='tranrick',
    group=args.DATASET,
    #job_type='semi_supervised',
    job_type=args.task,
    offline=False,)

sync_batch=True if len(kwargs["gpus"]) > 1 else False
wandb_logger.watch(model, log="all", log_freq=50)
trainer = Trainer(accelerator='gpu', auto_select_gpus=False, gpus=kwargs["gpus"],
                   logger=wandb_logger, max_epochs=kwargs["epochs"], auto_lr_find=kwargs["auto_lr_find"], strategy="ddp", sync_batchnorm=sync_batch)

if __name__ == '__main__':
    print("start training")
    print(f"Weights : {args.weight_path}")
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    print("end training")




