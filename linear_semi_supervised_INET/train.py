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
parser.add_argument("-b", "--batch_size",type=int, default=512,  help="batch_size for evaluation")
parser.add_argument("-d", "--metric",type=str, default="accuracy_1_5_torchmetric", choices=["accuracy_1_5_torchmetric",  "accuracy_1_5", "Mean_average_per_cls"], help="Which metric to use")
parser.add_argument("-t", "--task", choices=["linear", "finetune"], help="linear_eval or finetune")
parser.add_argument("-ep", "--epochs", type= int, default =60, help="number of iterations")
parser.add_argument("-wed", "--weight_decay", type=float, default=5e-7, help="The amount of weight decay to use")
parser.add_argument("-ra", "--RandAug", type=bool, default=False, help="linear_eval or finetune")
parser.add_argument("-lr", "--Init_lr", type=float, default=1e-2, help="The initial learning rate")
parser.add_argument("-lr_sch", "--lr_scheduler", type=str, default='step', help="Scheduler lr value during evaluation")
args = parser.parse_args()

## Some Parameters setting depend Machine training
kwargs = {
    "num_classes": 1000,
    "precision": 16,
    "lars": False,
    "auto_lr_find": False,# auto
    "exclude_bias_n_norm": False,
    "gpus": 2,    # Number of GPUs
    "lr_decay_steps": [30, 55, 75], #[30, 6, 75],
    "num_workers": 20,
    "num_transfs": 2,  
    "magni_transfs": 5,
    "batch_size": args.batch_size,
    "num_workers": 20,
    "metric": args.metric,
    "task": args.task,
    "backbone_weights": args.weight_path, 
    "RandAug": args.RandAug, 
    "num_transfs":2,  
    "magni_transfs": 5,
    "epochs": args.epochs,
    "lr": args.Init_lr,
    "weight_decay": args.weight_decay,
    "scheduler": args.lr_scheduler,
    # "root_dir": args.root_dir,
}

model = DownstreamLinearModule(**kwargs)

dataloader = DownstreamDataloader(args.DATASET,root_dir=args.root_dir, download=False, task=kwargs['task'], batch_size=kwargs["batch_size"], num_workers=kwargs['num_workers'], 
RandAug=kwargs['RandAug'],num_transfs=kwargs['num_transfs'], magni_transfs=kwargs['magni_transfs'], )

wandb_logger = WandbLogger(
    # name of the experiment
    #name=f'{METHOD}_semi-supervised_{DATASET}_lr={kwargs["lr"]}_lr_schedu={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}',
    name=f'{args.METHOD}_{args.task}_{args.DATASET}_lr={kwargs["lr"]}_lr_sched={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}_batch={kwargs["batch_size"]}_optim_{args.optim_type}',
    project="HAPiCLR_Downstream_Tasks",  # name of the wandb project
    entity='tranrick',
    group=args.DATASET,
    #job_type='semi_supervised',
    job_type='linear_eval',
    offline=False,)

wandb_logger.watch(model, log="all", log_freq=50)
trainer = Trainer(accelerator='gpu', auto_select_gpus=False, gpus=kwargs["gpus"],
                   logger=wandb_logger, max_epochs=kwargs["epochs"], auto_lr_find=kwargs["auto_lr_find"], strategy="ddp")

if __name__ == '__main__':
    print("start training")
    print(f"Weights : {args.weight_path}")
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    print("end training")

# def train():
#     print("start training")
#     print(f"Weights : {WEIGHTS}")
#     # print(train_loader)
#     # trainer.fit(model, train_loader, val_loader)
#     # if VAL_PATH != TEST_PATH:
#         # trainer.validate(model, test_loader)
#     print("end training")

# if __name__ == '__main__':
#     train()

