import argparse
from distutils.command.config import config
import torch
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet50
from downstream_modules import DownstreamLinearModule_sweep
from prepare_imagenet_subset import DownstreamDataloader


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--DATASET", type=str, default="one_per", help="SSL Backbone pretrained weight")
parser.add_argument("-w", "--METHOD", type=str, default="HAPiCLR_SimCLR", help="SSL Backbone pretrained weight")
parser.add_argument("-w", "--root_dir", type=str, default="/img_data/one_per", help="SSL Backbone pretrained weight")
parser.add_argument("-w", "--weight_path", type=str, default='/data/downstream_tasks/HAPiCLR/Classification/mscrl-imagenet-simclr+pixel_level_contrastive_background-dim1024-paperep=99.ckpt', help="SSL Backbone pretrained weight")
parser.add_argument("-b", "--batch_size",type=int, default=512,  help="batch_size for evaluation")
parser.add_argument("-d", "--metric",type=str, default="accuracy_1_5_torchmetric", choices=["accuracy_1_5_torchmetric",  "accuracy_1_5", "Mean_average_per_cls"], help="Which metric to use")
parser.add_argument("-t", "--task", help="linear_eval or finetune")
parser.add_argument("-ra", "--RandAug", type=bool, default=False, help="linear_eval or finetune")
args = parser.parse_args()

kwargs = {
    "num_classes": 1000,
    "cifar": False,
    "precision": 16,
    "lars": False,
    "auto_lr_find": False,# auto
    "exclude_bias_n_norm": False,
    "gpus": 2,    # Number of GPUs
    "lr_decay_steps": [15, 25, 35], #[30, 45, 65] --> For Linear Evaluation, Semi-Supervised
    "batch_size": args.batch_size,
    "num_workers": 20,
    "metric": args.metric,
    "task": args.task,
    "backbone_weights": args.weight_path, 
    "RandAug": args.RandAug, 
    "num_transfs":2,  
    "magni_transfs": 5,
}
hyperparameter_default= dict(
    lr=1e-2, 
    weight_decay= 1e-6, 
    epochs=30, 
    lr_scheduler="reduce",
)

wandb.init(config= hyperparameter_default, 
    name= f"{args.DATASET}, {args.METHOD}", 
    project= "HAPiCLR_Downstream_task", 
    entity='mlbrl', 
)

config= wandb.config 
model = DownstreamLinearModule_sweep(hypers=config, **kwargs)
dataloader = DownstreamDataloader(args.DATASET, root_dir=args.root_dir, download=False, task=kwargs['task'], batch_size=kwargs["batch_size"], num_workers=kwargs['num_workers'], 
RandAug=kwargs['RandAug'],num_transfs=kwargs['num_transfs'], magni_transfs=kwargs['magni_transfs'])

wandb_logger = WandbLogger(
    # name of the experiment
    #name=f'{METHOD}_semi-supervised_{DATASET}_lr={kwargs["lr"]}_lr_schedu={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}',
    # name=f'{METHOD}_linear_eval_{DATASET}_lr={kwargs["lr"]}_lr_sched={kwargs["scheduler"]}_{str(kwargs["lr_decay_steps"])}_wd={kwargs["weight_decay"]}_batch={kwargs["batch_size"]}_RA={kwargs["num_transfs"], kwargs["magni_transfs"]}_opti_Adamw',
    # project="MNCRL_downstream_tasks_1",  # name of the wandb project
    entity='mlbrl',
    group=args.DATASET,
    offline=False,
)
wandb_logger.watch(model, log="all", log_freq=50)
trainer = Trainer(accelerator='gpu', auto_select_gpus=False, gpus=kwargs["gpus"],
                   logger=wandb_logger, max_epochs=config.epochs, auto_lr_find=kwargs["auto_lr_find"], strategy="ddp")

if __name__ == '__main__':
    print("start training")
    print(f"Weights : {args.weight_path}")
    trainer.fit(model, dataloader)
    #trainer.test(model, dataloader)
    print("end training")

# def train():
#     print("start training")
#     print(f"Weights : {WEIGHTS}")
#     # print(train_loader)
#     # trainer.fit(model, train_loader, val_loader)
#     # if VAL_PATH != TEST_PATH:
#         # trainer.validate(model, test_loader)
#     print("end training")