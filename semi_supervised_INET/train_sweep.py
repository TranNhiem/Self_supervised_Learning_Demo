from argparse import Namespace
from distutils.command.config import config
import torch
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet50
from downstream_modules import DownstreamDataloader, DownstreamLinearModule_sweep
#from config_training_linear import *
from configure_finetune import *

kwargs = {
    "num_classes": 1000,
    "cifar": False,
    "precision": 16,
    "lars": False,
    "auto_lr_find": False,# auto
    "exclude_bias_n_norm": False,
    "gpus": 2,    # Number of GPUs
    "lr_decay_steps": [15, 25, 35], #[30, 45, 65] --> For Linear Evaluation, Semi-Supervised
    "batch_size": batch_size,
    "num_workers": 20,
    "metric": metric,
    "task": task,
    "backbone_weights": WEIGHTS, 
    "RandAug": RandAug, 
    "num_transfs": num_transfs,  
    "magni_transfs": magni_transfs
}
hyperparameter_default= dict(
    lr=1e-2, 
    weight_decay= 1e-6, 
    epochs=30, 
    lr_scheduler="reduce",
)

wandb.init(config= hyperparameter_default, 
    name= f"{DATASET}, {METHOD}", 
    project= "HAPiCLR_Downstream_task", 
    entity='mlbrl', 
)

config= wandb.config 
model = DownstreamLinearModule_sweep(hypers=config, **kwargs)
dataloader = DownstreamDataloader(DATASET, download=False, task=kwargs['task'], batch_size=kwargs["batch_size"], num_workers=kwargs['num_workers'], 
RandAug=kwargs['RandAug'],num_transfs=kwargs['num_transfs'], magni_transfs=kwargs['magni_transfs'])

wandb_logger = WandbLogger(
    # name of the experiment
    #name=f'{METHOD}_semi-supervised_{DATASET}_lr={kwargs["lr"]}_lr_schedu={kwargs["scheduler"]}_wd={kwargs["weight_decay"]}',
    # name=f'{METHOD}_linear_eval_{DATASET}_lr={kwargs["lr"]}_lr_sched={kwargs["scheduler"]}_{str(kwargs["lr_decay_steps"])}_wd={kwargs["weight_decay"]}_batch={kwargs["batch_size"]}_RA={kwargs["num_transfs"], kwargs["magni_transfs"]}_opti_Adamw',
    # project="MNCRL_downstream_tasks_1",  # name of the wandb project
    entity='mlbrl',
    group=DATASET,
    offline=False,
)
wandb_logger.watch(model, log="all", log_freq=50)
trainer = Trainer(accelerator='gpu', auto_select_gpus=False, gpus=kwargs["gpus"],
                   logger=wandb_logger, max_epochs=config.epochs, auto_lr_find=kwargs["auto_lr_find"], strategy="ddp")

if __name__ == '__main__':
    print("start training")
    print(f"Weights : {WEIGHTS}")
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