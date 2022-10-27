python train.py --METHOD 'MoCo_V2' --DATASET 'one_per' --task 'finetune' --Init_lr 0.0785159 --weight_decay 0.000082788 --lr_sch "step" 
# python train.py --METHOD 'MoCo_V2' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499 --weight_decay 0.0000373650 --lr_sch "step" 
python train.py --METHOD 'MoCo_V2' --DATASET 'imagenet' --task 'ImageNet_linear' --Init_lr --weight_decay 5e-06 --lr_sch "step" --epoch 90 
python train.py --METHOD 'SimCLR' --DATASET 'one_per' --task 'finetune' --Init_lr 0.0785159 --weight_decay 0.000082788 --lr_sch "step" 
# python train.py --METHOD 'SimCLR' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499 --weight_decay 0.0000373650 --lr_sch "step" 
python train.py --METHOD 'SimCLR' --DATASET 'imagenet' --task 'ImageNet_linear' --Init_lr 0.2 --weight_decay 5e-06 --lr_sch "step" --epoch 90 
