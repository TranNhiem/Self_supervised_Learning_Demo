python train.py --METHOD 'HAPiCLR_m' --DATASET 'one_per' --task 'finetune' --Init_lr 0.0785159 --weight_decay 0.000082788 --lr_sch "step" 
# python train.py --METHOD 'HAPiCLR_m' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499 --weight_decay 0.0000373650 --lr_sch "step" 
python train.py --METHOD 'HAPiCLR_m' --DATASET 'imagenet' --task 'ImageNet_linear' --Init_lr 0.2 --weight_decay 5e-06 --lr_sch "step" --epoch 90
python train.py --METHOD 'HAPiCLR_m' --DATASET 'one_per' --task 'finetune' --Init_lr 0.0497011 --weight_decay 0.00001893 --lr_sch "reduce" 
# python train.py --METHOD 'HAPiCLR_m' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499 --weight_decay 0.0000373650 --lr_sch "reduce" 
python train.py --METHOD 'HAPiCLR_m' --DATASET 'imagenet' --task 'ImageNet_linear' --Init_lr 0.2 --weight_decay 5e-06 --lr_sch "reduce" --epoch 90
python train.py --METHOD 'HAPiCLR_s' --DATASET 'one_per' --task 'finetune' --Init_lr 0.0785159 --weight_decay 0.000082788 --lr_sch "step" 
# python train.py --METHOD 'HAPiCLR_s' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499 --weight_decay 0.0000373650 --lr_sch "step"
python train.py --METHOD 'HAPiCLR_s' --DATASET 'imagenet' --task 'ImageNet_linear' --Init_lr 0.2 --weight_decay 5e-06 --lr_sch "step" --epoch 90
python train.py --METHOD 'HAPiCLR_s' --DATASET 'one_per' --task 'finetune' --Init_lr 0.04970118 --weight_decay 0.00001893 --lr_sch "reduce" 
# python train.py --METHOD 'HAPiCLR_s' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499 --weight_decay 0.0000373650 --lr_sch "reduce" 
python train.py --METHOD 'HAPiCLR_s' --DATASET 'imagenet' --task 'ImageNet_linear' --Init_lr 0.2 --weight_decay 5e-06 --lr_sch "reduce" --epoch 90

