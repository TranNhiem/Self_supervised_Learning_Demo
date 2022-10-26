python train.py --METHOD 'DenseCLR' --DATASET 'one_per' --task 'finetune' --lr 0.0785159  --weight_decay 0.000082788 --lr_sch "step" 
python train.py --METHOD 'DenseCLR' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499 --weight_decay 0.0000373650 --lr_sch "step" 
python train.py --METHOD 'DenseCLR' --DATASET 'imagenet' --task 'linear' --lr 0.2  --weight_decay 5e-06 --lr_sch "step" --epoch 90 
python train.py --METHOD 'PixelPro' --DATASET 'one_per' --task 'finetune' --lr 0.0785159  --weight_decay 0.000082788 --lr_sch "step" 
python train.py --METHOD 'PixelPro' --DATASET 'ten_per' --task 'finetune' --lr 0.0959499  --weight_decay 0.0000373650 --lr_sch "step" 
python train.py --METHOD 'PixelPro' --DATASET 'imagenet' --task 'linear' --lr 0.2 --weight_decay 5e-06 --lr_sch "step" --epoch 90 
