python train.py --method 'DenseCLR' --dataset 'cifar10' --task 'linear_eval' --lr 0.01 --weight_decay 0.00002124 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'cifar10' --task 'finetune' --lr 0.05 --weight_decay 0.000018 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'cifar100' --task 'linear_eval' --lr 0.01 --weight_decay 0.00002124 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'cifar100' --task 'finetune' --lr 0.05 --weight_decay 0.000018 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'food-101' --task 'linear_eval' --lr 0.01292 --weight_decay 0.000000558 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'food-101' --task 'finetune' --lr 0.002784 --weight_decay 0.00001129  --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'SUN397' --task 'linear_eval' --lr 0.0145  --weight_decay 0.0001184 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'SUN397' --task 'finetune' --lr 0.1001  --weight_decay 0.00005988 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'DTD' --task 'linear_eval' --lr 0.04093 --weight_decay 0.000893 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'DTD' --task 'finetune' --lr 0.05723 --weight_decay 0.00006045 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'cars196' --task 'linear_eval' --lr 0.01292  --weight_decay 0.00000558 --lr_sch "reduce"
python train.py --method 'DenseCLR' --dataset 'cars196' --task 'finetune' --lr 0.002784 --weight_decay 0.00001129 --lr_sch "reduce"
