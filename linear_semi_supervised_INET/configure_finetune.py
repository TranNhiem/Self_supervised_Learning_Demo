DATASET = 'ten_per'
METHOD = 'HAPiCLR_SimCLR'
#SCHEDULER = 'step'
#INIT_LR=0.05
batch_size=512
#EPOCHS=60
#weight_decay=1e6
RandAug= False
num_transfs= 1
magni_transfs= 7
task="finetune"
metric="accuracy_1_5_torchmetric" # accuracy_1_5 , Mean_average_per_cls
#WEIGHTS = '/data/downstream_tasks/MNCRL_lr0.2-beta_cosine_0.9-R50-imagenet-1000ep-mask-cropping0.3-999.ckpt' 
#WEIGHTS = '/data/downstream_tasks/semi_supervised/MV_MASSL_SimCLR_RA_FA_224_2_94_3_CropRatio_0.3_1.0_0.1_0.3_Lossobj_2_res50_imagenet-100ep.ckpt' 
#WEIGHTS='/data/downstream_tasks/HAPiCLR/moco_v2_800ep_pretrain.pth.tar'
#WEIGHTS='/data/downstream_tasks/HAPiCLR/Classification/imagenet-mocov2plus+pixel_level_contrastive_background_singal-dim2048-ep=99.ckpt'
WEIGHTS='/data/downstream_tasks/HAPiCLR/Classification/mscrl-imagenet-simclr+pixel_level_contrastive_background-dim1024-paperep=99.ckpt'