DATASET = 'ImageNet' # for the Linear Evaluation 
METHOD = 'HAPiCLR-MoCo v2' #
#SCHEDULER = 'step'#
#INIT_LR=0.2 #
batch_size=2048 #
#EPOCHS=90 #
task="ImageNet_linear"
metric="accuracy_1_5_torchmetric"
#weight_decay=0 #5e-7
RandAug= False
num_transfs= 1
magni_transfs= 10
#WEIGHTS = '/data/downstream_tasks/MNCRL_lr0.2-beta_cosine_0.9-R50-imagenet-1000ep-mask-cropping0.3-999.ckpt' 
#WEIGHTS = '/data/downstream_tasks/semi_supervised/MV_MASSL_SimCLR_RA_FA_224_2_94_3_CropRatio_0.3_1.0_0.1_0.3_Lossobj_2_res50_imagenet-100ep.ckpt' 
#WEIGHTS='/data/downstream_tasks/semi_supervised/MV_MASSL_SimCLR_RA_FA_224_2_94_4_CropRatio_0.3_1.0_0.1_0.3_Lossobj_2_res50_imagenet-300ep_Plus_RA-37u4oovj-ep=299.ckpt'
#WEIGHTS='/data/downstream_tasks/simclr-300epoch-2vkp21wv-ep=299.ckpt'
#WEIGHTS='/data/downstream_tasks/MASSL_2MLP_512_290.ckpt'
#WEIGHTS='/data/downstream_tasks/HAPiCLR/moco_v2_800ep_pretrain.pth.tar'
WEIGHTS='/data/downstream_tasks/HAPiCLR/Classification/imagenet-mocov2plus+pixel_level_contrastive_background_singal-dim2048-ep=99.ckpt'