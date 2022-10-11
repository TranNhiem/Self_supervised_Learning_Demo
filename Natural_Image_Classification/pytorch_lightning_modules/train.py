from downstream_modules import DatasetTrainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--method", help="MASSL or Baseline")
parser.add_argument("-d", "--dataset", help="Which dataset")
parser.add_argument("-t", "--task", help="linear_eval or finetune")
args = parser.parse_args()

WEIGHTS = {
    'MASSL': '/home/rick/offline_finetune/Weights/MASSL_2MLP_512_290.ckpt',
    'Baseline': '/home/rick/offline_finetune/Weights/Baseline_300epoch.ckpt',
}

kwargs = {
    'max_steps': 5000,
    'lr': 0.01,
    'weight_decay': 0.0001,
    'scheduler': 'step',
    'lr_decay_steps': [24, 48, 72],
    'replica_batch_size': 64,
    'batch_size': 256,
    'num_workers': 4,
    'gpus': [1, 2, 3, 4],
}

trainer = DatasetTrainer(method=args.method, dataset_name=args.dataset, task=args.task, backbone_weights=WEIGHTS[args.method], **kwargs)

trainer.run()