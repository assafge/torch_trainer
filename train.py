import argparse
import torch
import os
from TorchTrainer import TorchTrainer


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch training module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('g', 'gpu_index', help='index of gpu (if exist, torch indexing)', type=int, default=0)
    new = parser.add_argument_group('new model')
    new.add_argument('-m', '--model_cfg', help='path to model cfg file')
    new.add_argument('-o', '--optimizer_cfg', help='path to optimizer cfg file')
    new.add_argument('-m', '--dataset_cfg', help='path to dataset cfg file')
    new.add_argument('w', 'output_dir', help='path to output directory')
    retrain = parser.add_argument_group('warm startup')
    retrain.add_argument('r', 'model_path', help='path to pre-trained model')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    trainer = TorchTrainer.new_train(model_cfg=args.model_cfg, optimizer_cfg=args.optimizer_cfg,
                                     dataset_cfg=args.dataset_cfg, gpu_index=args.gpu_index)
    trainer.train()

if __name__ == '__main__':
    main()
