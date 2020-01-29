import torch
import os
from glob import glob
from argparse import ArgumentParser
import yaml

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('nets_folder', type=str, help='path to trained networks folder')
    parser.add_argument('-p', '--parameters', nargs='+', default=None)
    args = parser.parse_args()
    device = torch.device('cpu')
    for net_path in glob(os.path.join(args.nets_folder, '*')):
        out_str = net_path +' : '
        for cp_name, cp_type in zip(['last_checkpoint.pth', 'checkpoint.pth'], ['last', 'best']):
            cp_path = os.path.join(net_path, 'checkpoints', cp_name)
            if os.path.exists(cp_path):
                checkpoint = torch.load(cp_path, map_location=device)
                epoch = checkpoint['epoch']
                out_str += '{}: at epoch {} |'.format(cp_type, epoch)
        print(out_str)
        if args.parameters is not None:
            prm_path = os.path.join(net_path, 'cfg.yaml')
            with open(prm_path) as f:
                lines = f.readlines()
            for line in lines:
                if any(p in line for p in args.parameters):
                    print(line.strip())


