#! /home/assaf/EDOF/venv/bin/python

import argparse
from TorchTrainer import TorchTrainer
from time import time
from image_utils import pad_2d
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path
import sys
sys.path.append('../ISP')
import tabel_detect
plt.switch_backend('tkagg')


def crop_center(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch training module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', help='path to pre-trained model')
    parser.add_argument('-i', '--inference_img', nargs='+')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing)', type=int, default=0)
    parser.add_argument('-gt', '--GT', default=None)
    parser.add_argument('-dm', '--demosaicing', action='store_true', default=False)
    parser.add_argument('-o', '--out_path', default=None)
    parser.add_argument('-c', '--ref_checker', default=None)
    parser.add_argument('-rp', '--random_patch', type=int, default=None)
    parser.add_argument('-r', '--rot90', action='store_true', default=False)

    args = parser.parse_args()
    return args

def demosaic():
    pass

def main():

    args = get_args()
    factors = None
    if args.ref_checker:
        ref = cv2.imread(args.ref_checker)
        factors = tabel_detect.calc_factors(ref)
    trainer = TorchTrainer.warm_startup(root=args.model_path, gpu_index=args.gpu_index)
    trainer.model.eval()
    with torch.no_grad():
        if args.inference_img:
            for im_path in args.inference_img:
                t0 = time()
                if 'tif' in im_path:
                    if args.demosaicing:
                        im_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
                        max16bit = np.iinfo(np.uint16).max
                        im_raw = np.clip(im_raw, 0, max16bit)
                        im_raw = ((im_raw / (2 ** 16)) * 255).astype(np.uint8)
                        color = cv2.demosaicing(im_raw, cv2.COLOR_BayerBG2RGB)
                    else:
                        im_raw = cv2.imread(im_path).astype(np.uint8)
                        color = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB)
                else:
                    color = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)

                # color = crop_center(color, 4092, 2048)
                color = pad_2d(color, 32)
                if factors is not None:
                    fixed = color.astype(np.int32)
                    for i in range(3):
                        fixed[:,:, i] = fixed[:,:, i] * factors[i]
                    color = np.clip(fixed, 0, 255)
                if args.rot90:
                    color = np.ascontiguousarray(np.rot90(color))
                inputs = torch.from_numpy(np.transpose(color/255, (2, 0, 1))).float().to(trainer.device)
                inputs = inputs.unsqueeze(0)
                out = trainer.model(inputs)
                if out.shape[1] > 3:  # segmentation
                    outs = out.argmax(dim=1).squeeze()
                    out_im = outs.cpu().numpy()
                    out_im = out_im.astype(np.uint8)
                else:
                    out_np = out.cpu().numpy()
                    out_im = np.squeeze(out_np, axis=0)
                    out_im = out_im.transpose(1, 2, 0)
                    out_im = np.clip(out_im, 0, 1)
                    out_im = (out_im * 255).astype(np.uint8)

                print('calc time = %.2f' % (time() - t0))

                if args.out_path is not None:
                    out_name = os.path.basename(im_path)
                    out_name = out_name.split('.')[0] + '.png'
                    out_path = os.path.join(args.out_path, out_name)
                    out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_path, out_im * (255 / np.max(out_im)))
                else:
                    if args.GT:
                        gt = trainer.dataset.depth_read(args.GT)
                        rows = 3
                    else:
                        rows = 2
                    fig, axes = plt.subplots(nrows=1, ncols=rows, sharex=True, sharey=True)
                    axes[0].imshow(color), axes[0].set_title('in')
                    axes[1].imshow(out_im, cmap='jet'), axes[1].set_title('out')
                    if args.GT:
                        axes[2].imshow(gt), axes[2].set_title('GT')
                    plt.show()
        elif args.random_patch:
            _, _ = trainer.dataset.get_data_loaders(batch_size=1)
            trainer.dataset.sigma = 0
            trainer.dataset.augmentations = []
            for p_id in np.random.choice(len(trainer.dataset), args.random_patch, replace=False):
                img, lbl = trainer.dataset[p_id]
                inputs = torch.from_numpy(img).float().to(trainer.device)
                inputs = inputs.unsqueeze(0)
                out = trainer.model(inputs)
                out_np = out.cpu().numpy()
                out_im = np.squeeze(out_np, axis=0)
                out_im = out_im.transpose(1, 2, 0)
                out_im = np.clip(out_im, 0, 1)
                fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
                axes[0].imshow(img.transpose(1, 2, 0)), axes[0].set_title('in')
                axes[1].imshow(lbl.transpose(1, 2, 0)), axes[1].set_title('lbl')
                axes[2].imshow(out_im), axes[2].set_title('out')
                plt.show()

if __name__ == '__main__':
    main()
