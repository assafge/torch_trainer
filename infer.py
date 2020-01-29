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
from glob import glob
sys.path.append('../ISP')
import tabel_detect
plt.switch_backend('tkagg')


def crop_center(img,cropy,cropx):
    y, x = img.shape[:2]
    sx = x//2-(cropx//2)
    sy = y//2-(cropy//2)
    return img[sy:sy+cropy, sx:sx+cropx]


def inference_image(trainer: TorchTrainer, factors: np.ndarray,
                    im_path: str, demosaic: bool, rotate: bool, bit_depth: int):
    max_bit = (2**bit_depth) - 1
    if 'tif' in im_path and demosaic:
        im_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        im_raw = cv2.demosaicing(im_raw, cv2.COLOR_BayerBG2RGB).astype(np.float32)

    else:
        im_raw = cv2.imread(im_path)
        im_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB).astype(np.float32)
    if factors is not None:
        im_raw = im_raw * factors

    in_img = np.clip(im_raw, 0, max_bit) / max_bit
    if rotate:
        in_img = np.rot90(in_img)
    in_img = crop_center(in_img, min(2048, in_img.shape[0]), min(2048, in_img.shape[1]))
    in_img = pad_2d(in_img, 32).astype(np.float32)

    inputs = torch.from_numpy(np.transpose(in_img, (2, 0, 1))).float().to(trainer.device)
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

    return_img = (in_img * 255).astype(np.uint)
    return out_im, return_img


def inference_random_patch(trainer, num_images):
    _, _ = trainer.dataset.get_data_loaders(batch_size=1)
    trainer.dataset.sigma = 0
    trainer.dataset.augmentations = []
    for p_id in np.random.choice(len(trainer.dataset), num_images, replace=False):
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


def save_image(img, in_im_path, out_dir):
    out_name = os.path.basename(in_im_path)
    out_name = out_name.split('.')[0] + '.png'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_name)
    out_im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_im)


def display_result(GT_path, trainer, out_im, in_img, rot90):
    if rot90:
        out_im = np.rot90(out_im)
        in_img = np.rot90(in_img)
    if out_im.ndim == 2:
        cmap = 'jet'
    else:
        cmap = 'gray'
    if GT_path:
        gt = trainer.dataset.depth_read(GT_path)
        if rot90:
            gt = np.rot90(gt)
        rows = 3
    else:
        rows = 2
    fig, axes = plt.subplots(nrows=1, ncols=rows, sharex=True, sharey=True)
    axes[0].imshow(in_img), axes[0].set_title('in')
    axes[1].imshow(out_im, cmap=cmap), axes[1].set_title('out')
    if rows == 3:
        axes[2].imshow(gt, cmap='jet'), axes[2].set_title('GT')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch training module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', help='path to pre-trained model')
    parser.add_argument('-i', '--images_path', nargs='+')
    parser.add_argument('-f', '--factors', nargs='+', type=float)
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing)', type=int, default=0)
    parser.add_argument('-gt', '--GT', default=None)
    parser.add_argument('-dm', '--demosaic', action='store_true', default=False)
    parser.add_argument('-o', '--out_path', default=None, help='path to output folder')
    parser.add_argument('-c', '--ref_checker', default=None)
    parser.add_argument('-rp', '--random_images', type=int, default=None,
                        help='select the number of random images, taken from a data set')
    parser.add_argument('-r', '--rot90', action='store_true', default=False)
    parser.add_argument('-b', '--bit_depth', type=int, default=8, help='input image bit depth')
    parser.add_argument('-m', '--im_pattern', default='*mask.*', help='images regex pattern')
    # parser.add_argument('--check_patt', default='*mask.tif', help='input image file pattern')




    args = parser.parse_args()
    return args


def main():
    args = get_args()
    trainer = TorchTrainer.warm_startup(root=args.model_path, gpu_index=args.gpu_index)
    trainer.model.eval()
    with torch.no_grad():
        if args.images_path:
            if len(args.images_path) == 1 and os.path.isdir(args.images_path[0]):
                # given path is a directory
                images = glob(os.path.join(args.images_path[0], args.im_pattern))
                assert len(images) > 0, "ERROR - images lis is empty, check parameters \n{}".format(vars(args))
            else:
                # list of images
                images = args.images_path
            if args.ref_checker:
                ref = cv2.imread(args.ref_checker)
                factors = tabel_detect.calc_factors(ref/((2**args.bit_depth) - 1))
                print('INFO - factors:', factors)
            elif args.factors:
                factors = np.array(args.factors)
            else:
                factors = None
            for im_path in images:
                out_im, in_img = inference_image(trainer, factors, im_path, args.demosaic, args.rot90, args.bit_depth)
                if args.out_path is not None:
                    save_image(out_im, im_path, out_dir=args.out_path)
                else:
                    display_result(args.GT, trainer, out_im, in_img, args.rot90)

        elif args.random_patch:
            inference_random_patch(trainer, args.random_images)

if __name__ == '__main__':
    main()
