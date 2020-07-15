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
import scipy.io as sio
from general_utils import print_progress
# plt.switch_backend('tkagg')
import colour_demosaicing



def crop_center(img,cropy,cropx):
    y, x = img.shape[:2]
    sx = x//2-(cropx//2) + cropx % 2
    sy = y//2-(cropy//2) + cropy % 2
    return img[sy:sy+cropy, sx:sx+cropx]


def inference_image(trainer: TorchTrainer, factors: np.ndarray, im_path: str,
                    demosaic: bool, rotate: bool, bit_depth: int, raw_result: bool, do_crop: bool, gray: bool):
    max_bit = (2**bit_depth) - 1
    if demosaic:
        print('dm')
        # im_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        im_raw = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        # im_raw = cv2.demosaicing(im_raw, cv2.COLOR_BayerBG2RGB).astype(np.float32)
        im_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(im_raw, pattern='GRBG')
        # im_raw = cv2.demosaicing(im_raw, cv2.COLOR_BayerRG2RGB).astype(np.float32)

    elif gray:
        print('gray')
        im_raw = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    else:
        im_raw = cv2.imread(im_path)
        im_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(im_raw.ndim)

    if factors is not None:
        im_raw = im_raw * factors

    in_img = np.clip(im_raw, 0, max_bit) / max_bit
    if rotate:
        in_img = np.rot90(in_img)
    org_shape = in_img.shape[:2]
    if do_crop:
        in_img = crop_center(in_img, min(2048, in_img.shape[0]), min(2048, in_img.shape[1]))
    return_img = (in_img * 255).astype(np.uint8)
    if not do_crop:
        in_img = pad_2d(in_img, 32).astype(np.float32)

    if in_img.ndim > 2:
        in_img = np.transpose(in_img, (2, 0, 1))
    inputs = torch.from_numpy(in_img).float().to(trainer.device)
    inputs = inputs.unsqueeze(0)
    if inputs.ndim < 4:
        inputs = inputs.unsqueeze(0)
    out = trainer.model(inputs)
    if raw_result:
        out_im = out.squeeze().cpu().numpy()
        out_im = out_im.transpose(1, 2, 0)
        return out_im, return_img
    if out.ndim > 3 and out.shape[1] > 3:  # segmentation
        outs = out.argmax(dim=1).squeeze()
        out_im = outs.cpu().numpy()
        out_im = out_im.astype(np.uint8)

    elif out.shape[1] == 3:  # im2im
        out_np = out.cpu().numpy()
        if out_np.ndim > 2:
            out_np = np.squeeze(out_np, axis=0)
            out_np = out_np.transpose(1, 2, 0)
        out_im = np.clip(out_np, 0, 1)
        out_im = (out_im * 255).astype(np.uint8)
    else:
        out_np = out.squeeze().cpu().numpy()
        out_im = np.clip(out_np, -4, 10)
        out_im = ((out_im + 4) * (255/15)).astype(np.uint8)

    # out_im = np.pad(out_im, pad_width=((2, 2), (1, 1), (0, 0)), mode='constant')
    # remove the padding
    if not do_crop:
        if out_im.shape[:2] != org_shape[:2]:
            print('cropping')
            out_im = crop_center(out_im, org_shape[0], org_shape[1])
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


def save_image_type(img: np.ndarray, in_im_path: str, out_dir: str, mat_out: bool):
    out_name = os.path.basename(in_im_path)
    out_name = out_name.split('.')[0]
    out_name += '.mat' if mat_out else '.png'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_name)
    if mat_out:
        sio.savemat(out_path, {'dpt': img})
    else:
        out_im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out_im)

def save_image(img: np.ndarray, in_im_path: str, model_name: str, out_dir: str, mat_out: bool):
    out_name = os.path.basename(in_im_path)
    out_name = out_name.split('.')[0] + '_' + model_name
    out_name += '.mat' if mat_out else '.png'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_name)
    if mat_out:
        sio.savemat(out_path, {'dpt': img})
    else:
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
    parser.add_argument('-o', '--out_type', default=None, help='write the image in sub directory of the input image')
    parser.add_argument('-w', '--out_path', default=None, help='path to output file')
    parser.add_argument('-f', '--factors', nargs='+', type=float)
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing)', type=int, default=0)
    parser.add_argument('-gt', '--GT', default=None)
    parser.add_argument('-dm', '--demosaic', action='store_true', default=False)
    parser.add_argument('-gr', '--gray', action='store_true', default=False)
    parser.add_argument('-c', '--ref_checker', default=None)
    parser.add_argument('-rp', '--random_images', type=int, default=None,
                        help='select the number of random images, taken from a data set')
    parser.add_argument('-t', '--rot90', action='store_true', default=False)
    parser.add_argument('-cr', '--do_crop', action='store_true', default=False)
    parser.add_argument('-r', '--mat_out', action='store_true', default=False,
                        help='output the classification results (without argmax)')
    parser.add_argument('-b', '--bit_depth', type=int, default=8, help='input image bit depth')
    parser.add_argument('-m', '--im_pattern', default='*mask*', help='images regex pattern')
    # parser.add_argument('--check_patt', default='*mask.tif', help='input image file pattern')
    args = parser.parse_args()
    assert not (args.mat_out and not (args.mat_out ^ (args.out_type is None))), 'out path is required for mat'
    for in_path in args.images_path:
        assert os.path.exists(in_path), 'ERROR - path is not exist %s' % in_path

    return args


def main():
    args = get_args()

    print('reading model...')
    trainer = TorchTrainer.warm_startup(root=args.model_path, gpu_index=args.gpu_index, strict=True)
    trainer.model.eval()
    with torch.no_grad():
        if args.images_path:
            if args.ref_checker:
                ref = cv2.imread(args.ref_checker)
                factors = tabel_detect.calc_factors(ref / ((2 ** args.bit_depth) - 1))
                print('INFO - factors:', factors)
            elif args.factors:
                factors = np.array(args.factors)
            else:
                factors = None

            model_name = os.path.basename(os.path.normpath(args.model_path))
            images = []
            for in_path in args.images_path:
                if os.path.isdir(in_path):
                    # given path is a directory
                    glob_patt = os.path.join(in_path, args.im_pattern)
                    images.extend(glob(glob_patt))
                else:
                    images.append(in_path)
                if len(images) == 0:
                    print("WARNING - images list is empty, check glob input: {}".format(glob_patt))
                for i, im_path in enumerate(images):
                    print_progress(i, total=len(images), suffix='inference {}{}'.format(im_path, ' '*10), length=20)
                    out_im, in_img = inference_image(trainer, factors=factors, im_path=im_path,
                    demosaic=args.demosaic, rotate=args.rot90, bit_depth=args.bit_depth, raw_result=args.mat_out,
                    do_crop=args.do_crop, gray=args.gray)

                    if args.out_type is not None:
                        out_dir = os.path.join(in_path, args.out_type, model_name)
                        save_image_type(out_im, im_path, out_dir=out_dir, mat_out=args.mat_out)
                    elif args.out_path is not None:
                        save_image(out_im, im_path, model_name, args.out_path, mat_out=args.mat_out)
                        save_image(in_img, im_path, 'org', args.out_path, mat_out=args.mat_out)
                    else:
                        display_result(GT_path=args.GT, trainer=trainer, out_im=out_im, in_img=in_img,rot90=args.rot90)
                print_progress(len(images), total=len(images), suffix='inferenced {} images {}'.format(len(images), ' ' * 80), length=20)

        elif args.random_images:
            inference_random_patch(trainer, args.random_images)

if __name__ == '__main__':
    main()
