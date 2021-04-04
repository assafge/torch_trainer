#!/usr/bin/env python
import argparse
from TorchTrainer import TorchTrainer
# from time import time
from image_utils import pad_2d, num_of_channels
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path
from pathlib import Path
from glob import glob
# sys.path.append('../ISP')
# import tabel_detect
import scipy.io as sio
from general_utils import print_progress
# plt.switch_backend('tkagg')
import colour_demosaicing
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse


def crop_center(img,cropy,cropx):
    y, x = img.shape[:2]
    cropy, cropx = min(cropy, img.shape[0]), min(cropx, img.shape[1])

    sx = x//2-(cropx//2)
    sx -= sx % 2
    sy = y//2-(cropy//2)
    sy -= sy % 2
    return img[sy:sy+cropy, sx:sx+cropx]


def run_model(trainer, in_img):
    inputs = torch.from_numpy(in_img).float().to(trainer.device)
    inputs = inputs.unsqueeze(0)
    if inputs.ndim < 4:
        inputs = inputs.unsqueeze(0)

    out = trainer.model(inputs)
    return out


def inference_image(trainer: TorchTrainer, im_path: str, factors: np.ndarray = None,
                    demosaic: bool =False, rotate: bool = False, bit_depth: int = 8,
                    raw_result: bool = False, do_crop: bool = False, gray: bool = False,
                    fliplr: bool = False, boost: bool = False, split_inference: bool = False):
    max_bit = (2**bit_depth) - 1
    if demosaic:
        # im_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        im_raw = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        # im_raw = cv2.demosaicing(im_raw, cv2.COLOR_BayerBG2RGB).astype(np.float32)
        im_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(im_raw, pattern='GRBG')
        # im_raw = cv2.demosaicing(im_raw, cv2.COLOR_BayerRG2RGB).astype(np.float32)

    elif gray:
        im_raw = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    else:
        im_raw = cv2.imread(im_path)
        im_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB).astype(np.float32)

    if fliplr:
        im_raw = np.fliplr(im_raw)

    if factors is not None:
        im_raw = im_raw * factors

    if boost:
        p = np.percentile(im_raw, 98)
        # print('p', p)
        im_raw = im_raw.astype(np.float) * (max_bit/p)
    in_img = np.clip(im_raw, 0, max_bit) / max_bit
    if rotate:
        in_img = np.rot90(in_img)
    org_shape = in_img.shape[:2]
    if do_crop:
        in_img = crop_center(in_img, 2048, 2048)
    return_img = (in_img * 255).astype(np.uint8)
    if not do_crop:
        in_img = pad_2d(in_img, 32).astype(np.float32)
    if in_img.ndim > 2:
        in_img = np.transpose(in_img, (2, 0, 1))

    if split_inference:
        im_shape = np.array(in_img.shape)
        mid_point = im_shape // 2
        mid_point = mid_point - (mid_point % 2)

    else:
        out = run_model(trainer, in_img)
    if raw_result:
        out_im = out.squeeze().cpu().numpy()
        out_im = out_im.transpose(1, 2, 0)
        return out_im, return_img
    if out.ndim > 3 and out.shape[1] > 5:  # segmentation
        outs = out.argmax(dim=1).squeeze()
        out_im = outs.cpu().numpy()
        out_im = out_im.astype(np.uint8)

    elif out.ndim > 3:  # im2im
        out_np = out.cpu().numpy()
        if out_np.ndim > 2:
            out_np = np.squeeze(out_np, axis=0)
            out_np = out_np.transpose((1, 2, 0))
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
            out_im = crop_center(out_im, org_shape[0], org_shape[1])
    if fliplr:
        out_im = np.fliplr(out_im)
        return_img = np.fliplr(return_img)
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
        out_im = out_im.transpose((1, 2, 0))
        out_im = np.clip(out_im, 0, 1)
        fig, axes = plt.subplots(nrows=1, ncols=3, sharex='true', sharey='true')
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

def save_image(img: np.ndarray, in_img: np.ndarray, in_im_path: str, model_name: str, out_dir: str,
               mat_out: bool, do_crop: bool):
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
        # out_im = img
        cv2.imwrite(out_path, out_im)
    if do_crop:
        if in_img.ndim > 2 and\
                not os.path.exists(in_im_path.replace('.png', '_bilinear-demosaic_crop.png')):
            in_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(in_im_path.replace('.png', '_bilinear-demosaic_crop.png'), in_img)
        if not os.path.exists(in_im_path.replace('_crop.png', '_center_crop.png')):
            cv2.imwrite(in_im_path.replace('_crop.png', '_center_crop.png'), in_img)


def display_result(gt_path, trainer, out_im: np.ndarray, in_img: np.ndarray, rot90, do_crop):
    print(in_img.shape)
    if rot90:
        out_im = np.rot90(out_im)
        in_img = np.rot90(in_img)

    if gt_path:
        # gt = trainer.dataset.depth_read(GT_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if num_of_channels(gt) == 3:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        if rot90:
            gt = np.rot90(gt)
        if do_crop:
            gt = crop_center(gt, 2048, 2048)
        cols = 3
    else:
        cols = 2
    if out_im.shape[2] == 4:
        cols += 1
    fig, axes = plt.subplots(nrows=1, ncols=cols, sharex=True, sharey=True)
    axes[0].imshow(in_img), axes[0].set_title('in')
    if out_im.ndim == 2:
        axes[1].imshow(out_im, cmap='jet', interpolation=None)
    else:
        axes[1].imshow(out_im[:, :, :3])
    axes[1].set_title('out')
    if cols >= 3:
        if gt_path:
            axes[2].imshow(gt), axes[2].set_title('GT')
    if out_im.shape[2] == 4:
        axes[-1].imshow(out_im[:, :, 3], cmap='gray'), axes[-1].set_title('IR')

    plt.show()


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', help='path to pre-trained model')
    parser.add_argument('-i', '--images_path', nargs='+', help='list of image or folder or txt file')
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
    parser.add_argument('-lr', '--fliplr', action='store_true', default=False,
                        help='flip before inference and flip back afterwards - in order to use RGGB pattern on GRBG')
    parser.add_argument('-r', '--mat_out', action='store_true', default=False,
                        help='output the classification results (without argmax)')
    parser.add_argument('-b', '--bit_depth', type=int, default=8, help='input image bit depth')
    parser.add_argument('-m', '--im_pattern', default='*ids_crop.png', help='images regex pattern')
    parser.add_argument('-s', '--boost_image', action='store_true', help='auto gain per image')
    parser.add_argument('-ti', '--test_images', action='store_true', help='infer test images')
    # parser.add_argument('--check_patt', default='*mask.tif', help='input image file pattern')
    args = parser.parse_args()
    assert not (args.mat_out and not (args.mat_out ^ (args.out_type is None))), 'out path is required for mat'
    if not args.test_images:
        for in_path in args.images_path:
            assert os.path.exists(in_path), 'ERROR - path is not exist %s' % in_path

    return args

def main():
    args = get_args()

    print('reading model...')
    trainer = TorchTrainer.warm_startup(root=args.model_path, gpu_index=args.gpu_index, strict=True, best=True)
    trainer.model.eval()
    with torch.no_grad():
        if args.images_path:
            # if args.ref_checker:
            #     ref = cv2.imread(args.ref_checker)
                # factors = tabel_detect.calc_factors(ref / ((2 ** args.bit_depth) - 1))
                # print('INFO - factors:', factors)
            if args.factors:
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
                elif in_path[-4:] == '.txt':
                    with open(in_path) as f:
                        for _ in range(15):
                            images.append(f.readline().strip())
                else:
                    images.append(in_path)
            assert len(images) > 0, 'WARNING - images list is empty, check glob input: {}'.format(glob_patt)
            for i, im_path in enumerate(images):
                print_progress(i, total=len(images), suffix='inference {}{}'.format(im_path, ' '*20), length=20)
                out_im, in_img = inference_image(trainer, im_path=im_path, factors=factors,
                demosaic=args.demosaic, rotate=args.rot90, bit_depth=args.bit_depth, raw_result=args.mat_out,
                do_crop=args.do_crop, gray=args.gray, fliplr=args.fliplr, boost=args.boost_image)

                if args.out_type is not None:
                    out_dir = os.path.join(in_path, args.out_type, model_name)
                    save_image_type(out_im, im_path, out_dir=out_dir, mat_out=args.mat_out)
                elif args.out_path is not None:
                    save_image(out_im, in_img, im_path, model_name, args.out_path,
                               mat_out=args.mat_out, do_crop=args.do_crop)
                    # save_image(in_img, im_path, 'org', args.out_path, mat_out=args.mat_out)
                else:
                    display_result(gt_path=args.GT, trainer=trainer, out_im=out_im, in_img=in_img, rot90=args.rot90,
                                   do_crop=args.do_crop)
            print_progress(len(images), total=len(images), suffix='inferenced {} images {}'.format(len(images), ' ' * 80), length=20)

        elif args.random_images:
            inference_random_patch(trainer, args.random_images)
        elif args.test_images:
            model_dir = Path(args.model_path)
            test_df = pd.read_csv(model_dir.joinpath('test_images.csv'))
            test_dir = model_dir.joinpath('test_images')
            test_dir.mkdir(exist_ok=True)
            mse_rgb_ir = []
            ssim_rgb_ir = []
            psnr_rgb_ir = []
            for _, im_path in tqdm(test_df.iterrows()):
                out_im, in_img = inference_image(trainer, im_path=im_path.full)
                ref_rgb = cv2.cvtColor(cv2.imread(im_path.rgb, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                ref_ir = cv2.imread(im_path.ir, cv2.IMREAD_GRAYSCALE)
                mse_rgb_ir.append((mse(ref_rgb, out_im[:, :, :3]), mse(ref_ir, out_im[:, :, 3])))
                rgb_range = out_im[:, :, :3].max() - out_im[:, :, :3].min()
                ir_range = out_im[:, :, 3].max() - out_im[:, :, 3].min()
                ssim_rgb_ir.append((ssim(ref_rgb, out_im[:, :, :3], data_range=rgb_range, multichannel=True),
                                    ssim(ref_ir, out_im[:, :, 3], data_range=ir_range)))
                psnr_rgb_ir.append(10 * np.log10((255.0**2) / np.array(mse_rgb_ir[-1])))
                rgb_out = test_dir.joinpath(Path(im_path.rgb).stem + '_est.png')
                ir_out = test_dir.joinpath(Path(im_path.ir).stem + '_est.png')
                cv2.imwrite(str(rgb_out), cv2.cvtColor(out_im[:, :, :3], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(ir_out), out_im[:, :, 3])
            test_df[['mse_rgb', 'mse_ir']] = mse_rgb_ir
            test_df[['ssim_rgb', 'ssim_ir']] = ssim_rgb_ir
            test_df[['psnr_rgb', 'psnr_ir']] = psnr_rgb_ir
            test_df.to_csv(model_dir.joinpath('test_images.csv'), index_label=False, index=False)


if __name__ == '__main__':
    main()
