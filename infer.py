#!/usr/bin/env python
try:
    from .trainer import TorchTrainer
    from .image_utils import pad_2d, num_of_channels, mosaic_image
    from .general_utils import print_progress
except ImportError:
    from trainer import TorchTrainer
    from image_utils import pad_2d, num_of_channels, mosaic_image
    from general_utils import print_progress

import argparse
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
# plt.switch_backend('tkagg')
import colour_demosaicing
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse


def crop_center(img, cropy, cropx, even_pos=True):
    y, x = img.shape[:2]
    if (x, y) == (cropx, cropy):
        return img
    cropy, cropx = min(cropy, img.shape[0]), min(cropx, img.shape[1])
    if even_pos:
        sx = x//2-(cropx//2)
        sx -= sx % 2
        sy = y//2-(cropy//2)
        sy -= sy % 2
    return img[sy:sy+cropy, sx:sx+cropx]


def run_model(trainer, in_img):
    if in_img.ndim > 2 and in_img.shape[2] == 3:
        in_img = np.transpose(in_img, (2, 0, 1))
    inputs = torch.from_numpy(in_img).unsqueeze(0).to(trainer.device)
    if inputs.ndim < 4:
        inputs = inputs.unsqueeze(0)
    with torch.no_grad():
        return trainer.model(inputs)


def pred_to_img(preds: torch.Tensor):
    pred_np = preds.cpu().numpy()
    pred_np = np.squeeze(pred_np, axis=0)
    pred_np = pred_np.transpose((1, 2, 0))
    out_im = np.clip(pred_np, 0, 1)
    return (out_im * 255).astype(np.uint8)


def run_model_image_tiles(trainer, in_img, out_channels, tile_div=3, divisor=32):
    h, w = im_shape = np.array(in_img.shape[:2])
    tile_shape = im_shape // tile_div
    ty, tx = tile_shape = tile_shape - (tile_shape % divisor)
    out_im = np.zeros((h, w, out_channels), dtype=np.uint8)

    for iy in range(h//ty):
        for ix in range(w//tx):
            sy, ey = iy * ty, (iy + 1) * ty
            sx, ex = ix * tx, (ix + 1) * tx
            out_im[sy:ey, sx:ex] = pred_to_img(run_model(trainer, in_img[sy:ey, sx:ex]))
    residue = im_shape % tile_shape
    if np.sum(residue == 0) == 0:
        iy += 1
        sy, ey = iy * ty, min((iy + 1) * ty, h)
        tile = pad_2d(in_img[sy:ey, 0:ex], divisor)
        out_im[sy:ey, 0:ex] = crop_center(pred_to_img(run_model(trainer, tile)), ey-sy, ex)
        ix += 1
        sx, ex = ix * tx, min((ix + 1) * tx, w)
        tile = pad_2d(in_img[0:ey, sx:ex], divisor)
        out_im[0:ey, sx:ex] = crop_center(pred_to_img(run_model(trainer, tile)), ey, ex-sx)

    return out_im


def inference_image(trainer: TorchTrainer, im_path: str, factors: np.ndarray = None, rotate: bool = False,
                    raw_result: bool = False, do_crop: bool = False, gray: bool = False, fliplr: bool = False,
                    boost: bool = False, do_mosaic: bool = False):
    in_channels = trainer.cfg.model.in_channels
    out_channels = trainer.cfg.model.out_channels
    src = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    max_bit = (2 ** 16) - 1 if src.dtype == np.uint16 else 255
    if in_channels == 3 and src.ndim == 2 and not gray:
        # im_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(src, pattern='RGGB')
        im_raw = cv2.demosaicing(src, cv2.COLOR_BAYER_BG2RGB)
        # ratio = im_raw.max() / src.max()
        # im_flt = im_raw / max_bit
        im_flt = (im_raw / max_bit).astype(np.float32)

    elif src.ndim > 2 and src.shape[2] == 3:
        im_flt = src[..., ::-1].astype(np.float32) / max_bit  # cv2.COLOR_BGR2RGB
    else:
        im_flt = src.astype(np.float32) / max_bit
    if do_mosaic:
        im_flt = mosaic_image(im_flt, pattern='rggb')
    if fliplr:
        im_flt = np.fliplr(im_flt)
    if factors is not None:
        im_flt = im_flt * factors

    if boost:
        p = np.percentile(im_flt, 98)
        # print('p', p)
        im_flt = im_flt * (1/p)
    in_img = np.clip(im_flt, 0, 0.9999999)
    if rotate:
        in_img = np.rot90(in_img)
    return_img = (in_img * 255).astype(np.uint8)
    if not do_crop:
        in_img = pad_2d(in_img, 32)
        preds = run_model(trainer, in_img)
        if preds.ndim > 3 and preds.shape[1] > 7:  # segmentation
            outs = preds.argmax(dim=1).squeeze()
            out_im = outs.cpu().numpy()
            out_im = out_im.astype(np.uint8)

        elif preds.ndim > 2:  # im2im
            out_im = pred_to_img(preds)
        else:
            out_np = preds.squeeze().cpu().numpy()
            out_im = np.clip(out_np, -4, 10)
            out_im = ((out_im + 4) * (255 / 15)).astype(np.uint8)
    else:
        out_im = run_model_image_tiles(trainer, in_img, out_channels)
    if not do_crop and raw_result:
        out_im = preds.squeeze().cpu().numpy()
        out_im = out_im.transpose(1, 2, 0)

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


def save_image_type(img: np.ndarray, in_im_path: Path, out_dir: Path, mat_out: bool):
    if not out_dir.exists():
        os.makedirs(out_dir.as_posix())
    if mat_out:
        sio.savemat(out_dir.joinpath(in_im_path.stem+'.mat'), {'dpt': img})
    else:
        out_path = out_dir.joinpath(in_im_path.stem)
        save_color_image(img, out_path)


def save_image(img: np.ndarray, in_img: np.ndarray, in_im_path: Path, model_name: str, out_dir: str,
               mat_out: bool, do_crop: bool):
    p = Path(in_im_path)
    out_path = p.parent.joinpath(p.stem + '_' + model_name)
    if not p.parent.is_dir():
        p.parent.mkdir()
    if mat_out:
        out_name = out_path.as_posix() + '.mat'
        sio.savemat(out_name, {'dpt': img})
    else:
        save_color_image(img, out_path)
    if do_crop:
        if in_img.ndim > 2 and\
                not os.path.exists(in_im_path.replace('.png', '_bilinear-demosaic_crop.png')):
            in_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(in_im_path.replace('.png', '_bilinear-demosaic_crop.png'), in_img)
        if not os.path.exists(in_im_path.replace('_crop.png', '_center_crop.png')):
            cv2.imwrite(in_im_path.replace('_crop.png', '_center_crop.png'), in_img)


def save_color_image(img, out_path):
    if img.ndim > 2:
        if img.shape[2] == 4:
            rgb_im = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            ir_im = img[:, :, 3]
            cv2.imwrite(out_path.as_posix() + '_rgb_est.png', rgb_im)
            cv2.imwrite(out_path.as_posix() + '_ir_est.png', ir_im)
        if img.shape[2] == 6:
            rgb_im = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            ir_im = cv2.cvtColor(img[:, :, 3:], cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path.as_posix() + '_rgb_est.png', rgb_im)
            cv2.imwrite(out_path.as_posix() + '_ir_est.png', ir_im)
    else:
        out_im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # out_im = img
        out_name = out_path.as_posix() + '.png'
        cv2.imwrite(out_name, out_im)


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
    if out_im.shape[2] == 4 or out_im.shape[2] == 6:
        cols += 1
    fig, axes = plt.subplots(nrows=1, ncols=cols, sharex=True, sharey=True)
    axes[0].imshow(in_img), axes[0].set_title('in')
    if out_im.ndim == 2:
        axes[1].imshow(out_im, cmap='jet', interpolation=None)
    else:
        axes[1].imshow(out_im[:, :, :3])
    axes[1].set_title('out')
    cv2.imwrite('rgb.png', cv2.cvtColor(out_im[:, :, :3], cv2.COLOR_RGB2BGR))
    if cols >= 3:
        if gt_path:
            axes[2].imshow(gt), axes[2].set_title('GT')
    if out_im.shape[2] == 4:
        axes[-1].imshow(out_im[:, :, 3], cmap='gray'), axes[-1].set_title('IR')
    if out_im.shape[2] == 6:
        axes[-1].imshow(out_im[:, :, 3:]), axes[-1].set_title('IR')
        cv2.imwrite('ir.png', cv2.cvtColor(out_im[:, :, 3:], cv2.COLOR_RGB2BGR))

    plt.show()


if __name__ == '__main__':
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
    parser.add_argument('-m', '--im_pattern', default=None, help='images regex pattern')
    parser.add_argument('-s', '--boost_image', action='store_true', help='auto gain per image')
    parser.add_argument('-ti', '--test_images', action='store_true', help='infer test images')
    parser.add_argument('-mo', '--mosaic_images', action='store_true', help='perform mosaicing to input images')

    # parser.add_argument('--check_patt', default='*mask.tif', help='input image file pattern')
    args = parser.parse_args()
    assert not (args.mat_out and not (args.mat_out ^ (args.out_type is None))), 'out path is required for mat'
    if not args.test_images:
        for in_path in args.images_path:
            assert os.path.exists(in_path), 'ERROR - path is not exist %s' % in_path


    print('reading model...')
    trainer = TorchTrainer.warm_startup(in_path=args.model_path, gpu_index=args.gpu_index, best=True)
    trainer.model.eval()

    if args.images_path is not None:
        if len(args.images_path) == 1 and Path(args.images_path[0]).is_dir():
            assert args.im_pattern is not None, 'ERROR - please set image pattern argument -m'
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
        for in_path_str in args.images_path:
            in_path = Path(in_path_str)
            if in_path.is_dir():
                # given path is a directory
                images.extend(in_path.glob(args.im_pattern))
            elif in_path.suffix == '.txt':
                with in_path.open() as f:
                    for _ in range(15):
                        images.append(f.readline().strip())
            else:
                images.append(in_path)
        assert len(images) > 0, 'WARNING - images list is empty, check glob input: {}'.format(args.im_pattern)
        for i, im_path in enumerate(images):
            print_progress(i, total=len(images), suffix='inference {}{}'.format(im_path, ' '*20), length=20)
            out_im, in_img = inference_image(trainer, im_path=im_path.as_posix(), factors=factors,
                                             rotate=args.rot90, raw_result=args.mat_out,
                                             do_crop=args.do_crop, gray=args.gray, fliplr=args.fliplr,
                                             boost=args.boost_image, do_mosaic=args.mosaic_images)

            if args.out_type is not None:
                out_dir = in_path.joinpath(args.out_type, model_name)
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
        in_path = Path(args.model_path)
        model_dir = in_path if in_path.is_dir() else in_path.parent.parent
        test_df = pd.read_csv(model_dir.joinpath('test_images.csv'))
        test_dir = model_dir.joinpath('test_images')
        test_dir.mkdir(exist_ok=True)
        mse_rgb_ir = []
        ssim_rgb_ir = []
        psnr_rgb_ir = []
        for _, im_path in tqdm(test_df.iterrows()):
            if not Path(im_path.full).exists():
                print('ERROR', im_path.full, 'is missing')
                continue
            out_im, in_img = inference_image(trainer, im_path=im_path.full, rotate=args.rot90, raw_result=args.mat_out,
                                             do_crop=args.do_crop, gray=args.gray, fliplr=args.fliplr,
                                             boost=args.boost_image, do_mosaic=args.mosaic_images)
            ref_rgb = cv2.cvtColor(cv2.imread(im_path.rgb, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            rgb_est = out_im[..., :3]
            if out_im.shape[2] == 4:
                ref_ir = cv2.imread(im_path.ir, cv2.IMREAD_GRAYSCALE)
                ir_est = out_im[..., 3]
                mse_rgb_ir.append((mse(ref_rgb, rgb_est), mse(ref_ir, ir_est)))
            elif out_im.shape[2] == 6 or out_im.shape[2] == 3:
                if out_im.shape[2] == 6:
                    ir_est = out_im[:, :, 3:]
                else:
                    ir_est = in_img - out_im
                ref_ir = cv2.cvtColor(cv2.imread(im_path.ir), cv2.COLOR_BGR2RGB)
                mse_rgb_ir.append((mse(ref_rgb, rgb_est), mse(ref_ir, ir_est)))
            ir_range = ir_est.max() - ir_est.min()
            rgb_range = rgb_est.max() - rgb_est.min()
            ssim_rgb_ir.append((ssim(ref_rgb, rgb_est, data_range=rgb_range, multichannel=True),
                                ssim(ref_ir, ir_est, data_range=ir_range, multichannel=out_im.shape[2] != 4)))
            psnr_rgb_ir.append(10 * np.log10((255.0**2) / np.array(mse_rgb_ir[-1])))
            rgb_out = test_dir.joinpath(Path(im_path.rgb).stem + '_est.png')
            ir_out = test_dir.joinpath(Path(im_path.ir).stem + '_est.png')
            cv2.imwrite(rgb_out.as_posix(), cv2.cvtColor(rgb_est, cv2.COLOR_RGB2BGR))
            if out_im.shape[2] == 4:
                cv2.imwrite(ir_out.as_posix(), ir_est)
            else:
                cv2.imwrite(ir_out.as_posix(), cv2.cvtColor(ir_est, cv2.COLOR_RGB2BGR))
        test_df[['mse_rgb', 'mse_ir']] = mse_rgb_ir
        test_df[['ssim_rgb', 'ssim_ir']] = ssim_rgb_ir
        test_df[['psnr_rgb', 'psnr_ir']] = psnr_rgb_ir
        test_df.to_csv(model_dir.joinpath('test_images.csv'), index_label=False, index=False)


