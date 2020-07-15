import re
import os.path
import random
import numpy as np
from skimage.io import imread
# from PIL import Image
from collections import defaultdict
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
import sys
sys.path.append('../')
from image_utils import random_dual_augmentation, coll_fn_rand_rot90_float, random_crop, base_collate_fn
# from skimage.io import imsave
from typing import List, Dict
from dataclasses import dataclass
# import torch
import cv2
import matplotlib.pyplot as plt
import random
import pickle
import colour_demosaicing


__author__ = "Assaf Genosar"




def file_name_order(path):
    fname = os.path.basename(path)
    idx = fname.split('.')[0]
    if idx.isdigit():
        return int(fname.split('.')[0])
    else:
        return idx

def list_images(folder_path: str, pattern = '') -> List[str]:
    """return a sorted list of png files"""
    files_list = glob(os.path.join(folder_path, '*'+pattern+'*'))
    files_list.sort(key=file_name_order)
    return files_list



def calc_bounding_rect(img1, img2, do_plot=False, min_match_count=10, ratio_thresh=0.7, ransac_th=5.0):
    """given a reference and taken image"""
    detector = cv2.xfeatures2d_SIFT.create()
    kp1, descriptors1 = detector.detectAndCompute(img1, None)
    kp2, descriptors2 = detector.detectAndCompute(img2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    # -- Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    if len(good_matches) < min_match_count:
        print("Not enough matches are found - %d/%d" % (len(good_matches), min_match_count))
        return None
    else:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_th)
        matches_mask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        bounding_pts = cv2.perspectiveTransform(pts, transform)

        if do_plot:
            debug_img = img2.copy()
            debug_img = cv2.polylines(debug_img, [np.int32(bounding_pts)], True, 255, 3, cv2.LINE_AA)
            debug_img = cv2.drawMatches(img1, kp1, debug_img, kp2, good_matches, None, matchColor=(0, 255, 0),
                                  singlePointColor=None, matchesMask=matches_mask, flags=2)
            plt.imshow(debug_img), plt.show()
        return transform, bounding_pts.astype(np.int).squeeze(1)

def crop_roi(org_path, img_path, ref_path, border=5):
    org = cv2.imread(org_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)  # assumed to be a bayer image
    ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2GRAY)  # assumed to be a rgb image
    if any([im is None for im in [org, img, ref]]):
        return None
    try:
        h, pts = calc_bounding_rect(org, img)
        if h is None:
            return None
    except Exception:
        return None
    y0 = max(pts[0][1], pts[3][1]) + border
    x0 = max(pts[0][0], pts[1][0]) + border
    y1 = min(pts[1][1], pts[2][1]) - border
    x1 = min(pts[2][0], pts[3][0]) - border
    if y1 < y0 or x1 < x0:
        return None
    return y0, x0, y1, x1

def split_to_patches(img_roi, ref_roi, patch_size):
    """returns pairs of (img, ref) patches"""
    h, w = img_roi.shape[:2]
    start_y = (h % patch_size) // 2
    start_x = (w % patch_size) // 2
    # can be randomized
    patches = []
    for i in range((h//patch_size)-1):
        for j in range((w//patch_size)-1):
            y0 = start_y + i * patch_size
            x0 = start_x + j * patch_size
            y1 = start_y + ((i + 1) * patch_size)
            x1 = start_x + ((j + 1) * patch_size)
            patches.append((img_roi[y0:y1, x0:x1], ref_roi[y0:y1, x0:x1]))
    return patches


def shift_calc(clip: np.ndarray, template: np.ndarray, debug: bool):
    result = cv2.matchTemplate(clip, template, cv2.TM_CCORR)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    min_loc = np.array(max_loc)
    result_shape = np.array(result.shape[:2])
    shift = min_loc - (result_shape // 2)
    if debug:
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(clip), ax2.imshow(template), ax3.imshow(result), plt.show()
        print('clip shape={}, tmp shape={}, res shape={}, loc={}, shift={}'.format(clip.shape, template.shape,
                                                                                   result_shape, min_loc, shift))
    return shift

@dataclass
class DatasetParams:
    """stores dataset parameters and validate input"""
    org_dir: str
    img_dir: str
    ref_suffix: str   # jai
    img_suffix: str   # ids
    # align_img: str
    # align_ref: str
    prepared: str = ''

class DualCamDataset(Dataset):
    """ Texture data format dataset """
    def __init__(self, data_sets: dict, patch_size: int, bi_linear_demosaic: bool, seed=42, shuffle=True, train_split=0.8):
        """
        Parameters
        ----------
        data_sets:
            org_dir: str - directory original images
            img_dir: str - directory of sampled images
            ref_suffix: str
            img_suffix: str
            align_img: str
            align_ref: str
            prepared: str - prepared data set (empty if not avilable)
        patch_size: int
        shuffle: bool
            shuffle the samples order
        train_split:
            portion of the training samples
        bi_linear_demosaic:
        """
        self.data_sets: Dict[str, DatasetParams] = {}
        for set_name, prm in data_sets.items():
            self.data_sets[set_name] = DatasetParams(**prm)
        self.data_map = {}
        self.data: List = []   # list of patches
        self.train_split = train_split
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None
        self.train_split = train_split
        self.shuffle = shuffle
        self.bi_linear_demosaic = bi_linear_demosaic
        self.seed = seed
        self.patch_size = patch_size
        self.i = 0

    def prepare_data(self):
        last_idx = 0
        for dataset_name, prm in self.data_sets.items():
            if prm.prepared is not None:
                with open(prm.prepared, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                # align_img = cv2.imread(prm.align_img, cv2.IMREAD_GRAYSCALE)
                # align_ref = cv2.imread(prm.align_ref, cv2.IMREAD_GRAYSCALE)
                for img_path in list_images(prm.img_dir, prm.img_suffix):
                    # org_name = os.path.basename(img_path.replace(prm.img_suffix, ''))
                    # org_path = os.path.join(prm.org_dir, org_name)
                    ref_path = img_path.replace(prm.img_suffix, prm.ref_suffix)
                    # bbx_path = img_path.replace(prm.img_suffix + '.png', prm.bbx_suffix)
                    # if os.path.exists(org_path) and os.path.exists(ref_path):
                    # bbx = crop_roi(org_path, img_path, ref_path)
                    # if bbx is None:
                    #     continue
                    #     else:
                    if os.path.exists(ref_path):
                        if cv2.imread(img_path) is not None and cv2.imread(ref_path) is not None:
                            self.data.append((img_path, ref_path))

            # print('\rimage {} added {} patches to a total of {}'.format(org_name, len(patches), len(self.data)), end='\r')
            # print()
            print('added {} images from {} dataset'.format(len(self.data) - last_idx, dataset_name))
            last_idx = len(self.data)
        assert len(self.data) > 0, 'dataset is empty'
        if prm.prepared is None:
            with open('/tmp/data.pth', 'wb') as f:
                pickle.dump(self.data, f)
            print('saved to /tmp/data.pth')
        if self.shuffle:
            # random.seed = self.seed
            random.shuffle(self.data)
        train_size = int(len(self.data) * self.train_split)
        self.train_idx = np.arange(train_size)
        self.test_idx = np.arange(train_size, len(self.data))
        with open('/tmp/test_images.txt', 'w') as f:
            for idx in self.test_idx:
                f.write(self.data[idx][0] + os.linesep)
        print('wrote test images list to /tmp/test_images.txt')

    # def crop_roi(images_path, ref_images_path, to_32=False):
    #     ids_calib_path = os.path.join(images_path, 'align_target_ids.png')
    #     jai_calib_path = os.path.join(images_path, 'align_target_jai.png')
    #     assert os.path.exists(ids_calib_path), 'path {} is not exist'.format(ids_calib_path)
    #     assert os.path.exists(jai_calib_path), 'path {} is not exist'.format(jai_calib_path)
    #     ids_calib = np.fliplr(cv2.imread(ids_calib_path, cv2.IMREAD_GRAYSCALE))
    #     jai_calib = cv2.cvtColor(cv2.imread(jai_calib_path), cv2.COLOR_BGR2GRAY)
    #     jai2ids = calc_homography(jai_calib, ids_calib, do_plot=True, ratio_thresh=0.4, ransac_th=5)
    #     target_w = cv2.warpPerspective(jai_calib, jai2ids, (ids_calib.shape[1], ids_calib.shape[0]))
    #     fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    #     ax[0].imshow(ids_calib), ax[0].set_title('ids')
    #     ax[1].imshow(target_w), ax[1].set_title('jai wrapped')
    #     ax[2].imshow(cv2.absdiff(ids_calib, target_w)), ax[2].set_title('diff')
    #     plt.show()
    #     for ids_path in glob(images_path + '/*ids.png'):
    #         print(ids_path)
    #         if 'calib' in ids_path or 'target' in ids_path:
    #             continue
    #         ref_path = ids_path.replace(images_path, ref_images_path).replace('_ids', '')
    #         jai_path = ids_path.replace('ids', 'jai')
    #         if not all([os.path.exists(p) for p in [jai_path, ref_path]]):
    #             print('1')
    #             continue
    #         ids = cv2.imread(ids_path, cv2.IMREAD_GRAYSCALE)
    #         jai = cv2.cvtColor(cv2.imread(jai_path), cv2.COLOR_BGR2RGB)
    #         ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2GRAY)
    #         if any([im is None for im in [ids, jai, ref]]):
    #             print('2')
    #             continue
    #         jai_proj = cv2.warpPerspective(jai, jai2ids, (ids.shape[1], ids.shape[0]))
    #         roi1 = calc_roi(ids, ref, to_32=to_32)
    #         if roi1 is None:

    def random_sample_patch(self, im, lbl, do_transpose=True, registration=False, debug=True):
        if self.bi_linear_demosaic:
            im = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(im, pattern='GRBG')
            im = np.clip(im, 0, 255)
            y0, y1, x0, x1 = 2, im.shape[0]-2, 2, im.shape[1]-2
        else:
            y0, y1, x0, x1 = 0, im.shape[0], 0, im.shape[1]
        sy = random.randint(y0, y1 - self.patch_size)
        sy -= sy % 2
        sx = random.randint(x0, x1 - self.patch_size)
        sx -= sx % 2

        im_p = im[sy:sy+self.patch_size, sx:sx+self.patch_size]
        lbl_p = lbl[sy:sy+self.patch_size, sx:sx+self.patch_size]

        if registration:  #reg
            lbl_reg = lbl[sy-5:5+sy+self.patch_size, sx-5:5+sx+self.patch_size]
            shift = shift_calc(im_p[:-1, :-1], lbl_reg, False)
            sx -= shift[0]
            sy -= shift[1]
            lbl_p = lbl[sy:sy+self.patch_size, sx:sx+self.patch_size]

        if debug:
            f, a = plt.subplots(1, 2, sharex=True, sharey=True)
            a[0].imshow(im_p.astype(np.uint8))
            a[1].imshow(lbl_p)
            plt.show()
        # cv2.imwrite('/home/assaf/data/dual_cam_results/7_july/train_patches_bilinear/sample_{}.png'.format(self.i),
        #             cv2.cvtColor(im_p.astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.imwrite('/home/assaf/data/dual_cam_results/7_july/train_patches_bilinear/reference_{}.png'.format(self.i),
        #             cv2.cvtColor(lbl_p, cv2.COLOR_RGB2BGR))
        # self.i += 1

        if do_transpose:
            if self.bi_linear_demosaic:
                im_p = np.transpose(im_p, (2, 0, 1))
            lbl_p = np.transpose(lbl_p, (2, 0, 1))
        return im_p.astype(np.float32) / 255, lbl_p.astype(np.float32) / 255

    def __getitem__(self, item):
        img_path, ref_path = self.data[item]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
        ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        sample, label = self.random_sample_patch(img, ref)
        return sample, label

    def __len__(self):
        return len(self.data)

    def get_data_loaders(self, batch_size):
        self.prepare_data()
        train_loaders = [DataLoader(dataset=Subset(self, indices=self.train_idx), pin_memory=True, num_workers=4,
                                    batch_size=batch_size, collate_fn=base_collate_fn, shuffle=True)]
        test_loaders = [DataLoader(dataset=Subset(self, indices=self.test_idx), pin_memory=False, shuffle=False,
                                   batch_size=batch_size, collate_fn=base_collate_fn)]
        return train_loaders, test_loaders


