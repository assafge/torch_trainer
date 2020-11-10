__author__ = "Assaf Genosar"

from data_sets.base_dataset import BaseDataset
from image_utils import base_collate_fn, num_of_channels
import os.path
import numpy as np
from glob import glob
from torch.utils.data import DataLoader, Subset
from typing import List, Dict
from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt
import random
import pickle
import colour_demosaicing


def file_name_order(path):
    fname = os.path.basename(path)
    idx = fname.split('.')[0]
    if idx.isdigit():
        return int(fname.split('.')[0])
    else:
        return idx


def list_images(folder_path: str, pattern='') -> List[str]:
    """return a sorted list of png files"""
    files_list = glob(os.path.join(folder_path, '*' + pattern + '*'))
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
    for i in range((h // patch_size) - 1):
        for j in range((w // patch_size) - 1):
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
class DualCamParams:
    """stores dataset parameters and validate input"""
    org_dir: str
    img_dir: str
    ref_suffix: str  # jai
    img_suffix: str  # ids
    mix_inputs: bool = False
    prepared: str = ''


class DualCamDataset(BaseDataset):
    """ dual cameras dataset """

    def __init__(self, root_dir: str, in_channels: int, out_channels: int, train_split: float,
                 data_sets: dict, patch_size: int, seed=42,):
        super().__init__(root_dir, in_channels, out_channels, train_split)
        self.data_sets: Dict[str, DualCamParams] = {}
        for set_name, prm in data_sets.items():
            self.data_sets[set_name] = DualCamParams(**prm)
        self.data_map = {}
        self.data: List = []  # list of patches
        self.train_idx: np.ndarray = None
        self.val_idx: np.ndarray = None
        self.seed = seed
        self.patch_size = patch_size
        self.i = 0
        self.colors = {}

    def prepare_data(self):
        last_idx = 0
        for dataset_name, prm in self.data_sets.items():
            if prm.prepared is not None:
                with open(prm.prepared, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                for img_path in list_images(prm.img_dir, prm.img_suffix):
                    ref_name = os.path.basename(img_path.replace(prm.img_suffix, prm.ref_suffix))
                    ref_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), ref_name)
                    if os.path.exists(ref_path):
                        self.data.append((img_path, ref_path))
            print('added {} images from {} dataset'.format(len(self.data) - last_idx, dataset_name))
            last_idx = len(self.data)
        assert len(self.data) > 0, 'dataset is empty'
        if prm.prepared is None:
            with open('/tmp/data.pth', 'wb') as f:
                pickle.dump(self.data, f)
            print('saved to /tmp/data.pth')
        random.shuffle(self.data)
        train_size = int(len(self.data) * self.train_split)
        self.train_idx = np.arange(train_size)
        self.val_idx = np.arange(train_size, len(self.data))
        val_txt = os.path.join(self.root_dir, 'validation_images.txt')
        with open(val_txt, 'w') as f:
            for idx in self.val_idx:
                f.write(self.data[idx][0] + os.linesep)
        print(f'wrote test images list to {val_txt}')

    def random_sample_patch(self, raw, lbl, debug=False):
        if num_of_channels(raw) < self.in_channels:
            im = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, pattern='GRBG')
            im = np.clip(im, 0, 255)
            # in bilinear demosaicing the edges are noisy
            y0, y1, x0, x1 = 2, im.shape[0] - 2, 2, im.shape[1] - 2
        else:
            im = raw
            y0, y1, x0, x1 = 0, im.shape[0], 0, im.shape[1]
        sy = random.randint(y0, y1 - self.patch_size)
        sx = random.randint(x0, x1 - self.patch_size)
        if im is raw:
            # keep bayer structure
            sy -= sy % 2
            sx -= sx % 2

        im_p = im[sy:sy + self.patch_size, sx:sx + self.patch_size]
        lbl_p = lbl[sy:sy + self.patch_size, sx:sx + self.patch_size]

        if debug:
            r2g = im_p[:, :, 0] / im_p[:, :, 1]
            r2b = im_p[:, :, 0] / im_p[:, :, 2]
            r2g_l = lbl_p[:, :, 0] / lbl_p[:, :, 1]
            r2b_l = lbl_p[:, :, 0] / lbl_p[:, :, 2]
            R2G = np.mean(cv2.absdiff(r2g, r2g_l))
            R2B = np.mean(cv2.absdiff(r2b, r2b_l))
            # print(R2G, R2B)
            if R2G > 1 or R2B > 1:
                print(sy, sx)
                f, a = plt.subplots(2, 2)
                a[0, 0].imshow(im_p.astype(np.uint8))
                a[0, 1].imshow(im.astype(np.uint8))
                a[1, 0].imshow(lbl)
                a[1, 1].imshow(raw)
                f.suptitle('R2G={:.3f}  | R2B={:.3f}'.format(R2G, R2B))
                plt.show()

        if im_p.ndim > 2:
            im_p = np.transpose(im_p, (2, 0, 1))
        if lbl_p.ndim > 2:
            lbl_p = np.transpose(lbl_p, (2, 0, 1))
        return im_p.astype(np.float32) / 255, lbl_p.astype(np.float32) / 255

    def __getitem__(self, item):
        img_path, ref_path = self.data[item]
        if self.out_channels == 3:
            ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        else:
            ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        sample, label = self.random_sample_patch(img, ref)
        return sample, label

    def __len__(self):
        return len(self.data)

    def get_data_loaders(self, batch_size):
        self.prepare_data()
        train_loaders = [DataLoader(dataset=Subset(self, indices=self.train_idx),
                                    batch_size=batch_size, shuffle=True
                                    # num_workers=1, collate_fn=base_collate_fn, pin_memory=True,
                         )]
        test_loaders = [DataLoader(dataset=Subset(self, indices=self.val_idx), pin_memory=False, shuffle=False,
                                   batch_size=batch_size, collate_fn=base_collate_fn)]
        return train_loaders, test_loaders
