from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
import spectral
# from dataclasses import dataclass
import pandas
import numpy as np

import matplotlib.pyplot as plt

# spectral.settings.envi_support_nonlowercase_params = True

# @dataclass
# class HyperSpectralParams:
#     """stores dataset parameters and validate input"""
#     input_dir: str
#     sensors_response_csv: str

def hyperspectral_to_rgb(hdr, raw, sensor_response_csv):
    spec_img = spectral.io.envi.open(hdr, raw)
    response = pandas.read_csv(sensor_response_csv)
    img = np.zeros(spec_img.shape[:2] + (3,), dtype=np.float)
    # r, g, b = (700, 530, 470)
    r,g,b = (7, 16, 23)
    centers = np.array(spec_img.bands.centers)
    for i, wavelength in enumerate([r, g, b]):
        # im_id = np.argmin(abs(centers - wavelength))
        # res_in = np.argmin(abs(response.wavelength - wavelength))
        # img[:, :, i] = spec_img.read_band(im_id)
        img[:, :, i] = spec_img.read_band(wavelength)
    return img


def convert_hyperspectral_to_rgb(hdr, raw, sensor_response_csv, cutoff=700):
    spec_img = spectral.io.envi.open(hdr, raw)
    response = pandas.read_csv(sensor_response_csv)
    img = np.zeros(spec_img.shape[:2] + (3,), dtype=np.float)
    for i, row in response.iterrows():
        if row['wavelength'] > cutoff:
            break
        wavelength_id = spec_img.bands.centers.index(row['wavelength'])
        band = spec_img.read_band(wavelength_id)
        img[:, :, 0] += row.R * band
        img[:, :, 1] += row.G * band
        img[:, :, 2] += row.B * band
    img[:, :, 0] /= np.max(response.R)
    img[:, :, 1] /= np.max(response.G)
    img[:, :, 2] /= np.max(response.B)

    return img

class DualCamDataset(Dataset):
    """ dual cameras dataset """

    def __init__(self, spec_img_dir: str, sensor_csv: str, bands_factor: int, patch_size: int, seed=42, shuffle=True,
                 train_split=0.8):

        self.data: List = []  # list of patches
        self.train_split = train_split
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None
        self.train_split = train_split
        self.shuffle = shuffle
        self.bi_linear_demosaic = bi_linear_demosaic
        self.seed = seed
        self.patch_size = patch_size
        self.i = 0
        self.colors = {}



hdr = '/home/assaf/data/Datasets/BGU_hyperspectral/4cam_0411-1640-1.hdr'
raw = '/home/assaf/data/Datasets/BGU_hyperspectral/4cam_0411-1640-1.raw'
csv = '/home/assaf/data/Datasets/sensos_response/imx264C_resp.csv'
# convert_hyperspectral_to_rgb(hdr, raw, csv)
hyperspectral_to_rgb(hdr, raw, csv)