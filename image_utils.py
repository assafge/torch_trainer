import numpy as np
import torch
from datetime import datetime
from skimage.io import imsave


def pad_2d(img: np.ndarray, divisor) -> np.ndarray:
    """return padded image to shape of nearest divisor"""
    expand = divisor - (np.array(img.shape[:2]) % divisor)
    expand = np.bitwise_xor(expand, divisor)
    top, left = expand // 2
    bottom, right = expand - (expand // 2)
    if img.ndim > 2:
        return np.pad(img, pad_width=((top, bottom), (left, right), (0, 0)), mode='constant')
    else:
        return np.pad(img, pad_width=((top, bottom), (left, right)), mode='constant')


def random_dual_augmentation(image, label, sigma, pad_divisor=0, img_min=0, im_max=1, align_to_tensor=True,
                             augmentations=[np.flipud, np.fliplr]):
    """torch transformations cannot be applied to a pair of images, so I've decided to implement in numpy."""
    if sigma > 0:
        gauss = np.random.normal(0, sigma, image.shape).astype(image.dtype)
        image = image + gauss
        image = np.clip(image, img_min, im_max)

    did_augment = False
    for augment in augmentations:
        if np.random.rand() > 0.5:
            image = augment(image)
            label = augment(label)
            did_augment = True
    if align_to_tensor:
        if pad_divisor > 0:
            image = pad_2d(image, pad_divisor)
            label = pad_2d(label, pad_divisor)
        # elif did_augment:
        #     image = np.ascontiguousarray(image)
        #     label = np.ascontiguousarray(label)
        if image.ndim > 2:
            image = np.transpose(image, (2, 0, 1))
        if label.ndim > 2:
            label = np.transpose(label, (2, 0, 1))
    return image, label


def collate_fn_random_rot90(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, depth).
            - image: torch tensor of shape (3, w, h) torch.float64.
            - depth: torch tensor of shape (w, h) torch.long.
    Returns:
        images: torch tensor of shape (batch_size, 3, w, h).
        depths: torch tensor of shape (batch_size, w, h).
    """
    images, depths = zip(*data)
    # Merge arrays (from tuple of 3D to 4D).
    images = np.stack(images, 0)
    depths = np.stack(depths, 0)
    if np.random.rand() > 0.5:
        images = np.rot90(images, axes=(2, 3))
        depths = np.rot90(depths, axes=(1, 2))
    # # depths = np.squeeze(depths, dim=1)

    im_tensor = torch.as_tensor(data=np.ascontiguousarray(images), dtype=torch.float32)
    dp_tensor = torch.as_tensor(data=np.ascontiguousarray(depths), dtype=torch.long)
    return im_tensor, dp_tensor


def myplot(im, ref):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.transpose(im, (1, 2, 0)))
    ax2.imshow(ref)
    plt.show()
