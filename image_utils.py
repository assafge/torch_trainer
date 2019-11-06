import numpy as np
from torch import as_tensor


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
        coin = np.random.rand()
        if coin > 0.5:
            image = augment(image)
            label = augment(label)
            did_augment = True
    if align_to_tensor:
        if pad_divisor > 0:
            image = pad_2d(image, pad_divisor)
            label = pad_2d(label, pad_divisor)
        elif did_augment:
            image = np.ascontiguousarray(image)
            label = np.ascontiguousarray(label)
        if image.ndim > 2:
            image = np.transpose(image, (2, 0, 1))
        if label.ndim > 2:
            label = np.transpose(label, (2, 0, 1))
    return image, label

