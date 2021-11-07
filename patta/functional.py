from typing import Optional, Tuple, Union

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F


def rot90(x, k=1):
    """rotate batch of images by 90 degrees k times"""
    try:
        x = paddle.to_tensor(x).numpy()
    except:
        x = x.numpy()
    rot = np.rot90(x, k, (2, 3))
    return paddle.to_tensor(rot)


def hflip(x):
    """flip batch of images horizontally"""
    return x.flip([3])


def vflip(x):
    """flip batch of images vertically"""
    return x.flip([2])


def hshift(x, shifts=0):
    """shift batch of images horizontally"""
    return paddle.roll(x, int(shifts*x.shape[3]), axis=3)


def vshift(x, shifts=0):
    """shift batch of images vertically"""
    return paddle.roll(x, int(shifts*x.shape[2]), axis=2)


def sum(x1, x2):
    """sum of two tensors"""
    return x1 + x2


def add(x, value):
    """add value to tensor"""
    return x + value


def max(x1, x2):
    """compare 2 tensors and take max values"""
    return paddle.max(paddle.concat([x1, x2]))


def min(x1, x2):
    """compare 2 tensors and take min values"""
    return paddle.min(paddle.concat([x1, x2]))


def multiply(x, factor):
    """multiply tensor by factor"""
    return x * factor


def scale(x, scale_factor, interpolation="nearest", align_corners=False):
    """scale batch of images by `scale_factor` with given interpolation mode"""
    h, w = x.shape[2:]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    return F.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=align_corners)


def resize(x, size, interpolation="nearest", align_corners=False):
    """resize batch of images to given spatial size with given interpolation mode"""
    return F.interpolate(x, size=size, mode=interpolation, align_corners=align_corners)


def crop(x, x_min=None, x_max=None, y_min=None, y_max=None):
    """perform crop on batch of images"""
    return x[:, :, y_min:y_max, x_min:x_max]


def crop_lt(x, crop_h, crop_w):
    """crop left top corner"""
    return x[:, :, 0:crop_h, 0:crop_w]


def crop_lb(x, crop_h, crop_w):
    """crop left bottom corner"""
    return x[:, :, -crop_h:, 0:crop_w]


def crop_rt(x, crop_h, crop_w):
    """crop right top corner"""
    return x[:, :, 0:crop_h, -crop_w:]


def crop_rb(x, crop_h, crop_w):
    """crop right bottom corner"""
    return x[:, :, -crop_h:, -crop_w:]


def center_crop(x, crop_h, crop_w):
    """make center crop"""

    center_h = x.shape[2] // 2
    center_w = x.shape[3] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[:, :, y_min:y_max, x_min:x_max]


def adjust_contrast(x, contrast_factor: float=1.):
    """adjusts contrast for batch of images"""
    table = np.array([
        (i - 74) * contrast_factor + 74
        for i in range(0, 256)
    ]).clip(0, 255).astype(np.uint8)
    try:
        x = x.paddle.to_tensor(x).numpy()
    except:
        x = x.numpy()
    x = x.clip(0,255).astype(np.uint8)
    x = cv2.LUT(x, table)
    x = x.astype(np.float32)
    return paddle.to_tensor(x)


def adjust_brightness(x, brightness_factor: float=1.):
    """adjusts brightness for batch of images"""
    table = np.array([
        i * brightness_factor
        for i in range(0, 256)
    ]).clip(0, 255).astype(np.uint8)
    try:
        x = x.paddle.to_tensor(x).numpy()
    except:
        x = x.numpy()
    x = x.clip(0,255).astype(np.uint8)
    x = cv2.LUT(x, table)
    x = x.astype(np.float32)
    return paddle.to_tensor(x)

def filter2d(x, kernel):
    """applies an arbitrary linear filter to an image"""
    C = x.shape[1]
    kernel = kernel.reshape((1, 1, *kernel.shape))
    kernel = paddle.concat([kernel for _ in range(C)], axis=0)
    return F.conv2d(x, kernel, stride=1, padding="same", groups=C)


def average_blur(x, kernel_size: Union[Tuple[int, int], int] = 3):
    """smooths the input image"""
    cv2.blur
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if kernel_size == (1, 1):
        return x
    assert (
        kernel_size[0] > 0 and kernel_size[1] > 0 and kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    ), "kernel_size both must be positive and odd"
    kernel = np.ones(kernel_size[0] * kernel_size[1], dtype=np.float32)
    kernel = kernel / (kernel_size[0] * kernel_size[1])
    kernel = kernel.reshape((kernel_size[1], kernel_size[0]))
    assert kernel.shape == (kernel_size[1], kernel_size[0])
    kernel = paddle.to_tensor(kernel)
    return filter2d(x, kernel)


def gaussian_blur(
    x,
    kernel_size: Union[Tuple[int, int], int] = 3,
    sigma: Optional[Union[Tuple[float, float], float]] = None
):
    """smooths the input image with the specified Gaussian kernel"""
    cv2.GaussianBlur
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if kernel_size == (1, 1):
        return x
    if sigma is None:
        sigma = (kernel_size[0] - 1) / 4.0
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)
    assert kernel_size[0] > 0 and kernel_size[1] > 0 and kernel_size[0] % 2 == 1 and \
           kernel_size[1] % 2 == 1, "kernel_size both must be positive and odd"
    kernel_x = cv2.getGaussianKernel(kernel_size[0], sigma[0])
    kernel_y = cv2.getGaussianKernel(kernel_size[1], sigma[1])
    kernel = kernel_y @ kernel_x.transpose()
    kernel = kernel.astype(np.float32)
    assert kernel.shape == (kernel_size[1], kernel_size[0])
    kernel = paddle.to_tensor(kernel)
    return filter2d(x, kernel)


def sharpen(x, kernel_size: int = 3):
    """sharpen the input image"""
    if kernel_size == 1:
        return x
    assert kernel_size > 0 and kernel_size % 2 == 1, "kernel_size both must be positive and odd"
    kernel = get_laplacian_kernel(kernel_size).astype(np.float32)
    kernel = kernel.astype(np.float32)
    kernel = paddle.to_tensor(kernel)
    x_laplacian = filter2d(x, kernel)
    x = x - x_laplacian
    x = x.clip(0, 255)
    return x


def get_laplacian_kernel(kernel_size):
    assert kernel_size > 0 and kernel_size % 2 == 1, "kernel_size both must be positive and odd"
    kernel = np.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    return kernel


def _disassemble_keypoints(keypoints):
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    return x, y


def _assemble_keypoints(x, y):
    return paddle.stack([x, y], axis=-1)


def keypoints_hflip(keypoints):
    x, y = _disassemble_keypoints(keypoints)
    return _assemble_keypoints(1. - x, y)


def keypoints_vflip(keypoints):
    x, y = _disassemble_keypoints(keypoints)
    return _assemble_keypoints(x, 1. - y)


def keypoints_hshift(keypoints, shifts):
    x, y = _disassemble_keypoints(keypoints)
    return _assemble_keypoints((x + shifts) % 1, y)


def keypoints_vshift(keypoints, shifts):
    x, y = _disassemble_keypoints(keypoints)
    return _assemble_keypoints(x, (y + shifts) % 1)


def keypoints_rot90(keypoints, k=1):

    if k not in {0, 1, 2, 3}:
        raise ValueError("Parameter k must be in [0:3]")
    if k == 0:
        return keypoints
    x, y = _disassemble_keypoints(keypoints)

    if k == 1:
        xy = [y, 1. - x]
    elif k == 2:
        xy = [1. - x, 1. - y]
    elif k == 3:
        xy = [1. - y, x]

    return _assemble_keypoints(*xy)
