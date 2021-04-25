import random
import numpy as np
import cv2

from skimage import filters, transform
from scipy import ndimage
from skimage.transform import AffineTransform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def fix_classes(label):
    imgMax = np.max(label)
    imgMin = np.min(label)

    label[label < (imgMax / 2)] = imgMin
    label[label >= (imgMax / 2)] = imgMax

    return label


def add_noise(img, label):
    size = img.shape
    s = random.randint(0, 5)
    noise = np.random.normal(0, s, size)
    return (img + noise, label)


def rotate(img, label):
    angle = random.randint(1, 359)

    # Assume image is same x,y shape
    cmin = int(img.shape[0] / 3)
    cmax = 2 * int(img.shape[0] / 3)
    center_x = random.randint(cmin, cmax)
    center_y = random.randint(cmin, cmax)

    _img = transform.rotate(img, angle=angle, center=(center_x, center_y), mode='reflect')
    _label = transform.rotate(label, angle=angle, center=(center_x, center_y), mode='reflect')
    _label = fix_classes(_label)

    return (_img, _label)


def shear(img, label):
    shear_val = random.uniform(-1.5, 1.5)

    tf = AffineTransform(shear=shear_val)
    _img = transform.warp(img, tf, order=1, preserve_range=True, mode='reflect')
    _label = transform.warp(label, tf, order=1, preserve_range=True, mode='reflect')
    _label = fix_classes(_label)

    return (_img, _label)


def zoom(img, label):
    size = img.shape

    zoom_factor = random.uniform(1.25, 2.25)

    # print('Zoom: ', zoom_factor)

    img_zoom = ndimage.zoom(img, zoom_factor)
    label_zoom = ndimage.zoom(label, zoom_factor)

    cmin = int(img_zoom.shape[0] / 3)
    cmax = 2 * int(img_zoom.shape[0] / 3)
    center_x = random.randint(cmin, cmax)
    center_y = random.randint(cmin, cmax)

    startx = int(center_x - (size[0] / 2))
    starty = int(center_y - (size[0] / 2))

    _img = img_zoom[starty:(starty + size[0]), startx:(startx + size[0])]
    _label = label_zoom[starty:(starty + size[0]), startx:(startx + size[0])]
    _label = fix_classes(_label)
    return (_img, _label)


def elastic(img, label):
    alpha = random.uniform(100, 140)
    sigma = 10
    alpha_affine = random.uniform(0, 20)

    random_state = np.random.RandomState(None)

    shape = img.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    _img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    _label = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    _img = map_coordinates(_img, indices, order=0, mode='reflect').reshape(shape)
    _label = map_coordinates(_label, indices, order=0, mode='reflect').reshape(shape)
    _label = fix_classes(_label)

    return (_img, _label)


funcs12 = [add_noise, rotate, shear, zoom, elastic]

if __name__ == '__main__':
    k = np.random.choice(funcs12)
    print(k)