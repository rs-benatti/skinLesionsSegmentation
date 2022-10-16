import numpy as np
from skimage import color

def lab_color_space(Y, L):
    img = LYL(Y, L)
    img = color.rgb2lab(img)
    return img

def LYL(Y, L):
    shape = (Y.shape[0], Y.shape[1], 3)
    img = np.empty(shape)
    img[:, :, 0] = L
    img[:, :, 1] = Y
    img[:, :, 2] = L
    return img

def ab_color_space(Y, L):
    input_image = lab_color_space(Y, L)
    shape = (Y.shape[0], Y.shape[1], 2)
    img = np.empty(shape)
    img[:, :, 0] = input_image[:, :, 1]
    img[:, :, 1] = input_image[:, :, 2]
    return img