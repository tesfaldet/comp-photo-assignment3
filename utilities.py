import numpy as np
from scipy.ndimage import *


def normalize(input, a=0, b=1):
    return ((input - np.min(input)) * (b - a)) / \
           (np.max(input) - np.min(input))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # MATLAB style


def laplacianImages(I, multiple=True):
    weights = [1, -2, 1]
    axis_x = 1
    axis_y = 0
    if multiple:
        axis_x = 2
        axis_y = 1
    fxx = correlate1d(I, weights=weights, axis=axis_x, mode='constant')
    fyy = correlate1d(I, weights=weights, axis=axis_y, mode='constant')
    return np.abs(fxx + fyy)


def discreteGradientMagImages(I, multiple=True):
    weights = [-1, 0, 1]
    axis_x = 1
    axis_y = 0
    if multiple:
        axis_x = 2
        axis_y = 1
    fx = correlate1d(I, weights=weights, axis=axis_x, mode='constant')
    fy = correlate1d(I, weights=weights, axis=axis_y, mode='constant')
    return np.sqrt(fx**2 + fy**2)


def gaussLapImages(I, sigma):
    return np.abs(gaussian_laplace(I, sigma))


def sobelGradientMagImages(I):
    return generic_gradient_magnitude(I, sobel)


# np.choose has a regression where a compile-time constant NPY_MAXARGS
# limitss the number of arrays to be broadcasted to be less than 32
# This is my custom implementation that's slower *shakes fist*
def customChoose(choices, I):
    rows = choices.shape[0]
    cols = choices.shape[1]
    return np.array([I[choices[i, j], i, j]
                     for i in range(rows)
                     for j in range(cols)]).reshape((rows, cols, -1))


def tenengrad(I):
    weights = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    fx = convolve(I, weights=weights, mode='constant')
    fy = convolve(I, weights=weights.T, mode='constant')
    return np.mean(fx**2 + fy**2)


def gauss2d_kernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gaussBlurImage(I, sigma):
    return gaussian_filter(I, sigma=sigma)


def gauss_impulse_response(I, sigma):
    k = np.zeros_like(I)
    k[(k.shape[0]-1)/2, (k.shape[1]-1)/2] = 1.0
    return gaussBlurImage(k, sigma)
