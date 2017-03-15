import numpy as np
from scipy.ndimage import *


def normalize(input, a=0, b=1):
    return ((input - np.min(input)) * (b - a)) / \
           (np.max(input) - np.min(input))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # MATLAB style


def laplacianImage(I):
    weights = [1, -2, 1]
    fxx = correlate1d(I, weights=weights, axis=2, mode='constant')
    fyy = correlate1d(I, weights=weights, axis=1, mode='constant')
    return np.abs(fxx + fyy)


def discreteGradientMagImage(I):
    weights = [-1, 0, 1]
    fx = correlate1d(I, weights=weights, axis=2, mode='constant')
    fy = correlate1d(I, weights=weights, axis=1, mode='constant')
    return np.sqrt(fx**2 + fy**2)


def gaussLapImage(I, sigma):
    return np.abs(gaussian_laplace(I, sigma))


def sobelGradientMagImage(I):
    return generic_gradient_magnitude(I, sobel)
