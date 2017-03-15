import numpy as np
import matplotlib
matplotlib.use('tkagg')  # for rendering to work
import matplotlib.pyplot as plt
from scipy import misc
import glob
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
    return fxx + fyy


def gradientMagImage(I):
    weights = [-1, 1]
    fx = correlate1d(I, weights=weights, axis=2, mode='constant')
    fy = correlate1d(I, weights=weights, axis=1, mode='constant')
    return np.sqrt(fx**2 + fy**2)


def activityMap(G, L):
    A = G / (np.abs(L) + 1e-8)
    # A = median_filter(A, size=3)
    return A


filelist = glob.glob('dataset-fly/fly/frame*')
filelist.sort()
images = np.stack([rgb2gray(misc.imread(fname)) for fname in filelist])

limages = laplacianImage(images)
gimages = gradientMagImage(images)
amap = activityMap(gimages, limages)
amax = np.amax(amap, axis=0)
amin = np.amin(amap, axis=0)
asum = np.ndarray.sum(amap, axis=0)
dmap = amax / (amin + 1e-8)

for i in range(20):
    im = amap[i] / asum
    misc.imsave('dataset-fly/fly/amap%02d.png' % i, im)
