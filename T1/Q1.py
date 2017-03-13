import numpy as np
import matplotlib
matplotlib.use('tkagg')  # for rendering to work
import matplotlib.pyplot as plt
from scipy import misc
import glob
import scipy.ndimage.filters as filters


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # MATLAB style


def laplacianImage(I):
    return filters.laplace(I)


def gradientMagImage(I):
    return filters.generic_gradient_magnitude(I)


filelist = glob.glob('dataset-fly/fly/frame*')
images = np.stack([rgb2gray(misc.imread(fname)) for fname in filelist])
limages = laplacianImage(images)
gimages = gradientMagImage(images)
