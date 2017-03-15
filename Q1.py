import numpy as np
import matplotlib
matplotlib.use('macosx')  # for rendering to work
import matplotlib.pyplot as plt
from scipy import misc
import glob
from utilities import *


def focusImage(images):
    # compute laplacian of images and gradient mag of images
    limages = laplacianImage(rgb2gray(images))
    lsum = np.sum(limages, axis=0)
    gimages = gaussLapImage(rgb2gray(images), 2)
    gsum = np.sum(gimages, axis=0)

    # get indices of image corresponding to max laplacian response, per-pixel
    max_indices = np.argmax(limages, axis=0)
    plt.imshow(max_indices, cmap=plt.cm.get_cmap('Set1'))
    plt.colorbar(cmap=plt.cm.get_cmap('Set1'))
    plt.show()

    # get rid of noise using median filter
    max_indices = median_filter(max_indices, 10)  # slow
    plt.imshow(max_indices, cmap=plt.cm.get_cmap('Set1'))
    plt.colorbar(cmap=plt.cm.get_cmap('Set1'))
    plt.show()

    # per-pixel, measure variance of gradients across images
    gvar = normalize(np.var(gimages, axis=0))
    plt.imshow(gvar)
    plt.show()

    # if variance > threshold, use pixel from image[max_indices]
    # else use a weighted combination of pixels from all images
    dmap = gvar.copy()
    dmap[dmap <= 1e-3] = 0
    dmap[dmap > 1e-3] = 1
    dmap = dmap.astype('uint8')
    plt.imshow(dmap, cmap='gray')
    plt.show()

    # create weighted image
    wimage = np.sum(gimages[..., np.newaxis] * images, axis=0) / \
        gsum[..., np.newaxis]
    wimage = wimage.astype('uint8')
    plt.imshow(wimage)
    plt.show()

    # create max image
    mimage = np.ndarray.choose(max_indices[..., np.newaxis], images)
    plt.imshow(mimage)
    plt.show()

    # create focused image (not as good as just using max image)
    fimage = np.ndarray.choose(dmap[..., np.newaxis],
                               np.stack([wimage, mimage]))
    plt.imshow(fimage)
    plt.show()

    return fimage


# retrieve images
filelist = glob.glob('T1/dataset-fly/fly/frame*')
filelist.sort()

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# compute focused image
misc.imsave('T1/dataset-fly/fly/fimage.png', focusImage(images))

filelist = glob.glob('T1/dataset-watch/watch/frame*')
filelist.sort()

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# compute focused image
misc.imsave('T1/dataset-watch/watch/fimage.png', focusImage(images))
