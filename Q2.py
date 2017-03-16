import numpy as np
import matplotlib
matplotlib.use('tkagg')  # for rendering to work
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
from utilities import *
from skimage import restoration


# read in image
im = misc.imread('T2/crosses.bmp').astype('float32')

# initialize tenengrad
block_size = 128
tgrad = np.zeros((im.shape[0] / block_size, im.shape[1] / block_size))

# find the brightest feature
sums = np.zeros((im.shape[0] / block_size, im.shape[1] / block_size))
for i in range(sums.shape[0]):
    for j in range(sums.shape[1]):
        sums[i, j] = np.sum(im[i*block_size:(i+1)*block_size,
                               j*block_size:(j+1)*block_size])
brightest_index = np.unravel_index(sums.argmax(), sums.shape)
brightest_feat = \
    im[brightest_index[0]*block_size:(brightest_index[0]+1)*block_size,
       brightest_index[1]*block_size:(brightest_index[1]+1)*block_size]

# write it to disk
misc.imsave('T2/brightest_feature.png', brightest_feat)

# retrieve its DC component
DC_max = np.fft.fft2(brightest_feat)[0, 0]

# transfer its DC component to all other features (intensity normalization)
for i in range(sums.shape[0]):
    for j in range(sums.shape[1]):
        if i == brightest_index[0] and j == brightest_index[1]:
            continue
        f = np.fft.fft2(im[i*block_size:(i+1)*block_size,
                           j*block_size:(j+1)*block_size])
        f[0, 0] = DC_max
        im[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
            np.fft.ifft2(f)

# compute tenegrad over 128x128 blocks around each feature
for i in range(tgrad.shape[0]):
    for j in range(tgrad.shape[1]):
        tgrad[i, j] = tenengrad(im[i*block_size:(i+1)*block_size,
                                   j*block_size:(j+1)*block_size])

# plot the tenengrad response
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, tgrad.shape[1], 1)
Y = np.arange(0, tgrad.shape[0], 1)
X, Y = np.meshgrid(X, Y)
Z = tgrad
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.get_cmap('jet'))
fig.colorbar(surf)
plt.show()

# retrieve exemplar (i.e. sharpest) feature
exemplar_index = np.unravel_index(tgrad.argmax(), tgrad.shape)
exemplar_feat = \
    im[exemplar_index[0]*block_size:(exemplar_index[0]+1)*block_size,
       exemplar_index[1]*block_size:(exemplar_index[1]+1)*block_size]

# write it to disk
misc.imsave('T2/infocus_feature.png', exemplar_feat)

# given exemplar, compute k blurred templates with increasing sigma
resolution = 100
sigmas = np.array([((1.0/2.0)*i) / 12.0 for i in range(1, resolution+1)])
blurred_templates = [gaussBlurImage(exemplar_feat, sigma) for sigma in sigmas]

# template match each blurred template with each feature
templates = [tenengrad(blurred) for blurred in blurred_templates]
diffs = np.stack([np.abs(tgrad - template) for template in templates])
min_indices = np.argmin(diffs, axis=0)
PSF = np.take(sigmas, min_indices)

# plot the PSF map
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, PSF.shape[1], 1)
Y = np.arange(0, PSF.shape[0], 1)
X, Y = np.meshgrid(X, Y)
Z = PSF
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.get_cmap('jet'))
fig.colorbar(surf)
plt.show()

# build a synthetic version of the original input using only the exemplar
# feature and estimated blurred kernel for each feature
synthetic = np.zeros((im.shape[0], im.shape[1]))
for i in range(PSF.shape[0]):
    for j in range(PSF.shape[1]):
        synthetic[i*block_size:(i+1)*block_size,
                  j*block_size:(j+1)*block_size] = \
                  gaussBlurImage(exemplar_feat, PSF[i, j])

# write it to disk
misc.imsave('T2/synthetic_image.png', synthetic)

# apply inverse filtering to each feature using its estimated blur kernel to
# construct a deblurred version of the original image
deblurred = im.copy()
for i in range(PSF.shape[0]):
    for j in range(PSF.shape[1]):
        f = im[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        h = gauss_impulse_response(f, PSF[i, j])
        f_deblurred = restoration.wiener(f, h, 1, clip=False)
        deblurred[i*block_size:(i+1)*block_size,
                  j*block_size:(j+1)*block_size] = f_deblurred

# write it to disk
misc.imsave('T2/deblurred_image.png', deblurred)
