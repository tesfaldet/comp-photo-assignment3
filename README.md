# comp-photo-assignment3

#### Task 1: Focus Stacking
For this task, I first computed the laplacian of the images and their gradient magnitudes as well. Then I obtained the index (at each pixel) of the image corresponding to max laplacian response. This produced a map where each pixel maps to the index of the image with the largest laplacian response. I chose the laplacian since it gives zero-crossing edge responses. I got rid of noise by using a median filter.

Next, for each pixel, I measured the gradient variance across all images. I made a decision map based off of this, where at each pixel, assign 1 if the gradient variance is greater than some threshold, and assign 0 otherwise.

After, I created an image that is a weighted combination of all images, where the weighting for each image (to obtain a single pixel value) is determined by its gradient intensity. For each pixel, the resulting values were divided by the sum of the gradient intensities across all images.

Finally, to create the focused image, at each pixel, if the decision value is 1 then the pixel from the image corresponding to the max laplacian response is taken. Otherwise, the weighted combination of the pixel value across all images is taken.

#### Task 2: Defocus Blur Detection
This task follows the paper to a T, pretty much. First, divide the picture into 128x128 blocks covering each feature. Find the brightest feature using the sum of pixel intensities in each feature. Retrieve its DC component and transfer it to all other features via Fourier space. Next, compute the tenengrad for each feature (and plot the response) and use that measure to obtain the sharpest (exemplar) feature. Given the exemplar, compute k blurred templates with increasing sigma. Template-match each blurred template with each feature, assigning the sigma for each that gives the closest match. Plot the PSF sigma map. Build a synthetic version of the original input using only the exemplar feature and the estimated blurred kernel for each feature. Apply inverse filtering (using the wiener filter) to each feature using its estimated blur kernel (via its sigma) to construct a deblurred version of the original image.
