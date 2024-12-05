import sys
from skimage import color
from skimage import io
from scipy.signal import convolve2d as conv2
from scipy.ndimage.filters import generic_filter as gf
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw


def fspecial_gaussian(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    source: https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def main(argv):

    #
    # usage: python harris.py 'image.png'
    #
    imgpath = argv[1]
    threshold = 1e-3

    # im = color.rgb2gray(io.imread(str(imgpath)))
    im = io.imread(str(imgpath))
    if len(im.shape) == 3:
        im = color.rgb2gray(im)
        
    g1 = fspecial_gaussian([9, 9], 1)  # Gaussian with sigma_d
    g2 = fspecial_gaussian([11, 11], 1.5)  # Gaussian with sigma_i

    img1 = conv2(im, g1, 'same')  # blur image with sigma_d
    Ix = conv2(img1, np.array([[-1, 0, 1]]), 'same')  # take x derivative
    Iy = conv2(img1, np.transpose(np.array([[-1, 0, 1]])), 'same')  # take y derivative

    # Compute elements of the Harris matrix H
    # we can use blur instead of the summing window
    Ix2 = conv2(np.multiply(Ix, Ix), g2, 'same')
    Iy2 = conv2(np.multiply(Iy, Iy), g2, 'same')
    IxIy = conv2(np.multiply(Ix, Iy), g2, 'same')
    eps = 2.2204e-16
    R = np.divide(np.multiply(Ix2, Iy2) - np.multiply(IxIy, IxIy),(Ix2 + Iy2 + eps))

    # don't want corners close to image border
    R[0:15] = 0  # all columns from the first 15 lines
    R[-16:] = 0  # all columns from the last 15 lines
    R[:, 0:15] = 0  # all lines from the first 15 columns
    R[:, -16:] = 0  # all lines from the last 15 columns

    # non-maxima suppression within 3x3 windows
    Rmax = gf(R, np.max, footprint=np.ones((3, 3)))
    Rmax[Rmax != R] = 0  # suppress non-max
    Rmax[Rmax < threshold] = 0
    v = Rmax[Rmax != 0]
    y, x = np.nonzero(Rmax)

    # show 'em
    for xp, yp in zip(x, y):
        rr, cc = draw.circle_perimeter(yp, xp, radius=6, shape=im.shape)
        im[rr, cc] = 1
    plt.imshow(im)
    plt.show()

    return v, x, y


if __name__ == "__main__":
    main(sys.argv)
