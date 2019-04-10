""" 3 Corner detection """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from scipy.signal import convolve2d, convolve
from skimage import color, io
import pdb
import scipy as sp
from scipy import signal
import math
from mpl_toolkits.mplot3d import Axes3D

# Load the image, convert to float and grayscale
img = io.imread('chessboard.jpg')
img = color.rgb2gray(img)

# 3.1
# Write a function myharris(image) which computes the harris corner for each pixel in the image. The function should return the R
# response at each location of the image.
# HINT: You may have to play with different parameters to have appropriate R maps.
# Try Gaussian smoothing with sigma=0.2, Gradient summing over a 5x5 region around each pixel and k = 0.1.)
def myharris(image, w_size, sigma, k):
    # This function computes the harris corner for each pixel in the image
    # INPUTS
    # @image    : a 2-D image as a numpy array
    # @w_size   : an integer denoting the size of the window over which the gradients will be summed
    # sigma     : gaussian smoothing sigma parameter
    # k         : harris corner constant
    # OUTPUTS
    # @R        : 2-D numpy array of same size as image, containing the R response for each image location

    ### your code should go here ###

    gauss = sp.signal.windows.gaussian(w_size, sigma)
    gauss = gauss.reshape(1, len(gauss))
    gauss_2d = convolve2d(gauss, np.transpose(gauss))
    image = convolve2d(image, gauss_2d, 'same')

    # Compute derivatives of image
    dx = np.array([-1, 0, 1])
    dx = dx.reshape(1, 3)
    dy = np.transpose(dx)
    ix = convolve2d(image, dx, 'same')
    iy = convolve2d(image, dy, 'same')

    sx = convolve2d(ix * ix, gauss_2d, 'same')
    sy = convolve2d(iy * iy, gauss_2d, 'same')
    sxy = convolve2d(ix * iy, gauss_2d, 'same')

    det = (sx * sy) - k * sxy**2
    trace = sx + sy
    R = det / trace

    return R


# 3.2
# Evaluate myharris on the image
R = myharris(img, 5, 0.2, 0.1)
plt.imshow(R)
plt.colorbar()
plt.show()


# 3.3
# Repeat with rotated image by 45 degrees
# HINT: Use scipy.ndimage.rotate() function
R_rotated = sp.ndimage.rotate(img, 45, (0, 1))
R_rotated = myharris(R_rotated, 13, 6, 0.1)
plt.imshow(R_rotated)
plt.colorbar()
plt.show()


# 3.4
# Repeat with downscaled image by a factor of half
# HINT: Use scipy.misc.imresize() function
R_scaled = sp.misc.imresize(img, 0.5)
R_scaled = myharris(R_scaled, 13, 6, 0.1)
plt.imshow(R_scaled)
plt.colorbar()
plt.show()
