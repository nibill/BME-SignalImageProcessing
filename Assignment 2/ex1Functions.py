""" 1 Linear filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['image.cmap'] = 'gray'
import time
import pdb



# 1.1
def boxfilter(n):
    # this function returns a box filter of size nxn

    ### your code should go here ###
    box_filter = np.ones((n, n)) / n**2
    return box_filter

# 1.2
# Implement full convolution


def myconv2(image, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    ### your code should go here ###

    # get shure that the filter has 2 dimensions
    if np.ndim(filt) < 2:
        filt = filt.reshape(1, len(filt))
        #print("filt dim < 2")
    if np.ndim(image) < 2:
        image = image.reshape(1, len(image))
        #print("img dim < 2")
    if np.ndim(filt) >= 2:  # kernel has 2 dimensions
        # flip kernel
        filt = np.flip(filt, 0)
        filt = np.flip(filt, 1)
        # print(image.shape)
        # print(filt.shape)
        filtered_img = np.zeros(
            (image.shape[0] + filt.shape[0] - 1, image.shape[1] + filt.shape[1] - 1))
        image = np.pad(image, ((filt.shape[0] - 1, filt.shape[0] - 1),
                               (filt.shape[1] - 1, filt.shape[1] - 1)), mode='constant')
        # print(image)
        # print(image.shape)
        # print(filtered_img.shape)
        for row in range(filtered_img.shape[0]):
            for col in range(filtered_img.shape[1]):
                #print(str(row) + ", " + str(col))
                filtered_img[row, col] = np.sum(np.multiply(
                    image[row:row + filt.shape[0], col:col + filt.shape[1]], filt))
                # filtered_img[row + int((filt.shape[0]-1) / 2), col +
                # int((filt.shape[1]-1) / 2)] =
    else:  # kernel has just 1 dimension
        # flip kernel
        # filt = np.flip(filt,0)
        # filtered_img = np.zeros((image.shape[0]+filt.shape[0]-1, image.shape[1]+filt.shape[1]-1))
        print("convolution error")
    # print(filtered_img)
    return filtered_img





# 1.4
# create a function returning a 1D gaussian kernel
def gauss1d(sigma, filter_length=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    ### your code should go here ###
    if filter_length % 2 != 0:
        x = np.linspace(int(-filter_length / 2),
                        int(filter_length / 2), filter_length)
    else:
        filter_length += 1
        x = np.linspace(int(-filter_length / 2),
                        int(filter_length / 2), filter_length)
    gauss_filter = np.exp(-(x**2) / (2 * sigma**2))
    gauss_filter = gauss_filter / sum(gauss_filter)
    # print(sum(gauss_filter))
    # plt.plot(gauss_filter)
    # plt.show()
    # print(gauss_filter)
    return gauss_filter


# 1.5
# create a function returning a 2D gaussian kernel
def gauss2d(sigma, filter_size=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_size   : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    ### your code should go here ###
    g1d = gauss1d(sigma, filter_size)
    g1dT = g1d.reshape(len(g1d), 1)
    g2d = myconv2(g1d, g1dT)
    gauss2d_filter = g2d
    return gauss2d_filter



# 1.6
# Convoltion with gaussian filter

def gconv(image, sigma):
    # INPUTS
    # image           : 2d image
    # @ sigma         : sigma of gaussian distribution
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    ### your code should go here ###
    filt = gauss2d(sigma, 30)
    img_filtered = myconv2(image, filt)

    return img_filtered
