#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np


def spatial_kernel(X, Y, x, y, sigma_center, sigma_surround, inverse=1):
    """Creates a spatial kernel for the LGN cells' receptive field.

    The kernel is defined as the difference of Gaussians.

    Reference: Eq 2.45, Theoretical Neuroscience Dayan & Abbot (2011).

    Args:
        X (ndarray of floats): x coordinates of the mesh grid.
        Y (ndarray of floats): y coordinates of the mesh grid.
        x (float): x coordinate of the position of the kernel on the grid. 
        y (float): y coordinate of the position of the kernel on the grid.
        sigma_center (float): size of the central region (should be less than 
            sigma_surround).
        sigma_surround (float): size of the surround region (should be greater
            than sigma_center).
        inverse (int, optional): determines if the type of the kernel (ON/OFF). 
            Takes values -1 or 1 only. If the given value is 1 the kernel is of 
            type ON, otherwise OFF. Defaults to 1.

    Returns:
        ndarray of float: the spatial kernel, normalized so that for the perfect
            stimulus (for the case of type ON:1 in the center and -1 in the 
            surround region) the firing rate is equal to 1.
    """

    Z = ((X - x)**2 + (Y - y)**2)  # move center

    center = (17.0 / (2*np.pi * (sigma_center**2))) * \
        np.exp(-Z / (2*(sigma_center**2)))

    surround = (16.0 / (2*np.pi * (sigma_surround**2))) * \
        np.exp(-Z / (2*(sigma_surround**2)))

    Z = center - surround

    # points outside circle/receptive field center are set to zero
    pts = (X-x)**2+(Y-y)**2 >= (2*sigma_surround)**2
    Z[pts] = 0

    # normalize by multiplying with scale factor(dividing number of points/area)
    return (Z * inverse) / np.sum(np.absolute(Z))


if __name__ == "__main__":
    import plots
    from matplotlib import pyplot as plt

    dx = dy = 0.05
    lx = ly = 5.0
    x = np.arange(-lx/2, lx/2, dx)
    y = np.arange(-ly/2, ly/2, dy)
    X, Y = np.meshgrid(x, y)
    xp = yp = 0

    sigma_center = 10.6 * (1/60)
    sigma_surround = 31.8 * (1/60)

    k = spatial_kernel(X, Y, xp, yp, sigma_center, sigma_surround, inverse=1)
    plots.plot_kernel_3D(k, X, Y)
    plt.show()
