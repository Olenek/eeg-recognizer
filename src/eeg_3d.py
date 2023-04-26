'''
Created by Pouya bashivan
This code has been created by p. bashivan source : https://github.com/pbashivan/EEGLearn
'''

__author__ = 'Pouya Bashivan'

import numpy as np

np.random.seed(123)

from scipy.interpolate import griddata
import math as m


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)


def gen_image(locs, features, resolution):
    """
    Generates EEG Images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs:        An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features:    Feature vector as [n_electrodes x n_bands]
                        Features corresponding to each frequency band are concatenated.
                        (alpha_1, alpha_2, ..., beta1, beta2,...)
    :param resolution:  Number of pixels in the output images
    :return:            Tensor [H, W, channels] containing generated image.
    """
    eeg_channels = locs.shape[0]  # number of electrodes

    # test whether the feature vector length is divisible by number of electrodes
    assert features.shape[0] % eeg_channels == 0

    img_channels = int(features.shape[0] / eeg_channels)

    # split by bands (channels)
    feat_array_temp = [features[c * eeg_channels: eeg_channels * (c + 1)] for c in range(img_channels)]

    min_x, min_y = np.min(locs, axis=0)
    max_x, max_y = np.max(locs, axis=0)
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):resolution * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):resolution * 1j
                     ]

    # locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
    # for c in range(img_channels):
    #     feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros(4))

    temp_interp = [np.zeros([resolution, resolution])] * img_channels

    # Generate edgeless images

    # Interpolating
    for c in range(img_channels):
        temp_interp[c] = griddata(locs, feat_array_temp[c], (grid_x, grid_y),
                                        method='cubic', fill_value=0)

    # swap axes [channels x H x W] -> [H x channels x W] -> [H x W x channels]
    return np.asarray(temp_interp)