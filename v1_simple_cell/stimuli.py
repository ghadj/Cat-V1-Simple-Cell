#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np


def sine_grating(A, K, phi, omega, X, Y, t, theta=0.0, solid=True):
    """Generates a sinusoidal grating for every time step.

    Slightly modified Eq. 2.18, Theoretical Neuroscience Dayan & Abbot (2011).

    Note: The rotation of the stimulus can be set to 0 degrees, and instead
          rotate the position of the receptive field centers.

    Args:
        A (float): amplitude.
        K (float): spatial frequency.
        phi (float): phase.
        omega (float): temporal frequency.
        X (ndarray of float): x coordinates of the mesh grid.
        Y (ndarray of float): y coordinates of the mesh grid.
        t (list of float): time.
        theta (float, optional): rotation angle. Defaults to 0.0.
        solid (bool, optional): if true the stimulus will have only 1 or -1
            values, otherwise it will contain intermediate values. Defaults to
            True.

    Returns:
        ndarray of floats: stimulus; a sinusoidal grating for every time step.
    """

    stimulus = np.zeros((t.size, X[0].size, Y[0].size))

    for i, ti in enumerate(t):  # for each time-step
        stimulus[i, ...] = A * np.cos((K * X * np.cos(theta) -
                                       K * Y * np.sin(theta) - phi) +
                                      omega * 2 * np.pi * ti)

    if solid:
        stimulus[stimulus < 0] = -1
        stimulus[stimulus >= 0] = 1

    return stimulus
