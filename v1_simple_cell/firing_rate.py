#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np


def calculate_firing_rate(kernels, stimulus, t, r0=0.0, L0=0.0, G=1.0):
    """Calculation of the firing rate of LGN cells based on the given kernels 
    and stimulus.

    Reference: Eq. 2.8 & 2.9, Theoretical Neuroscience Dayan and Abott (2011).  

    Args:
        kernels (list of float ndarray): list of kernels.
        stimulus (list of float ndarray): the stimulus at each time step.
        t (list of float): list of time instaces.
        r0 (float, optional): any background firing that may occur when stimulus
            is zero. Defaults to 0.0.
        L0 (float, optional): the threshold value that must be attained before 
            firing begins. Defaults to 0.0.
        G (float, optional): constant of proportionality. Defaults to 1.0.

    Returns:
        ndarray of float: the firing rate for each time step.
    """

    rate = np.zeros([t.size, len(kernels)])

    # Multiply spatial-kernel with sinusoidal stimulus at each time step
    for i, t in enumerate(t):
        rate[i, ...] = [np.sum(kernel * stimulus[i, ...])
                        for kernel in kernels]

    # Add background firing
    rate += r0
    # Rectify firing rates
    rate[rate < L0] = 0
    # Multiply with the constant of proportionality G
    return G*rate
