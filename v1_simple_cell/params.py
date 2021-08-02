#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from brian2 import *

import stimuli

# Grid parameters
dx = dy = 0.05              # resolution - step size (degrees)
lx = ly = 8.0               # size (degrees)
x = np.arange(-lx/2, lx/2, dx)
y = np.arange(-ly/2, ly/2, dy)
X, Y = np.meshgrid(x, y)    # grid

# Center-Surround parameters (values from Archie and Mel, 2000)
# Convert from minutes(') to degrees x 1/60
sigma_center = 10.6 * (1/60)    # size of the central region (degrees)
sigma_surround = 31.8 * (1/60)  # size of the surround region (degrees)
# number of RF-centers per type(ON/OFF and ex/inh)
num_rfc = 10

# Time parameters
# set dt for all objects (check Brian2 doc)
defaultclock.dt = 0.000244 * second
dt = defaultclock.dt/second          # time-step
# default value for defaultclock.dt = 0.0001 sec
# in experimental data time-step was 0.000244sec
dur = 4*second                       # duration of the simulation
t = np.arange(0., dur/second, dt)    # array of time instances

# Sine grating - spatial parameters
# so that an ON-center part is on the light and OFF-surround part on dark spot
K = 0.25/(0.39) * (2 * np.pi)   # spatial frequency (cycle per degree)
Phi = 0 * np.pi                 # spatial phase (radians)
# maximum degree of difference between light and dark areas
A = 1         # amplitude of sine grating
omega = 4     # temporal frequency (Hz)
Theta = [0, pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6, pi]  # orientation (radians)
stimulus = stimuli.sine_grating(A, K, Phi, omega, X, Y, t)

# Simple cell parameters
# In experimental data the first spike of each burst has lower threshold (-55*mV)
# than the rest. (firing threshold for all the spikes -49.2197*mV).
# Lower the threshold for all spikes to that of the first spike
Vth = -55*mV  # threshold potential equal to the Vth for first spike in experimental
Vreset = -55*mV
Vrest = -76.3369*mV    # resting potential
tau = 7.5*ms           # time constant for LIF
tau_syn_ex = 1.0*ms    # time constant for ex. synapses
tau_syn_in = 1.0*ms    # time contant for inh. synapses

# Remember to set refractory period shorter than the time for burst detection
refract = 6*ms         # refractory period
we = 4.6*mV            # weight of excitatory synapses
wi = -1.4*mV           # weight of inhibitory synapses

# Positions of RF centers
# ratio of a subfield of the RF (see Jones and Palmer 1987)
ratio = 1.7
rf_w = (2 * np.pi)/K   # width of RF
rf_l = ratio * rf_w  # length of RF

# At each point there is an excitatory ON/OFF-center and an inhibitory OFF/ON-center
xs_on_ex = xs_off_inh = np.full(num_rfc, 0)
xs_off_ex = xs_on_inh = np.full(num_rfc, 0.5 * rf_w)
ys_on_ex = ys_off_inh = np.arange(-rf_l/2, rf_l/2, rf_l/num_rfc)
ys_off_ex = ys_on_inh = np.arange(-rf_l/2, rf_l/2, rf_l/num_rfc)

# Firing rate params
r0 = 0       # any background firing that may occur when s = 0
L0 = 0       # the threshold value that L must attain before firing begins
G_ex = 135   # constant of proportionality (upper limit) - excitatory
G_inh = 100  # constant of proportionality (upper limit) - inhibitory
