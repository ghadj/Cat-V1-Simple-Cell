#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import utils as local_utils
import firing_rate
import kernel
import plots

import numpy as np
import math
from tqdm import tqdm  # progress bar
from brian2 import *
# Suppress resolution conflict warnings
BrianLogger.suppress_name('resolution_conflict')


def execute(Vth, Vreset, Vrest, theta,
            refract, dur, tau, tau_syn_ex, tau_syn_in,
            xs_on_ex, ys_on_ex, xs_off_ex, ys_off_ex,
            xs_on_inh, ys_on_inh, xs_off_inh, ys_off_inh,
            X, Y, lx, dx, ly, dy, sigma_center, sigma_surround,
            stimulus, t, we, wi, num_rfc, r0=0.0, L0=0.0, G_ex=100.0, G_inh=None,
            plot_results=True, ax_pot=None, ax_rf=None, ax_firing_rate=None):
    """Execute the simulation based on the given parameters.

    Args:
        Vth (float): firing threshold.
        Vreset (float): reset potential.
        Vrest (float): resting potential.
        theta (float): stimulus orientation.
        refract (float): duration of refractory period.
        dur (float): duration of the simulation.
        tau (float): membrane potential's time constant.
        tau_syn_ex (float): presynaptic excitatory time constant.
        tau_syn_in (float): presynaptic inhibitory time constant.
        xs_on_ex (list of float): x coordinates of excitatory ON receptive
            fields.
        ys_on_ex (list of float): x coordinates of excitatory ON receptive
            fields.
        xs_off_ex (list of float): x coordinates of excitatory OFF receptive
            fields.
        ys_off_ex (list of float): x coordinates of excitatory OFF receptive
            fields.
        xs_on_inh (list of float): x coordinates of inhibitory ON receptive
            fields.
        ys_on_inh (list of float): x coordinates of inhibitory ON receptive
            fields.
        xs_off_inh (list of float): x coordinates of inhibitory OFF receptive
            fields.
        ys_off_inh (list of float): x coordinates of inhibitory OFF receptive
            fields.
        X (ndarray of floats): x coordinates of the mesh grid.
        Y (ndarray of floats): y coordinates of the mesh grid.
        lx (float): size of the grid (x axis)
        dx (float): resolution of the grid (x axis).
        ly (float): size of the grid (y axis)
        dy (float): resolution of the grid (y axis).
        sigma_center (float): size of the central region (should be less than
            sigma_surround).
        sigma_surround (float): size of the surround region (should be greater
            than sigma_center).
        stimulus (list of float ndarray): the stimulus at each time step.
        t (list of float): time.
        we (float): weight of excitatory presynaptic input.
        wi (float): weight of inhibitory presynaptic input.
        num_rfc (int): number of receptive fields.
        r0 (float, optional): any background firing that may occur when stimulus
            is zero. Defaults to 0.0.
        L0 (float, optional): the threshold value that must be attained before
            firing begins. Defaults to 0.0.
        G_ex (float, optional): constant of proportionality for the excitation.
            Defaults to 100.
        G_inh (float, optional): constant of proportionality for the inhibition.
            Defaults to None.
        plot_results (bool, optional): if True makes the related plots. Defaults
             to True.
        ax_pot (axes object, optional): specified target axes for the membrane
            potential. Defaults to None.
        ax_rf (axes object, optional): specified target axes for the receptive
            fields. Defaults to None.
        ax_firing_rate (axes object, optional): specified target axes for the
            firing rate. Defaults to None.

    Returns:
        (brian object, brian object, brian object, brian object): state monitors
            for: (1) LIF neuron, (2) LIF spike monitor, (3) excitatory and (4)
            inhibitory presynaptic spikes monitors.
    """

    # default value for G_inh is the same as G_ex
    if G_inh is None:
        G_inh = G_ex

    # Definition of neuron model
    eqs = '''
          dv/dt = (Vrest-v+x+w)/tau : volt

          dx/dt = (y-x)/tau_syn_ex : volt
          dy/dt = -y/tau_syn_ex : volt

          dw/dt = (z-w)/tau_syn_in : volt
          dz/dt = -z/tau_syn_in : volt
          '''

    lif = NeuronGroup(1, eqs, threshold='v>Vth', reset='v=Vreset',
                      method='exact', refractory=refract, dt=defaultclock.dt)
    lif.v = Vrest

    xs_on_ex_rot, ys_on_ex_rot = local_utils.rotate_point_arr(
        xs_on_ex, ys_on_ex, theta)
    xs_off_ex_rot, ys_off_ex_rot = local_utils.rotate_point_arr(
        xs_off_ex, ys_off_ex, theta)
    xs_on_inh_rot, ys_on_inh_rot = local_utils.rotate_point_arr(
        xs_on_inh, ys_on_inh, theta)
    xs_off_inh_rot, ys_off_inh_rot = local_utils.rotate_point_arr(
        xs_off_inh, ys_off_inh, theta)

    kernels_on_ex = []
    for i, (x, y) in enumerate(zip(xs_on_ex_rot, ys_on_ex_rot)):
        kernels_on_ex.append(kernel.spatial_kernel(
            X, Y, x, y, sigma_center, sigma_surround, inverse=1))

    kernels_on_inh = []
    for i, (x, y) in enumerate(zip(xs_on_inh_rot, ys_on_inh_rot)):
        kernels_on_inh.append(kernel.spatial_kernel(
            X, Y, x, y, sigma_center, sigma_surround, inverse=1))

    kernels_off_ex = []
    for i, (x, y) in enumerate(zip(xs_off_ex_rot, ys_off_ex_rot)):
        kernels_off_ex.append(kernel.spatial_kernel(
            X, Y, x, y, sigma_center, sigma_surround, inverse=-1))

    kernels_off_inh = []
    for i, (x, y) in enumerate(zip(xs_off_inh_rot, ys_off_inh_rot)):
        kernels_off_inh.append(kernel.spatial_kernel(
            X, Y, x, y, sigma_center, sigma_surround, inverse=-1))

    kernels_ex = kernels_on_ex + kernels_off_ex
    kernels_inh = kernels_off_inh + kernels_on_inh

    firing_rate_ex = firing_rate.calculate_firing_rate(
        kernels_ex, stimulus, t, r0, L0, G_ex)
    firing_rate_inh = firing_rate.calculate_firing_rate(
        kernels_inh, stimulus, t, r0, L0, G_inh)

    firing_rate_ex_arr = TimedArray(firing_rate_ex*Hz, dt=defaultclock.dt)
    firing_rate_inh_arr = TimedArray(firing_rate_inh*Hz, dt=defaultclock.dt)

    poisson_gr_ex = PoissonGroup(
        2*num_rfc, rates='firing_rate_ex_arr(t,i)', dt=defaultclock.dt)
    poisson_gr_inh = PoissonGroup(
        2*num_rfc, rates='firing_rate_inh_arr(t,i)', dt=defaultclock.dt)

    # The on_pre keyword defines what happens when a presynaptic spike
    # arrives at a synapse.
    # excitatory synapses & inhibitory synapses
    synapses_ex = Synapses(poisson_gr_ex, lif, on_pre='y += we*exp(1)')
    synapses_inh = Synapses(poisson_gr_inh, lif, on_pre='z += wi*exp(1)')

    synapses_ex.connect()
    synapses_inh.connect()

    lif_state_monitor = StateMonitor(lif, ['v'], record=True)
    lif_spike_monitor = SpikeMonitor(lif)
    # excitatory presynaptic action potentials
    epsp_monitor = SpikeMonitor(poisson_gr_ex)
    # inhibitory presynaptic action potentials
    ipsp_monitor = SpikeMonitor(poisson_gr_inh)

    # Create a Network object in order to prevent "Magic Error" from Brian
    # Reference: Brian2 MagicNetwork documentation
    net = Network(lif)
    net.add(synapses_ex, synapses_inh,
            poisson_gr_inh, poisson_gr_ex,
            lif_state_monitor, lif_spike_monitor,
            epsp_monitor, ipsp_monitor)

    # Run simulation
    net.run(dur)

    if plot_results:
        plots.plot_mem(lif_state_monitor, epsp_monitor,
                       ipsp_monitor, show_pre_spikes=True,
                       ax=ax_pot,
                       kwargs={'title': (str(round(math.degrees(theta))) + "°")})

        plots.plot_receptive_field(kernels_on_ex, kernels_off_ex,
            kernels_on_inh, kernels_off_inh, stimulus, lx, ly, t=0, ax=ax_rf,
            kwargs={'title': (str(round(math.degrees(theta))) + "°")})

        plots.plot_firing_rate_per_rfc(firing_rate_ex,
            np.concatenate((xs_on_ex_rot, xs_off_ex_rot), axis=0),
            np.concatenate((ys_on_ex_rot, ys_off_ex_rot), axis=0), t,
            ax=ax_firing_rate)
        plots.plot_firing_rate_per_rfc(firing_rate_inh,
            np.concatenate((xs_on_inh_rot, xs_off_inh_rot), axis=0),
            np.concatenate((ys_on_inh_rot, ys_off_inh_rot), axis=0), t,
            ax=ax_firing_rate)

    return lif_state_monitor, lif_spike_monitor, epsp_monitor, ipsp_monitor


if __name__ == "__main__":
    import params as pms

    # Declare plots
    # Number of subplots per axis
    axes_x = 4
    axes_y = 2

    # Membrane potential plots
    fig_pot, axs_pot = plt.subplots(axes_y, axes_x, sharex=True, sharey=True)
    fig_pot.suptitle('Membrane Potential')
    axs_pot[axes_y-1, axes_x-1].set_visible(False)  # remove last subplot

    # Receptive field plots
    fig_rf, axs_rf = plt.subplots(axes_y, axes_x, sharex=True, sharey=True)
    fig_rf.suptitle('LGN Receptive Fields')
    axs_rf[axes_y-1, axes_x-1].set_visible(False)  # remove last subplot

    # Firing rate plots
    fig_fire_rate, axs_fire_rate = plt.subplots(
        axes_y, axes_x, sharex=True, sharey=True)
    fig_fire_rate.suptitle('Firing Rate')
    axs_fire_rate[axes_y-1, axes_x-1].set_visible(False)  # remove last subplot

    for index, theta_i in enumerate(tqdm(pms.Theta)):
        lif_state_monitor, lif_spike_monitor, epsp_monitor, ipsp_monitor = execute(
            pms.Vth, pms.Vreset, pms.Vrest, theta_i,
            pms.refract, pms.dur, pms.tau, pms.tau_syn_ex, pms.tau_syn_in,
            pms.xs_on_ex, pms.ys_on_ex, pms.xs_off_ex, pms.ys_off_ex,
            pms.xs_on_inh, pms.ys_on_inh, pms.xs_off_inh, pms.ys_off_inh,
            pms.X, pms.Y, pms.lx, pms.dx, pms.ly, pms.dy,
            pms.sigma_center, pms.sigma_surround,
            pms.stimulus, pms.t, pms.we, pms.wi, pms.num_rfc,
            pms.r0, pms.L0, pms.G_ex, pms.G_inh, plot_results=True,
            ax_pot=axs_pot[index//axes_x, index % axes_x],
            ax_rf=axs_rf[index//axes_x, index % axes_x],
            ax_firing_rate=axs_fire_rate[index//axes_x, index % axes_x])

    plt.show()
