#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from brian2 import second, mV, plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_kernels(kernels, lx, ly,  kwargs={}, ax=None):
    """Plots the given kernels.

    Args:
        kernels (list of ndarray of float): list of kernels.
        lx (float): dimensions (x axes).
        ly (float): dimensions (y axes).
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
        ax (axes object, optional): specified target axes. Defaults to None.
    """

    if not kernels:
        return

    if ax is None:
        fig, ax = plt.subplots()

    kernels_sum = np.sum(kernels, axis=0)
    masked_data = np.ma.masked_where(kernels_sum == 0.0, kernels_sum)
    ax.imshow(masked_data, extent=[-lx/2, lx/2, ly/2, -ly/2],
              interpolation='none', **kwargs)


def plot_kernel_3D(kernel, X, Y, kwargs={}, ax=None):
    """Plots single given kernel in 3D plot.

    Args:
        kernel (ndarray of float): spatial kernel of the receptive field.
        X (ndarray of floats): x coordinates of the mesh grid.
        Y (ndarray of floats): y coordinates of the mesh grid.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
        ax (axes object, optional): specified target axes. Defaults to None.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    kernel[kernel == 0] = np.nan
    ax.plot_surface(X, Y, kernel, rstride=1, cstride=1, alpha=0, linewidth=0.5,
                    edgecolors='k')

    cmap = colors.ListedColormap(['blue', 'red'])
    bounds = [-1, 0, 1]
    norm = colors.BoundaryNorm(bounds, 2)
    ax.contourf(X, Y, kernel, 10, offset=-0.002, cmap=cmap,
                norm=norm, alpha=0.4)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.002, 0.01)
    ax.view_init(10, 45)
    ax.set_xlabel('x (degrees)')
    ax.set_ylabel('y (degrees)')
    ax.set_zlabel('Ds')
    # ax.set_zticks([0])


def plot_stimulus(stimulus, t, lx, ly, ax=None):
    """Plots the given stimulus at the specified index/time.

    Args:
        stimulus (ndarray of float): stimulus for every time step.
        t (int): time index.
        lx (float): dimensions (x axes).
        ly (float): dimensions (y axes).
        ax (axes object, optional): specified target axes. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(stimulus[t, :], extent=[-lx/2, lx/2, ly/2, -ly/2],
              cmap='binary_r')


def plot_firing_rate_per_rfc(rates, xs, ys, t, ax=None, kwargs={}):
    """Plots the given firing rates in time.

    Args:
        rates (ndarray of float): the firing rate for every time step of each
            LGN cell.
        xs (list of float): x coordinates of the receptive fields.
        ys (list of float): y coordinates of the receptive fields.
        t (list of float): time.
        ax (axes object, optional): specified target axes. Defaults to None.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
    """

    if ax is None:
        fig, ax = plt.subplots()

    for index, (x, y) in enumerate(zip(xs, ys)):
        ax.plot(t, rates[:, index], label="x={} y={}".format(x, y))

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Firing rate (Hz)')

    ax.set(**kwargs)


def plot_stimulus_per_rfc(stimulus, xs, ys, t, lx, ly, dx, dy, ax=None):
    """Plots stimulus value per receptive field based on the position.

    Args:
        stimulus (ndarray of float): stimulus for every time step.
        xs (list of float): x coordinates of the receptive fields.
        ys (list of float): y coordinates of the receptive fields.
        t (list of float): time.
        lx (float): length of x axis of the stimulus.
        ly (float): length of y axis of the stimulus.
        dx (float): resolution of x axis of the stimulus.
        dy (float): resolution of y axis of the stimulus.
        ax (axes object, optional): specified target axes. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots()

    for x, y in zip(xs, ys):
        sine_xy = stimulus[:, int((y+ly/2)/dy), int((x+lx/2)/dx)]
        ax.plot(t, sine_xy, label='x={} y={}'.format(x, y))


def plot_postition_per_rfc(xs, ys, ax=None):
    """Plots a scatter plot of the positions of the centers of each receptive
    field.

    Args:
        xs (list of float): x coordinates of the receptive fields.
        ys (list of float): y coordinates of the receptive fields.
        ax (axes object, optional): specified target axes. Defaults to None.
    """

    # plot position per RF
    if ax is None:
        fig, ax = plt.subplots()

    for x, y in zip(xs, ys):
        ax.scatter(x, y, label='x={} y={}'.format(x, y))


def plot_receptive_field(kernels_on_ex, kernels_off_ex,
                         kernels_on_inh, kernels_off_inh,
                         stimulus, lx, ly, t=0, ax=None, kwargs={}):
    """Plots receptive fields and the stimulus on 2D plot.

    Args:
        kernels_on_ex (ndarray of float): kernels for excitatory center ON
            receptive field.
        kernels_off_ex (ndarray of float): kernels for excitatory center OFF
            receptive field.
        kernels_on_inh (ndarray of float): kernels for inhibitory center ON
            receptive field.
        kernels_off_inh (ndarray of float): kernels for inhibitory center OFF
            receptive field.
        stimulus (ndarray of float): stimulus for every time step.
        lx (float): dimensions (x axes).
        ly (float): dimensions (y axes).
        t (int, optional): time index. Defaults to 0.
        ax (axes object, optional): specified target axes. Defaults to None.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
    """

    if ax is None:
        fig, ax = plt.subplots()

    cmap = colors.ListedColormap(['blue', 'red'])
    bounds = [-1, 0, 1]
    norm = colors.BoundaryNorm(bounds, 2)
    kernel_kwargs = {'alpha': 0.3, 'cmap': cmap, 'norm': norm}

    plot_stimulus(stimulus, t, lx, ly, ax)
    plot_kernels(kernels_on_ex, lx, ly, kernel_kwargs, ax)
    plot_kernels(kernels_off_ex, lx, ly, kernel_kwargs, ax)
    #plot_kernels(kernels_on_inh, lx, ly, kernel_kwargs, ax)
    #plot_kernels(kernels_off_inh, lx, ly, kernel_kwargs, ax)
    ax.set(**kwargs)

    ax.set_xticks([])
    ax.set_yticks([])


def plot_mem(lif_state_monitor, ex_presyn_spike_monitor, in_presyn_spike_monitor,
             show_pre_spikes=True, ax=None, kwargs={}):
    """Plots the membrane potential (black) and the time of presynaptic
    excitatory (blue) and inhibitory (red) input.

    Args:
        lif_state_monitor (brian object): state monitor of the LIF neuron.
        ex_presyn_spike_monitor (brian object): excitatory presynaptic spike
            monitor.
        in_presyn_spike_monitor (brian object): inhibitory  presynaptic spike
            monitor.
        show_pre_spikes (bool, optional): if true plots the presynaptic spikes,
            below the membrane potential, otherwise not. Defaults to True.
        ax (axes object, optional): specified target axes. Defaults to None.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Plot membrane potential
    ax.plot(lif_state_monitor.t/second, lif_state_monitor[0].v/mV, color='k')
    ax.set(**kwargs)

    if not show_pre_spikes:
        return

    # Plot time of pre-synaptic spikes
    offset = -80
    for spike_ex, spike_inh in zip(ex_presyn_spike_monitor.spike_trains().values(),
                                   in_presyn_spike_monitor.spike_trains().values()):

        ax.plot(spike_ex/second, np.zeros_like(spike_ex/second) +
                offset, '|', color='b', alpha=0.4)
        ax.plot(spike_inh/second, np.zeros_like(spike_inh/second) +
                offset, '|', color='r', alpha=0.4)

        offset -= 1

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Membrane Potential (mV)')


def plot_bursts(lif_state_monitor, bursts_all, burst_groups, dt, ax=None,
                kwargs={}):
    """Plots the membrane potential indicating the bursts (red line) and
    burst-groups (green line).

    Args:
        lif_state_monitor (brian object): state monitor of the LIF neuron.
        bursts_all (list of int): indices of all bursts.
        burst_groups (list of int): indices of burst-groups.
        dt (float): time step.
        ax (axes object, optional): specified target axes. Defaults to None.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
    """

    if ax is None:
        fig, ax = plt.subplots()

    for b in bursts_all:
        plt.axvline(x=b*dt, color='r')

    for bg in burst_groups:
        plt.axvline(x=bg*dt, color='g')

    ax.plot(lif_state_monitor.t/second, lif_state_monitor[0].v/mV, color='k')
    ax.set(**kwargs)


def plot_pold(pold_ex, pold_in, prespiking_potential, time,
              thresh, ax=None, kwargs={}):
    """Plots the membrane potential with different color each period i.e.
    excitatory(blue), inhibitory(orange) and pre-spiking(green).

    Args:
        pold_ex (list of float): membrane potential of excitatory
            period.
        pold_in (list of float): membrane potential of inhibitory
            period.
        prespiking_potential (list of float): membrane potential of pre-spiking
            period.
        time (list of float): time.
        thresh (float): firing threshold potential.
        ax (axes object, optional): specified target axes. Defaults to None.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
    """

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(time, pold_ex, label='excited', linewidth=1)
    ax.plot(time, pold_in, label='inhibited', linewidth=1)
    ax.plot(time, prespiking_potential, label='pre-spiking', linewidth=1)
    ax.axhline(y=thresh, color='r', linestyle='-', label='Resting potential')

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Pold (mV)')

    ax.legend()
    ax.set(**kwargs)


def plot_pnew(pnew_ex, pnew_in, pnew_prespiking, pold_all, time, interval,
              ax=None, kwargs={}):
    """Plots Pnew  with different color each period i.e. excitatory(blue),
    inhibitory(orange) and pre-spiking(green).

    Args:
        pnew_ex (list of float): Pnew values for excitatory period.
        pnew_in (list of float): Pnew values for inhibitory period.
        pnew_prespiking (list of float): Pnew values for pre-spiking period.
        pold_all (list of float): membrane potential (Pold).
        time (list of float): time.
        interval (float): duration of interval used for Pnew calculation.
        ax (axes object, optional): specified target axes. Defaults to None.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
    """

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(time, pold_all, 'k-', alpha=0.2)
    #ax.plot(time, pold_all, 'ko', markersize=1, alpha=0.3)

    ax.axhline(y=0, color='k', alpha=0.2)

    ax.plot(time, pnew_ex, linewidth=1)
    ax.plot(time, pnew_in, linewidth=1)
    ax.plot(time, pnew_prespiking, linewidth=1)

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Pnew (mV)')

    ax.set_title('interval {:.3f} msec'.format(interval))
    ax.set_xlim(0, 4)

    ax.set(**kwargs)


def plot_std(std_ex, std_in, std_prespiking, std_all, intervals,
             std_exp_df=None, ax=None, kwargs={}):
    """Plots the standard deviation of Pnew  with different color each period
    i.e. excitatory(blue), inhibitory(orange) and pre-spiking(green).

    Args:
        std_ex (list of float): standard deviation of Pnew, during the
            excitatory period for all the given intervals.
        std_in (list of float): standard deviation of Pnew, during the
            inhibitory period for all the given intervals.
        std_prespiking (list of float): standard deviation of Pnew, during the
            pre-spiking period for all the given intervals.
        std_all (list of float): standard deviation of Pnew, during the
            total duration for all the given intervals.
        intervals (list of floats): duration of intervals used for Pnew
            calculation.
        std_exp_df (pandas dataframe, optional): The corresponding results of
            the experimental data. Defaults to None.
        ax (axes object, optional): specified target axes. Defaults to None.
        kwargs (dict, optional): pyplot arguments. Defaults to {}.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if not (std_exp_df is None):
        ax.plot(intervals, std_exp_df['std_in'],
                color='C1', label='inhibitory (real)')
        ax.plot(intervals, std_exp_df['std_prespiking'],
                color='C2', label='pre-spiking (real)')

    ax.plot(intervals, std_in, color='C1', linestyle='--', alpha=0.4,
            label='inhibitory (simulation)')
    ax.plot(intervals, std_prespiking, color='C2', linestyle='--', alpha=0.4,
            label='pre-spiking (simulation)')

    ax.set_xlabel('Δt (msec)')
    ax.set_ylabel('Standard deviation')

    ax.legend()
    ax.set(**kwargs)
