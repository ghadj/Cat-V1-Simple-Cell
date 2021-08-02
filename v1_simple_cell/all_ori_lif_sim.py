#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
from matplotlib import pyplot as plt

import stimuli
import plots
import utils as local_utils


def get_active_rfc(xs_on, ys_on, xs_off, ys_off, time, total_rfc):
    # Returns the number of the active RF centers, i.e. ON on light and
    # OFF on dark
    count_active_rfc_exc = np.zeros(len(time))
    for x, y in zip(xs_on, ys_on):
        count_active_rfc_exc += stimulus[:, int((y+ly/2)/dy), int((x+lx/2)/dx)]

    for x, y in zip(xs_off, ys_off):
        count_active_rfc_exc += np.abs(stimulus[:, int((y+ly/2)/dy), int((x+lx/2)/dx)]-1)

    return count_active_rfc_exc, total_rfc-count_active_rfc_exc


def plot_active_rfc(active_rfc_exc, active_rfc_inh, total_rfc, time, ax=None, kwargs={}):
    # Plots percentage of active rf centers
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(time, (100*active_rfc_exc)/(total_rfc), label='excitatory')
    ax.plot(time, (100*active_rfc_inh)/(total_rfc), label ='inhibitory')
    ax.set_yticks(range(0, 101, 10))
    ax.set(**kwargs)


def plot_rf(stimulus, lx, ly, time, xs_on, ys_on, xs_off, ys_off, ax=None, kwargs={}):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    plots.plot_stimulus(stimulus, 1000, lx, ly, ax)
    ax.scatter(xs_on, ys_on, marker='x', color='r', s=200, label='ON')
    ax.scatter(xs_off, ys_off, marker='x', color='b', s=200, label='OFF' )
    ax.set(**kwargs)


def plot_firing_rate(time, rate_exc, rate_inh, rate_max_exc, rate_max_inh,
                     ax=None, kwargs={}):
    # Plots percentage of active rf centers
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(time, rate_exc, label='excitatory')
    ax.plot(time, rate_inh, label ='inhibitory')
    ax.set_yticks(range(0, max(rate_max_exc, rate_max_inh)+200, 200))
    ax.set(**kwargs)


def presynaptic_firing_rate(xs_on, ys_on, xs_off, ys_off, total_rfc, stimulus,
                            time, lx, ly, theta, rate_max_exc, rate_max_inh,
                            ax_rfc=None, ax_active=None, ax_rate=None,
                            plot_results=True):
    # Returns the firing rate for excitation and inhibition, based on the
    # percentage of active RF centers

    # Rotate receptive field
    xs_on, ys_on = local_utils.rotate_point_arr(
            xs_on, ys_on, theta)
    xs_off, ys_off = local_utils.rotate_point_arr(
            xs_off, ys_off, theta)

    # Count active rfc at each time step
    active_rfc_exc, active_rfc_inh = get_active_rfc(xs_on, ys_on, xs_off, ys_off,
                                                   time, total_rfc)

    rate_exc = (active_rfc_exc/total_rfc)*rate_max_exc
    rate_inh = (active_rfc_inh/total_rfc)*rate_max_inh

    if plot_results:
        plot_active_rfc(active_rfc_exc, active_rfc_inh, total_rfc, time, ax_active,
                        kwargs={'title': (str(round(math.degrees(theta))) + "°")})
        plot_rf(stimulus, lx, ly, time, xs_on, ys_on, xs_off, ys_off, ax_rfc,
                kwargs={'title': (str(round(math.degrees(theta))) + "°")})
        plot_firing_rate(time, rate_exc, rate_inh, rate_max_exc, rate_max_inh, ax_rate,
                         kwargs={'title': (str(round(math.degrees(theta))) + "°")})

    return rate_exc, rate_inh


if __name__ == "__main__":

    import lif_params as lp
    import lif_sim as ls

    # Grid parameters
    dx = dy = 0.1                   # resolution - step size (degrees)
    lx = ly = 8.0                   # size (degrees)
    x = np.arange(-lx/2, lx/2, dx)
    y = np.arange(-ly/2, ly/2, dy)
    X, Y = np.meshgrid(x, y)        # grid

    total_rfc = 7 # number of RF centers /2

    # Sine grating - spatial parameters
    # so that an ON-center part is on the light and OFF-surround part on dark spot
    K = 0.25/(0.39) * (2 * np.pi)   # spatial frequency (cycle per degree)
    Phi = 0 * np.pi                 # spatial phase (radians)
    A = 1                           # amplitude of sine grating
    omega = 4                       # temporal frequency (Hz)
    all_orientations = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi]
    stimulus = stimuli.sine_grating(A, K, Phi, omega, X, Y, lp.time)
    stimulus[stimulus < 0] = 0      # by default values < 0 were set to -1

    # Positions of RF centers
    # ratio of a subfield of the RF (see Jones and Palmer 1987)
    ratio = 1.5
    rf_w = (2 * np.pi)/K   # width of RF
    rf_l = ratio * rf_w  # length of RF

    xs_on = np.full(total_rfc, 0)
    xs_off = np.full(total_rfc, 0.5 * rf_w)
    ys_on = np.arange(-rf_l/2, rf_l/2, rf_l/total_rfc)
    ys_off = np.arange(-rf_l/2, rf_l/2, rf_l/total_rfc)

    # Plots parameters
    axes_x = 4
    axes_y = 2

    # Receptive field plot
    fig_rfc, axs_rfc = plt.subplots(axes_y, axes_x)
    fig_rfc.suptitle('Stimulus and Position of RF centers')
    axs_rfc[axes_y-1, axes_x-1].set_visible(False)   # remove last subplot

    # Percentage of active RF centers
    fig_active, axs_active = plt.subplots(axes_y, axes_x)
    fig_active.suptitle('Percentage of Active RF centers')
    axs_active[axes_y-1, axes_x-1].set_visible(False)  # remove last subplot
    fig_active.text(0.5, 0.04, 'Time (sec)', ha='center')
    fig_active.text(0.04, 0.5, 'Percentage of activated RF centers (%)',
                    va='center', rotation='vertical')

    # Firing rate
    fig_rate, axs_rate = plt.subplots(axes_y, axes_x)
    fig_rate.suptitle('Firing rate')
    axs_rate[axes_y-1, axes_x-1].set_visible(False) # remove last subplot
    fig_rate.text(0.5, 0.04, 'Time (sec)', ha='center')
    fig_rate.text(0.04, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')

    # LIF potential
    fig_pot, axs_pot = plt.subplots(axes_y, axes_x)
    fig_pot.suptitle('LIF Potential (mV)')
    axs_pot[axes_y-1, axes_x-1].set_visible(False) # remove last subplot
    fig_pot.text(0.5, 0.04, 'Time (sec)', ha='center')
    fig_pot.text(0.04, 0.5, 'LIF potential (mV)', va='center', rotation='vertical')

    v_all = np.zeros((len(all_orientations)+1, len(lp.time)), dtype=float)
    v_all[0] = lp.time

    # Run simulation
    for index, theta in enumerate(all_orientations):
        # Presynaptic firing rate
        rate_exc, rate_inh = presynaptic_firing_rate(xs_on, ys_on, xs_off, ys_off,
                                2*total_rfc, stimulus, lp.time, lx, ly, theta,
                                lp.rate_max_exc, lp.rate_max_inh,
                                ax_rfc=axs_rfc[index//axes_x, index % axes_x],
                                ax_active=axs_active[index//axes_x, index % axes_x],
                                ax_rate=axs_rate[index//axes_x, index % axes_x])

        # Injected current
        Iinj = ls.injected_current(lp.alpha_exc, lp.alpha_inh, lp.amp_exc, lp.amp_inh,
                                rate_exc, rate_inh,
                                lp.dt, lp.time, lp.dur)

        # Add noise to the injected current
        #Iinj = ls.awgn(Iinj-lp.Vrest, 20)+lp.Vrest

        # Leaky Integreate and Fire
        v = ls.lif_dynamics(Iinj, lp.Vrest, lp.g_L, lp.tau_m, lp.time, lp.dt)
        v_all[index+1] = v

        # Plot potential
        ax_pot = axs_pot[index//axes_x, index % axes_x]
        ax_pot.plot(lp.time, v, label='potential', color='black', linewidth=1)
        ax_pot.hlines(lp.Vrest, xmin=lp.time[0], xmax=lp.time[-1],
                      label='Vrest', color='black', alpha=0.3)
        ax_pot.hlines(lp.Vth, xmin=lp.time[0], xmax=lp.time[-1],
                      label='Vth', alpha=0.3, color='red')
        ax_pot.set_yticks(range(-90, 0, 10))
        ax_pot.set(**{'title':(str(round(math.degrees(theta))) + "°")})

    # Add legend to the plots
    handles, labels = axs_rfc[0,0].get_legend_handles_labels()
    fig_rfc.legend(handles, labels, loc='center right')

    handles, labels = axs_active[0,0].get_legend_handles_labels()
    fig_active.legend(handles, labels, loc='center right')

    handles, labels = axs_rate[0,0].get_legend_handles_labels()
    fig_rate.legend(handles, labels, loc='center right')

    handles, labels = axs_pot[0,0].get_legend_handles_labels()
    fig_pot.legend(handles, labels, loc='center right')

    # Export to csv
    np.savetxt("../../../Data/Simulation/simulation_all_orientations_04_30_nonoise.csv", v_all.T, delimiter=",")

    plt.show()
