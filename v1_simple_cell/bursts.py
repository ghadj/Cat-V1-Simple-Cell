#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np


def detect_bursts(spikes, dt):
    """Returns the indices of the first spikes of burst-groups, all bursts
    and the indices of the previous spike of each burst.

    The time that determines the duration of each spike is empirically 
    predefined.

    Args:
        spikes (numpy array of int): indices of spikes.
        dt (float): time-step.

    Returns:
        (list, list, list): lists containing the indices of the first spikes of 
        (1) burst-groups, (2) all bursts and (3) the indices of the previous 
        spike of each burst.
    """

    b = 0.007/dt        # 0.007 sec maximum time between spikes in a burst
    b_group = 0.05/dt   # 0.05 sec maximum time between spikes in a burst-group

    bursts_all = [spikes[0]]    # set first burst to the first spike
    burst_groups = [spikes[0]]  # set first burst to the first spike
    previous_spike = spikes[0]  # index of the previous spike
    pre_burst_spikes = [0]      # set index of first spike before a burst

    for spike in spikes[1:]:
        # Find beginning of a burst
        if spike-previous_spike > b:
            bursts_all.append(spike)
            pre_burst_spikes.append(int(previous_spike/dt))

        # Find beginning of a burst-group
        if spike-previous_spike > b_group:
            burst_groups.append(spike)

        previous_spike = spike

    return burst_groups, bursts_all, pre_burst_spikes


if __name__ == "__main__":
    import params as pms
    import simulation
    import plots

    import matplotlib.pyplot as plt
    from brian2 import second

    lif_state_monitor, lif_spike_monitor, epsp_monitor, ipsp_monitor = simulation.execute(
        pms.Vth, pms.Vreset, pms.Vrest, 0,
        pms.refract, pms.dur, pms.tau, pms.tau_syn_ex, pms.tau_syn_in,
        pms.xs_on_ex, pms.ys_on_ex, pms.xs_off_ex, pms.ys_off_ex,
        pms.xs_on_inh, pms.ys_on_inh, pms.xs_off_inh, pms.ys_off_inh,
        pms.X, pms.Y, pms.lx, pms.dx, pms.ly, pms.dy,
        pms.sigma_center, pms.sigma_surround,
        pms.stimulus, pms.t, pms.we, pms.wi, pms.num_rfc,
        pms.r0, pms.L0, pms.G_ex, pms.G_inh)

    burst_groups, bursts_all, pre_burst_spikes = detect_bursts(
        ((lif_spike_monitor.spike_trains()[0]/pms.dt)/second).astype(int),
        pms.dt)

    plots.plot_bursts(lif_state_monitor, bursts_all, burst_groups, pms.dt)

    plt.show()
