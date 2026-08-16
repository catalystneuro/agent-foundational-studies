"""Shared plotting helpers and style."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
})

REGION_COLORS = {
    "visual cortex": "#2166ac",
    "visual thalamus": "#f4a259",
    "hippocampus": "#8c8c8c",
}
REGION_ORDER = ["visual cortex", "visual thalamus", "hippocampus"]


def psth(spike_times, event_times, window=(-0.5, 2.5), bin_size=0.025):
    """Trial-averaged rate (spikes/s) of one unit around a set of events."""
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    counts = np.zeros(len(edges) - 1)
    for t0 in event_times:
        rel = spike_times[
            np.searchsorted(spike_times, t0 + window[0]) : np.searchsorted(spike_times, t0 + window[1])
        ] - t0
        counts += np.histogram(rel, edges)[0]
    return edges[:-1] + bin_size / 2, counts / (len(event_times) * bin_size)


def raster_rows(spike_times, event_times, window=(-0.5, 2.5)):
    """Return (relative_time, trial_index) arrays for a raster plot."""
    xs, ys = [], []
    for i, t0 in enumerate(event_times):
        rel = spike_times[
            np.searchsorted(spike_times, t0 + window[0]) : np.searchsorted(spike_times, t0 + window[1])
        ] - t0
        xs.append(rel)
        ys.append(np.full(len(rel), i))
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def polar_tuning(ax, angles_deg, rates, color="#2166ac", period=360, label=None):
    """Closed polar tuning curve. `period` is 360 for direction, 180 for orientation."""
    th = np.deg2rad(angles_deg * (360.0 / period))
    th = np.concatenate([th, th[:1]])
    r = np.concatenate([rates, rates[:1]])
    ax.plot(th, r, "-o", color=color, ms=3, lw=1.4, label=label)
    ax.fill(th, r, color=color, alpha=0.15)
    ax.set_theta_zero_location("E")
    ax.set_rlabel_position(135)
    ax.tick_params(pad=0.5)
