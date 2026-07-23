"""Fast peri-event spike utilities (numpy searchsorted) for large trial counts."""
import numpy as np


def perievent_rel_times(spikes, events, window):
    """Relative spike times (s) of `spikes` around each event, within `window`.

    Returns (rel_times, trial_index). Assumes window length < min inter-event
    interval so each spike falls in at most one window.
    """
    lo, hi = window
    spikes = np.asarray(spikes)
    ev = np.asarray(events)
    j = np.searchsorted(ev, spikes, side="right") - 1
    rel, tri = [], []
    for off in (0, 1):  # preceding event (rel>=0) and following event (rel<0)
        k = j + off
        ok = (k >= 0) & (k < len(ev))
        r = np.full(len(spikes), np.nan)
        r[ok] = spikes[ok] - ev[k[ok]]
        m = ok & (r >= lo) & (r < hi)
        rel.append(r[m])
        tri.append(k[m])
    return np.concatenate(rel), np.concatenate(tri)


def psth(spikes, events, window, bin_size):
    """Trial-averaged firing rate (Hz) in bins spanning `window`."""
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    rel, _ = perievent_rel_times(spikes, events, window)
    counts, _ = np.histogram(rel, edges)
    return edges[:-1] + bin_size / 2, counts / len(events) / bin_size


def window_counts(spikes, events, window):
    """Spike count per event in [event+window[0], event+window[1])."""
    spikes = np.asarray(spikes)
    lo = np.searchsorted(spikes, np.asarray(events) + window[0], side="left")
    hi = np.searchsorted(spikes, np.asarray(events) + window[1], side="left")
    return hi - lo
