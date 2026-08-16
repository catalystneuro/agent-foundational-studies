"""Build a 20 ms binned spike-count / kinematics matrix restricted to trial intervals."""
import numpy as np, pandas as pd, pynapple as nap

BIN = 0.02

def build(spikes, hand_vel, cursor, tr, bin_size=BIN):
    ep = nap.IntervalSet(start=tr['start_time'].values, end=tr['stop_time'].values)
    counts = spikes.count(bin_size, ep=ep)                       # (n_bins, n_units)
    vel = hand_vel.bin_average(bin_size, ep=ep)
    pos = cursor.bin_average(bin_size, ep=ep)
    t = counts.index.values
    trial_idx = ep.in_interval(nap.Ts(t))                        # which trial each bin belongs to
    assert len(vel) == len(counts) == len(pos)
    out = dict(t=t, counts=np.asarray(counts.values), vel=np.asarray(vel.values),
               pos=np.asarray(pos.values), trial=np.asarray(trial_idx).astype(int), bin_size=bin_size)
    out['speed'] = np.hypot(out['vel'][:, 0], out['vel'][:, 1])
    out['vel_angle'] = np.arctan2(out['vel'][:, 1], out['vel'][:, 0])
    return out

def shift_by_lag(M, lag_bins):
    """Pair spike counts at bin i with kinematics at bin i+lag_bins, never crossing trials.
    Positive lag_bins => neural activity LEADS the kinematics it predicts."""
    n = len(M['t'])
    src = np.arange(n) + lag_bins
    ok = (src >= 0) & (src < n)
    ok[ok] &= M['trial'][src[ok]] == M['trial'][np.arange(n)[ok]]
    return np.arange(n)[ok], src[ok]
