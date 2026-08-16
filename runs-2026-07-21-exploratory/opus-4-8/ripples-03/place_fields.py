"""Place-field (1D tuning curve) computation during track running, via Pynapple."""
import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d


def build_pynapple(d):
    """Return (spikes TsGroup, position Tsd, epochs dict) from cached data."""
    spikes = d["spikes"]
    cell_type = d["cell_type"]
    tsd_dict = {i: nap.Ts(t=np.asarray(spikes[i], dtype=float)) for i in range(len(spikes))}
    tsg = nap.TsGroup(tsd_dict)
    tsg.set_info(cell_type=cell_type, location=d["location"], shank=d["shank_id"])
    pos = nap.Tsd(t=d["pos_t"].astype(float), d=d["pos"].astype(float))
    epl, eps, epe = d["ep_label"], d["ep_start"], d["ep_stop"]
    epochs = {l: nap.IntervalSet(start=float(s), end=float(e))
              for l, s, e in zip(epl, eps, epe)}
    return tsg, pos, epochs


def running_intervals(pos, speed_thr=0.03, min_dur=0.5):
    """IntervalSet where |velocity| exceeds threshold (m/s)."""
    t = pos.index.values
    x = pos.values
    v = np.gradient(x, t)
    speed = np.abs(v)
    speed = gaussian_filter1d(speed, 5)
    run = speed > speed_thr
    # to intervals
    edges = np.diff(run.astype(int))
    starts = t[np.where(edges == 1)[0] + 1]
    stops = t[np.where(edges == -1)[0] + 1]
    if run[0]:
        starts = np.r_[t[0], starts]
    if run[-1]:
        stops = np.r_[stops, t[-1]]
    keep = (stops - starts) >= min_dur
    return nap.IntervalSet(start=starts[keep], end=stops[keep]), nap.Tsd(t=t, d=v)


def direction_epochs(pos, vel, run_ep):
    """Split running into rightward (v>0) and leftward (v<0) IntervalSets."""
    t = vel.index.values
    v = vel.values
    right = nap.Tsd(t=t, d=(v > 0).astype(float)).threshold(0.5).time_support.intersect(run_ep)
    left = nap.Tsd(t=t, d=(v < 0).astype(float)).threshold(0.5).time_support.intersect(run_ep)
    return right, left


def compute_place_fields(tsg, pos, ep, n_bins=50, smooth=1.5, pyramidal_only=True):
    grp = tsg[tsg.cell_type == "excitatory"] if pyramidal_only else tsg
    bins = np.linspace(np.nanmin(pos.values), np.nanmax(pos.values), n_bins + 1)
    tc = nap.compute_1d_tuning_curves(grp, pos, nb_bins=n_bins, ep=ep, minmax=(bins[0], bins[-1]))
    centers = (bins[:-1] + bins[1:]) / 2
    tc_sm = tc.copy()
    for c in tc_sm.columns:
        tc_sm[c] = gaussian_filter1d(tc_sm[c].values, smooth)
    return tc_sm, centers
