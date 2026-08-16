"""Bayesian decoding of SWR events: is the sequence content a spatial trajectory?"""

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm

from dandi_io import open_session
from swr_lib import FS, bayesian_decode, linearize_position, running_speed

SESSION = "Achilles_10252013"
BIN = 0.020            # decoding bin (s)
MIN_EVENT = 0.100      # short ripples are padded to this length
MIN_CELLS = 5          # active place cells required in an event
MIN_BINS = 5           # time bins required
N_SHUF = 500
RNG = np.random.default_rng(1)

nwbfile, nwb, h5 = open_session(SESSION)
units = nwb["units"]
pyr = units[units.cell_type == "excitatory"]
pos, TRACK = linearize_position(nwb)
speed = running_speed(pos)

pf = np.load("place_fields_Achilles_10252013.npz")
centers = pf["centers"]
all_units = pf["units"]
place_units = pf["place_units"]
sel = np.isin(all_units, place_units)
templates = {"rightward": np.nan_to_num(pf["tc_right"])[:, sel],
             "leftward": np.nan_to_num(pf["tc_left"])[:, sel]}
cells = all_units[sel]
print(f"{len(cells)} place cells in the decoding template")

rip = np.load("ripples_Achilles_10252013.npz")
ripples = nap.IntervalSet(start=rip["start"], end=rip["end"])

ep_df = nwbfile.epochs.to_dataframe()
epochs = {lab.replace("Epoch", ""): nap.IntervalSet(
    start=ep_df.query("label==@lab").start_time.values,
    end=ep_df.query("label==@lab").stop_time.values) for lab in ep_df.label}

# candidate events: ripples outside running
run_ep = speed.threshold(0.05).time_support.merge_close_intervals(0.5)
ok = np.ones(len(ripples), bool)
for s, e in zip(run_ep.start, run_ep.end):
    ok &= ~((ripples.end > s) & (ripples.start < e))
c_start, c_end = ripples.start[ok], ripples.end[ok]
pad = np.maximum(MIN_EVENT - (c_end - c_start), 0) / 2      # pad short ripples
cand = nap.IntervalSet(start=c_start - pad, end=c_end + pad)
print(f"{len(cand)} candidate events (immobility ripples)")

# ------------------------------------------------ flat spike array for fast slicing
spike_t = np.concatenate([pyr[u].times() for u in cells])
spike_c = np.concatenate([np.full(pyr[u].shape[0], i) for i, u in enumerate(cells)])
o = np.argsort(spike_t)
spike_t, spike_c = spike_t[o], spike_c[o]


def event_counts(t0, t1):
    """(n_time_bins, n_cells) spike-count matrix for one event."""
    n = int(np.floor((t1 - t0) / BIN))
    if n < MIN_BINS:
        return None
    i0, i1 = np.searchsorted(spike_t, [t0, t0 + n * BIN])
    if i1 - i0 == 0:
        return None
    tb = np.minimum(((spike_t[i0:i1] - t0) / BIN).astype(int), n - 1)
    C = np.zeros((n, len(cells)))
    np.add.at(C, (tb, spike_c[i0:i1]), 1)
    return C


def weighted_corr_batch(P, x, t):
    """Weighted correlation between decoded position and time for a stack of posteriors.

    P: (S, T, X) posteriors (each row already normalised over X).
    """
    P = P / P.sum(axis=(1, 2), keepdims=True)
    tt = t[None, :, None]
    xx = x[None, None, :]
    mt = (P * tt).sum(axis=(1, 2), keepdims=True)
    mx = (P * xx).sum(axis=(1, 2), keepdims=True)
    cov = (P * (tt - mt) * (xx - mx)).sum(axis=(1, 2))
    vt = (P * (tt - mt) ** 2).sum(axis=(1, 2))
    vx = (P * (xx - mx) ** 2).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vt * vx)


def column_cycle_shuffle(post, n_shuf, rng):
    """Circularly shift the posterior of each time bin by an independent random offset."""
    T, X = post.shape
    shifts = rng.integers(0, X, size=(n_shuf, T))
    idx = (np.arange(X)[None, None, :] + shifts[:, :, None]) % X
    return np.take_along_axis(post[None, :, :].repeat(n_shuf, 0), idx, axis=2)


def id_shuffle_corr(C, tmpl, tcent, n_shuf, rng):
    """Weighted correlations after randomly reassigning place fields to cells.

    Preserves each cell's spike train and the set of fields, destroying only the
    cell-to-field mapping, so it tests whether the *specific* sequence matters.
    """
    L = np.log(np.clip(tmpl, 1e-3, None)).T          # (n_cells, n_pos)
    offset = BIN * np.clip(tmpl, 1e-3, None).sum(axis=1)[None, None, :]
    perms = np.stack([rng.permutation(L.shape[0]) for _ in range(n_shuf)])
    ll = np.einsum("tn,snx->stx", C, L[perms]) - offset
    ll -= ll.max(axis=2, keepdims=True)
    post = np.exp(ll)
    post /= post.sum(axis=2, keepdims=True)
    return weighted_corr_batch(post, centers, tcent)


results = []
for k in tqdm(range(len(cand)), desc="decoding events"):
    t0, t1 = cand.start[k], cand.end[k]
    C = event_counts(t0, t1)
    if C is None:
        continue
    n_active = int((C.sum(axis=0) > 0).sum())
    if n_active < MIN_CELLS:
        continue
    tcent = (np.arange(C.shape[0]) + 0.5) * BIN

    row = dict(start=t0, end=t1, n_active=n_active, n_bins=C.shape[0])
    for d, tmpl in templates.items():
        post = bayesian_decode(C, tmpl, BIN)
        r = float(weighted_corr_batch(post[None], centers, tcent)[0])
        cyc = weighted_corr_batch(column_cycle_shuffle(post, N_SHUF, RNG),
                                  centers, tcent)
        ids = id_shuffle_corr(C, tmpl, tcent, N_SHUF, RNG)
        row[f"r_{d}"] = r
        row[f"p_cycle_{d}"] = float((np.abs(cyc) >= abs(r)).mean())
        row[f"p_id_{d}"] = float((np.abs(ids[np.isfinite(ids)]) >= abs(r)).mean())
    results.append(row)

import pandas as pd
res = pd.DataFrame(results)
res["direction"] = np.where(res.r_rightward.abs() >= res.r_leftward.abs(),
                            "rightward", "leftward")
res["r"] = np.where(res.direction == "rightward", res.r_rightward, res.r_leftward)
res["p_cycle"] = np.where(res.direction == "rightward",
                          res.p_cycle_rightward, res.p_cycle_leftward)
res["p_id"] = np.where(res.direction == "rightward",
                       res.p_id_rightward, res.p_id_leftward)
res["significant"] = (res.p_cycle < 0.05) & (res.p_id < 0.05)
# forward = decoded position advances in the direction of travel of the template
res["forward"] = np.where(res.direction == "rightward", res.r > 0, res.r < 0)

for name, ep in epochs.items():
    m = np.zeros(len(res), bool)
    for s, e in zip(ep.start, ep.end):
        m |= (res.start >= s) & (res.start < e)
    res.loc[m, "epoch"] = name

print(res.groupby("epoch").agg(n=("r", "size"), n_sig=("significant", "sum"),
                               frac_sig=("significant", "mean")))
res.to_csv("replay_events_Achilles_10252013.csv", index=False)

sig = res[res.significant]
print(f"{len(sig)}/{len(res)} events with significant trajectory content "
      f"({len(sig)/len(res)*100:.0f}%); "
      f"{sig.forward.mean()*100:.0f}% forward, {(1-sig.forward.mean())*100:.0f}% reverse")
print("expected under the null: 5%")
