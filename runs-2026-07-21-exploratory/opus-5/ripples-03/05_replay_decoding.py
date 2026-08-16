"""Stage 5: Bayesian decoding of replay inside sharp-wave ripples.

Candidate events are population bursts that coincide with a detected ripple.
Each event is decoded with the two direction-specific place-field templates from
stage 4 and scored by the weighted correlation between decoded position and time.
The event statistic is the larger |wcorr| of the two templates, and the null is
the same maximum taken over matched shuffles, so the model selection is paid for.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import swr_utils as su

SESSION = "Achilles-10252013"
BIN = 0.020            # decoding bin (s)
MUA_BIN = 0.010
MUA_SIGMA = 0.015      # s
MUA_PEAK_Z = 3.0
PBE_MIN, PBE_MAX = 0.100, 0.500
MIN_ACTIVE = 5
N_SHUFFLE = 400
ALPHA = 0.05
RNG = np.random.default_rng(1)


def population_bursts(spk, ep, ripple_peaks):
    """Population burst events (MUA peak > 3 z) that contain a ripple peak."""
    counts = spk.count(MUA_BIN, ep).sum(axis=1)
    mua = np.asarray(counts, dtype=float)
    sm = gaussian_filter1d(mua, MUA_SIGMA / MUA_BIN)
    z = nap.Tsd(t=np.asarray(counts.index), d=(sm - sm.mean()) / sm.std(), time_support=ep)
    cand = z.threshold(0.0, "above").time_support
    dur = cand.end - cand.start
    cand = cand[(dur >= PBE_MIN) & (dur <= PBE_MAX)]

    keep = []
    for i in range(len(cand)):
        seg = z.get(cand.start[i], cand.end[i])
        if len(seg) and seg.values.max() >= MUA_PEAK_Z:
            if np.any((ripple_peaks >= cand.start[i]) & (ripple_peaks <= cand.end[i])):
                keep.append(i)
    return cand[np.array(keep)], z


def weighted_corr(post, x, t):
    """Weighted correlation and slope of decoded position against time."""
    w = post / post.sum()
    mx = (w * x[:, None]).sum()
    mt = (w * t[None, :]).sum()
    dx, dt = x[:, None] - mx, t[None, :] - mt
    cov = (w * dx * dt).sum()
    vx = (w * dx ** 2).sum()
    vt = (w * dt ** 2).sum()
    return cov / np.sqrt(vx * vt), cov / vt  # correlation, slope in cm/s


def shuffle_scores(post, x, t, n=N_SHUFFLE):
    """Null |wcorr| from a circular position shuffle and from a time permutation."""
    nx, nt = post.shape
    circ, perm = np.empty(n), np.empty(n)
    ar = np.arange(nx)[:, None]
    for i in range(n):
        rows = (ar - RNG.integers(0, nx, size=nt)[None, :]) % nx
        circ[i] = abs(weighted_corr(np.take_along_axis(post, rows, axis=0), x, t)[0])
        perm[i] = abs(weighted_corr(post[:, RNG.permutation(nt)], x, t)[0])
    return circ, perm


def decode_event(spk, tc_df, iset):
    """Bayesian decoding within one event; returns posterior (position x time)."""
    _, proba = nap.decode_bayes(tuning_curves=tc_df, data=spk, epochs=iset, bin_size=BIN)
    return np.asarray(proba).T, np.asarray(proba.index)


def score_event(spk, templates, centers, iset):
    """Score one event: max |wcorr| over templates, with a matched shuffle null."""
    obs, nulls, keep = {}, {}, {}
    for k, tc in templates.items():
        post, tt = decode_event(spk, tc, iset)
        if post.shape[1] < 5 or not np.isfinite(post).all():
            return None
        t = tt - tt[0]
        r, slope = weighted_corr(post, centers, t)
        circ, perm = shuffle_scores(post, centers, t)
        obs[k] = (r, slope)
        nulls[k] = (circ, perm)
        keep[k] = (post, tt)
    best = max(obs, key=lambda k: abs(obs[k][0]))
    r, slope = obs[best]
    # null for the max-over-templates statistic, paired across shuffle draws
    circ_max = np.maximum(*[nulls[k][0] for k in templates])
    perm_max = np.maximum(*[nulls[k][1] for k in templates])
    p_circ = (np.sum(circ_max >= abs(r)) + 1) / (N_SHUFFLE + 1)
    p_perm = (np.sum(perm_max >= abs(r)) + 1) / (N_SHUFFLE + 1)
    return dict(template=best, wcorr=r, slope=slope, p_circ=p_circ, p_perm=p_perm,
                max_p=max(p_circ, p_perm)), keep[best]


if __name__ == "__main__":
    h5 = su.open_session(SESSION)
    epochs = su.load_epochs(h5)
    units = su.load_units(h5)
    pyr = units.getby_category("cell_type")["excitatory"]

    pf = np.load("place_fields.npz")
    centers, is_place = pf["centers"], pf["is_place"]
    ids = pf["unit_ids"][is_place]
    spk = pyr[list(ids)]
    templates = {
        k: xr.DataArray(pf[k][:, is_place].T, dims=["unit", "position"],
                        coords={"unit": ids, "position": centers})
        for k in ("right", "left")
    }
    print("%d place cells in the decoding template" % len(ids))

    peak_t = np.load("ripples.npz")["peak_t"]

    # ---- decoder sanity check: decode the animal's actual position -------------
    pos = su.load_position(h5)
    pos = pos[np.isfinite(pos.values)]
    speed = su.compute_speed(pos)
    run_speed = float(np.median(speed.restrict(nap.IntervalSet(pf["run_start"], pf["run_end"])).values))
    err = {}
    for k in ("right", "left"):
        ep_dir = nap.IntervalSet(pf[k + "_start"], pf[k + "_end"])
        dec, _ = nap.decode_bayes(tuning_curves=templates[k], data=spk, epochs=ep_dir, bin_size=0.25)
        err[k] = np.abs(np.asarray(dec) - np.asarray(pos.interpolate(dec)))
    all_err = np.concatenate(list(err.values()))
    chance = np.median(np.abs(RNG.uniform(0, 160, 20000) - RNG.uniform(0, 160, 20000)))
    print("decoder median error while running: %.1f cm (chance %.1f cm); median run speed %.0f cm/s"
          % (np.median(all_err), chance, run_speed))

    # ---- score every ripple-associated population burst ------------------------
    results, posteriors = [], {}
    for ep_name in ("PRE", "POST"):
        ep = epochs[ep_name]
        pbe, _ = population_bursts(spk, ep, peak_t)
        n_rip = int(((peak_t >= ep.start[0]) & (peak_t <= ep.end[-1])).sum())
        print("%s: %d ripple-associated population bursts (of %d ripples)" % (ep_name, len(pbe), n_rip))
        for i in tqdm(range(len(pbe)), desc="decoding " + ep_name):
            iset = nap.IntervalSet(pbe.start[i], pbe.end[i])
            n_active = int((np.asarray(spk.restrict(iset).count(PBE_MAX)) > 0).sum())
            if n_active < MIN_ACTIVE:
                continue
            out = score_event(spk, templates, centers, iset)
            if out is None:
                continue
            rec, post = out
            rec.update(epoch=ep_name, event=i, t0=float(pbe.start[i]),
                       dur=float(pbe.end[i] - pbe.start[i]), n_active=n_active)
            results.append(rec)
            posteriors[(ep_name, i)] = post

    df = pd.DataFrame(results)
    df["significant"] = df["max_p"] < ALPHA
    df.to_csv("replay_events.csv", index=False)

    # ---- control: same pipeline after shuffling place-field identity -----------
    ctrl_idx = RNG.choice(len(df), size=min(400, len(df)), replace=False)
    perm_ids = RNG.permutation(len(ids))
    shuf = {k: xr.DataArray(v.values[perm_ids, :], dims=["unit", "position"],
                            coords={"unit": ids, "position": centers})
            for k, v in templates.items()}
    ctrl = []
    for j in tqdm(ctrl_idx, desc="cell-ID shuffle control"):
        row = df.iloc[j]
        iset = nap.IntervalSet(row["t0"], row["t0"] + row["dur"])
        out = score_event(spk, shuf, centers, iset)
        if out is not None:
            ctrl.append(out[0])
    ctrl = pd.DataFrame(ctrl)
    ctrl["significant"] = ctrl["max_p"] < ALPHA
    ctrl.to_csv("replay_control.csv", index=False)

    print("\n--- replay summary ---")
    for ep_name, g in df.groupby("epoch"):
        print("%s: %d candidate events, %d significant (%.1f%%), median |wcorr| %.2f"
              % (ep_name, len(g), g["significant"].sum(), 100 * g["significant"].mean(),
                 g["wcorr"].abs().median()))
    print("cell-identity-shuffled control: %.1f%% significant (n=%d), median |wcorr| %.2f"
          % (100 * ctrl["significant"].mean(), len(ctrl), ctrl["wcorr"].abs().median()))
    sig = df[df["significant"]]
    fwd = ((sig.template == "right") & (sig.slope > 0)) | ((sig.template == "left") & (sig.slope < 0))
    print("significant events: %d forward, %d reverse" % (fwd.sum(), (~fwd).sum()))
    print("median |replay speed|: %.0f cm/s = %.0fx the animal's median running speed"
          % (sig["slope"].abs().median(), sig["slope"].abs().median() / run_speed))

    np.savez("decoder_check.npz", err_right=err["right"], err_left=err["left"],
             chance=chance, run_speed=run_speed)
    np.savez("posteriors.npz", **{f"{k[0]}_{k[1]}": v[0] for k, v in posteriors.items()})
    np.savez("posterior_times.npz", **{f"{k[0]}_{k[1]}": v[1] for k, v in posteriors.items()})
    print("saved replay_events.csv, replay_control.csv, posteriors.npz")
