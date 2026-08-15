"""End-to-end replay pipeline for a single session, reusable across sessions.

Consolidates the steps prototyped in scripts 01-05:
  behaviour -> place fields -> ripples -> population bursts -> Bayesian decoding
  -> weighted-correlation sequence test against two shuffles.
"""

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import replaylib as R

# ---- parameters, shared by every session -----------------------------------
BIN_CM = 4.0
SMOOTH_BINS = 1.5
MIN_SPEED = 0.05
MIN_PEAK_RATE = 1.0
MIN_RUN_SPIKES = 50   # spikes emitted while running

RIP_LOW, RIP_HIGH = 3.0, 6.0   # robust SD (median/MAD) of the ripple envelope
RIP_MIN_DUR, RIP_MAX_DUR = 0.03, 0.30

DEC_BIN = 0.020
MUA_BIN = 0.001
MUA_SIGMA = 0.010
MUA_HIGH = 3.0
EVT_MIN, EVT_MAX = 0.10, 0.50
MIN_ACTIVE = 5
MIN_BINS = 5
N_SHUFFLE = 500
P_THRESH = 0.025          # per-shuffle, Bonferroni-corrected for two templates

BLOCK, PAD = 1800.0, 5.0


def _tuning(group, position, ep, edges):
    tc = nap.compute_1d_tuning_curves(group, position, nb_bins=len(edges) - 1,
                                      ep=ep, minmax=(edges[0], edges[-1]))
    arr = np.nan_to_num(tc.values.T, nan=0.0)
    return gaussian_filter1d(arr, SMOOTH_BINS, axis=1, mode="nearest")


def spatial_info(rates, occ):
    p = occ / occ.sum()
    mean_r = (p * rates).sum()
    if mean_r <= 0:
        return 0.0
    r = np.clip(rates, 1e-6, None)
    return float((p * (r / mean_r) * np.log2(r / mean_r)).sum())


def build_templates(h5, nwb, nwbfile):
    """Direction-specific place-field templates from the maze epoch."""
    position, _ = R.load_position(h5)
    speed = R.compute_speed(position)
    run_dirs = R.direction_intervals(position, speed, min_speed=MIN_SPEED)
    run_all = run_dirs["right"].union(run_dirs["left"])

    lo, hi = np.percentile(position.restrict(run_all).d, [0.2, 99.8])
    n_pos = int(np.ceil((hi - lo) / (BIN_CM / 100)))
    edges = np.linspace(lo, lo + n_pos * BIN_CM / 100, n_pos + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    cell_type = np.array(nwbfile.units["cell_type"][:])
    units = nwb["units"]
    pyr = np.array(units.index)[cell_type == "excitatory"]
    pyr_units = units[list(pyr)]

    tc = {d: _tuning(pyr_units, position, run_dirs[d], edges) for d in ("right", "left")}
    tc_all = _tuning(pyr_units, position, run_all, edges)
    occ, _ = np.histogram(position.restrict(run_all).d, bins=edges)
    si = np.array([spatial_info(tc_all[i], occ) for i in range(len(pyr))])
    # Selection is on firing, not on spatial information.  A fixed information
    # threshold left as few as 11 cells in some sessions, which more than
    # doubled the cross-validated decoding error; the Bayesian decoder tolerates
    # weakly tuned cells because their near-flat tuning curves contribute an
    # almost position-independent term to the likelihood.
    n_run_spikes = np.array([len(pyr_units[i].restrict(run_all)) for i in pyr_units.index])
    is_place = (tc_all.max(axis=1) >= MIN_PEAK_RATE) & (n_run_spikes >= MIN_RUN_SPIKES)

    return dict(position=position, speed=speed, run_dirs=run_dirs, run_all=run_all,
                edges=edges, centers=centers, place_ids=pyr[is_place],
                templates={d: tc[d][is_place] for d in tc},
                tc_all=tc_all[is_place], si=si[is_place], si_all=si,
                n_pyr=len(pyr))


def detect_epoch_ripples(h5, ep, ch, rate, rem):
    """Ripple events over one epoch, streamed in blocks; REM excluded."""
    start, stop = float(ep.start[0]), float(ep.end[0])
    ts, es = [], []
    for b in tqdm(np.arange(start, stop, BLOCK), desc="  LFP", leave=False):
        seg = R.read_lfp(h5, [ch], max(start, b - PAD), min(stop, b + BLOCK + PAD))
        _, env = R.ripple_envelope(seg.values[:, 0], rate)
        keep = (seg.t >= b) & (seg.t < min(b + BLOCK, stop))
        ts.append(seg.t[keep]); es.append(env[keep])
    t, env = np.concatenate(ts), np.concatenate(es)
    iv, pk_t, pk_z = R.detect_ripples(env, t, rate, RIP_LOW, RIP_HIGH,
                                      RIP_MIN_DUR, RIP_MAX_DUR)
    in_rem = np.zeros(len(pk_t), dtype=bool)
    for s, e in zip(rem.start, rem.end):
        in_rem |= (pk_t >= s) & (pk_t <= e)
    return (nap.IntervalSet(start=iv.start[~in_rem], end=iv.end[~in_rem]),
            pk_t[~in_rem], pk_z[~in_rem])


def population_bursts(units, ep, ripple_peaks):
    start, stop = float(ep.start[0]), float(ep.end[0])
    spikes = np.concatenate([units[i].t for i in units.index])
    spikes = np.sort(spikes[(spikes >= start) & (spikes <= stop)])
    edges = np.arange(start, stop + MUA_BIN, MUA_BIN)
    mua = gaussian_filter1d(np.histogram(spikes, bins=edges)[0].astype(np.float32),
                            MUA_SIGMA / MUA_BIN)
    z = (mua - mua.mean()) / mua.std()
    t = edges[:-1] + MUA_BIN / 2
    above = (z > 0).astype(np.int8)
    s_i = np.where(np.diff(np.r_[0, above]) == 1)[0]
    e_i = np.where(np.diff(np.r_[above, 0]) == -1)[0]
    peak = np.array([z[s:e + 1].max() for s, e in zip(s_i, e_i)])
    ok = peak >= MUA_HIGH
    s, e, peak = t[s_i[ok]], t[e_i[ok]], peak[ok]
    dur = e - s
    ok2 = (dur >= EVT_MIN) & (dur <= EVT_MAX)
    s, e, peak = s[ok2], e[ok2], peak[ok2]
    keep = np.array([np.any((ripple_peaks >= a) & (ripple_peaks <= b))
                     for a, b in zip(s, e)], dtype=bool)
    return nap.IntervalSet(start=s[keep], end=e[keep]), peak[keep]


def weighted_fit(post, pos, tt):
    r = R.weighted_correlation(post, pos, tt)
    sw = post.sum()
    mt = (post * tt[:, None]).sum() / sw
    mx = (post * pos[None, :]).sum() / sw
    cov = (post * (tt[:, None] - mt) * (pos[None, :] - mx)).sum() / sw
    vt = (post * (tt[:, None] - mt) ** 2).sum() / sw
    return r, (cov / vt if vt > 0 else np.nan)


def shuffle_pvals(post, pos, tt, r_obs, rng):
    n_t, n_x = post.shape
    cols = np.arange(n_x)
    shift = rng.integers(0, n_x, size=(N_SHUFFLE, n_t))
    idx = (cols[None, None, :] + shift[:, :, None]) % n_x
    r_cyc = R.weighted_correlation_stack(
        np.take_along_axis(np.broadcast_to(post, (N_SHUFFLE, n_t, n_x)), idx, axis=2),
        pos, tt)
    order = np.argsort(rng.random((N_SHUFFLE, n_t)), axis=1)
    r_perm = R.weighted_correlation_stack(post[order], pos, tt)
    a = np.abs(r_obs)
    return np.mean(np.abs(r_cyc) >= a), np.mean(np.abs(r_perm) >= a)


def score_events(units, events, burst_peak, templates, centers, rng, desc=""):
    counts = units.count(DEC_BIN, events)
    which = np.searchsorted(events.end, counts.t)
    out = []
    for ei in tqdm(range(len(events)), desc=desc, leave=False):
        sel = which == ei
        c = counts.values[sel]
        if c.shape[0] < MIN_BINS or (c.sum(axis=0) > 0).sum() < MIN_ACTIVE:
            continue
        tt = counts.t[sel] - counts.t[sel][0]
        rec = dict(event=ei, t_start=float(events.start[ei]),
                   t_end=float(events.end[ei]), n_bins=c.shape[0],
                   n_active=int((c.sum(axis=0) > 0).sum()),
                   n_spikes=int(c.sum()), burst_z=float(burst_peak[ei]))
        for d in ("right", "left"):
            post = R.bayesian_decode(c, templates[d], DEC_BIN)
            r, slope = weighted_fit(post, centers, tt)
            p_cyc, p_perm = shuffle_pvals(post, centers, tt, r, rng)
            rec.update({f"r_{d}": r, f"slope_{d}": slope,
                        f"pcyc_{d}": p_cyc, f"pperm_{d}": p_perm})
        out.append(rec)
    df = pd.DataFrame(out)
    if len(df) == 0:
        return df
    best = np.where(np.abs(df.r_right) >= np.abs(df.r_left), "right", "left")
    df["direction"] = best
    for col in ("r", "slope", "pcyc", "pperm"):
        df[col] = [df[f"{col}_{d}"].iloc[i] for i, d in enumerate(best)]
    df["significant"] = (df.pcyc < P_THRESH) & (df.pperm < P_THRESH)
    return df


def run_session(session, seed=0, with_control=True):
    """Full pipeline for one session.  Returns (summary dict, dict of DataFrames)."""
    rng = np.random.default_rng(seed)
    h5, nwbfile, nwb = R.open_session(session)
    epochs = {row.label: nap.IntervalSet(start=row.start_time, end=row.stop_time)
              for row in nwbfile.epochs.to_dataframe().itertuples()}
    states = nwbfile.processing["behavior"]["states"].to_dataframe()
    rem = nap.IntervalSet(start=states.loc[states.label == "REM", "start_time"].values,
                          end=states.loc[states.label == "REM", "stop_time"].values)
    rate, _, n_t, n_ch = R.lfp_meta(h5)

    pf = build_templates(h5, nwb, nwbfile)
    units = nwb["units"][list(pf["place_ids"])]

    # Pick the ripple channel on a 2 min slice of POST sleep.
    p0 = float(epochs["POSTEpoch"].start[0]) + 300.0
    scores = []
    for ch in tqdm(range(n_ch), desc="  channels", leave=False):
        x = R.read_lfp(h5, [ch], p0, p0 + 120.0).values[:, 0]
        if np.allclose(x, 0):
            scores.append(np.nan)
            continue
        _, env = R.ripple_envelope(x, rate)
        scores.append(R.ripple_density(env))
    best_ch = int(np.nanargmax(scores))

    frames, summary = {}, dict(
        session=session, n_units=len(nwbfile.units), n_pyr=pf["n_pyr"],
        n_place=len(pf["place_ids"]), n_pos_bins=len(pf["centers"]),
        track_len_m=float(pf["edges"][-1] - pf["edges"][0]),
        run_time_s=float(pf["run_all"].tot_length()), best_ch=best_ch)

    for name in ("POSTEpoch", "PREEpoch"):
        rip_iv, pk_t, _ = detect_epoch_ripples(h5, epochs[name], best_ch, rate, rem)
        summary[f"n_ripples_{name[:4]}"] = len(rip_iv)
        summary[f"ripple_rate_{name[:4]}"] = len(rip_iv) / epochs[name].tot_length()
        events, peak = population_bursts(units, epochs[name], pk_t)
        df = score_events(units, events, peak, pf["templates"], pf["centers"], rng,
                          desc=f"  decode {name}")
        frames[name] = df
        summary[f"n_events_{name[:4]}"] = len(df)
        summary[f"frac_sig_{name[:4]}"] = float(df.significant.mean())
        summary[f"median_r_{name[:4]}"] = float(df.r.abs().median())
        if name == "POSTEpoch" and with_control:
            perm = rng.permutation(len(pf["place_ids"]))
            shuf = {d: pf["templates"][d][perm] for d in pf["templates"]}
            dfc = score_events(units, events, peak, shuf, pf["centers"], rng,
                               desc="  decode control")
            frames["control"] = dfc
            summary["frac_sig_ctrl"] = float(dfc.significant.mean())
            summary["median_r_ctrl"] = float(dfc.r.abs().median())

    sig = frames["POSTEpoch"][frames["POSTEpoch"].significant]
    summary["median_speed_mps"] = float(sig.slope.abs().median())
    summary["frac_forward"] = float((sig.slope > 0).mean())
    return summary, frames
