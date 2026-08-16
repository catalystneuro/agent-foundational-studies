"""End-to-end SWR + replay pipeline for one session of DANDI:000044."""

import os

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

from dandi_io import open_session
from swr_lib import (FS, bayesian_decode, detect_ripples, linearize_position,
                     load_lfp_channel, pick_ripple_channel, ripple_envelope,
                     ripple_peak_frequency, running_speed)

BIN = 0.020
MIN_EVENT = 0.100
MIN_CELLS = 5
MIN_BINS = 5
N_SHUF = 500
N_POS_BINS = 50
CORR_BIN = 0.100


# ------------------------------------------------------------------ components
def session_epochs(nwbfile):
    ep_df = nwbfile.epochs.to_dataframe()
    return {lab.replace("Epoch", ""): nap.IntervalSet(
        start=ep_df.query("label==@lab").start_time.values,
        end=ep_df.query("label==@lab").stop_time.values) for lab in ep_df.label}


def sleep_states(nwbfile):
    st = nwbfile.processing["behavior"]["states"].to_dataframe()
    return {lab: nap.IntervalSet(start=st.query("label==@lab").start_time.values,
                                 end=st.query("label==@lab").stop_time.values)
            for lab in st.label.unique()}


LFP_CACHE = os.environ.get("LFP_CACHE", "/tmp/swr_lfp_cache")


def get_lfp(nwbfile, session, channel):
    """One LFP channel for the whole session, cached outside the output directory."""
    os.makedirs(LFP_CACHE, exist_ok=True)
    cache = os.path.join(LFP_CACHE, f"lfp_{session}_ch{channel}.npy")
    if os.path.exists(cache):
        d = np.load(cache)
        return nap.Tsd(t=np.arange(d.size) / FS, d=d)
    lfp = load_lfp_channel(nwbfile, channel)
    np.save(cache, lfp.values)
    return lfp


def spatial_information(rate, occupancy):
    p = occupancy / occupancy.sum()
    mean_rate = np.sum(p * rate)
    if mean_rate <= 0:
        return 0.0
    nz = rate > 0
    return float(np.sum(p[nz] * rate[nz] / mean_rate * np.log2(rate[nz] / mean_rate)))


def place_fields(pyr, pos, track, run_ep):
    """Direction-split tuning curves plus a place-cell selection table."""
    disp = np.array([pos.restrict(run_ep[i]).values[-1] - pos.restrict(run_ep[i]).values[0]
                     for i in range(len(run_ep))])
    keep = np.abs(disp) > 0.5 * track
    dir_ep = {"rightward": run_ep[np.where(keep & (disp > 0))[0]],
              "leftward": run_ep[np.where(keep & (disp < 0))[0]]}
    bins = np.linspace(0, track, N_POS_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    dt_pos = float(np.median(np.diff(pos.times())))

    tc, occ, half = {}, {}, {}
    for d, ep in dir_ep.items():
        tc[d] = nap.compute_1d_tuning_curves(pyr, pos, N_POS_BINS,
                                             minmax=(0, track), ep=ep)
        occ[d] = np.histogram(pos.restrict(ep).values, bins=bins)[0] * dt_pos
        n = len(ep)
        a = nap.compute_1d_tuning_curves(pyr, pos, N_POS_BINS, minmax=(0, track),
                                         ep=ep[: n // 2])
        b = nap.compute_1d_tuning_curves(pyr, pos, N_POS_BINS, minmax=(0, track),
                                         ep=ep[n // 2:])
        half[d] = {u: np.corrcoef(np.nan_to_num(a[u].values),
                                  np.nan_to_num(b[u].values))[0, 1] for u in pyr.index}

    rows = []
    for d in dir_ep:
        for u in pyr.index:
            r = np.nan_to_num(tc[d][u].values)
            rows.append(dict(unit=u, direction=d, peak=r.max(),
                             si=spatial_information(r, occ[d]),
                             com=centers[np.argmax(r)], stability=half[d][u]))
    stats = pd.DataFrame(rows)
    stats["place_cell"] = ((stats.peak >= 1.0) & (stats.si >= 0.4)
                           & (stats.stability >= 0.3))
    return tc, occ, centers, stats, dir_ep, disp, keep


def weighted_corr_batch(P, x, t):
    P = P / P.sum(axis=(1, 2), keepdims=True)
    tt, xx = t[None, :, None], x[None, None, :]
    mt = (P * tt).sum(axis=(1, 2), keepdims=True)
    mx = (P * xx).sum(axis=(1, 2), keepdims=True)
    cov = (P * (tt - mt) * (xx - mx)).sum(axis=(1, 2))
    vt = (P * (tt - mt) ** 2).sum(axis=(1, 2))
    vx = (P * (xx - mx) ** 2).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vt * vx)


def column_cycle_shuffle(post, n_shuf, rng):
    T, X = post.shape
    shifts = rng.integers(0, X, size=(n_shuf, T))
    idx = (np.arange(X)[None, None, :] + shifts[:, :, None]) % X
    return np.take_along_axis(post[None, :, :].repeat(n_shuf, 0), idx, axis=2)


def id_shuffle_corr(C, tmpl, centers, tcent, n_shuf, rng):
    L = np.log(np.clip(tmpl, 1e-3, None)).T
    offset = BIN * np.clip(tmpl, 1e-3, None).sum(axis=1)[None, None, :]
    perms = np.stack([rng.permutation(L.shape[0]) for _ in range(n_shuf)])
    ll = np.einsum("tn,snx->stx", C, L[perms]) - offset
    ll -= ll.max(axis=2, keepdims=True)
    post = np.exp(ll)
    post /= post.sum(axis=2, keepdims=True)
    return weighted_corr_batch(post, centers, tcent)


def decode_events(cand, templates, centers, spike_t, spike_c, n_cells, rng,
                  progress=True):
    def event_counts(t0, t1):
        n = int(np.floor((t1 - t0) / BIN))
        if n < MIN_BINS:
            return None
        i0, i1 = np.searchsorted(spike_t, [t0, t0 + n * BIN])
        if i1 == i0:
            return None
        tb = np.minimum(((spike_t[i0:i1] - t0) / BIN).astype(int), n - 1)
        C = np.zeros((n, n_cells))
        np.add.at(C, (tb, spike_c[i0:i1]), 1)
        return C

    rows = []
    it = range(len(cand))
    for k in (tqdm(it, desc="decoding events") if progress else it):
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
            cyc = weighted_corr_batch(column_cycle_shuffle(post, N_SHUF, rng),
                                      centers, tcent)
            ids = id_shuffle_corr(C, tmpl, centers, tcent, N_SHUF, rng)
            row[f"r_{d}"] = r
            row[f"p_cycle_{d}"] = float((np.abs(cyc) >= abs(r)).mean())
            row[f"p_id_{d}"] = float((np.abs(ids[np.isfinite(ids)]) >= abs(r)).mean())
        rows.append(row)

    res = pd.DataFrame(rows)
    res["direction"] = np.where(res.r_rightward.abs() >= res.r_leftward.abs(),
                                "rightward", "leftward")
    for col in ["r", "p_cycle", "p_id"]:
        res[col] = np.where(res.direction == "rightward",
                            res[f"{col}_rightward"], res[f"{col}_leftward"])
    res["significant"] = (res.p_cycle < 0.05) & (res.p_id < 0.05)
    res["forward"] = np.where(res.direction == "rightward", res.r > 0, res.r < 0)
    return res


def partial_corr(x, y, z):
    rxy, rxz, ryz = (np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1],
                     np.corrcoef(y, z)[0, 1])
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def explained_variance(group, epochs, states, run_ep):
    """Kudrimoti-style EV / reverse-EV of run-time pairwise correlations."""
    nrem = states["Non-REM"]
    win = {"pre": nrem.intersect(epochs["PRE"]), "run": run_ep,
           "post": nrem.intersect(epochs["POST"])}
    iu = np.triu_indices(len(group), 1)
    v = {}
    for k, ep in win.items():
        C = group.count(CORR_BIN, ep=ep).values.astype(float)
        C = (C - C.mean(0)) / np.where(C.std(0) > 0, C.std(0), 1)
        v[k] = ((C.T @ C) / C.shape[0])[iu]
    return (partial_corr(v["run"], v["post"], v["pre"]) ** 2,
            partial_corr(v["run"], v["pre"], v["post"]) ** 2, v)


# ------------------------------------------------------------------- full run
def analyze_session(session, progress=True):
    nwbfile, nwb, h5 = open_session(session)
    epochs = session_epochs(nwbfile)
    states = sleep_states(nwbfile)

    # ripple channel from the longest post-task non-REM bout
    nrem_post = states["Non-REM"].intersect(epochs["POST"])
    t0 = float(nrem_post.start[np.argmax(nrem_post.end - nrem_post.start)])
    channel, sd = pick_ripple_channel(nwbfile, t0)

    lfp = get_lfp(nwbfile, session, channel)
    filt, env = ripple_envelope(lfp)

    pos, track = linearize_position(nwb)
    speed = running_speed(pos)
    run_ep = (speed.threshold(0.05).time_support
              .merge_close_intervals(0.5).drop_short_intervals(0.5))
    session_ep = nap.IntervalSet(start=0.0, end=float(lfp.times()[-1]))
    immobile = session_ep.set_diff(run_ep)

    ripples, peaks = detect_ripples(env, session_ep, ref_ep=immobile)
    pfreq = ripple_peak_frequency(filt, peaks)

    units = nwb["units"]
    pyr = units[units.cell_type == "excitatory"]
    tc, occ, centers, stats, dir_ep, disp, keep = place_fields(pyr, pos, track, run_ep)
    place_units = sorted(stats.query("place_cell").unit.unique())

    templates = {d: np.stack([np.nan_to_num(tc[d][u].values) for u in place_units], 1)
                 for d in tc}
    cells = np.array(place_units)
    spike_t = np.concatenate([pyr[u].times() for u in cells])
    spike_c = np.concatenate([np.full(pyr[u].shape[0], i) for i, u in enumerate(cells)])
    o = np.argsort(spike_t)
    spike_t, spike_c = spike_t[o], spike_c[o]

    ok = np.ones(len(ripples), bool)
    for s, e in zip(run_ep.start, run_ep.end):
        ok &= ~((ripples.end > s) & (ripples.start < e))
    pad = np.maximum(MIN_EVENT - (ripples.end[ok] - ripples.start[ok]), 0) / 2
    cand = nap.IntervalSet(start=ripples.start[ok] - pad, end=ripples.end[ok] + pad)

    rng = np.random.default_rng(1)
    res = decode_events(cand, templates, centers, spike_t, spike_c, len(cells), rng,
                        progress)
    for name, ep in epochs.items():
        m = np.zeros(len(res), bool)
        for s, e in zip(ep.start, ep.end):
            m |= (res.start >= s) & (res.start < e)
        res.loc[m, "epoch"] = name

    ev, rev, _ = explained_variance(pyr[list(cells)], epochs, states, run_ep)

    rate = {}
    for lab in ["Non-REM", "REM", "Awake"]:
        ep = states[lab].intersect(immobile)
        rate[lab] = len(nap.Ts(peaks.times()).restrict(ep)) / ep.tot_length() * 60
    rate["running"] = len(nap.Ts(peaks.times()).restrict(run_ep)) / \
        max(run_ep.tot_length(), 1e-9) * 60

    summary = dict(
        session=session, track=track, channel=channel,
        n_units=len(units), n_pyr=len(pyr), n_place=len(place_units),
        n_laps=int(keep.sum()), n_ripples=len(ripples),
        ripple_rate=len(ripples) / session_ep.tot_length() * 60,
        ripple_dur_ms=float(np.median(ripples.end - ripples.start) * 1000),
        ripple_freq=float(np.nanmedian(pfreq)),
        rate_nrem=rate["Non-REM"], rate_rem=rate["REM"],
        rate_awake=rate["Awake"], rate_run=rate["running"],
        n_events=len(res), ev=ev, rev=rev)
    for e in ["PRE", "Maze", "POST"]:
        sub = res[res.epoch == e]
        summary[f"frac_sig_{e}"] = float(sub.significant.mean()) if len(sub) else np.nan
        summary[f"n_{e}"] = int(len(sub))
    summary["frac_forward"] = float(res[res.significant].forward.mean())

    result = dict(summary=summary, res=res, stats=stats, tc=tc, centers=centers,
                place_units=place_units, ripples=ripples, peaks=peaks,
                pfreq=pfreq, lfp=lfp, filt=filt, env=env, pos=pos, speed=speed,
                  epochs=epochs, states=states, run_ep=run_ep, track=track,
                  channel_sd=sd, nwbfile=nwbfile, nwb=nwb, pyr=pyr)
    return result


LINEAR_SESSIONS = ["Achilles_10252013", "Buddy_06272013", "Cicero_09012014",
                   "Cicero_09172014", "Gatsby_08022013"]


def run_all_sessions(sessions=LINEAR_SESSIONS, cache_dir="."):
    """Analyse every session, caching per-session event tables and summaries."""
    summaries, events = [], []
    for s in sessions:
        ev_path = os.path.join(cache_dir, f"cache_events_{s}.csv")
        sm_path = os.path.join(cache_dir, f"cache_summary_{s}.csv")
        if os.path.exists(ev_path) and os.path.exists(sm_path):
            res = pd.read_csv(ev_path)
            summ = pd.read_csv(sm_path).iloc[0].to_dict()
        else:
            print(f"=== {s} ===")
            out = analyze_session(s)
            res, summ = out["res"], out["summary"]
            res.to_csv(ev_path, index=False)
            pd.DataFrame([summ]).to_csv(sm_path, index=False)
        res["session"] = s
        events.append(res)
        summaries.append(summ)
    return pd.DataFrame(summaries), pd.concat(events, ignore_index=True)
