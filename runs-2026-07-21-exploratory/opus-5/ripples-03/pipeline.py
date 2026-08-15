"""Session-level pipeline: ripple detection, place fields, replay scoring.

This is the same analysis prototyped in scripts 01-06, packaged so it can be run
over several sessions of DANDI:000044.
"""

import numpy as np
import pandas as pd
import pynapple as nap
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert, welch
from tqdm import tqdm

import swr_utils as su

# ripple detection
LOW, HIGH = 130.0, 250.0
PEAK_Z, EDGE_Z = 4.0, 2.0
MIN_DUR, MAX_DUR, MERGE_GAP = 0.020, 0.200, 0.020
SPEED_THRESH = 4.0

# place fields
BIN_CM = 3.2
SPEED_MIN = 5.0
SMOOTH_BINS = 1.5
MIN_PEAK_RATE = 1.0
MIN_SPATIAL_INFO = 0.3

# replay
DEC_BIN = 0.020
MUA_BIN, MUA_SIGMA, MUA_PEAK_Z = 0.010, 0.015, 3.0
PBE_MIN, PBE_MAX = 0.100, 0.500
MIN_ACTIVE = 5
N_SHUFFLE = 400
ALPHA = 0.05


# --------------------------------------------------------------------- signals
def bandpass(x, lo, hi, fs, order=4):
    """Zero-phase band-pass along the time axis (axis 0 for multi-channel input)."""
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def envelope(x, fs, chunk=2 ** 22):
    """Hilbert envelope in overlapping chunks, to bound memory on 9 h of LFP."""
    pad = int(2 * fs)
    out = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), chunk):
        a, b = max(0, i - pad), min(len(x), i + chunk + pad)
        e = np.abs(hilbert(x[a:b].astype(np.float64)))
        out[i:min(i + chunk, len(x))] = e[i - a: i - a + min(chunk, len(x) - i)]
    return out


def pick_ripple_channel(h5, epochs, states, probe_dur=200.0):
    """Channel with the largest ripple-band envelope SD during post-task non-REM."""
    _, fs, _ = su.lfp_meta(h5)
    nrem = states["Non-REM"].intersect(epochs["POST"])
    i = int(np.argmax(nrem.end - nrem.start))
    t_start = float(nrem.start[i]) + 5
    probe = su.load_lfp_window(h5, t_start, t_start + probe_dur)
    env = np.abs(hilbert(bandpass(probe.values, LOW, HIGH, fs), axis=0))
    sd = env.std(axis=0)
    group = su._decode(h5["general/extracellular_ephys/electrodes/group_name"][:])
    best = int(np.argmax(sd))
    other = [c for c in range(len(sd)) if group[c] != group[best]]
    alt = int(other[int(np.argmax(sd[other]))])
    return best, alt, sd, group


def detect_ripples(env_z):
    """Threshold-crossing detection: edges at 2 SD, kept if the peak exceeds 4 SD."""
    cand = env_z.threshold(EDGE_Z, "above").time_support
    starts, ends = list(cand.start), list(cand.end)
    ms, me = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - me[-1] < MERGE_GAP:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
    cand = nap.IntervalSet(start=np.array(ms), end=np.array(me))
    dur = cand.end - cand.start
    cand = cand[(dur >= MIN_DUR) & (dur <= MAX_DUR)]

    keep, peak_t, peak_v = [], [], []
    for i in range(len(cand)):
        seg = env_z.get(cand.start[i], cand.end[i])
        if len(seg) == 0:
            continue
        j = int(np.argmax(seg.values))
        if seg.values[j] >= PEAK_Z:
            keep.append(i)
            peak_t.append(float(seg.index[j]))
            peak_v.append(float(seg.values[j]))
    return cand[np.array(keep)], np.array(peak_t), np.array(peak_v)


def ripples_from_channel(h5, channel, pos):
    """Full detection chain for one channel, excluding locomotion."""
    _, fs, _ = su.lfp_meta(h5)
    lfp = su.load_lfp_channel(h5, int(channel))
    filt = bandpass(lfp.values, LOW, HIGH, fs)
    env = envelope(filt, fs)
    env_z = nap.Tsd(t=lfp.index, d=(env - env.mean()) / env.std())
    ripples, peak_t, peak_z = detect_ripples(env_z)

    speed = su.compute_speed(pos)
    run_ep = speed.threshold(SPEED_THRESH, "above").time_support.drop_short_intervals(0.5)
    in_run = np.zeros(len(peak_t), dtype=bool)
    for s, e in zip(run_ep.start, run_ep.end):
        in_run |= (peak_t >= s) & (peak_t <= e)
    return (ripples[~in_run], peak_t[~in_run], peak_z[~in_run],
            np.asarray(lfp.values), filt, fs, float(lfp.index[0]))


def ripple_stats(filt, fs, t0, peak_t, half=0.25):
    """Snippet matrix, duration-independent intra-ripple frequency."""
    n = int(half * fs)
    idx = np.round((peak_t - t0) * fs).astype(int)
    idx = idx[(idx > n) & (idx < len(filt) - n - 1)]
    snips = filt[idx[:, None] + np.arange(-n, n + 1)]
    core = snips[:, n - int(0.04 * fs): n + int(0.04 * fs)]
    fx, px = welch(core, fs=fs, nperseg=core.shape[1], axis=1)
    band = (fx >= 100) & (fx <= 300)
    return fx[band][np.argmax(px[:, band], axis=1)]


# ----------------------------------------------------------------- place fields
def spatial_information(tc, occupancy):
    """Skaggs information rate in bits per spike."""
    p = occupancy / occupancy.sum()
    r = np.asarray(tc)
    rbar = (p[:, None] * r).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p[:, None] * (r / rbar) * np.log2(r / rbar)
    return np.nansum(terms, axis=0)


def running_epochs(pos):
    """Rightward and leftward traversals of the track."""
    speed = su.compute_speed(pos)
    t, d = np.asarray(pos.index), np.asarray(pos.values, float)
    good = np.isfinite(d)
    grid = np.asarray(speed.index)
    dt = float(np.median(np.diff(grid)))
    vel = np.gradient(gaussian_filter1d(np.interp(grid, t[good], d[good]), 0.25 / dt), dt)
    run = speed.threshold(SPEED_MIN, "above").time_support.drop_short_intervals(0.5)
    v = nap.Tsd(t=grid, d=vel)
    right = v.restrict(run).threshold(0, "above").time_support.drop_short_intervals(0.3)
    left = v.restrict(run).threshold(0, "below").time_support.drop_short_intervals(0.3)
    return run, right, left


def place_fields(pyr, pos, right, left, track_len):
    nbins = int(round(track_len / BIN_CM))
    tcs, occs = {}, {}
    for name, ep in (("right", right), ("left", left)):
        tc = nap.compute_1d_tuning_curves(pyr, pos, nb_bins=nbins, ep=ep, minmax=(0, track_len))
        occ, edges = np.histogram(pos.restrict(ep).values, bins=nbins, range=(0, track_len))
        tcs[name] = np.clip(gaussian_filter1d(np.asarray(tc, float), SMOOTH_BINS, axis=0,
                                              mode="nearest"), 1e-3, None)
        occs[name] = occ.astype(float)
    centers = 0.5 * (edges[1:] + edges[:-1])
    si = {k: spatial_information(tcs[k], occs[k]) for k in tcs}
    is_place = np.zeros(tcs["right"].shape[1], dtype=bool)
    for k in tcs:
        is_place |= (tcs[k].max(0) >= MIN_PEAK_RATE) & (si[k] >= MIN_SPATIAL_INFO)
    return tcs, centers, si, is_place


# ----------------------------------------------------------------------- replay
def population_bursts(spk, ep, ripple_peaks):
    """Population bursts (MUA peak > 3 z, 100-500 ms) containing a ripple peak."""
    counts = spk.count(MUA_BIN, ep).sum(axis=1)
    sm = gaussian_filter1d(np.asarray(counts, float), MUA_SIGMA / MUA_BIN)
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
    return cand[np.array(keep)] if keep else cand[np.array([], dtype=int)]


def weighted_corr(post, x, t):
    """Weighted correlation and slope of decoded position against time."""
    w = post / post.sum()
    mx = (w * x[:, None]).sum()
    mt = (w * t[None, :]).sum()
    dx, dt = x[:, None] - mx, t[None, :] - mt
    cov = (w * dx * dt).sum()
    return cov / np.sqrt((w * dx ** 2).sum() * (w * dt ** 2).sum()), cov / (w * dt ** 2).sum()


def shuffle_scores(post, x, t, rng, n=N_SHUFFLE):
    """Null |wcorr| from a circular position shuffle and from a time permutation."""
    nx, nt = post.shape
    circ, perm = np.empty(n), np.empty(n)
    ar = np.arange(nx)[:, None]
    for i in range(n):
        rows = (ar - rng.integers(0, nx, size=nt)[None, :]) % nx
        circ[i] = abs(weighted_corr(np.take_along_axis(post, rows, axis=0), x, t)[0])
        perm[i] = abs(weighted_corr(post[:, rng.permutation(nt)], x, t)[0])
    return circ, perm


def make_template(values, ids, centers):
    return xr.DataArray(values.T, dims=["unit", "position"],
                        coords={"unit": ids, "position": centers})


def score_event(spk, templates, centers, iset, rng):
    """Max |wcorr| over the two templates, with a matched max-over-templates null."""
    obs, nulls, keep = {}, {}, {}
    for k, tc in templates.items():
        _, proba = nap.decode_bayes(tuning_curves=tc, data=spk, epochs=iset, bin_size=DEC_BIN)
        post, tt = np.asarray(proba).T, np.asarray(proba.index)
        if post.shape[1] < 5 or not np.isfinite(post).all():
            return None
        t = tt - tt[0]
        obs[k] = weighted_corr(post, centers, t)
        nulls[k] = shuffle_scores(post, centers, t, rng)
        keep[k] = (post, tt)
    best = max(obs, key=lambda k: abs(obs[k][0]))
    r, slope = obs[best]
    circ_max = np.maximum(*[nulls[k][0] for k in templates])
    perm_max = np.maximum(*[nulls[k][1] for k in templates])
    p_circ = (np.sum(circ_max >= abs(r)) + 1) / (N_SHUFFLE + 1)
    p_perm = (np.sum(perm_max >= abs(r)) + 1) / (N_SHUFFLE + 1)
    return dict(template=best, wcorr=r, slope=slope, p_circ=p_circ, p_perm=p_perm,
                max_p=max(p_circ, p_perm)), keep[best]


def score_epoch(spk, templates, centers, pbe, ep_name, rng, progress=True):
    rows, posts = [], {}
    it = range(len(pbe))
    if progress:
        it = tqdm(it, desc="decoding " + ep_name)
    for i in it:
        iset = nap.IntervalSet(pbe.start[i], pbe.end[i])
        n_active = int((np.asarray(spk.restrict(iset).count(PBE_MAX)) > 0).sum())
        if n_active < MIN_ACTIVE:
            continue
        out = score_event(spk, templates, centers, iset, rng)
        if out is None:
            continue
        rec, post = out
        rec.update(epoch=ep_name, event=i, t0=float(pbe.start[i]),
                   dur=float(pbe.end[i] - pbe.start[i]), n_active=n_active)
        rows.append(rec)
        posts[(ep_name, i)] = post
    return rows, posts


def explained_variance(spk, ripple_ep, run_ep, epochs, bin_size=0.100):
    """Kudrimoti-style EV / REV on pairwise co-firing, independent of decoding."""
    def vec(ep):
        c = np.asarray(spk.count(bin_size, ep))
        r = np.corrcoef(c.T)
        return r[np.triu_indices(r.shape[0], 1)]

    v_run, v_pre, v_post = (vec(run_ep), vec(ripple_ep.intersect(epochs["PRE"])),
                            vec(ripple_ep.intersect(epochs["POST"])))
    ok = np.isfinite(v_run) & np.isfinite(v_pre) & np.isfinite(v_post)
    v_run, v_pre, v_post = v_run[ok], v_pre[ok], v_post[ok]

    def partial(a, b, c):
        rab, rac, rbc = (np.corrcoef(a, b)[0, 1], np.corrcoef(a, c)[0, 1], np.corrcoef(b, c)[0, 1])
        return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))

    return partial(v_run, v_post, v_pre) ** 2, partial(v_run, v_pre, v_post) ** 2, len(v_run)


# ------------------------------------------------------------------ full session
def run_session(session, seed=1, progress=True):
    rng = np.random.default_rng(seed)
    h5 = su.open_session(session)
    epochs = su.load_epochs(h5)
    states = su.load_states(h5)
    units = su.load_units(h5)
    pos = su.load_position(h5)
    pyr = units.getby_category("cell_type")["excitatory"]
    track_len = float(np.ceil(np.nanmax(pos.values)))

    best_ch, alt_ch, sd, group = pick_ripple_channel(h5, epochs, states)
    ripples, peak_t, peak_z, raw, filt, fs, t0 = ripples_from_channel(h5, best_ch, pos)
    freq = ripple_stats(filt, fs, t0, peak_t)

    rates = {}
    for name, ep in list(epochs.items()) + list(states.items()):
        n = int(((peak_t[:, None] >= ep.start) & (peak_t[:, None] <= ep.end)).any(1).sum())
        rates[name] = n / ep.tot_length()

    pos_run = pos[np.isfinite(pos.values)].restrict(epochs["MAZE"])
    run, right, left = running_epochs(pos_run)
    tcs, centers, si, is_place = place_fields(pyr, pos_run, right, left, track_len)
    ids = np.array(list(pyr.keys()))[is_place]
    spk = pyr[list(ids)]
    templates = {k: make_template(tcs[k][:, is_place], ids, centers) for k in tcs}

    # decoder check against the animal's real position
    errs = []
    for k, ep_dir in (("right", right), ("left", left)):
        dec, _ = nap.decode_bayes(tuning_curves=templates[k], data=spk, epochs=ep_dir, bin_size=0.25)
        errs.append(np.abs(np.asarray(dec) - np.asarray(pos_run.interpolate(dec))))
    err = np.concatenate(errs)

    rows, posts = [], {}
    for ep_name in ("PRE", "POST"):
        pbe = population_bursts(spk, epochs[ep_name], peak_t)
        r, p = score_epoch(spk, templates, centers, pbe, ep_name, rng, progress)
        rows += r
        posts.update(p)
    df = pd.DataFrame(rows)
    df["significant"] = df["max_p"] < ALPHA

    # cell-identity shuffle control
    perm_ids = rng.permutation(len(ids))
    shuf = {k: make_template(tcs[k][:, is_place][:, perm_ids], ids, centers) for k in tcs}
    ctrl_idx = rng.choice(len(df), size=min(400, len(df)), replace=False)
    ctrl = []
    it = tqdm(ctrl_idx, desc="cell-ID shuffle") if progress else ctrl_idx
    for j in it:
        row = df.iloc[j]
        out = score_event(spk, shuf, centers, nap.IntervalSet(row.t0, row.t0 + row.dur), rng)
        if out is not None:
            ctrl.append(out[0])
    ctrl = pd.DataFrame(ctrl)
    ctrl["significant"] = ctrl["max_p"] < ALPHA

    ev, rev, npairs = explained_variance(spk, ripples, run, epochs)
    speed = su.compute_speed(pos_run)
    run_speed = float(np.median(speed.restrict(run).values))
    sig = df[df.significant]

    summary = dict(
        session=session, track_len=track_len, n_pyr=len(pyr), n_place=int(is_place.sum()),
        n_ripples=len(peak_t), ripple_rate=len(peak_t) / (epochs["POST"].end[-1] - epochs["PRE"].start[0]),
        rate_nrem=rates["Non-REM"], rate_rem=rates["REM"], rate_awake=rates["Awake"],
        rate_pre=rates["PRE"], rate_post=rates["POST"], rate_maze=rates["MAZE"],
        ripple_dur_ms=float(np.median((ripples.end - ripples.start) * 1000)),
        ripple_freq_hz=float(np.median(freq)),
        decoder_err_cm=float(np.median(err)), run_speed=run_speed,
        n_pre=int((df.epoch == "PRE").sum()), n_post=int((df.epoch == "POST").sum()),
        frac_pre=float(df[df.epoch == "PRE"].significant.mean()),
        frac_post=float(df[df.epoch == "POST"].significant.mean()),
        frac_ctrl=float(ctrl.significant.mean()),
        replay_speed=float(sig.slope.abs().median()) if len(sig) else np.nan,
        ev=ev, rev=rev, n_pairs=npairs,
    )
    return summary, df, ctrl, dict(centers=centers, tcs=tcs, is_place=is_place, ids=ids,
                                   posteriors=posts, ripples=ripples, peak_t=peak_t,
                                   peak_z=peak_z, err=err, rates=rates, freq=freq)
