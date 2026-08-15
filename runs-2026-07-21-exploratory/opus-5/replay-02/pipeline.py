"""Reusable hippocampal-replay pipeline for DANDI:000044 (Grosmark & Buzsaki 2016)."""
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d
import pynapple as nap
import common

# ----------------------------------------------------------------- parameters
NBINS = 50                 # position bins over the 1.6 m track
TRACK = 1.6
RIP_BAND = (140, 230)
RIP_PEAK_Z, RIP_EDGE_Z = 4.0, 1.0
RIP_MIN, RIP_MAX = 0.020, 0.250
MUA_PEAK_Z, MUA_MIN, MUA_MAX = 3.0, 0.050, 0.600
BIN = 0.020                # decoding bin
MIN_BINS, MIN_CELLS = 5, 5
N_SHUF = 500


def bandpass(x, lo, hi, fs, order=4):
    return sosfiltfilt(butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos"), x)


# ----------------------------------------------------------------- loading
def load_session(session):
    nwbfile, h = common.open_session(session)
    nwb = nap.NWBFile(nwbfile)
    ep_df = nwbfile.epochs.to_dataframe()
    epochs = {r.label: nap.IntervalSet(start=r.start_time, end=r.stop_time) for r in ep_df.itertuples()}
    st = nwbfile.processing["behavior"]["states"].to_dataframe()
    states = {lab: nap.IntervalSet(start=g.start_time.values, end=g.stop_time.values)
              for lab, g in st.groupby("label")}
    udf = nwbfile.units.to_dataframe()
    lfp_grp = nwbfile.processing["ecephys"]["LFP"]["LFP"]
    d = dict(nwb=nwb, nwbfile=nwbfile, h5=h, epochs=epochs, states=states, udf=udf,
             units=nwb["units"], fs=lfp_grp.rate, conv=lfp_grp.conversion,
             lfp_ds=h["processing/ecephys/LFP/LFP/data"], session=session)
    d["pyr"] = udf.index[udf.cell_type == "excitatory"].values
    return d


def find_linearized(d):
    """Name of the linearised position series (the maze differs between sessions)."""
    for mod in d["nwbfile"].processing["behavior"].data_interfaces.values():
        for k in getattr(mod, "spatial_series", {}):
            if "Linearized" in k:
                return k
    raise KeyError("no linearised position series in this session")


def linear_position(d):
    """Linearised position restricted to the intervals where tracking is defined."""
    ls = d["nwb"][find_linearized(d)]
    t, x = ls.t, np.asarray(ls.d).squeeze()
    g = ~np.isnan(x)
    tg, xg = t[g], x[g]
    dt = np.median(np.diff(t))
    brk = np.where(np.diff(tg) > 5 * dt)[0]
    s_i, e_i = np.r_[0, brk + 1], np.r_[brk, len(tg) - 1]
    n = e_i - s_i + 1
    s_i, e_i = s_i[n >= 5], e_i[n >= 5]
    laps = nap.IntervalSet(start=tg[s_i] - dt / 2, end=tg[e_i] + dt / 2)
    pos = nap.Tsd(t=tg, d=xg, time_support=laps)
    track = float(np.ceil(np.nanmax(xg) * 20) / 20)      # round up to 5 cm
    return pos, laps, dt, track


def split_laps(pos, laps, track=TRACK):
    """Classify traversals by direction, keeping full-track runs."""
    direction, speed, span = [], [], []
    for s, e in zip(laps.start, laps.end):
        p = pos.get(s, e)
        direction.append(np.sign(p.d[-1] - p.d[0]))
        speed.append(abs(p.d[-1] - p.d[0]) / max(p.t[-1] - p.t[0], 1e-9))
        span.append(np.ptp(p.d))
    direction, speed, span = map(np.array, (direction, speed, span))
    ok = (span > 0.6 * track) & (speed > 0.1)
    out = {}
    for name, sgn in [("rightward", 1), ("leftward", -1)]:
        m = ok & (direction == sgn)
        out[name] = (nap.IntervalSet(start=laps.start[m], end=laps.end[m]), np.where(m)[0])
    return out, ok, direction, speed


def tuning_curves(group, pos, ep, track=TRACK, sigma=1.0):
    tc = nap.compute_1d_tuning_curves(group=group, feature=pos.restrict(ep), nb_bins=NBINS,
                                      minmax=(0, track), ep=ep)
    return np.apply_along_axis(lambda v: gaussian_filter1d(v, sigma, mode="nearest"), 0, tc.values)


def spatial_info(rate, occ):
    p = occ / occ.sum()
    mr = (p * rate).sum()
    if mr <= 0:
        return 0.0
    r = np.where(rate > 0, rate, np.nan)
    return float(np.nansum(p * r / mr * np.log2(r / mr)))


# ----------------------------------------------------------------- ripples
def pick_ripple_channel(d, probe_ep, seconds=300):
    fs, ds, conv = d["fs"], d["lfp_ds"], d["conv"]
    dur = probe_ep.end - probe_ep.start
    k = int(np.argmax(dur))
    i0 = int(probe_ep.start[k] * fs); i1 = int(min(probe_ep.end[k], probe_ep.start[k] + seconds) * fs)
    n = ds.shape[1]
    rip, hf = np.zeros(n), np.zeros(n)
    for ch in range(n):
        x = ds[i0:i1, ch].astype(np.float64) * conv * 1e6
        rip[ch] = np.std(bandpass(x, *RIP_BAND, fs))
        hf[ch] = np.std(bandpass(x, 300, 500, fs))
    return int(np.argmax(rip * rip / np.maximum(hf, 1e-9))), rip


def detect_ripples(d, channel, nrem, eps):
    fs, conv = d["fs"], d["conv"]
    raw = d["lfp_ds"][:, channel].astype(np.float64) * conv * 1e6
    t = np.arange(len(raw)) / fs
    filt = bandpass(raw, *RIP_BAND, fs)
    env = gaussian_filter1d(np.abs(hilbert(filt)), 0.0075 * fs)
    ref = nap.Tsd(t=t, d=env).restrict(nrem).d
    z = nap.Tsd(t=t, d=(env - ref.mean()) / ref.std())
    out = {}
    for name, ep in eps.items():
        zz = z.restrict(ep)
        cand = zz.threshold(RIP_EDGE_Z, "above").time_support.merge_close_intervals(0.020)
        cand = cand[(cand.end - cand.start) >= RIP_MIN]
        keep, pk, pt = [], [], []
        for s, e in zip(cand.start, cand.end):
            seg = zz.get(s, e)
            if len(seg) == 0:
                continue
            m = int(np.argmax(seg.d))
            if seg.d[m] >= RIP_PEAK_Z and (e - s) <= RIP_MAX:
                keep.append((s, e)); pk.append(seg.d[m]); pt.append(seg.t[m])
        keep = np.array(keep)
        out[name] = dict(ep=nap.IntervalSet(start=keep[:, 0], end=keep[:, 1]),
                         peak_z=np.array(pk), peak_t=np.array(pt))
    return out, nap.Tsd(t=t, d=raw), nap.Tsd(t=t, d=filt), z


def detect_pbes(d, ep, ripples):
    """Population-burst events (MUA) that coincide with a detected ripple."""
    grp = d["units"][list(d["pyr"])]
    mua = grp.count(0.001, ep=ep)
    rate = gaussian_filter1d(np.asarray(mua.values, float).sum(1), 10) * 1000.0
    z = nap.Tsd(t=mua.t, d=(rate - rate.mean()) / rate.std(), time_support=ep)
    cand = z.threshold(0.0, "above").time_support
    cand = cand[(cand.end - cand.start) >= MUA_MIN]
    pk = np.array([z.get(s, e).d.max() for s, e in zip(cand.start, cand.end)])
    sel = (pk >= MUA_PEAK_Z) & ((cand.end - cand.start) <= MUA_MAX)
    pbe = nap.IntervalSet(start=cand.start[sel], end=cand.end[sel])
    r = ripples["ep"]
    ov = np.array([np.any((r.start < e) & (r.end > s)) for s, e in zip(pbe.start, pbe.end)])
    return nap.IntervalSet(start=pbe.start[ov], end=pbe.end[ov]), z, ov.mean()


# ----------------------------------------------------------------- decoding
def _norm(ll):
    ll = ll - ll.max(-1, keepdims=True)
    p = np.exp(ll)
    return p / p.sum(-1, keepdims=True)


def _wstats(P, tvec, centers):
    sw = P.sum((-2, -1))
    mt = (P * tvec[:, None]).sum((-2, -1)) / sw
    mx = (P * centers[None, :]).sum((-2, -1)) / sw
    ct = tvec[:, None] - mt[..., None, None]
    cx = centers[None, :] - mx[..., None, None]
    cov = (P * ct * cx).sum((-2, -1)) / sw
    vt = (P * ct ** 2).sum((-2, -1)) / sw
    vx = (P * cx ** 2).sum((-2, -1)) / sw
    return cov / np.sqrt(vt * vx + 1e-12), cov / (vt + 1e-12)


def decode_events(group, templates, events, centers, n_shuf=N_SHUF, seed=0,
                  shuffle_observed=False, progress=None):
    """Bayesian decoding + replay statistics for a set of candidate events."""
    from tqdm import tqdm
    X = len(centers); U = len(group)
    rng = np.random.default_rng(seed)
    templates = {k: np.maximum(v, 1e-3) for k, v in templates.items()}
    obs_tc = templates
    if shuffle_observed:
        q = rng.permutation(U)
        obs_tc = {k: v[:, q] for k, v in templates.items()}
    perms = np.stack([rng.permutation(U) for _ in range(n_shuf)])
    shuf = {k: (np.ascontiguousarray(np.log(v)[:, perms].transpose(2, 1, 0).reshape(U, n_shuf * X)),
                v[:, perms].sum(2).T.reshape(n_shuf * X)) for k, v in templates.items()}
    cnt = group.count(BIN, ep=events)
    cd = np.asarray(cnt.values, dtype=np.float64)
    ev = np.searchsorted(events.start, cnt.t, side="right") - 1
    rows, posteriors = [], {}
    it = range(len(events))
    if progress:
        it = tqdm(it, desc=progress, mininterval=3.0)
    for i in it:
        N = cd[ev == i]; T = N.shape[0]
        if T < MIN_BINS:
            continue
        nact = int((N.sum(0) > 0).sum())
        if nact < MIN_CELLS or N.sum() < 2 * MIN_CELLS:
            continue
        tvec = np.arange(T) * BIN
        best = None; id_max = np.zeros(n_shuf); cyc_max = np.zeros(n_shuf)
        for k in templates:
            tc = obs_tc[k]
            P = _norm(N @ np.log(tc).T - BIN * tc.sum(1)[None, :])
            c, s = _wstats(P, tvec, centers)
            if best is None or abs(c) > abs(best[1]):
                best = (k, float(c), float(s), P)
            lt, st = shuf[k]
            Ps = _norm((N @ lt - BIN * st[None, :]).reshape(T, n_shuf, X).transpose(1, 0, 2))
            cs, _ = _wstats(Ps, tvec, centers)
            id_max = np.maximum(id_max, np.abs(cs))
            shift = rng.integers(0, X, size=(n_shuf, T))
            idx = (np.arange(X)[None, None, :] - shift[:, :, None]) % X
            Pc = np.take_along_axis(np.broadcast_to(P, (n_shuf, T, X)), idx, axis=2)
            cc, _ = _wstats(Pc, tvec, centers)
            cyc_max = np.maximum(cyc_max, np.abs(cc))
        k, c, s, P = best
        com = P @ centers
        rows.append(dict(event=i, start=events.start[i], end=events.end[i],
                         dur=events.end[i] - events.start[i], nbins=T, ncells=nact,
                         nspikes=int(N.sum()), template=k, wcorr=c, slope=s, speed=abs(s),
                         p_id=(1 + (id_max >= abs(c)).sum()) / (n_shuf + 1),
                         p_cyc=(1 + (cyc_max >= abs(c)).sum()) / (n_shuf + 1),
                         start_pos=float(com[0]), end_pos=float(com[-1]),
                         span=float(abs(com[-1] - com[0]))))
        posteriors[i] = P
    df = pd.DataFrame(rows)
    df["sig"] = (df.p_id < 0.05) & (df.p_cyc < 0.05)
    return df, posteriors
