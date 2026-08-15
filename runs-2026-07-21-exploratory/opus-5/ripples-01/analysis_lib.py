"""Ripple detection, place-field estimation and replay decoding.

The single-session scripts (02, 04, 05) and the multi-session sweep (07) both call
these functions, so every session is processed identically.
"""

import numpy as np
import pandas as pd
import pynapple as nap
import scipy.signal as sg
from scipy.ndimage import gaussian_filter1d

import swr_lib as L

# ---------------------------------------------------------------- ripple detection
DET_BAND = (130.0, 250.0)
PEAK_Z = 5.0
EDGE_Z = 2.0
MIN_DUR, MAX_DUR = 0.020, 0.200
MERGE_GAP = 0.030


def select_ripple_channel(h5, nrem_ep, n_windows=6, win=20.0):
    """Return the channel with the largest ripple-band spectral bump during non-REM.

    Raw 140-230 Hz power alone would just pick the noisiest site, so the score is the
    height of the ripple peak above a baseline interpolated from 100-120 and 280-320 Hz.
    """
    long_nrem = nrem_ep[(nrem_ep.end - nrem_ep.start) > 60]
    starts = long_nrem.start[np.linspace(0, len(long_nrem) - 1, n_windows).astype(int)]
    dset = L.lfp_dataset(h5)
    conv = L.lfp_conversion(h5)
    score = np.zeros(dset.shape[1])
    for ch in range(dset.shape[1]):
        x = np.concatenate(
            [dset[int(s * L.LFP_FS) : int(s * L.LFP_FS) + int(win * L.LFP_FS), ch]
             for s in starts]
        ).astype(float) * conv
        f, p = sg.welch(x, fs=L.LFP_FS, nperseg=2048)
        lp = np.log10(p)
        peak = lp[(f >= 140) & (f <= 220)].max()
        base = np.interp(180, [110, 300],
                         [lp[(f >= 100) & (f <= 120)].mean(), lp[(f >= 280) & (f <= 320)].mean()])
        score[ch] = peak - base
    return int(np.argmax(score)), score


def detect_ripples(raw, nrem_mask):
    """Threshold the smoothed ripple-band envelope; return the events and the traces.

    Amplitude is z-scored against non-REM sleep so that running theta cannot drag the
    threshold around. Events must reach PEAK_Z, last between MIN_DUR and MAX_DUR, and
    not ride on a broadband deflection (movement or chewing artefact).
    """
    n = raw.size
    rip_filt, rip_env = L.band_envelope(raw, *DET_BAND)
    env_s = L.gaussian_smooth(rip_env, 0.008).astype(np.float32)
    mu, sd = env_s[nrem_mask].mean(), env_s[nrem_mask].std()
    z = (env_s - mu) / sd
    raw_sd = raw[nrem_mask].std()

    above = z > EDGE_Z
    edges = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    stops = np.flatnonzero(edges == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        stops = np.r_[stops, n - 1]

    ms, me = [starts[0]], [stops[0]]
    for s, e in zip(starts[1:], stops[1:]):
        if (s - me[-1]) / L.LFP_FS < MERGE_GAP:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)

    rows = []
    for s, e in zip(ms, me):
        seg = z[s:e]
        if seg.size == 0:
            continue
        pk = int(seg.argmax())
        dur = (e - s) / L.LFP_FS
        if seg[pk] < PEAK_Z or not (MIN_DUR <= dur <= MAX_DUR):
            continue
        if np.abs(raw[s:e]).max() > 12 * raw_sd:
            continue
        fseg = rip_filt[s:e].astype(np.float64)
        phase = np.unwrap(np.angle(sg.hilbert(fseg)))
        rows.append(
            dict(start=s / L.LFP_FS, stop=e / L.LFP_FS, peak_t=(s + pk) / L.LFP_FS,
                 peak_z=float(seg[pk]), duration=dur,
                 peak_amp=float(np.abs(fseg).max()),
                 peak_freq=float(np.median(np.diff(phase) * L.LFP_FS / (2 * np.pi))))
        )
    return pd.DataFrame(rows), rip_filt, z


def in_intervals(times, iset):
    m = np.zeros(len(times), dtype=bool)
    idx = np.searchsorted(iset.start, times, side="right") - 1
    ok = idx >= 0
    m[ok] = times[ok] <= iset.end[idx[ok]]
    return m


def tag_events(rip, states, epochs):
    rip = rip.copy()
    rip["state"] = "other"
    for lab in states:
        rip.loc[in_intervals(rip["peak_t"].values, states[lab]), "state"] = lab
    rip["epoch"] = "other"
    for lab in epochs:
        rip.loc[in_intervals(rip["peak_t"].values, epochs[lab]), "epoch"] = lab.replace("Epoch", "")
    return rip


def nrem_mask_from(states, n_samples):
    m = np.zeros(n_samples, dtype=bool)
    for s, e in zip(states["Non-REM"].start, states["Non-REM"].end):
        m[int(s * L.LFP_FS) : int(e * L.LFP_FS)] = True
    return m


# ------------------------------------------------------------------- place fields
def tracking_blocks(position, min_len=5):
    """Split the linearised position into contiguous tracked stretches."""
    ok = np.flatnonzero(~np.isnan(position.d))
    blocks = np.split(ok, np.flatnonzero(np.diff(ok) > 2) + 1)
    return [b for b in blocks if len(b) >= min_len]


def behaviour(position, dt):
    blocks = tracking_blocks(position)
    t, x, v, iv = [], [], [], []
    for b in blocks:
        tb, xb = position.t[b], position.d[b]
        t.append(tb)
        x.append(xb)
        v.append(np.gradient(xb, tb))
        iv.append((tb[0] - dt / 2, tb[-1] + dt / 2))
    t = np.concatenate(t)
    support = nap.IntervalSet(start=[a for a, _ in iv], end=[b for _, b in iv])
    return (nap.Tsd(t=t, d=np.concatenate(x), time_support=support),
            nap.Tsd(t=t, d=np.concatenate(v), time_support=support),
            support)


def place_fields(group, pos, ep, n_bins, track_len, dt, smooth_bins=1.0):
    tc = nap.compute_tuning_curves(group, pos, bins=n_bins, range=[(0.0, track_len)],
                                   epochs=ep, fs=1.0 / dt, return_pandas=True)
    tc[:] = gaussian_filter1d(np.nan_to_num(tc.values), smooth_bins, axis=0, mode="nearest")
    return tc


def select_template_cells(tc, mean_rate, min_peak=1.0, min_mean=0.05):
    """Pyramidal cells whose maze tuning curve is worth putting into the decoder.

    Deliberately a rate criterion and not a spatial-information criterion. An
    information cut tuned on one session throws away most of the population in the
    lower-yield ones, and the decoder does not need cells to be certified place cells:
    it weights each cell by its own tuning curve, and the shuffles below are what
    establish that the resulting sequences mean something.
    """
    return (tc.values.max(0) >= min_peak) & (mean_rate >= min_mean)


def skaggs_information(tc, occ_s):
    p = occ_s / occ_s.sum()
    r = tc.values.T
    rbar = (p[None, :] * r).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p[None, :] * (r / rbar[:, None]) * np.log2(r / rbar[:, None])
    return np.nansum(term, axis=1), rbar


# ------------------------------------------------------------ replay decoding
TAU = 0.020
PAD = 0.025
MIN_BINS = 5
MIN_ACTIVE = 5
EPS = 1e-3


def build_event_counts(cand, group, unit_ids):
    """Bin the place-cell spikes of every candidate event at TAU."""
    counts, keep, nbins = [], [], []
    spikes = {u: group[u].t for u in unit_ids}
    for r in cand.itertuples():
        t0, t1 = r.start - PAD, r.stop + PAD
        nb = int(np.floor((t1 - t0) / TAU))
        if nb < MIN_BINS:
            continue
        edges = t0 + np.arange(nb + 1) * TAU
        c = np.stack([np.histogram(spikes[u], bins=edges)[0] for u in unit_ids], axis=1)
        if (c.sum(0) > 0).sum() < MIN_ACTIVE:
            continue
        counts.append(c.astype(np.float64))
        keep.append(r.Index)
        nbins.append(nb)
    return counts, keep, nbins


class ReplayDecoder:
    """Memoryless Bayesian decoding of a stack of events against directional templates."""

    def __init__(self, counts, xbins, templates):
        self.E = len(counts)
        self.Tmax = max(c.shape[0] for c in counts)
        self.C = counts[0].shape[1]
        self.X = len(xbins)
        self.xgrid = np.asarray(xbins)
        self.tgrid = np.arange(self.Tmax) * TAU
        N = np.zeros((self.E, self.Tmax, self.C))
        M = np.zeros((self.E, self.Tmax), dtype=bool)
        for i, c in enumerate(counts):
            N[i, : c.shape[0]] = c
            M[i, : c.shape[0]] = True
        self.flatN = N.reshape(-1, self.C)
        self.M = M
        self.templates = {k: np.maximum(np.nan_to_num(v), EPS) for k, v in templates.items()}
        self.keys = list(self.templates)

    def decode(self, Fmat):
        # this matmul raises spurious FP flags from the BLAS kernel on finite input,
        # so the result is checked explicitly instead of trusting the warnings
        with np.errstate(all="ignore"):
            logl = self.flatN @ np.log(Fmat) - TAU * Fmat.sum(0)[None, :]
        assert np.isfinite(logl).all()
        logl = logl.reshape(self.E, self.Tmax, self.X)
        logl -= logl.max(axis=2, keepdims=True)
        p = np.exp(logl)
        p /= p.sum(axis=2, keepdims=True)
        return p * self.M[:, :, None]

    def weighted_corr(self, p):
        sw = p.sum(axis=(1, 2))
        mt = (p * self.tgrid[None, :, None]).sum(axis=(1, 2)) / sw
        mx = (p * self.xgrid[None, None, :]).sum(axis=(1, 2)) / sw
        dt = self.tgrid[None, :, None] - mt[:, None, None]
        dx = self.xgrid[None, None, :] - mx[:, None, None]
        ctt = (p * dt**2).sum(axis=(1, 2)) / sw
        cxx = (p * dx**2).sum(axis=(1, 2)) / sw
        ctx = (p * dt * dx).sum(axis=(1, 2)) / sw
        return ctx / np.sqrt(np.maximum(ctt * cxx, 1e-30)), ctx / np.maximum(ctt, 1e-30)

    def run(self, n_shuffles=500, seed=1, progress=None):
        rng = np.random.default_rng(seed)
        post = {k: self.decode(self.templates[k]) for k in self.keys}
        rs, ss = zip(*[self.weighted_corr(post[k]) for k in self.keys])
        r_real, s_real = np.stack(rs), np.stack(ss)
        best = np.argmax(np.abs(r_real), axis=0)
        r_best = r_real[best, np.arange(self.E)]
        s_best = s_real[best, np.arange(self.E)]

        cnt = dict(column=np.zeros(self.E), placefield=np.zeros(self.E),
                   timebin=np.zeros(self.E))
        it = range(n_shuffles)
        if progress is not None:
            it = progress(it, desc="shuffles")
        for _ in it:
            out = []
            for k in self.keys:
                sh = rng.integers(0, self.X, size=(self.E, self.Tmax))
                idx = (np.arange(self.X)[None, None, :] + sh[:, :, None]) % self.X
                out.append(self.weighted_corr(np.take_along_axis(post[k], idx, axis=2))[0])
            cnt["column"] += np.abs(np.stack(out)).max(0) >= np.abs(r_best)

            perm = rng.permutation(self.C)
            out = [self.weighted_corr(self.decode(self.templates[k][perm]))[0] for k in self.keys]
            cnt["placefield"] += np.abs(np.stack(out)).max(0) >= np.abs(r_best)

            order = np.argsort(rng.random((self.E, self.Tmax)) + (~self.M) * 10, axis=1)
            out = []
            for k in self.keys:
                q = np.take_along_axis(post[k], order[:, :, None].repeat(self.X, axis=2), axis=1)
                out.append(self.weighted_corr(q)[0])
            cnt["timebin"] += np.abs(np.stack(out)).max(0) >= np.abs(r_best)

        p = {k: (v + 1) / (n_shuffles + 1) for k, v in cnt.items()}
        sig = (p["column"] < 0.05) & (p["placefield"] < 0.05) & (p["timebin"] < 0.05)
        return dict(r=r_best, slope=s_best,
                    direction=np.array(self.keys)[best],
                    p_column=p["column"], p_placefield=p["placefield"],
                    p_timebin=p["timebin"], significant=sig, posteriors=post)
