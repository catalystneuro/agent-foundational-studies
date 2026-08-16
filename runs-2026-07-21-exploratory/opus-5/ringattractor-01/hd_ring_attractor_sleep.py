# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The head-direction system as a continuous ring attractor, maintained during sleep
#
# The head-direction (HD) cells of the anterodorsal thalamus and post-subiculum each
# fire over a narrow range of the animal's azimuthal heading. Attractor models predict
# something stronger than a collection of independent tuned cells: that the population
# is wired so that only a one-parameter family of activity patterns is stable, a bump of
# activity sitting somewhere on a ring of cells ordered by preferred direction. Two
# properties follow from that, and neither follows from tuning alone.
#
# The first is that the set of population states the network visits should be
# one-dimensional and closed. Each cell has its own firing rate, so a population of *N*
# cells lives in an *N*-dimensional space, but if the network is a ring attractor almost
# all of that space is unreachable and the states should fall on a curve that closes on
# itself.
#
# The second is that this structure is intrinsic to the circuit, not imposed by the
# senses. If the ring is built into the connectivity, it must still be there when the
# animal is asleep, motionless, and receiving no vestibular or visual evidence about
# its heading. The bump should still exist, still be localised, and still move
# continuously, even though it is no longer anchored to the world.
#
# This notebook tests both claims on real recordings from the DANDI Archive, using
# open-field foraging to define each cell's place on the ring and then asking what the
# same population does during REM and non-REM sleep.
#
# ## Dataset
#
# **DANDI:000056**, *Internally organized mechanisms of the head direction sense*
# (Peyrache, Lacroix, Petersen & Buzsaki; CC-BY-4.0). Extracellular silicon-probe
# recordings from the anterior thalamus and post-subiculum of freely moving mice, with
# open-field foraging blocks interleaved with long sleep sessions in a rest box, and
# sleep states (awake / REM / non-REM) already scored. Files are streamed from S3 with
# LINDI and cached locally; nothing is downloaded in full.
#
# ## What is computed
#
# 1. Head direction from the two head-mounted LEDs; HD cells identified against a
#    circular time-shift shuffle.
# 2. Pairwise correlation structure of the HD population as a function of the angular
#    offset between preferred directions, separately in wake, REM and non-REM.
# 3. Internal tuning curves: the angle read out from half the HD cells is used to score
#    the other half, so that each cell is measured against an angle it did not help
#    produce. During sleep this is the only available notion of heading.
# 4. Isomap embeddings of the population state in each brain state, with a
#    cell-identity-shuffled control, plus embedding-dimension and radial-spread
#    measures of how ring-like the state cloud is.
# 5. Bayesian decoding of the internal angle, its step-size and autocorrelation, and
#    the cross-correlogram timescale of similarly tuned pairs.
# 6. The whole pipeline repeated over six sessions from five animals.

# %%
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import Isomap
from tqdm.auto import tqdm

import h5py
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
H = A = sys.modules[__name__]   # the helper functions below live in this namespace

np.random.seed(0)
print("pynapple", nap.__version__)

# %% [markdown]
# ## Streaming the data and reconstructing the behavioural variables
#
# Head direction is not stored in these files; it is recovered from the two LEDs on the
# head stage. The tracker writes -1 when an LED is not detected, and the LEDs sit only
# about 20 px apart, so the coordinates are smoothed over ~0.15 s before the angle is
# taken.
#
# The animal alternates between a small rest box and the open field. Only the open-field
# blocks give a usable sample of head directions, and the map can rotate between visits
# that are hours apart, so tuning curves are estimated from a single visit
# (``main_exploration``) and the other visits are kept aside as a check.

# %%
DANDISET = "000056"
LINDI_TMPL = "https://lindi.neurosift.org/dandi/dandisets/{d}/assets/{a}/nwb.lindi.json"
CACHE_DIR = "cache/lindi"

# Sessions used. asset ids resolved from the DANDI API (see README).
SESSIONS = {
    "Mouse28-140313": "56d2e2d7-ba41-40a1-b017-b81871f3f0c3",
    "Mouse28-140312": "ff614bbb-8194-4372-9c7e-bcc0966bd453",
    "Mouse28-140317": "3c4d6960-bdea-41da-a98f-842be83abf41",
    "Mouse25-140130": "9c49d0c6-6e94-405b-b4e9-cb2ac744a36a",
    "Mouse24-131216": "14dafd30-f9fb-4aac-bc8e-e0d46d1b35e6",
    "Mouse17-130204": "f92f2709-4469-4c1d-a883-75eb66ee898a",
}


def open_session(asset_id):
    """Stream one NWB file from DANDI through LINDI with a local cache."""
    url = LINDI_TMPL.format(d=DANDISET, a=asset_id)
    f = lindi.LindiH5pyFile.from_lindi_file(
        url, local_cache=lindi.LocalCache(cache_dir=CACHE_DIR)
    )
    return NWBHDF5IO(file=f, mode="r").read()


def _smooth(x, w):
    """NaN-aware boxcar smoothing along a 1-D array."""
    k = np.ones(w) / w
    num = np.convolve(np.nan_to_num(x), k, "same")
    den = np.convolve(np.isfinite(x).astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan
    return out


def head_direction(nwbfile, smooth_s=0.15):
    """Head direction (rad, in [0, 2pi)) from the two head-mounted LEDs.

    The tracker writes -1 for frames where an LED was not detected; those become NaN.
    LED coordinates are lightly smoothed before taking the angle, because the two
    LEDs are only ~21 px apart and single-frame jitter is large relative to that.
    """
    blue = nwbfile.acquisition["BlueLED"]
    b = np.asarray(blue.data[:], dtype=float)
    r = np.asarray(nwbfile.acquisition["RedLED"].data[:], dtype=float)
    fs = float(blue.rate)
    t = np.arange(b.shape[0]) / fs + float(blue.starting_time)

    bad = (b <= 0).any(1) | (r <= 0).any(1)
    b[bad] = np.nan
    r[bad] = np.nan

    w = max(1, int(round(smooth_s * fs)))
    b = np.column_stack([_smooth(b[:, i], w) for i in range(2)])
    r = np.column_stack([_smooth(r[:, i], w) for i in range(2)])

    ang = np.arctan2(b[:, 1] - r[:, 1], b[:, 0] - r[:, 0]) % (2 * np.pi)
    pos = (b + r) / 2.0

    ok = np.isfinite(ang)
    hd = nap.Tsd(t=t[ok], d=ang[ok])
    position = nap.TsdFrame(t=t[ok], d=pos[ok], columns=["x", "y"])
    return hd, position, fs


def load_units(nwbfile, min_spikes=100):
    """Spike trains as a pynapple TsGroup.

    Spike times are sorted explicitly: a few units in this dandiset contain a single
    out-of-order timestamp, which otherwise makes pynapple refuse to build the group.
    """
    col = nwbfile.units["spike_times"]
    data = {}
    for i in range(len(nwbfile.units)):
        s = np.asarray(col[i], dtype=float)
        if len(s) < min_spikes:
            continue
        data[int(nwbfile.units.id[i])] = nap.Ts(t=np.sort(s))
    return nap.TsGroup(data)


def sleep_states(nwbfile):
    """Return {'Awake': IntervalSet, 'REM': ..., 'Non-REM': ...} from the scored states."""
    df = nwbfile.processing["behavior"]["states"].to_dataframe()
    out = {}
    for label, sub in df.groupby("label"):
        out[label] = nap.IntervalSet(
            start=sub["start_time"].values, end=sub["stop_time"].values
        )
    return out


def exploration_epoch(position, awake, min_extent=100.0, bin_s=60.0, min_dur=180.0):
    """Find the open-field foraging blocks.

    The animal alternates between a small sleep box and a 53x46 cm open field. The
    field is identified by the spatial extent covered in each 60 s block; blocks whose
    x- or y-range exceeds ``min_extent`` pixels, and that form a bout of at least
    ``min_dur`` seconds, are taken as open-field foraging.
    """
    t = position.t
    edges = np.arange(t[0], t[-1] + bin_s, bin_s)
    idx = np.digitize(t, edges) - 1
    keep = []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < 20:
            continue
        x, y = position["x"].d[m], position["y"].d[m]
        ex = np.percentile(x, 97.5) - np.percentile(x, 2.5)
        ey = np.percentile(y, 97.5) - np.percentile(y, 2.5)
        if max(ex, ey) > min_extent:
            keep.append((edges[i], edges[i + 1]))
    if not keep:
        return nap.IntervalSet(start=[], end=[])
    keep = np.array(keep)
    ep = nap.IntervalSet(start=keep[:, 0], end=keep[:, 1]).merge_close_intervals(1.0)
    # keep only sustained foraging bouts; isolated 60 s blocks are usually the animal
    # being carried between the sleep box and the field
    ep = ep[(ep.end - ep.start) >= min_dur]
    return ep.intersect(awake)


def group_bouts(ep, max_gap=600.0):
    """Split an IntervalSet into groups of bouts separated by less than ``max_gap``."""
    if len(ep) == 0:
        return []
    groups, cur = [], [0]
    for i in range(1, len(ep)):
        if ep.start[i] - ep.end[i - 1] > max_gap:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    return [nap.IntervalSet(start=ep.start[g], end=ep.end[g]) for g in groups]


def main_exploration(ep, max_gap=600.0):
    """The longest block of open-field foraging, plus the other blocks separately.

    The head-direction map is anchored to external cues only while the animal is in
    the environment; between separate visits the whole map can rotate. Pooling visits
    would blur every tuning curve, so tuning is estimated from a single visit. Bouts
    more than ``max_gap`` apart are treated as separate visits.
    """
    groups = group_bouts(ep, max_gap)
    if not groups:
        return ep, []
    tot = [float(g.tot_length()) for g in groups]
    k = int(np.argmax(tot))
    return groups[k], [g for i, g in enumerate(groups) if i != k]


def split_epoch_by_time(ep):
    """Split an IntervalSet into two halves of equal total duration.

    The split point is found in cumulative-duration coordinates and intervals are cut
    where necessary, so this works even when the epoch is a single long interval.
    """
    dur = ep.end - ep.start
    cum = np.cumsum(dur)
    target = cum[-1] / 2
    i = int(np.searchsorted(cum, target))
    i = min(i, len(ep) - 1)
    prev = cum[i - 1] if i > 0 else 0.0
    cut = ep.start[i] + (target - prev)
    cut = min(max(cut, ep.start[i]), ep.end[i])
    span = ep.time_span()
    first = ep.intersect(nap.IntervalSet(start=span.start[0], end=cut))
    second = ep.intersect(nap.IntervalSet(start=cut, end=span.end[0]))
    return first, second


def circular_mean_direction(tc):
    """Preferred direction and mean-vector length from a tuning-curve DataFrame."""
    ang = tc.index.values
    r = tc.values  # (n_bins, n_units)
    w = np.clip(r, 0, None)
    tot = w.sum(0)
    tot[tot == 0] = np.nan
    vx = (w * np.cos(ang)[:, None]).sum(0) / tot
    vy = (w * np.sin(ang)[:, None]).sum(0) / tot
    return np.arctan2(vy, vx) % (2 * np.pi), np.hypot(vx, vy)


def circ_diff(a, b):
    """Signed circular difference a - b wrapped to (-pi, pi]."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi

# %% [markdown]
# ## Analysis routines
#
# ``hd_cell_selection`` compares each cell's occupancy-corrected mean vector length
# against a null built by circularly shifting that cell's own spike-count vector
# relative to the head-direction signal, which preserves both the cell's firing
# statistics and the animal's occupancy.
#
# ``internal_tuning_curves`` and ``split_half_coherence`` implement the held-out test:
# HD cells are sorted by preferred direction and split into two interleaved groups so
# that each group still tiles the circle, an angle is decoded from one group, and the
# other group is scored against it. No cell contributes to the angle it is judged by.
#
# ``manifold_input`` prepares the population states for Isomap. Counts are
# square-root transformed, lightly smoothed, z-scored per cell, and projected onto the
# unit sphere; the last step removes overall population gain, which during non-REM
# varies enormously between up and down states and would otherwise turn the ring into a
# cone. Bins in the lowest quartile of total population activity are dropped, because in
# those bins (non-REM down states) essentially no HD cell fires and the ring coordinate
# is undefined.

# %%
N_HD_BINS = 60
HD_BINS = np.linspace(0, 2 * np.pi, N_HD_BINS + 1)
HD_CENTERS = HD_BINS[:-1] + np.diff(HD_BINS) / 2


# --------------------------------------------------------------------------
# tuning curves and HD-cell selection
# --------------------------------------------------------------------------

def tuning_curves(units, hd, ep, nb_bins=N_HD_BINS, smooth_bins=1.0):
    """Occupancy-normalised HD tuning curves, circularly smoothed."""
    tc = nap.compute_1d_tuning_curves(
        units, hd, nb_bins=nb_bins, minmax=(0, 2 * np.pi), ep=ep
    )
    if smooth_bins:
        sm = gaussian_filter1d(tc.values, smooth_bins, axis=0, mode="wrap")
        tc = pd.DataFrame(sm, index=tc.index, columns=tc.columns)
    return tc


def _count_matrix(units, hd, ep):
    """Spike counts per head-direction sample, restricted to ``ep``.

    Returns (counts [n_units x n_samples], hd bin index per sample, dt).
    """
    hdr = hd.restrict(ep)
    t = hdr.t
    dt = np.median(np.diff(t))
    edges = np.concatenate([t - dt / 2, [t[-1] + dt / 2]])
    counts = np.zeros((len(units), len(t)))
    for i, u in enumerate(units.index):
        st = units[u].restrict(ep).t
        counts[i] = np.histogram(st, edges)[0]
    hdbin = np.clip(np.digitize(hdr.d, HD_BINS) - 1, 0, N_HD_BINS - 1)
    return counts, hdbin, dt


def hd_cell_selection(units, hd, ep, n_shuffle=100, alpha=0.01, rng=None):
    """Identify HD cells by a circular time-shift shuffle test on the mean vector length.

    The whole spike-count vector of a cell is circularly shifted relative to the head
    direction, which destroys the cell/direction relationship while preserving the
    cell's own firing statistics and the occupancy of the animal.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    counts, hdbin, dt = _count_matrix(units, hd, ep)
    n_samp = counts.shape[1]
    occ = np.bincount(hdbin, minlength=N_HD_BINS) * dt
    onehot = np.zeros((n_samp, N_HD_BINS))
    onehot[np.arange(n_samp), hdbin] = 1.0

    visited = occ > 0
    occ_safe = np.where(visited, occ, 1.0)

    def mvl_from_counts(C):
        with np.errstate(all="ignore"):           # BLAS matmul emits spurious flags
            tc = (C @ onehot) / occ_safe[None, :]  # n_units x n_bins
        w = np.clip(tc, 0, None) * visited[None, :]
        tot = w.sum(1)
        tot[tot == 0] = np.nan
        vx = (w * np.cos(HD_CENTERS)).sum(1) / tot
        vy = (w * np.sin(HD_CENTERS)).sum(1) / tot
        return np.hypot(vx, vy)

    obs = mvl_from_counts(counts)
    shifts = rng.integers(int(0.1 * n_samp), int(0.9 * n_samp), size=n_shuffle)
    null = np.empty((n_shuffle, len(units)))
    for k, s in enumerate(shifts):
        null[k] = mvl_from_counts(np.roll(counts, s, axis=1))

    p = (1 + (null >= obs[None, :]).sum(0)) / (n_shuffle + 1)
    rate = counts.sum(1) / (n_samp * dt)
    return pd.DataFrame(
        {"mvl": obs, "mvl_null_mean": null.mean(0), "mvl_null_p99": np.percentile(null, 99, axis=0),
         "p": p, "rate": rate},
        index=units.index,
    )


# --------------------------------------------------------------------------
# population activity matrices
# --------------------------------------------------------------------------

def population_counts(units, ep, bin_size):
    """Binned spike counts as a TsdFrame restricted to ``ep``."""
    return units.count(bin_size, ep)


def zscored_rates(counts, smooth_bins=1.0):
    """sqrt-transform, smooth and z-score binned counts (rows = time, cols = cells)."""
    x = np.sqrt(counts.values.astype(float))
    if smooth_bins:
        x = gaussian_filter1d(x, smooth_bins, axis=0, mode="nearest")
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


# --------------------------------------------------------------------------
# pairwise correlation structure
# --------------------------------------------------------------------------

def pair_correlations(counts, pref_dir):
    """Pearson correlations between all cell pairs, with their preferred-direction offset."""
    x = np.sqrt(counts.values.astype(float))
    C = np.corrcoef(x.T)
    n = C.shape[0]
    iu, ju = np.triu_indices(n, 1)
    d = np.abs(circ_diff(pref_dir[iu], pref_dir[ju]))
    return C, d, C[iu, ju]


def binned_profile(d, r, nbins=12, hi=np.pi):
    """Mean +/- SEM of ``r`` in bins of angular offset ``d``."""
    edges = np.linspace(0, hi, nbins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, nbins - 1)
    ctr, m, s, n = [], [], [], []
    for k in range(nbins):
        sel = idx == k
        if sel.sum() < 2:
            continue
        ctr.append((edges[k] + edges[k + 1]) / 2)
        m.append(r[sel].mean())
        s.append(r[sel].std(ddof=1) / np.sqrt(sel.sum()))
        n.append(sel.sum())
    return np.array(ctr), np.array(m), np.array(s), np.array(n)


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

def decode_hd(tc, units, ep, bin_size):
    """Poisson Bayesian decoding of head direction with a uniform prior."""
    decoded, proba = nap.decode_1d(
        tuning_curves=tc, group=units, ep=ep, bin_size=bin_size
    )
    return decoded, proba


def decoded_vector_strength(proba):
    """Concentration of the posterior on the ring: |sum_k p_k e^{i theta_k}|."""
    p = proba.values
    ang = np.asarray(proba.columns, dtype=float)
    vx = (p * np.cos(ang)).sum(1)
    vy = (p * np.sin(ang)).sum(1)
    return np.hypot(vx, vy)


def angular_step(decoded):
    """Absolute angular change between consecutive decoded samples (rad/bin)."""
    return np.abs(circ_diff(decoded.d[1:], decoded.d[:-1]))


def angular_step_null(decoded, rng=0):
    """Null step distribution: angular distance between randomly paired time bins.

    This is the distribution expected if successive decoded angles were unrelated,
    i.e. if the population state jumped anywhere on the ring from bin to bin.
    """
    rng = np.random.default_rng(rng)
    d = decoded.d
    i = rng.integers(0, len(d), len(d) - 1)
    j = rng.integers(0, len(d), len(d) - 1)
    return np.abs(circ_diff(d[i], d[j]))


# --------------------------------------------------------------------------
# coherence: can half the cells predict the other half through a single angle?
# --------------------------------------------------------------------------

def split_half_coherence(tc, units, ep, bin_size, pref_dir, rng=0):
    """Decode an angle from one half of the HD cells, predict the other half.

    Cells are split into two interleaved groups after sorting by preferred direction,
    so both halves tile the ring. If the population state is confined to a
    one-dimensional ring, the single angle read out from group A must predict the
    instantaneous rates of group B.

    Two controls are returned alongside the observed correlation. The time-shifted
    control breaks all temporal alignment. The identity-permuted control keeps the
    timing but predicts each held-out cell from another held-out cell's tuning curve,
    which preserves any common population-rate fluctuation (the up/down states of
    non-REM in particular) while destroying the cell-specific ring prediction.
    """
    order = np.argsort(pref_dir)
    ids = np.asarray(units.index)[order]
    # membership is what matters; ids are re-sorted because pynapple's TsGroup
    # indexing requires keys in ascending order
    grpA = sorted(int(i) for i in ids[0::2])
    grpB = sorted(int(i) for i in ids[1::2])
    out = {}
    for name, (fit, test) in {"A->B": (grpA, grpB), "B->A": (grpB, grpA)}.items():
        dec, _ = nap.decode_1d(
            tuning_curves=tc[fit], group=units[fit], ep=ep, bin_size=bin_size
        )
        obs = units[test].count(bin_size, ep).values.astype(float)
        # predicted rate of held-out cells at the decoded angle
        bin_idx = np.clip(np.digitize(dec.d, HD_BINS) - 1, 0, N_HD_BINS - 1)
        pred = tc[test].values[bin_idx, :]
        n = min(len(obs), len(pred))
        obs, pred = obs[:n], pred[:n]
        shift = max(50, n // 7)
        pred_ctrl = np.roll(pred, shift, axis=0)
        m = pred.shape[1]
        perm = (np.arange(m) + max(1, m // 2)) % m      # predict cell j from cell perm[j]
        pred_perm = pred[:, perm]
        r_obs = np.array([_corr(obs[:, j], pred[:, j]) for j in range(m)])
        r_ctl = np.array([_corr(obs[:, j], pred_ctrl[:, j]) for j in range(m)])
        r_prm = np.array([_corr(obs[:, j], pred_perm[:, j]) for j in range(m)])
        out[name] = (test, r_obs, r_ctl, r_prm)
    cells = np.concatenate([out["A->B"][0], out["B->A"][0]])
    r_obs = np.concatenate([out["A->B"][1], out["B->A"][1]])
    r_ctl = np.concatenate([out["A->B"][2], out["B->A"][2]])
    r_prm = np.concatenate([out["A->B"][3], out["B->A"][3]])
    return cells, r_obs, r_ctl, r_prm


def _corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------
# manifold structure
# --------------------------------------------------------------------------

def isomap_embed(X, n_components=2, n_neighbors=20, max_points=2500, rng=0):
    """Isomap embedding of population states (subsampled for tractability)."""
    rng = np.random.default_rng(rng)
    if len(X) > max_points:
        sel = np.sort(rng.choice(len(X), max_points, replace=False))
    else:
        sel = np.arange(len(X))
    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    Y = iso.fit_transform(X[sel])
    return Y, sel, iso


def isomap_residual_variance(X, dims=(1, 2, 3, 4, 5, 6), n_neighbors=20,
                             max_points=1500, rng=0):
    """Residual variance of the Isomap fit as a function of embedding dimension.

    The elbow indicates the number of coordinates needed to embed the manifold; a
    circle needs 2 (it is intrinsically 1-D but not flat).
    """
    rng = np.random.default_rng(rng)
    sel = np.sort(rng.choice(len(X), min(max_points, len(X)), replace=False))
    Xs = X[sel]
    res = []
    iso = Isomap(n_components=max(dims), n_neighbors=n_neighbors)
    iso.fit(Xs)
    Dg = iso.dist_matrix_
    iu = np.triu_indices(len(Xs), 1)
    dg = Dg[iu]
    for k in dims:
        Y = iso.embedding_[:, :k]
        dy = np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(1))
        res.append(1 - np.corrcoef(dg, dy)[0, 1] ** 2)
    return np.array(dims), np.array(res)


def ring_angle(Y):
    """Angular coordinate and normalised radial spread of a 2-D embedding."""
    Yc = Y - Y.mean(0)
    ang = np.arctan2(Yc[:, 1], Yc[:, 0]) % (2 * np.pi)
    rad = np.hypot(Yc[:, 0], Yc[:, 1])
    return ang, rad


def circ_corr(a, b):
    """Circular-circular correlation coefficient (Jammalamadaka & Sengupta)."""
    a = np.asarray(a); b = np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    am = np.angle(np.mean(np.exp(1j * a)))
    bm = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    den = np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2))
    return float(num / den)


# --------------------------------------------------------------------------
# controls and timescale
# --------------------------------------------------------------------------

def unit_normalise(X, eps=1e-9):
    """Project population-state vectors onto the unit sphere (removes overall gain)."""
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def shuffle_cells(X, rng=0):
    """Circularly shift each cell's own time course, destroying cross-cell structure."""
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    Xs = np.empty_like(X)
    for j in range(X.shape[1]):
        Xs[:, j] = np.roll(X[:, j], int(rng.integers(n // 10, 9 * n // 10)))
    return Xs


def manifold_input(units, ep, bin_size, smooth_bins=1.0, activity_pct=25.0):
    """Population states used for the manifold analyses.

    Counts are square-root transformed, smoothed, z-scored per cell and projected
    onto the unit sphere. Bins in the lowest ``activity_pct`` of total population
    spike count are dropped: during non-REM these are down states in which no cell
    fires and the ring coordinate is undefined.
    """
    cnt = units.count(bin_size, ep)
    total = cnt.values.sum(1)
    X = unit_normalise(zscored_rates(cnt, smooth_bins=smooth_bins))
    keep = total >= np.percentile(total, activity_pct)
    return X, keep, cnt


def pca_spectrum(X, n=10):
    """Fraction of variance carried by each principal component."""
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    v = s ** 2
    return (v / v.sum())[:n]


def decode_shuffled(tc, units, ep, bin_size, rng=0):
    """Decode after permuting which tuning curve belongs to which cell."""
    rng = np.random.default_rng(rng)
    perm = rng.permutation(tc.shape[1])
    tc_s = tc.copy()
    tc_s.columns = list(range(tc.shape[1]))
    tc_s = tc_s.iloc[:, perm]
    tc_s.columns = tc.columns
    return nap.decode_1d(tuning_curves=tc_s, group=units, ep=ep, bin_size=bin_size)


def mean_crosscorrelogram(units, ep, pref_dir, binsize=0.01, windowsize=1.0,
                          near=np.pi / 4, far=3 * np.pi / 4):
    """Mean normalised cross-correlograms for similarly and oppositely tuned pairs."""
    cc = nap.compute_crosscorrelogram(units, binsize, windowsize, ep=ep, norm=True)
    ids = np.asarray(units.index)
    pos = {int(u): i for i, u in enumerate(ids)}
    near_c, far_c = [], []
    for col in cc.columns:
        a, b = int(col[0]), int(col[1])
        d = abs(circ_diff(pref_dir[pos[a]], pref_dir[pos[b]]))
        v = cc[col].values
        if not np.all(np.isfinite(v)):
            continue
        (near_c if d < near else far_c if d > far else []).append(v)
    t = cc.index.values
    return t, np.array(near_c), np.array(far_c)


def internal_tuning_curves(tc, units, ep, bin_size, pref_dir, n_bins=N_HD_BINS):
    """Tuning of each cell to the ring angle read out from the *other* cells.

    Half the HD cells are used to decode an angle; the tuning curve of each held-out
    cell is then computed against that internally generated angle. During sleep there
    is no measured head direction, so this is the only way to ask whether a cell still
    fires at its preferred position on the ring. No held-out cell contributes to the
    angle it is scored against.
    """
    order = np.argsort(pref_dir)
    ids = np.asarray(units.index)[order]
    grpA = sorted(int(i) for i in ids[0::2])
    grpB = sorted(int(i) for i in ids[1::2])
    cells, curves = [], []
    for fit, test in [(grpA, grpB), (grpB, grpA)]:
        dec, _ = nap.decode_1d(
            tuning_curves=tc[fit], group=units[fit], ep=ep, bin_size=bin_size
        )
        obs = units[test].count(bin_size, ep).values.astype(float)
        n = min(len(obs), len(dec))
        idx = np.clip(np.digitize(dec.d[:n], HD_BINS) - 1, 0, n_bins - 1)
        occ = np.bincount(idx, minlength=n_bins) * bin_size
        occ_safe = np.where(occ > 0, occ, np.nan)
        for j, cell in enumerate(test):
            num = np.bincount(idx, weights=obs[:n, j], minlength=n_bins)
            cells.append(cell)
            curves.append(num / occ_safe)
    curves = np.array(curves)
    cells = np.array(cells)
    # align each curve so that bin 0 is the cell's wake preferred direction
    pos = {int(u): i for i, u in enumerate(np.asarray(units.index))}
    shift = np.array([int(round(pref_dir[pos[c]] / (2 * np.pi) * n_bins)) for c in cells])
    aligned = np.array([np.roll(c, -s) for c, s in zip(curves, shift)])
    aligned = np.roll(aligned, n_bins // 2, axis=1)   # centre the peak
    return cells, curves, aligned


def angle_autocorrelation(decoded, ep, bin_size, max_lag=4.0):
    """Circular autocorrelation <cos(theta(t+tau) - theta(t))> of the decoded angle.

    Computed within each continuous interval so that epoch boundaries never
    contribute a spurious jump.
    """
    n_lag = int(round(max_lag / bin_size))
    num = np.zeros(n_lag + 1)
    den = np.zeros(n_lag + 1)
    for s, e in zip(ep.start, ep.end):
        d = decoded.restrict(nap.IntervalSet(start=s, end=e)).d
        if len(d) < n_lag + 2:
            continue
        for k in range(n_lag + 1):
            num[k] += np.sum(np.cos(d[k:] - d[: len(d) - k]))
            den[k] += len(d) - k
    lags = np.arange(n_lag + 1) * bin_size
    with np.errstate(invalid="ignore"):
        ac = num / den
    return lags, ac


def halfwidth(lags, y):
    """Lag at which a curve first falls to half of its value at lag zero."""
    half = y[0] / 2
    below = np.where(y < half)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return 0.0
    x0, x1, y0_, y1 = lags[i - 1], lags[i], y[i - 1], y[i]
    return float(x0 + (y0_ - half) * (x1 - x0) / (y0_ - y1))


def circ_match(a, b):
    """How tightly two circular variables track each other up to rotation/reflection.

    Returns the mean resultant length of the offset (a - s*b), maximised over the two
    chiralities s = +-1. 1 means the two angles differ by a constant; 0 means no
    relationship. Unlike a circular correlation coefficient this does not assume the
    mapping advances at a uniform rate, which matters because Isomap does not preserve
    arc length around the ring.
    """
    a = np.asarray(a); b = np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(max(abs(np.mean(np.exp(1j * (a - s * b)))) for s in (1, -1)))

# %% [markdown]
# ## Per-session pipeline
#
# ``analyse_session`` runs everything above on one NWB file and returns a dictionary of
# results; ``summarise`` reduces it to the row used in the cross-session comparison.

# %%
BIN = 0.2                 # s, population bin for decoding / correlations
SMOOTH_BINS = 1.0         # Gaussian sigma (in bins) for the manifold analyses
ACTIVITY_PCT = 25.0       # drop the least active bins (non-REM down states)


def analyse_session(name, asset_id, bin_size=BIN, verbose=True, do_manifold=True):
    nwbfile = H.open_session(asset_id)
    hd, pos, fs = H.head_direction(nwbfile)
    states = H.sleep_states(nwbfile)
    units = H.load_units(nwbfile)
    all_expl = H.exploration_epoch(pos, states["Awake"])
    expl, other_visits = H.main_exploration(all_expl)
    ep1, ep2 = H.split_epoch_by_time(expl)

    sel = A.hd_cell_selection(units, hd, expl, n_shuffle=100)
    tc_all = A.tuning_curves(units, hd, expl)
    pref, mvl = H.circular_mean_direction(tc_all)
    sel["pref"] = pref
    is_hd = (sel.p < 0.01) & (sel.mvl > 0.3) & (sel.rate > 0.5)
    hd_ids = [int(i) for i in sel.index[is_hd]]
    non_hd_ids = [int(i) for i in sel.index[(~is_hd) & (sel.rate > 0.5)]]

    if verbose:
        print(f"[{name}] {len(units)} units, {len(hd_ids)} HD cells | "
              f"wake-explore {expl.tot_length():.0f}s, "
              f"REM {states['REM'].tot_length():.0f}s, "
              f"non-REM {states['Non-REM'].tot_length():.0f}s")

    hd_units = units[hd_ids]
    tc = tc_all[hd_ids]
    pd_hd = sel.loc[hd_ids, "pref"].values

    tc1 = A.tuning_curves(hd_units, hd, ep1)
    tc2 = A.tuning_curves(hd_units, hd, ep2)
    p1, m1 = H.circular_mean_direction(tc1)
    p2, m2 = H.circular_mean_direction(tc2)

    # preferred directions on a separate visit to the environment, hours later
    revisit = None
    if other_visits:
        far = max(other_visits, key=lambda g: g.tot_length())
        if far.tot_length() > 300:
            tcr = A.tuning_curves(hd_units, hd, far)
            pr, mr = H.circular_mean_direction(tcr)
            revisit = dict(pref=pr, mvl=mr, tc=tcr,
                           dur=float(far.tot_length()),
                           gap_h=float((far.start[0] - expl.end[-1]) / 3600))

    epochs = {"wake": expl, "REM": states["REM"], "non-REM": states["Non-REM"]}

    res = dict(
        name=name, asset_id=asset_id, bin_size=bin_size,
        sel=sel, tc_all=tc_all, tc=tc, tc1=tc1, tc2=tc2,
        hd_ids=hd_ids, non_hd_ids=non_hd_ids, pref=pd_hd,
        pref_block1=p1, pref_block2=p2, n_units=len(units),
        durations={k: float(v.tot_length()) for k, v in epochs.items()},
        hd=hd, position=pos, states=states, epochs=epochs, _units=units,
        all_exploration=all_expl, revisit=revisit,
    )

    # ---- pairwise correlation structure
    res["corr"], res["corr_nonhd"] = {}, {}
    for k, ep in epochs.items():
        cnt = hd_units.count(bin_size, ep)
        C, d, r = A.pair_correlations(cnt, pd_hd)
        res["corr"][k] = dict(C=C, d=d, r=r, profile=A.binned_profile(d, r))
        if len(non_hd_ids) >= 4:
            cnt_n = units[non_hd_ids].count(bin_size, ep)
            pdn = sel.loc[non_hd_ids, "pref"].values
            Cn, dn, rn = A.pair_correlations(cnt_n, pdn)
            res["corr_nonhd"][k] = dict(profile=A.binned_profile(dn, rn), r=rn, d=dn)
    res["corr_across_states"] = {
        k: float(np.corrcoef(res["corr"]["wake"]["r"], res["corr"][k]["r"])[0, 1])
        for k in epochs
    }
    res["corr_across_states_nonhd"] = {
        k: float(np.corrcoef(res["corr_nonhd"]["wake"]["r"], res["corr_nonhd"][k]["r"])[0, 1])
        for k in epochs if k in res["corr_nonhd"]
    } if res["corr_nonhd"] else {}

    # ---- decoding with wake tuning curves
    res["decode"] = {}
    for k, ep in epochs.items():
        dec, proba = A.decode_hd(tc, hd_units, ep, bin_size)
        dec_s, proba_s = A.decode_shuffled(tc, hd_units, ep, bin_size)
        res["decode"][k] = dict(
            decoded=dec, proba=proba,
            vs=A.decoded_vector_strength(proba),
            vs_shuf=A.decoded_vector_strength(proba_s),
            step=A.angular_step(dec),
            step_null=A.angular_step_null(dec),
            step_shuf=A.angular_step(dec_s),
        )
    dec2, _ = A.decode_hd(tc1, hd_units, ep2, bin_size)   # tuning from block 1
    signed = _decode_error(dec2, hd, ep2, bin_size)
    offset = float(np.angle(np.mean(np.exp(1j * signed))))
    res["wake_decode_err"] = np.abs(signed)
    res["wake_decode_err_aligned"] = np.abs(H.circ_diff(signed, offset))
    res["wake_decode_offset"] = offset

    # ---- split-half coherence
    res["coherence"] = {}
    for k, ep in epochs.items():
        cells, r_obs, r_ctl, r_prm = A.split_half_coherence(
            tc, hd_units, ep, bin_size, pd_hd)
        ic_cells, ic_raw, ic_aligned = A.internal_tuning_curves(
            tc, hd_units, ep, bin_size, pd_hd)
        wake_aligned = _align_wake(tc, pd_hd, ic_cells, hd_units)
        r_shape = np.array([_corr_nan(a, w) for a, w in zip(ic_aligned, wake_aligned)])
        res["coherence"][k] = dict(cells=cells, r_obs=r_obs, r_ctl=r_ctl, r_prm=r_prm,
                                   ic_cells=ic_cells, ic_aligned=ic_aligned,
                                   wake_aligned=wake_aligned, r_shape=r_shape)

    # ---- cross-correlogram timescale
    res["xcorr"] = {}
    for k, ep in epochs.items():
        t, near, far = A.mean_crosscorrelogram(hd_units, ep, pd_hd,
                                               binsize=0.02, windowsize=8.0)
        res["xcorr"][k] = dict(t=t, near=near.mean(0), far=far.mean(0),
                               near_n=len(near), far_n=len(far))

    # ---- drift timescale of the internal angle
    res["autocorr"] = {}
    for k, ep in epochs.items():
        lags, ac = A.angle_autocorrelation(res["decode"][k]["decoded"], ep, bin_size,
                                           max_lag=8.0)
        res["autocorr"][k] = dict(lags=lags, ac=ac, halfwidth=A.halfwidth(lags, ac))

    # ---- fine-grained sleep sweeps
    res["fine"] = {}
    for k in ["REM", "non-REM"]:
        dec_f, proba_f = A.decode_hd(tc, hd_units, epochs[k], 0.04)
        res["fine"][k] = dict(decoded=dec_f, proba=proba_f, bin_size=0.04)

    # ---- manifold structure
    if do_manifold:
        res["manifold"] = {}
        for k, ep in epochs.items():
            X, keep, cnt = A.manifold_input(hd_units, ep, bin_size,
                                            SMOOTH_BINS, ACTIVITY_PCT)
            dec = res["decode"][k]["decoded"]
            ang_ref = dec.d[:len(X)][keep]
            Xk = X[keep]
            Y, subsel, _ = A.isomap_embed(Xk, n_neighbors=25)
            ang, rad = A.ring_angle(Y)
            Ys, _, _ = A.isomap_embed(A.shuffle_cells(Xk), n_neighbors=25)
            angs, rads = A.ring_angle(Ys)
            dims, rv = A.isomap_residual_variance(Xk)
            entry = dict(
                Y=Y, sel=subsel, ring_angle=ang, radius=rad,
                decoded_at=ang_ref[subsel],
                circ_corr=A.circ_match(ang, ang_ref[subsel]),
                radial_cv=float(np.std(rad) / np.mean(rad)),
                Y_shuf=Ys, radial_cv_shuf=float(np.std(rads) / np.mean(rads)),
                circ_corr_shuf=A.circ_match(angs, ang_ref[subsel]),
                dims=dims, resvar=rv,
                resvar_shuf=A.isomap_residual_variance(A.shuffle_cells(Xk))[1],
                pca=A.pca_spectrum(Xk), pca_shuf=A.pca_spectrum(A.shuffle_cells(Xk)),
                n_states=int(keep.sum()),
            )
            if k == "wake":
                true_hd = hd.restrict(ep).bin_average(bin_size, ep)
                m = min(len(true_hd), len(X))
                th = np.full(len(X), np.nan)
                th[:m] = true_hd.d[:m]
                entry["true_at"] = th[keep][subsel]
                entry["circ_corr_true"] = A.circ_match(ang, entry["true_at"])
            res["manifold"][k] = entry

    return res


def summarise(res):
    """One-row summary of the quantities compared across sessions."""
    row = dict(session=res["name"], n_units=res["n_units"], n_hd=len(res["hd_ids"]))
    row.update({f"dur_{k}": v for k, v in res["durations"].items()})
    row["wake_decode_err_med_deg"] = float(np.degrees(np.median(res["wake_decode_err"])))
    row["wake_decode_err_aligned_med_deg"] = float(
        np.degrees(np.median(res["wake_decode_err_aligned"])))
    shift = H.circ_diff(res["pref_block2"], res["pref_block1"])
    rot = float(np.angle(np.mean(np.exp(1j * shift))))
    row["block_rotation_deg"] = float(np.degrees(rot) % 360)
    row["block_resid_deg"] = float(np.degrees(np.median(np.abs(H.circ_diff(shift, rot)))))
    if res.get("revisit") is not None:
        rshift = H.circ_diff(res["revisit"]["pref"], res["pref"])
        rrot = float(np.angle(np.mean(np.exp(1j * rshift))))
        row["revisit_rotation_deg"] = float(np.degrees(rrot) % 360)
        row["revisit_resid_deg"] = float(
            np.degrees(np.median(np.abs(H.circ_diff(rshift, rrot)))))
        row["revisit_gap_h"] = res["revisit"]["gap_h"]
    for k in res["epochs"]:
        row[f"corrmatch_{k}"] = res["corr_across_states"][k]
        row[f"coh_{k}"] = float(np.nanmean(res["coherence"][k]["r_obs"]))
        row[f"cohctl_{k}"] = float(np.nanmean(res["coherence"][k]["r_ctl"]))
        row[f"cohperm_{k}"] = float(np.nanmean(res["coherence"][k]["r_prm"]))
        row[f"shape_{k}"] = float(np.nanmean(res["coherence"][k]["r_shape"]))
        row[f"step_{k}_degs"] = float(np.degrees(np.median(res["decode"][k]["step"]))
                                      / res["bin_size"])
        row[f"stepnull_{k}_degs"] = float(np.degrees(np.median(res["decode"][k]["step_null"]))
                                          / res["bin_size"])
        row[f"vs_{k}"] = float(np.median(res["decode"][k]["vs"]))
        row[f"tau_{k}_s"] = float(res["autocorr"][k]["halfwidth"])
        row[f"xcorr_hw_{k}_s"] = float(A.halfwidth(
            res["xcorr"][k]["t"][res["xcorr"][k]["t"] >= 0],
            (res["xcorr"][k]["near"] - res["xcorr"][k]["far"])[res["xcorr"][k]["t"] >= 0]))
        if "manifold" in res:
            m = res["manifold"][k]
            row[f"circcorr_{k}"] = m["circ_corr"]
            row[f"circcorrshuf_{k}"] = m["circ_corr_shuf"]
            row[f"radcv_{k}"] = m["radial_cv"]
            row[f"radcvshuf_{k}"] = m["radial_cv_shuf"]
            row[f"resvar1_{k}"] = float(m["resvar"][0])
            row[f"resvar2_{k}"] = float(m["resvar"][1])
            row[f"pca2_{k}"] = float(m["pca"][:2].sum())
            row[f"pca2shuf_{k}"] = float(m["pca_shuf"][:2].sum())
    return row


def _align_wake(tc, pref_dir, cells, units):
    """Wake tuning curves of the same cells, aligned the same way for comparison."""
    n_bins = tc.shape[0]
    pos = {int(u): i for i, u in enumerate(np.asarray(units.index))}
    out = []
    for c in cells:
        s = int(round(pref_dir[pos[int(c)]] / (2 * np.pi) * n_bins))
        out.append(np.roll(np.roll(tc[int(c)].values, -s), n_bins // 2))
    return np.array(out)


def _corr_nan(a, b):
    """Pearson r ignoring bins where either curve is undefined."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _decode_error(dec, hd, ep, bin_size):
    """Signed decoding error, wrapped to (-pi, pi]."""
    true = hd.restrict(ep).bin_average(bin_size, ep)
    df = pd.DataFrame({"dec": dec.d}, index=np.round(dec.t, 6)).join(
        pd.DataFrame({"true": true.d}, index=np.round(true.t, 6)), how="inner"
    ).dropna()
    return H.circ_diff(df["dec"].values, df["true"].values)

# %% [markdown]
# ## Figure code

# %%
STATES = ["wake", "REM", "non-REM"]
SCOL = {"wake": "#2a7f62", "REM": "#c0392b", "non-REM": "#2c5f9e"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})


def _busiest_window(hd, ep, dur):
    """Start time of the ``dur``-second window inside ``ep`` with the most head turning."""
    best, best_t = -1, ep.start[0]
    for s, e in zip(ep.start, ep.end):
        t = s
        while t + dur <= e:
            h = hd.restrict(H.nap.IntervalSet(start=t, end=t + dur))
            if len(h) > 10:
                v = np.abs(H.circ_diff(h.d[1:], h.d[:-1])).sum()
                if v > best:
                    best, best_t = v, t
            t += dur / 2
    return best_t


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- figure 1
def fig_overview(res, path="fig01_dataset_overview.png"):
    hd, pos, states = res["hd"], res["position"], res["states"]
    expl = res["epochs"]["wake"]
    fig = plt.figure(figsize=(12, 8.5))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 0.85, 1.3],
                  hspace=0.55, wspace=0.32)

    # session structure
    ax = fig.add_subplot(gs[0, :])
    for lab, col in [("Awake", "#bdbdbd"), ("REM", SCOL["REM"]), ("Non-REM", SCOL["non-REM"])]:
        ep = states[lab]
        for s, e in zip(ep.start, ep.end):
            ax.axvspan(s / 60, e / 60, ymin=0.55, ymax=1.0, color=col, lw=0)
    for s, e in zip(expl.start, expl.end):
        ax.axvspan(s / 60, e / 60, ymin=0.0, ymax=0.45, color=SCOL["wake"], lw=0)
    ax.set_xlim(0, hd.t[-1] / 60)
    ax.set_yticks([0.22, 0.78])
    ax.set_yticklabels(["open field", "scored state"])
    ax.set_xlabel("time in session (min)")
    ax.set_title(f"{res['name']}: session structure "
                 f"({len(res['hd_ids'])} head-direction cells of {res['n_units']} units)")
    handles = [plt.Line2D([], [], color=c, lw=6, label=l) for l, c in
               [("awake", "#bdbdbd"), ("REM", SCOL["REM"]), ("non-REM", SCOL["non-REM"]),
                ("open-field foraging", SCOL["wake"])]]
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)

    # trajectory
    ax = fig.add_subplot(gs[1, 0])
    p = pos.restrict(expl)
    ax.plot(p["x"].d, p["y"].d, lw=0.3, color="0.35")
    ax.set_aspect("equal")
    ax.set_title("open-field trajectory")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    # head direction over the 60 s of foraging with the most turning
    t0 = _busiest_window(hd, expl, 60.0)
    win = H.nap.IntervalSet(start=t0, end=t0 + 60)
    ax = fig.add_subplot(gs[1, 1:])
    h = hd.restrict(win)
    ax.plot(h.t - t0, np.degrees(h.d), ".", ms=1.2, color="0.2")
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("head direction (deg)")
    ax.set_title("tracked head direction (60 s of foraging)")

    # raster sorted by preferred direction
    ax = fig.add_subplot(gs[2, :])
    order = np.argsort(res["pref"])
    ids = np.array(res["hd_ids"])[order]
    units = res["_units"]
    for row, u in enumerate(ids):
        st = units[int(u)].restrict(win).t - t0
        ax.plot(st, np.full_like(st, row), "|", ms=3.2, color="0.15", mew=0.6)
    ax2 = ax.twinx()
    ax2.plot(h.t - t0, np.degrees(h.d), ".", ms=1.6, color=SCOL["REM"], alpha=0.75)
    ax2.set_ylim(0, 360)
    ax2.set_ylabel("head direction (deg)", color=SCOL["REM"])
    ax2.tick_params(axis="y", colors=SCOL["REM"])
    ax2.spines["right"].set_visible(True)
    ax.set_ylim(-1, len(ids))
    ax.set_xlim(0, 60)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("HD cell (sorted by\npreferred direction)")
    ax.set_title("head-direction cells sweep through the population as the animal turns")
    _save(fig, path)


# ---------------------------------------------------------------- figure 2
def fig_tuning(res, path="fig02_tuning_curves.png"):
    sel, tc = res["sel"], res["tc"]
    order = np.argsort(res["pref"])
    ids = np.array(res["hd_ids"])[order]
    n = len(ids)
    ncol = 7
    nrow = int(np.ceil(n / ncol)) + 1
    fig = plt.figure(figsize=(13, 2.1 * nrow))
    gs = GridSpec(nrow, ncol, figure=fig, hspace=0.75, wspace=0.45)
    ang = tc.index.values
    ang_c = np.append(ang, ang[0])
    for i, u in enumerate(ids):
        ax = fig.add_subplot(gs[i // ncol, i % ncol], projection="polar")
        r = tc[int(u)].values
        ax.plot(ang_c, np.append(r, r[0]), color=SCOL["wake"], lw=1.3)
        ax.fill(ang_c, np.append(r, r[0]), color=SCOL["wake"], alpha=0.2)
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"#{int(u)}  {r.max():.0f} Hz\nR={sel.loc[int(u),'mvl']:.2f}",
                     fontsize=7.5, pad=6)
    # mean vector length vs shuffle
    ax = fig.add_subplot(gs[nrow - 1, :3])
    is_hd = sel.index.isin(res["hd_ids"])
    ax.scatter(sel.mvl_null_p99[~is_hd], sel.mvl[~is_hd], s=16, color="0.6",
               label="other units")
    ax.scatter(sel.mvl_null_p99[is_hd], sel.mvl[is_hd], s=18, color=SCOL["wake"],
               label="HD cells")
    lim = [0, max(sel.mvl.max(), sel.mvl_null_p99.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("99th pct of time-shift shuffle")
    ax.set_ylabel("observed mean vector length")
    ax.set_title("HD-cell criterion (p<0.01, R>0.3, rate>0.5 Hz)")
    ax.legend(frameon=False, fontsize=8)

    # tuning heat map
    ax = fig.add_subplot(gs[nrow - 1, 3:])
    M = tc[[int(u) for u in ids]].values.T
    M = M / np.nanmax(M, axis=1, keepdims=True)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 360, 0, n])
    ax.set_xlabel("head direction (deg)")
    ax.set_ylabel("HD cell (sorted)")
    ax.set_title("normalised tuning curves tile the circle")
    plt.colorbar(im, ax=ax, label="norm. rate", fraction=0.045)
    _save(fig, path)


# ---------------------------------------------------------------- figure 3
def fig_stability(res, path="fig03_tuning_stability.png"):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    p1, p2 = np.degrees(res["pref_block1"]), np.degrees(res["pref_block2"])
    ax = axes[0]
    ax.scatter(p1, p2, s=22, color=SCOL["wake"])
    ax.plot([0, 360], [0, 360], "k--", lw=0.8)
    ax.set_xlim(0, 360); ax.set_ylim(0, 360)
    ax.set_xlabel("preferred direction, 1st block (deg)")
    ax.set_ylabel("2nd block (deg)")
    cc = A.circ_corr(res["pref_block1"], res["pref_block2"])
    ax.set_title(f"tuning is stable across\nforaging blocks (circ. r = {cc:.3f})")

    ax = axes[1]
    shift = H.circ_diff(res["pref_block2"], res["pref_block1"])
    rot = np.angle(np.mean(np.exp(1j * shift)))
    ax.hist(np.degrees(shift), bins=np.arange(-180, 185, 10), color=SCOL["wake"])
    ax.axvline(np.degrees(rot), color="k", lw=1.2, ls="--")
    ax.set_xlabel("signed shift in preferred direction (deg)")
    ax.set_ylabel("cells")
    ax.set_title(f"common rotation {np.degrees(rot):+.0f} deg,\nresidual "
                 f"{np.degrees(np.median(np.abs(H.circ_diff(shift, rot)))):.0f} deg")

    ax = axes[2]
    err = np.degrees(res["wake_decode_err"])
    erra = np.degrees(res["wake_decode_err_aligned"])
    ax.hist(err, bins=np.arange(0, 185, 5), color="0.75", density=True, label="raw")
    ax.hist(erra, bins=np.arange(0, 185, 5), color="0.35", density=True,
            histtype="step", lw=1.6, label="after removing offset")
    ax.axvline(np.median(erra), color=SCOL["REM"], lw=1.5)
    ax.set_xlabel("|decoding error| (deg)")
    ax.set_ylabel("density")
    ax.set_title(f"cross-validated wake decoding\nmedian error {np.median(erra):.0f} deg "
                 f"(raw {np.median(err):.0f})")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[3]
    rv = res.get("revisit")
    if rv is None:
        ax.axis("off")
    else:
        pr = np.degrees(rv["pref"])
        rot = np.angle(np.mean(np.exp(1j * H.circ_diff(rv["pref"], res["pref"]))))
        resid = np.degrees(np.abs(H.circ_diff(rv["pref"], res["pref"] + rot)))
        ax.scatter(np.degrees(res["pref"]), pr, s=22, color="#8e44ad")
        x = np.array([0, 360])
        for off in (-360, 0, 360):
            ax.plot(x, x + np.degrees(rot) + off, color="#8e44ad", lw=1, ls="-")
        ax.plot([0, 360], [0, 360], "k--", lw=0.8)
        ax.set_xlim(0, 360); ax.set_ylim(0, 360)
        ax.set_xlabel("preferred direction, main visit (deg)")
        ax.set_ylabel(f"revisit {rv['gap_h']:.1f} h later (deg)")
        ax.set_title(f"the map rotates as a whole\n"
                     f"({np.degrees(rot) % 360:.0f} deg shift, residual "
                     f"{np.median(resid):.0f} deg)")
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------- figure 4/5
def fig_correlation_structure(res, path="fig04_correlation_structure.png"):
    order = np.argsort(res["pref"])
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.6, wspace=0.38, height_ratios=[1, 0.95])
    for j, k in enumerate(STATES):
        ax = fig.add_subplot(gs[0, j])
        C = res["corr"][k]["C"][np.ix_(order, order)]
        v = np.nanpercentile(np.abs(C - np.diag(np.diag(C))), 99)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        ax.set_title(f"{k}\npairwise correlation")
        ax.set_xlabel("cell (sorted by preferred dir.)")
        if j == 0:
            ax.set_ylabel("cell (sorted by preferred dir.)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 0])
    for k in STATES:
        c, m, s, _ = res["corr"][k]["profile"]
        ax.errorbar(np.degrees(c), m, s, color=SCOL[k], label=k, lw=1.6, capsize=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("|difference in preferred direction| (deg)")
    ax.set_ylabel("correlation (200 ms bins)")
    ax.set_title("HD cells")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    for k in STATES:
        if k in res["corr_nonhd"]:
            c, m, s, _ = res["corr_nonhd"][k]["profile"]
            ax.errorbar(np.degrees(c), m, s, color=SCOL[k], label=k, lw=1.6, capsize=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("|difference in preferred direction| (deg)")
    ax.set_ylabel("correlation (200 ms bins)")
    ax.set_title("control: non-HD units\n(nominal preferred directions)")

    ax = fig.add_subplot(gs[1, 2])
    rw = res["corr"]["wake"]["r"]
    for k in ["REM", "non-REM"]:
        ax.scatter(rw, res["corr"][k]["r"], s=9, alpha=0.55, color=SCOL[k],
                   label=f"{k}  r = {res['corr_across_states'][k]:.2f}")
    lim = [min(rw.min(), -0.3), max(rw.max(), 0.6)]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("pair correlation, wake")
    ax.set_ylabel("pair correlation, sleep")
    ax.set_title("the correlation pattern survives\ninto sleep pair by pair")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, path)


# ---------------------------------------------------------------- figure 5
def fig_internal_tuning(res, path="fig05_internal_tuning.png"):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    n_bins = res["tc"].shape[0]
    off = np.degrees(np.linspace(-np.pi, np.pi, n_bins, endpoint=False))
    for j, k in enumerate(STATES):
        ax = axes[j]
        c = res["coherence"][k]
        wa = c["wake_aligned"] / np.nanmax(c["wake_aligned"], axis=1, keepdims=True)
        ia = c["ic_aligned"] / np.nanmax(c["ic_aligned"], axis=1, keepdims=True)
        ax.plot(off, np.nanmean(wa, 0), color="0.35", lw=1.8, label="wake (measured HD)")
        m, s = np.nanmean(ia, 0), np.nanstd(ia, 0) / np.sqrt(len(ia))
        ax.plot(off, m, color=SCOL[k], lw=1.8, label=f"{k} (internal angle)")
        ax.fill_between(off, m - s, m + s, color=SCOL[k], alpha=0.25, lw=0)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("offset from preferred direction (deg)")
        if j == 0:
            ax.set_ylabel("normalised rate")
        ax.set_title(f"{k}\nshape match r = {np.nanmean(c['r_shape']):.2f}")
        ax.set_ylim(0, 1.12)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax = axes[3]
    x = np.arange(len(STATES))
    obs = [np.nanmean(res["coherence"][k]["r_obs"]) for k in STATES]
    sem = [np.nanstd(res["coherence"][k]["r_obs"]) / np.sqrt(len(res["coherence"][k]["r_obs"]))
           for k in STATES]
    ctl = [np.nanmean(res["coherence"][k]["r_ctl"]) for k in STATES]
    prm = [np.nanmean(res["coherence"][k]["r_prm"]) for k in STATES]
    ax.bar(x - 0.26, obs, 0.25, yerr=sem, color=[SCOL[k] for k in STATES], capsize=3,
           label="held-out cells")
    ax.bar(x, prm, 0.25, color="0.55", label="wrong-cell tuning curve")
    ax.bar(x + 0.26, ctl, 0.25, color="0.8", label="time-shifted")
    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("r (predicted vs observed rate)")
    ax.set_title("one angle from half the cells\npredicts the other half")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------- figure 6
def fig_manifold(res, path="fig06_ring_manifold.png"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for j, k in enumerate(STATES):
        m = res["manifold"][k]
        ax = axes[0, j]
        col = m["true_at"] if k == "wake" else m["decoded_at"]
        lab = "measured HD" if k == "wake" else "internally decoded angle"
        sc = ax.scatter(m["Y"][:, 0], m["Y"][:, 1], c=np.degrees(col), cmap="hsv",
                        s=5, vmin=0, vmax=360)
        ax.set_aspect("equal")
        ax.set_title(f"{k}  (n = {m['n_states']} bins)\n"
                     f"ring match = {m['circ_corr']:.2f}, radial CV = {m['radial_cv']:.2f}")
        ax.set_xlabel("Isomap 1")
        if j == 0:
            ax.set_ylabel("Isomap 2")
        cb = plt.colorbar(sc, ax=ax, fraction=0.046)
        cb.set_label(lab, fontsize=7.5)

        ax = axes[1, j]
        ax.scatter(m["Y_shuf"][:, 0], m["Y_shuf"][:, 1], c=np.degrees(col), cmap="hsv",
                   s=5, vmin=0, vmax=360)
        ax.set_aspect("equal")
        ax.set_title(f"{k}: cell-shuffled control\n"
                     f"ring match = {m['circ_corr_shuf']:.2f}, "
                     f"radial CV = {m['radial_cv_shuf']:.2f}")
        ax.set_xlabel("Isomap 1")
        if j == 0:
            ax.set_ylabel("Isomap 2")
    fig.suptitle("population states of HD cells lie on a ring in every state", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, path)


# ---------------------------------------------------------------- figure 7
def fig_dimensionality(res, path="fig07_dimensionality.png"):
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    ax = axes[0]
    for k in STATES:
        m = res["manifold"][k]
        ax.plot(m["dims"], m["resvar"], "-o", color=SCOL[k], ms=4, label=k)
        ax.plot(m["dims"], m["resvar_shuf"], "--", color=SCOL[k], alpha=0.5, lw=1)
    ax.set_xlabel("Isomap embedding dimension")
    ax.set_ylabel("residual variance")
    ax.set_title("2 coordinates suffice\n(dashed: cell-shuffled)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for k in STATES:
        m = res["manifold"][k]
        ax.plot(np.arange(1, len(m["pca"]) + 1), np.cumsum(m["pca"]), "-o",
                color=SCOL[k], ms=4, label=k)
        ax.plot(np.arange(1, len(m["pca_shuf"]) + 1), np.cumsum(m["pca_shuf"]), "--",
                color=SCOL[k], alpha=0.5, lw=1)
    ax.set_xlabel("principal component")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title("PCA spectrum")

    ax = axes[2]
    x = np.arange(len(STATES))
    ax.bar(x - 0.19, [res["manifold"][k]["radial_cv"] for k in STATES], 0.36,
           color=[SCOL[k] for k in STATES], label="data")
    ax.bar(x + 0.19, [res["manifold"][k]["radial_cv_shuf"] for k in STATES], 0.36,
           color="0.75", label="cell-shuffled")
    ax.set_xticks(x); ax.set_xticklabels(STATES)
    ax.set_ylabel("radial CV of embedding")
    ax.set_title("states sit at a\nconstant radius (hollow ring)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[3]
    ax.bar(x - 0.19, [res["manifold"][k]["circ_corr"] for k in STATES], 0.36,
           color=[SCOL[k] for k in STATES], label="data")
    ax.bar(x + 0.19, [res["manifold"][k]["circ_corr_shuf"] for k in STATES], 0.36,
           color="0.75", label="cell-shuffled")
    ax.set_xticks(x); ax.set_xticklabels(STATES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ring-angle match")
    ax.set_title("ring angle matches the\ndecoded head direction")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------- figure 8
def fig_bump_dynamics(res, path="fig08_bump_dynamics.png"):
    fig = plt.figure(figsize=(13, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.3, height_ratios=[1.1, 1])
    for j, k in enumerate(STATES):
        d = res["decode"][k]
        proba = d["proba"]
        ep = res["epochs"][k]
        # pick a window well inside a long bout
        lens = ep.end - ep.start
        i = int(np.argmax(lens))
        t0 = ep.start[i] + min(5, lens[i] / 4)
        t1 = min(t0 + 25, ep.end[i])
        sub = proba.restrict(H.nap.IntervalSet(start=t0, end=t1))
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(sub.values.T, aspect="auto", origin="lower", cmap="magma",
                  extent=[0, t1 - t0, 0, 360], vmin=0,
                  vmax=np.percentile(sub.values, 99.5))
        dec = d["decoded"].restrict(H.nap.IntervalSet(start=t0, end=t1))
        ax.plot(dec.t - t0, np.degrees(dec.d), ".", ms=1.6, color="w", alpha=0.8)
        if k == "wake":
            true = res["hd"].restrict(H.nap.IntervalSet(start=t0, end=t1))
            ax.plot(true.t - t0, np.degrees(true.d), ".", ms=1.2, color="#39ff14",
                    alpha=0.9, label="measured HD")
            ax.legend(frameon=False, fontsize=7.5, labelcolor="w", loc="upper right")
        ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel("head direction (deg)")
        ax.set_title(f"{k}: posterior over the ring")

    ax = fig.add_subplot(gs[1, 0])
    bs = res["bin_size"]
    bins = np.arange(0, 185, 5)
    for k in STATES:
        st = np.degrees(res["decode"][k]["step"])
        ax.hist(st, bins=bins, density=True, histtype="step", lw=1.7,
                color=SCOL[k], label=k)
    ax.hist(np.degrees(res["decode"]["non-REM"]["step_null"]), bins=bins, density=True,
            histtype="stepfilled", color="0.8", zorder=0, label="shuffled time")
    ax.set_xlabel(f"|change in decoded angle| per {int(bs*1000)} ms bin (deg)")
    ax.set_ylabel("density")
    ax.set_title("the internal angle moves in small steps")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(STATES))
    med = [np.degrees(np.median(res["decode"][k]["step"])) / bs for k in STATES]
    nul = [np.degrees(np.median(res["decode"][k]["step_null"])) / bs for k in STATES]
    ax.bar(x - 0.19, med, 0.36, color=[SCOL[k] for k in STATES], label="data")
    ax.bar(x + 0.19, nul, 0.36, color="0.75", label="shuffled time")
    ax.set_xticks(x); ax.set_xticklabels(STATES)
    ax.set_ylabel("median |angular velocity| (deg/s)")
    ax.set_title("drift speed of the internal angle")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    for k in STATES:
        a = res["autocorr"][k]
        ax.plot(a["lags"], a["ac"], color=SCOL[k], lw=1.7,
                label=f"{k}  (half-decay {a['halfwidth']:.2f} s)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel(r"$\langle \cos[\theta(t+\tau)-\theta(t)] \rangle$")
    ax.set_title("the internal angle decorrelates\n~10x faster in non-REM")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------- figure 9
def fig_sleep_sweeps(res, path="fig09_sleep_sweeps.png",
                     offsets=(50, 150, 250), dur=6.0):
    """Fine-grained decoded trajectories inside the longest sleep bouts."""
    fig, axes = plt.subplots(len(offsets), 2, figsize=(13, 2.1 * len(offsets)),
                             sharex=True)
    for c, k in enumerate(["REM", "non-REM"]):
        ep = res["epochs"][k]
        lens = ep.end - ep.start
        i = int(np.argmax(lens))
        proba = res["fine"][k]["proba"]
        for r, off in enumerate(offsets):
            ax = axes[r, c]
            t0 = ep.start[i] + off * min(1.0, (lens[i] - dur) / max(offsets))
            sub = proba.restrict(H.nap.IntervalSet(start=t0, end=t0 + dur))
            ax.imshow(sub.values.T, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, dur, 0, 360], vmin=0,
                      vmax=np.percentile(sub.values, 99.5))
            ax.set_ylabel("deg")
            if r == 0:
                ax.set_title(f"{k}: internal head direction, "
                             f"{int(res['fine'][k]['bin_size']*1000)} ms bins")
            if r == len(offsets) - 1:
                ax.set_xlabel("time (s)")
    fig.suptitle("during sleep the bump sweeps continuously around the ring", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, path)


# ---------------------------------------------------------------- figure 10
def fig_timescale(res, path="fig10_timescale.png"):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for j, k in enumerate(STATES):
        x = res["xcorr"][k]
        ax = axes[j]
        ax.plot(x["t"], x["near"], color=SCOL[k], lw=1.6,
                label=f"|dPD| < 45 deg (n={x['near_n']})")
        ax.plot(x["t"], x["far"], color="0.55", lw=1.6,
                label=f"|dPD| > 135 deg (n={x['far_n']})")
        ax.set_xlim(-3, 3)
        ax.axhline(1, color="k", lw=0.6, ls=":")
        ax.set_xlabel("lag (s)")
        if j == 0:
            ax.set_ylabel("normalised cross-correlation")
        ax.set_title(k)
        ax.legend(frameon=False, fontsize=7)
    ax = axes[3]
    for k in STATES:
        x = res["xcorr"][k]
        y = x["near"] - x["far"]
        y = y / np.max(y)
        hw = A.halfwidth(x["t"][x["t"] >= 0], y[x["t"] >= 0])
        ax.plot(x["t"], y, color=SCOL[k], lw=1.7, label=f"{k} (half-width {hw:.2f} s)")
    ax.set_xlim(-3, 3)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("normalised (near - far)")
    ax.set_title("same shape, compressed in time\nduring non-REM")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------- figure 11
def fig_multisession(df, path="fig11_multisession_summary.png"):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    x = np.arange(len(df))
    w = 0.27
    lab = [f"{r.session}\n({r.n_hd} HD cells)" for r in df.itertuples()]

    def ticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=6.5)

    ax = axes[0, 0]
    for i, k in enumerate(["REM", "non-REM"]):
        ax.bar(x + (i - 0.5) * w * 1.6, df[f"corrmatch_{k}"], w * 1.5,
               color=SCOL[k], label=k)
    ax.set_ylim(0, 1.15)
    ticks(ax)
    ax.set_ylabel("r (sleep vs wake pair correlations)")
    ax.set_title("correlation structure is preserved")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")

    ax = axes[0, 1]
    for i, k in enumerate(STATES):
        ax.bar(x + (i - 1) * w, df[f"shape_{k}"], w, color=SCOL[k], label=k)
    ax.set_ylim(0, 1.15)
    ticks(ax)
    ax.set_ylabel("r (internal vs wake tuning shape)")
    ax.set_title("held-out cells keep their place on the ring")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    ax = axes[0, 2]
    for i, k in enumerate(STATES):
        ax.bar(x + (i - 1) * w, df[f"circcorr_{k}"], w, color=SCOL[k], label=k)
        ax.plot(x + (i - 1) * w, df[f"circcorrshuf_{k}"], "k_", ms=7)
    ax.set_ylim(0, 1.15)
    ticks(ax)
    ax.set_ylabel("ring-angle match vs decoded HD")
    ax.set_title("ring geometry\n(black dashes: cell-shuffled control)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    ax = axes[1, 0]
    for i, k in enumerate(STATES):
        ax.bar(x + (i - 1) * w, df[f"xcorr_hw_{k}_s"], w, color=SCOL[k], label=k)
    ax.set_yscale("log")
    ticks(ax)
    ax.set_ylabel("cross-correlogram half-width (s)")
    ax.set_title("non-REM runs the same structure ~10x faster")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    ax = axes[1, 1]
    ax.bar(x - w / 2, df["block_rotation_deg"].apply(lambda v: min(v, 360 - v)), w,
           color="#8e44ad", label="common rotation")
    ax.bar(x + w / 2, df["block_resid_deg"], w, color="0.7",
           label="residual scatter")
    ticks(ax)
    ax.set_ylabel("degrees")
    ax.set_title("within-visit map drift is a rigid rotation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 2]
    ax.bar(x - w / 2, df["wake_decode_err_med_deg"], w, color="0.7", label="raw")
    ax.bar(x + w / 2, df["wake_decode_err_aligned_med_deg"], w, color=SCOL["wake"],
           label="after removing offset")
    ticks(ax)
    ax.set_ylabel("median |decoding error| (deg)")
    ax.set_title("cross-validated wake decoding")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, path)

# %% [markdown]
# ## Run the pipeline on the main session
#
# `Mouse28-140313` is used for the detailed figures: it has the largest number of
# simultaneously recorded HD cells and a long sleep session.

# %%
MAIN = "Mouse28-140313"
res = analyse_session(MAIN, SESSIONS[MAIN])
print({k: round(v) for k, v in res["durations"].items()})

# %% [markdown]
# ### Figure 1: what is in the recording
#
# The raster is the whole argument in miniature. Cells are ordered by the preferred
# direction measured over the whole foraging block, and the band of active cells moves
# up and down the ordering as the animal turns, without breaking apart. That is what a
# bump moving on a ring looks like when the ring is cut open and drawn as a line.

# %%
fig_overview(res)

# %% [markdown]
# ### Figure 2: head-direction cells and how they were selected
#
# Each cell is compared against its own circular time-shift null. Cells above the 99th
# percentile of that null with a mean vector length above 0.3 and a mean rate above
# 0.5 Hz are taken as HD cells. Their preferred directions tile the circle without gaps,
# which is what makes the population usable as a ring coordinate.

# %%
fig_tuning(res)

# %% [markdown]
# ### Figure 3: the tuning is stable, and the map rotates as a whole
#
# Within one visit to the open field the preferred directions barely move. Between
# visits separated by hours the entire map rotates by a common angle while the relative
# offsets are preserved. That rigid rotation is itself an attractor signature: the ring
# keeps its shape and loses only its anchoring to the room. It is also the reason
# tuning curves here are estimated from a single visit.
#
# The right-hand panel of the cross-validated decoding shows that wake head direction is
# recovered from these cells with a median error of about 12 degrees, which sets the
# scale for what the same decoder can be expected to do during sleep.

# %%
fig_stability(res)

# %% [markdown]
# ### Figure 4: the correlation structure is the ring, and it survives sleep
#
# Sorting the cells by preferred direction turns the correlation matrix into a band
# along the diagonal that wraps around at the corners. Cells with nearby preferred
# directions are positively correlated, cells with opposite preferred directions are
# anticorrelated, and the transition happens near 60-90 degrees, the width of a single
# tuning curve.
#
# The same matrix computed during REM and during non-REM has the same band and the same
# wrap-around, and the correlation of individual pair values between wake and sleep is
# above 0.9. Non-HD units, assigned nominal preferred directions by the same procedure,
# show no comparable structure.

# %%
fig_correlation_structure(res)

# %% [markdown]
# ### Figure 5: cells keep their place on the ring when the animal is asleep
#
# Here half the HD cells are used to decode an angle and the other half are scored
# against it. The resulting internal tuning curves during REM and non-REM peak at the
# cell's wake preferred direction and have close to the wake shape, even though the
# animal is motionless and the angle is entirely internally generated.

# %%
fig_internal_tuning(res)

# %% [markdown]
# ### Figure 6: the population state cloud is a ring in every state
#
# Isomap on the population states recovers a ring in wake and in REM, with the angular
# position around the ring matching the measured (wake) or decoded (sleep) head
# direction. Non-REM gives a noisier, partly filled ring: the angular ordering is still
# clear, but with 200 ms bins and a bump that moves an order of magnitude faster (see
# below), individual states are much noisier.
#
# Shuffling each cell's activity in time independently destroys the ring completely
# while leaving every single-cell statistic untouched, which is the control that
# separates population structure from the properties of individual neurons.

# %%
fig_manifold(res)

# %% [markdown]
# ### Figure 7: how many dimensions the state cloud needs
#
# Isomap residual variance falls sharply from one to two embedding dimensions and is
# then essentially flat, which is the signature of a closed one-dimensional curve: it is
# intrinsically 1-D but cannot be laid out without two coordinates. The radial spread of
# the embedding is small compared with the shuffled control, meaning the states sit near
# a fixed distance from the centre rather than filling a disc.

# %%
fig_dimensionality(res)

# %% [markdown]
# ### Figure 8: the internal angle moves continuously
#
# Decoding with the wake tuning curves gives a single localised posterior in every
# state. Its step size from bin to bin is far below what would be expected if the
# population jumped to unrelated positions, in sleep as well as in wake. The
# autocorrelation of the decoded angle shows the difference between the sleep states:
# REM drifts on roughly the wake timescale, non-REM about ten times faster.

# %%
fig_bump_dynamics(res)

# %% [markdown]
# ### Figure 9: watching the bump sweep during sleep
#
# At 40 ms resolution, inside the longest REM and non-REM bouts, the posterior is a
# narrow ridge that moves smoothly and passes through 0/360 without a break. Nothing in
# the decoder enforces continuity: each bin is decoded independently under a uniform
# prior, so a population that was not confined to the ring would produce a scatter of
# unrelated angles.

# %%
fig_sleep_sweeps(res)

# %% [markdown]
# ### Figure 10: the same structure, compressed in time during non-REM
#
# This uses spike times directly, with no decoding. Cross-correlograms of similarly
# tuned pairs are broad in wake and REM and roughly ten to twenty times narrower during
# non-REM, while oppositely tuned pairs dip at zero lag. The shape of the correlation
# structure is conserved; only its timescale changes.

# %%
fig_timescale(res)

# %% [markdown]
# ## The same analysis across sessions and animals
#
# Six sessions from five animals. Nothing is tuned per session: the same thresholds,
# bin sizes and epoch definitions are used throughout.

# %%
rows = []
for name, aid in tqdm(list(SESSIONS.items()), desc="sessions"):
    r = analyse_session(name, aid)
    rows.append(summarise(r))
    if name == MAIN:
        res = r
summary = pd.DataFrame(rows)
summary.to_csv("session_summary.csv", index=False)
cols = ["session", "n_units", "n_hd", "wake_decode_err_med_deg",
        "corrmatch_REM", "corrmatch_non-REM",
        "shape_REM", "shape_non-REM", "coh_REM", "coh_non-REM"]
summary[cols].round(3)

# %%
fig_multisession(summary)

# %%
tsc = summary[["session", "xcorr_hw_wake_s", "xcorr_hw_REM_s", "xcorr_hw_non-REM_s",
               "tau_wake_s", "tau_REM_s", "tau_non-REM_s",
               "step_wake_degs", "step_REM_degs", "step_non-REM_degs",
               "stepnull_wake_degs"]].round(3)
print(tsc.to_string(index=False))

# %% [markdown]
# ## Conclusions
#
# The head-direction population recorded here behaves as a continuous ring attractor,
# and the ring is maintained internally during sleep.
#
# The one-dimensional structure is visible three independent ways. Sorting cells by
# preferred direction turns the pairwise correlation matrix into a wrapped band, with
# positive correlation out to roughly one tuning width and anticorrelation beyond it.
# Isomap on the population states returns a closed ring whose angular coordinate tracks
# head direction, with residual variance that drops at two embedding dimensions and then
# stops improving, and with a radial spread far below the cell-shuffled control.
# Decoding recovers a single localised bump whose position changes by small steps from
# bin to bin rather than jumping around the circle.
#
# That structure does not depend on the senses. In REM and in non-REM the correlation
# matrix is the same one, pair by pair, as in wake. Cells scored against an angle
# decoded from other cells still fire at their wake preferred direction with close to
# their wake tuning shape, even though the animal is asleep and the angle is generated
# entirely inside the brain. The Isomap ring is still there in REM and, more noisily, in
# non-REM. The bump still moves continuously, wrapping through 0/360 rather than
# teleporting.
#
# The one thing that changes between states is the timescale. Cross-correlograms of
# similarly tuned pairs are an order of magnitude narrower during non-REM than during
# wake or REM, and the decoded angle decorrelates correspondingly faster, so the same
# trajectory on the same ring is being run through at roughly ten times normal speed.
# The two other things that change are also informative and are not failures of the
# model: between visits to the open field hours apart the whole map rotates rigidly, and
# during non-REM down states the population falls silent so that the ring coordinate is
# briefly undefined. Both are what one expects of an internally maintained ring that has
# lost its anchor to the world rather than lost its shape.
