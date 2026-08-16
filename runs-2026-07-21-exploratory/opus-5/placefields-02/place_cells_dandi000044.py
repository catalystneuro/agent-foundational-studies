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
# # Hippocampal place cells in rat dorsal CA1 (DANDI:000044)
#
# This notebook demonstrates hippocampal place cells using real extracellular
# recordings streamed from the DANDI Archive.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark &
# Buzsáki (2016), *"Diversity in neural firing dynamics supports both rigid and
# learned hippocampal sequences"*, Science 351:1440-1443 (the `hc-11` dataset).
# Four Long-Evans rats (Achilles, Buddy, Cicero, Gatsby) were implanted with
# bilateral silicon probes in dorsal CA1 and ran back and forth on a novel linear
# track for water reward at the two ends, flanked by pre- and post-run sleep
# sessions. Each NWB file contains sorted spike times with an excitatory /
# inhibitory label, 1250 Hz LFP on 128 channels, 2D and linearized position at
# ~39 Hz, and epoch boundaries.
#
# **What is demonstrated.**
#
# 1. Individual CA1 pyramidal cells fire in restricted portions of the track
#    (place fields), and the spatial selectivity far exceeds a circular-shift
#    null distribution.
# 2. Place fields tile the track, and are largely direction-selective, as
#    expected for a linear track.
# 3. The population code is accurate: a naive-Bayes decoder trained on half the
#    running passes reconstructs position on the held-out passes to within a few
#    centimetres.
# 4. Spikes precess to earlier theta phases as the animal crosses the field
#    (theta phase precession), the classic single-cell signature that
#    distinguishes hippocampal place coding from a generic sensory rate code.
#
# **Access.** NWB files are 5-9 GB each because they contain the full LFP, so
# everything is streamed with `remfile` + a local disk cache; only the units
# table, the position series, and one or a few LFP channels are ever transferred.
#
# **Scope note.** The dandiset has eight sessions. Three of them (Achilles
# 11012013, Cicero 09102014, Gatsby 08282013) used a circular maze rather than a
# linear track; the directional-pass logic below assumes a linear track with two
# ends, so those three are excluded and reported as excluded. The remaining five
# sessions (four rats) form the population analysed here.

# %%
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import norm, wilcoxon, mannwhitneyu
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
RNG = np.random.default_rng(20260731)

# %% [markdown]
# ## Configuration

# %%
ASSET_URL = ("https://api.dandiarchive.org/api/dandisets/000044/versions/draft/"
             "assets/{}/download/")
CACHE_DIR = "/tmp/remfile_cache_000044"

SESSIONS = {                                  # asset id -> session label
    "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d": "Achilles_10252013",
    "080a847a-5211-4101-bc05-fd45b7e77dbd": "Achilles_11012013",
    "185b8a36-d671-4688-ba05-9e89a902c486": "Buddy_06272013",
    "97252767-5e90-45cf-a1b8-8cbff2f2a3a3": "Cicero_09012014",
    "d6b8092b-94e2-4e37-8660-86f9e856aec8": "Cicero_09102014",
    "d9a1e010-af4a-4fba-812d-66e2d1a71e35": "Cicero_09172014",
    "93569d6c-781f-4422-938e-e935a62863de": "Gatsby_08022013",
    "402f78e1-9e7d-486c-8822-93d538ed6ccc": "Gatsby_08282013",
}
PROTOTYPE = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"   # Achilles, 1.6 m linear track

N_BINS = 40             # spatial bins across the track
SPEED_THRESH = 0.10     # m/s; running threshold
MIN_PASS_LEN = 0.6      # m; minimum net displacement of a directional pass
MIN_MEAN_RATE = 0.2     # Hz; minimum in-run rate for a unit to enter the analysis
MIN_SPIKES = 30         # in-run spikes needed for the shuffle test to have power
N_SHUFFLES = 500        # circular shifts for the spatial-information null
DECODE_BIN = 0.2        # s; decoding time bin
LFP_RATE = 1250.0
THETA_BAND = (6.0, 10.0)

# place-cell criteria (all three must hold, per direction)
P_CRIT, PEAK_CRIT, STAB_CRIT = 0.01, 1.0, 0.3

# %% [markdown]
# ## Loading helpers
#
# `remfile` turns the S3 object into a random-access file, so `h5py` reads only
# the byte ranges it needs; the disk cache makes re-runs cheap.

# %%
def open_session(asset_id):
    rem = remfile.File(ASSET_URL.format(asset_id),
                       disk_cache=remfile.DiskCache(CACHE_DIR))
    h5f = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    return io.read(), io, h5f


def maze_type(nwbfile):
    """'linear' or 'circular', plus the name of the linearized-position interface."""
    names = list(nwbfile.processing["behavior"].data_interfaces)
    lin = [n for n in names if n.endswith("LinearizedPosition")][0]
    return ("circular" if "Circular" in lin else "linear"), lin


def get_maze_epoch(nwbfile):
    ep = nwbfile.intervals["epochs"].to_dataframe()
    row = ep[ep["label"].str.contains("Maze")].iloc[0]
    return nap.IntervalSet(start=row["start_time"], end=row["stop_time"])


def get_linear_position(nwbfile):
    """Linearized position (Tsd, NaNs dropped), sample interval, track length.

    Linearized position is NaN whenever the animal is off the linear portion of
    the maze (the reward / turnaround areas at the two ends), so dropping NaNs
    leaves exactly the on-track samples.
    """
    beh = nwbfile.processing["behavior"]
    _, lin_name = maze_type(nwbfile)
    iface = beh[lin_name]
    series = (list(iface.spatial_series.values()) if hasattr(iface, "spatial_series")
              else list(iface.time_series.values()))[0]
    d = np.asarray(series.data[:]).squeeze()
    t = series.starting_time + np.arange(len(d)) / series.rate
    ok = np.isfinite(d)
    track = float(np.ceil(np.nanmax(d) * 20) / 20)       # round up to nearest 5 cm
    return nap.Tsd(t=t[ok], d=d[ok]), 1.0 / series.rate, track


def get_2d_position(nwbfile):
    beh = nwbfile.processing["behavior"]
    name = [n for n in beh.data_interfaces
            if n.endswith("Position") and "Linearized" not in n][0]
    ss = list(beh[name].spatial_series.values())[0]
    d = np.asarray(ss.data[:])
    t = ss.starting_time + np.arange(len(d)) / ss.rate
    ok = np.isfinite(d).all(axis=1)
    return nap.TsdFrame(t=t[ok], d=d[ok], columns=["x", "y"])


def get_units(nwbfile, cell_type="excitatory"):
    df = nwbfile.units.to_dataframe()
    if cell_type is not None:
        df = df[df["cell_type"] == cell_type]
    spikes = {int(u): nap.Ts(t=np.sort(np.asarray(df.loc[u, "spike_times"])))
              for u in df.index}
    return nap.TsGroup(spikes, metadata={"location": df["location"].values,
                                         "shank_id": df["shank_id"].values})


# %% [markdown]
# ## Behaviour segmentation
#
# Position is differentiated inside each contiguous on-track segment (never
# across the long gaps the animal spends in the end zones), and a *pass* is a
# maximal stretch of samples with a constant sign of velocity above the running
# threshold and at least `MIN_PASS_LEN` of net displacement.

# %%
def segment_position(pos, dt, gap_tol=2.5, smooth_sigma_s=0.15):
    t = pos.t
    brk = np.where(np.diff(t) > gap_tol * dt)[0]
    seg_idx = np.split(np.arange(len(t)), brk + 1)
    vel = np.full(len(t), np.nan)
    sigma = max(smooth_sigma_s / dt, 0.5)
    for s in seg_idx:
        if len(s) < 5:
            continue
        vel[s] = np.gradient(gaussian_filter1d(pos.values[s], sigma, mode="nearest"),
                             t[s])
    return seg_idx, nap.Tsd(t=t, d=np.abs(vel)), vel


def directional_passes(pos, dt, seg_idx, vel,
                       speed_thresh=SPEED_THRESH, min_len=MIN_PASS_LEN):
    out = {"right": [], "left": []}
    for s in seg_idx:
        if len(s) < 5:
            continue
        v = vel[s]
        key = np.where(np.abs(v) > speed_thresh, np.sign(v), 0).astype(int)
        for r in np.split(np.arange(len(s)), np.where(np.diff(key) != 0)[0] + 1):
            if key[r[0]] == 0 or len(r) < 3:
                continue
            ii = s[r]
            if abs(pos.values[ii[-1]] - pos.values[ii[0]]) < min_len:
                continue
            out["right" if key[r[0]] > 0 else "left"].append(
                (pos.t[ii[0]] - dt / 2, pos.t[ii[-1]] + dt / 2))
    return {k: nap.IntervalSet(start=np.array(v).reshape(-1, 2)[:, 0],
                               end=np.array(v).reshape(-1, 2)[:, 1])
            for k, v in out.items()}


# %% [markdown]
# ## Rate maps and place-field metrics

# %%
def bin_edges(track, nb=N_BINS):
    return np.linspace(0, track, nb + 1)


def bin_centers(track, nb=N_BINS):
    e = bin_edges(track, nb)
    return (e[:-1] + e[1:]) / 2


def occupancy(pos, ep, dt, track, nb=N_BINS):
    occ, _ = np.histogram(pos.restrict(ep).values, bins=bin_edges(track, nb))
    return occ * dt


def tuning_curves(units, pos, ep, dt, track, nb=N_BINS, sigma_bins=1.0, min_occ=0.1):
    """Occupancy-normalised firing-rate maps in Hz, DataFrame [bin_center x unit]."""
    tc = nap.compute_1d_tuning_curves(group=units, feature=pos, nb_bins=nb,
                                      ep=ep, minmax=(0, track))
    occ = occupancy(pos, ep, dt, track, nb)
    tc[occ < min_occ] = np.nan
    if sigma_bins > 0:                     # NaN-aware Gaussian smoothing
        v = tc.values.copy()
        nanm = np.isnan(v)
        v[nanm] = 0.0
        num = gaussian_filter1d(v, sigma_bins, axis=0, mode="nearest")
        den = gaussian_filter1d((~nanm).astype(float), sigma_bins, axis=0,
                                mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            v = num / den
        v[den < 1e-6] = np.nan
        tc = pd.DataFrame(v, index=tc.index, columns=tc.columns)
    return tc


def spatial_information(rate_map, occ):
    """Skaggs spatial information, bits per spike."""
    r, o = np.asarray(rate_map, float), np.asarray(occ, float)
    m = np.isfinite(r) & (o > 0)
    if m.sum() < 2:
        return np.nan
    p, r = o[m] / o[m].sum(), r[m]
    mean_r = np.sum(p * r)
    if mean_r <= 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * (r / mean_r) * np.log2(r / mean_r)
    return float(np.nansum(term[r > 0]))


def sparsity(rate_map, occ):
    r, o = np.asarray(rate_map, float), np.asarray(occ, float)
    m = np.isfinite(r) & (o > 0)
    if m.sum() < 2:
        return np.nan
    p, r = o[m] / o[m].sum(), r[m]
    den = np.sum(p * r ** 2)
    return float(np.sum(p * r) ** 2 / den) if den > 0 else np.nan


def field_bounds(rate_map, frac=0.5):
    """Index bounds of the contiguous region around the peak above `frac`*peak."""
    r = np.asarray(rate_map, float)
    if not np.isfinite(r).any() or np.nanmax(r) <= 0:
        return None
    ipk = int(np.nanargmax(r))
    above = np.nan_to_num(r, nan=0.0) >= frac * np.nanmax(r)
    lo = hi = ipk
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(r) - 1 and above[hi + 1]:
        hi += 1
    return lo, ipk, hi


def field_properties(rate_map, centers, frac=0.5):
    b = field_bounds(rate_map, frac)
    if b is None:
        return np.nan, np.nan, np.nan
    lo, ipk, hi = b
    binw = centers[1] - centers[0]
    return (float(np.nanmax(rate_map)), float(centers[ipk]),
            float((hi - lo + 1) * binw))


# %% [markdown]
# ### Circular-shift null for spatial information
#
# Each unit's spike train is circularly shifted by a random offset inside the
# maze epoch. This preserves the burstiness and overall rate of the train but
# destroys its relationship to position. `FastTuner` pre-digitises the position
# track so that 500 shuffles x ~100 units run in seconds.

# %%
class FastTuner:
    def __init__(self, pos, ep, dt, track, nb=N_BINS):
        p = pos.restrict(ep)
        self.t, self.dt, self.nb = p.t, dt, nb
        self.bin = np.clip(np.digitize(p.values, bin_edges(track, nb)) - 1, 0, nb - 1)
        self.occ = np.bincount(self.bin, minlength=nb) * dt

    def rate_map(self, spike_times):
        if len(spike_times) == 0 or len(self.t) < 2:
            return np.full(self.nb, np.nan)
        i = np.clip(np.searchsorted(self.t, spike_times), 1, len(self.t) - 1)
        i = np.where(np.abs(spike_times - self.t[i - 1]) <
                     np.abs(spike_times - self.t[i]), i - 1, i)
        keep = np.abs(spike_times - self.t[i]) <= self.dt
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = np.bincount(self.bin[i[keep]], minlength=self.nb) / self.occ
        rm[self.occ <= 0] = np.nan
        return rm


def shuffle_spatial_info(units, tuner, maze_ep, n_shuffles=N_SHUFFLES, rng=None,
                         min_shift=20.0, show_progress=False):
    rng = rng or np.random.default_rng(0)
    t0, t1 = float(maze_ep.start[0]), float(maze_ep.end[0])
    T = t1 - t0
    keys = list(units.keys())
    raw = {u: np.asarray(units[u].restrict(maze_ep).t) for u in keys}
    null = {u: np.empty(n_shuffles) for u in keys}
    it = tqdm(range(n_shuffles), desc="shuffles", leave=False) if show_progress \
        else range(n_shuffles)
    for k in it:
        offs = rng.uniform(min_shift, T - min_shift, size=len(keys))
        for j, u in enumerate(keys):
            st = np.sort(t0 + np.mod(raw[u] - t0 + offs[j], T))
            null[u][k] = spatial_information(tuner.rate_map(st), tuner.occ)
    return null


# %% [markdown]
# ## Theta LFP and phase precession

# %%
def load_lfp_channels(h5f, channels, t_start, t_stop, rate=LFP_RATE):
    """Read selected LFP channels over a time window (volts).

    The LFP dataset is chunked one channel at a time, so pulling a few channels
    over the whole maze epoch streams only a few MB out of a multi-GB file.
    """
    ds = h5f["processing/ecephys/LFP/LFP/data"]
    conv = ds.attrs["conversion"]
    i0, i1 = int(t_start * rate), int(min(t_stop * rate, ds.shape[0]))
    data = np.stack([ds[i0:i1, c].astype(np.float32) * conv for c in channels], axis=1)
    return nap.TsdFrame(t=np.arange(i0, i1) / rate, d=data, columns=list(channels))


def pick_theta_channel(h5f, candidates, run_ep, rate=LFP_RATE):
    """Channel with the largest theta (6-10 Hz) / delta (2-4 Hz) power ratio."""
    lfp = load_lfp_channels(h5f, candidates, float(run_ep.start[0]),
                            float(run_ep.end[-1]), rate).restrict(run_ep)
    ratios = {}
    for j, c in enumerate(candidates):
        f, p = welch(np.asarray(lfp.values[:, j], float), fs=rate,
                     nperseg=int(4 * rate))
        ratios[c] = p[(f >= 6) & (f <= 10)].mean() / p[(f >= 2) & (f <= 4)].mean()
    return max(ratios, key=ratios.get), ratios


def theta_phase(lfp_1ch, rate=LFP_RATE, band=THETA_BAND):
    b, a = butter(3, [band[0] / (rate / 2), band[1] / (rate / 2)], btype="band")
    filt = filtfilt(b, a, np.asarray(lfp_1ch.values, float))
    return (nap.Tsd(t=lfp_1ch.t, d=np.mod(np.angle(hilbert(filt)), 2 * np.pi)),
            nap.Tsd(t=lfp_1ch.t, d=filt))


def circ_lin_corr(phase, x, slope_range=(-4 * np.pi, 4 * np.pi), n_slopes=2001):
    """Kempter et al. (2012) circular-linear correlation. -> rho, p, slope, offset."""
    phase, x = np.asarray(phase, float), np.asarray(x, float)
    n = len(phase)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan
    slopes = np.linspace(*slope_range, n_slopes)
    R = np.abs(np.mean(np.exp(1j * (phase[None, :] - slopes[:, None] * x[None, :])),
                       axis=1))
    a = slopes[int(np.argmax(R))]
    phi0 = np.angle(np.mean(np.exp(1j * (phase - a * x))))
    theta = np.mod(np.abs(a) * x, 2 * np.pi)   # |a| so that sign(rho) is meaningful
    pb = np.angle(np.sum(np.exp(1j * phase)) / n)
    tb = np.angle(np.sum(np.exp(1j * theta)) / n)
    sp, st_ = np.sin(phase - pb), np.sin(theta - tb)
    den = np.sqrt(np.sum(sp ** 2) * np.sum(st_ ** 2))
    rho = float(np.sum(sp * st_) / den) if den > 0 else np.nan
    lam20, lam02, lam22 = np.mean(sp ** 2), np.mean(st_ ** 2), np.mean(sp ** 2 * st_ ** 2)
    z = np.sqrt(n * lam20 * lam02 / lam22) * rho if lam22 > 0 else np.nan
    p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    return rho, p, float(a), float(phi0)


# %% [markdown]
# ## The per-session pipeline
#
# One function that runs everything for a single session and returns a plain
# dictionary of results, so the same code serves the detailed prototype figures
# and the multi-session population summary.

# %%
def analyse_session(asset_id, n_shuffles=N_SHUFFLES, do_theta=True,
                    keep_heavy=False, show_progress=False):
    label = SESSIONS[asset_id]
    nwbfile, io, h5f = open_session(asset_id)
    mtype, _ = maze_type(nwbfile)
    maze = get_maze_epoch(nwbfile)
    pos, dt, track = get_linear_position(nwbfile)
    units_all = get_units(nwbfile, "excitatory")
    n_inh = len(get_units(nwbfile, "inhibitory"))

    seg_idx, speed, vel = segment_position(pos, dt)
    passes = directional_passes(pos, dt, seg_idx, vel)
    run_ep = passes["right"].union(passes["left"])

    rates = dict(zip(units_all.keys(), np.asarray(units_all.restrict(run_ep).rates)))
    keep = [u for u in units_all.keys() if rates[u] >= MIN_MEAN_RATE]
    units = units_all[keep]
    centers = bin_centers(track)

    tc, occ, rows = {}, {}, []
    for d in ("right", "left"):
        ep = passes[d]
        tc[d] = tuning_curves(units, pos, ep, dt, track)
        occ[d] = occupancy(pos, ep, dt, track)
        tuner = FastTuner(pos, ep, dt, track)
        null = shuffle_spatial_info(units, tuner, maze, n_shuffles,
                                    rng=np.random.default_rng(1),
                                    show_progress=show_progress)
        odd = nap.IntervalSet(start=ep.start[0::2], end=ep.end[0::2])
        even = nap.IntervalSet(start=ep.start[1::2], end=ep.end[1::2])
        tc_o = tuning_curves(units, pos, odd, dt, track)
        tc_e = tuning_curves(units, pos, even, dt, track)
        for u in units.keys():
            rm = tuner.rate_map(np.asarray(units[u].restrict(ep).t))
            si = spatial_information(rm, tuner.occ)
            pk, loc, width = field_properties(tc[d][u].values, centers)
            a, b = tc_o[u].values, tc_e[u].values
            m = np.isfinite(a) & np.isfinite(b)
            stab = (float(np.corrcoef(a[m], b[m])[0, 1])
                    if m.sum() > 5 and np.std(a[m]) > 0 and np.std(b[m]) > 0 else np.nan)
            rows.append(dict(session=label, subject=label.split("_")[0], unit=u,
                             direction=d, maze=mtype, track=track,
                             mean_rate=float(units[u].restrict(ep).rate),
                             si=si, si_null=float(np.nanmean(null[u])),
                             p=float(np.mean(null[u] >= si)),
                             sparsity=sparsity(rm, tuner.occ),
                             peak_rate=pk, peak_loc=loc, field_width=width,
                             stability=stab,
                             n_spikes=int(len(units[u].restrict(ep)))))
    metrics = pd.DataFrame(rows)
    metrics["is_place"] = ((metrics.p < P_CRIT) & (metrics.peak_rate >= PEAK_CRIT)
                           & (metrics.stability > STAB_CRIT))
    # a unit-direction with very few in-run spikes cannot beat the shuffle even if
    # it is spatially tuned; flag it so the yield can also be reported on the
    # subset where the test has power
    metrics["testable"] = metrics.n_spikes >= MIN_SPIKES

    # ---- cross-validated Bayesian decoding: train on odd passes, test on even ----
    decoding, dec_keep = [], {}
    for d in ("right", "left"):
        ep = passes[d]
        pc = metrics[(metrics.direction == d) & metrics.is_place]["unit"].values
        if len(pc) < 5 or len(ep) < 6:
            continue
        train = nap.IntervalSet(start=ep.start[0::2], end=ep.end[0::2])
        test = nap.IntervalSet(start=ep.start[1::2], end=ep.end[1::2])
        grp = units[[int(u) for u in pc]]
        tct = tuning_curves(grp, pos, train, dt, track).fillna(0.0) + 1e-3
        dec, proba = nap.decode_1d(tuning_curves=tct, group=grp, ep=test,
                                   bin_size=DECODE_BIN, feature=pos.restrict(test))
        true = pos.restrict(test).interpolate(dec)
        err = np.abs(dec.values - true.values)
        ok = np.isfinite(err)
        chance = np.mean([np.median(np.abs(RNG.permutation(dec.values[ok])
                                           - true.values[ok])) for _ in range(200)])
        decoding.append(dict(session=label, direction=d, n_cells=len(pc),
                             median_err=float(np.median(err[ok])),
                             mean_err=float(np.mean(err[ok])),
                             chance_err=float(chance), track=track,
                             n_bins=int(ok.sum())))
        if keep_heavy and d == "right":
            dec_keep = dict(decoded=dec, proba=proba, true=true, test=test, tc=tct,
                            grp=grp, train=train)

    # ---- theta phase precession ----
    pp_rows, pp_detail, theta_info = [], {}, {}
    if do_theta:
        cand = list(range(0, h5f["processing/ecephys/LFP/LFP/data"].shape[1], 8))
        best_ch, ratios = pick_theta_channel(h5f, cand, run_ep)
        lfp1 = load_lfp_channels(h5f, [best_ch], float(maze.start[0]),
                                 float(maze.end[0]))
        lfp_tsd = nap.Tsd(t=lfp1.t, d=np.asarray(lfp1.values[:, 0], float))
        ph, filt = theta_phase(lfp_tsd)
        theta_info = dict(channel=best_ch, ratio=float(ratios[best_ch]))
        edges = bin_edges(track)
        for _, r in metrics[metrics.is_place].iterrows():
            d, u = r.direction, int(r.unit)
            b = field_bounds(tc[d][u].values, frac=0.33)
            if b is None:
                continue
            lo, _, hi = b
            x0, x1 = edges[lo], edges[hi + 1]
            width = x1 - x0
            # well-isolated field, away from the reward zones at the track ends
            if not (0.15 <= width <= 0.45 * track) or x0 < 0.06 * track \
                    or x1 > 0.94 * track or r.peak_rate < 2.0:
                continue
            st = units[u].restrict(passes[d])
            if len(st) == 0:
                continue
            xs = pos.interpolate(st).values
            inside = np.isfinite(xs) & (xs >= x0) & (xs <= x1)
            if inside.sum() < 50:
                continue
            st_in = nap.Ts(t=st.t[inside])
            frac = (xs[inside] - x0) / width
            if d == "left":
                frac = 1 - frac            # fraction traversed through the field
            phs = st_in.value_from(ph).values
            rho, p, slope, phi0 = circ_lin_corr(phs, frac)
            pp_rows.append(dict(session=label, subject=label.split("_")[0], unit=u,
                                direction=d, n=int(inside.sum()), rho=rho, p=p,
                                slope_cycles=slope / (2 * np.pi), width=width,
                                x0=x0, x1=x1, peak_rate=r.peak_rate, si=r.si))
            if keep_heavy:
                pp_detail[(u, d)] = (frac, phs, slope, phi0)

    out = dict(label=label, subject=label.split("_")[0], maze=mtype, track=track,
               dt=dt, maze_dur=float(maze.tot_length()),
               n_exc=len(units_all), n_inh=n_inh, n_analysed=len(units),
               n_passes={k: len(v) for k, v in passes.items()},
               run_time=float(run_ep.tot_length()),
               metrics=metrics, decoding=pd.DataFrame(decoding),
               precession=pd.DataFrame(pp_rows), theta=theta_info,
               tc={d: tc[d] for d in tc}, occ=occ, centers=centers)
    if keep_heavy:
        out.update(pos=pos, speed=speed, passes=passes, units=units, maze_ep=maze,
                   pos2d=get_2d_position(nwbfile).restrict(maze),
                   dec=dec_keep, pp_detail=pp_detail,
                   lfp=lfp_tsd if do_theta else None,
                   theta_ph=ph if do_theta else None,
                   theta_filt=filt if do_theta else None,
                   h5f=h5f, io=io)
    else:
        io.close()
    return out


# %% [markdown]
# ## Prototype session: Achilles, 25 Oct 2013
#
# Load one session and check every data stream before analysing it.

# %%
S = analyse_session(PROTOTYPE, keep_heavy=True, show_progress=True)
print(f"session      : {S['label']}  ({S['maze']} maze, {S['track']} m)")
print(f"maze epoch   : {S['maze_dur']:.0f} s")
print(f"units        : {S['n_exc']} excitatory, {S['n_inh']} inhibitory; "
      f"{S['n_analysed']} analysed (>= {MIN_MEAN_RATE} Hz while running)")
print(f"passes       : {S['n_passes']['right']} rightward, "
      f"{S['n_passes']['left']} leftward; {S['run_time']:.0f} s of running")
print(f"theta channel: {S['theta']['channel']} "
      f"(theta/delta power ratio {S['theta']['ratio']:.1f})")
m = S["metrics"]
print(f"place cells  : {m.groupby('unit')['is_place'].any().sum()}/{S['n_analysed']} "
      f"units in >=1 direction")

# %% [markdown]
# ### Figure 1 — behaviour and raw spiking
#
# The 2D tracking shows the linear track plus the two end zones; linearized
# position is defined only on the track itself. The raster is ordered by each
# cell's place-field peak on rightward passes, so a place code should appear as
# diagonal sweeps of activity that reverse direction between passes.

# %%
pos, passes, units, speed = S["pos"], S["passes"], S["units"], S["speed"]
centers = S["centers"]
pc_right = m[(m.direction == "right") & m.is_place].sort_values("peak_loc")
order = pc_right["unit"].values

t0 = float(passes["right"].start[4]) - 4
win = 46.0
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(4, 3, height_ratios=[1.0, 0.55, 1.6, 1.5], width_ratios=[1, 1, 1],
              hspace=0.5, wspace=0.28)

ax = fig.add_subplot(gs[0, 0])
p2 = S["pos2d"]
ax.plot(p2["x"].values, p2["y"].values, ".", ms=0.6, color="0.7", alpha=0.5)
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("2D tracking (whole maze epoch)", fontsize=10)

ax = fig.add_subplot(gs[0, 1:])
ax.plot(pos.t, pos.values, "k.", ms=1.6)
for d, c in (("right", "tab:red"), ("left", "tab:blue")):
    for s, e in zip(passes[d].start, passes[d].end):
        ax.axvspan(s, e, color=c, alpha=0.20, lw=0)
ax.set_xlim(t0, t0 + win)
ax.set_ylabel("linearized\nposition (m)")
ax.set_title("Linearized position: rightward (red) and leftward (blue) passes",
             fontsize=10)

ax = fig.add_subplot(gs[1, 1:])
ax.plot(speed.t, speed.values, "k-", lw=0.9)
ax.axhline(SPEED_THRESH, color="tab:red", ls="--", lw=1)
ax.set_xlim(t0, t0 + win)
ax.set_ylabel("speed (m/s)")
ax.set_title(f"running threshold {SPEED_THRESH} m/s", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.hist(np.asarray(speed.restrict(S["maze_ep"]).values), bins=60, color="0.4")
ax.axvline(SPEED_THRESH, color="tab:red", ls="--")
ax.set_xlabel("speed (m/s)"); ax.set_ylabel("samples")
ax.set_title("on-track speed distribution", fontsize=10)

ax = fig.add_subplot(gs[2, :])
for i, u in enumerate(order):
    st = units[int(u)].restrict(nap.IntervalSet(t0, t0 + win))
    ax.plot(st.t, np.full(len(st), i), "|", color="k", ms=4.5, mew=0.9)
for d, c in (("right", "tab:red"), ("left", "tab:blue")):
    for s, e in zip(passes[d].start, passes[d].end):
        ax.axvspan(s, e, color=c, alpha=0.13, lw=0)
ax.set_xlim(t0, t0 + win)
ax.set_ylim(-1, len(order))
ax.set_xlabel("time (s)")
ax.set_ylabel("place cell\n(ordered by field peak)")
ax.set_title("CA1 spike raster, cells ordered by rightward place-field peak",
             fontsize=10)

# pass-aligned rasters: the same cells, each panel one traversal, time measured
# from the start of the pass. A place code shows up as a diagonal sweep.
sub = GridSpec(4, 6, height_ratios=[1.0, 0.55, 1.6, 1.5], hspace=0.5, wspace=0.12)
for j in range(6):
    axp = fig.add_subplot(sub[3, j])
    s, e = passes["right"].start[j + 4], passes["right"].end[j + 4]
    for i, u in enumerate(order):
        st = units[int(u)].restrict(nap.IntervalSet(s, e))
        axp.plot(st.t - s, np.full(len(st), i), "|", color="tab:red", ms=4, mew=0.9)
    axp.set_xlim(0, float(e - s))
    axp.set_ylim(-1, len(order))
    axp.set_xlabel("t from\npass onset (s)", fontsize=8)
    axp.tick_params(labelsize=8)
    if j == 0:
        axp.set_ylabel("place cell\n(ordered by field peak)", fontsize=9)
    else:
        axp.set_yticklabels([])
    axp.set_title(f"rightward pass {j + 5}", fontsize=8.5)
fig.suptitle(f"Figure 1 — {S['label']}: linear-track behaviour and CA1 spiking",
             fontsize=13, y=0.995)
fig.savefig("fig01_behavior_and_raster.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 2 — example place cells
#
# For each example cell: spikes plotted as position within each running pass
# (top), and the occupancy-normalised rate map for both directions (bottom).

# %%
def spike_pass_raster(ax, unit, d, color):
    ep = passes[d]
    st = units[int(unit)]
    for k in range(len(ep)):
        one = nap.IntervalSet(ep.start[k], ep.end[k])
        s = st.restrict(one)
        if len(s) == 0:
            continue
        x = pos.restrict(one).interpolate(s).values
        ax.plot(x, np.full(len(x), k), "|", color=color, ms=5, mew=1.0)
    ax.set_ylim(-1, len(ep))


# six place cells whose rightward fields are spread across the track; fields
# peaking in the first/last bin (the reward zones) are skipped so that the panel
# shows well-formed interior fields
rr = m[(m.direction == "right") & m.is_place].sort_values("peak_loc")
interior = rr[(rr.peak_loc > 0.12 * S["track"]) & (rr.peak_loc < 0.90 * S["track"])
              & (rr.peak_rate >= 3.0)]
pick_from = interior if len(interior) >= 6 else rr
idx = np.linspace(0, len(pick_from) - 1, 6).astype(int)
examples = pick_from.iloc[idx]

fig, axs = plt.subplots(2, 6, figsize=(16, 5.6), sharex=True,
                        gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.35,
                                     "wspace": 0.32})
for j, (_, r) in enumerate(examples.iterrows()):
    u = int(r.unit)
    spike_pass_raster(axs[0, j], u, "right", "tab:red")
    axs[0, j].set_title(f"unit {u}\nSI={r.si:.2f} bits/spk, p<{max(r.p,1/N_SHUFFLES):.3f}",
                        fontsize=9)
    if j == 0:
        axs[0, j].set_ylabel("rightward pass #")
    axs[1, j].plot(centers, S["tc"]["right"][u].values, color="tab:red", lw=2,
                   label="rightward")
    axs[1, j].plot(centers, S["tc"]["left"][u].values, color="tab:blue", lw=2,
                   label="leftward")
    axs[1, j].set_xlabel("position (m)")
    if j == 0:
        axs[1, j].set_ylabel("firing rate (Hz)")
        axs[1, j].legend(fontsize=7, frameon=False)
fig.suptitle(f"Figure 2 — {S['label']}: example CA1 place cells "
             "(spikes per pass, and rate maps)", fontsize=12, y=1.0)
fig.savefig("fig02_example_place_cells.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 3 — the population tiles the track
#
# Peak-normalised rate maps, sorted by the location of the peak on rightward
# passes. The same sort order is applied to the leftward maps, which reveals how
# direction-selective the fields are.

# %%
fig, axs = plt.subplots(1, 4, figsize=(16, 5.4),
                        gridspec_kw={"width_ratios": [1, 1, 1, 0.8], "wspace": 0.38})
pc_units = m.groupby("unit")["is_place"].any()
pc_units = pc_units[pc_units].index.values
tcR = S["tc"]["right"][pc_units].values.T
tcL = S["tc"]["left"][pc_units].values.T


def norm_rows(a):
    a = np.array(a, float)
    mx = np.nanmax(a, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / mx


srt = np.argsort(np.nanargmax(np.nan_to_num(tcR, nan=-1), axis=1))
for ax, mat, ttl in ((axs[0], norm_rows(tcR)[srt], "rightward passes"),
                     (axs[1], norm_rows(tcL)[srt], "leftward passes\n(same cell order)")):
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, S["track"], 0, len(srt)], vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("position (m)")
    ax.set_ylabel("place cell (sorted by rightward peak)")
    ax.set_title(ttl, fontsize=10)
plt.colorbar(im, ax=axs[1], label="rate / peak rate")

srtL = np.argsort(np.nanargmax(np.nan_to_num(tcL, nan=-1), axis=1))
im = axs[2].imshow(norm_rows(tcL)[srtL], aspect="auto", origin="lower", cmap="magma",
                   extent=[0, S["track"], 0, len(srtL)], vmin=0, vmax=1,
                   interpolation="nearest")
axs[2].set_xlabel("position (m)")
axs[2].set_ylabel("place cell (sorted by leftward peak)")
axs[2].set_title("leftward passes\n(sorted independently)", fontsize=10)

axs[3].hist(m[m.is_place]["peak_loc"], bins=np.linspace(0, S["track"], 21),
            color="0.4")
axs[3].set_xlabel("place-field peak (m)")
axs[3].set_ylabel("number of fields")
axs[3].set_title("field peaks tile the track", fontsize=10)
fig.suptitle(f"Figure 3 — {S['label']}: population rate maps", fontsize=12, y=1.02)
fig.savefig("fig03_population_maps.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 4 — spatial information against a circular-shift null

# %%
tuner_r = FastTuner(pos, passes["right"], S["dt"], S["track"])
null_r = shuffle_spatial_info(units, tuner_r, S["maze_ep"], N_SHUFFLES,
                              rng=np.random.default_rng(1), show_progress=True)
obs_r = {u: spatial_information(tuner_r.rate_map(
    np.asarray(units[u].restrict(passes["right"]).t)), tuner_r.occ)
    for u in units.keys()}

fig, axs = plt.subplots(1, 4, figsize=(16, 3.9), gridspec_kw={"wspace": 0.34})
allnull = np.concatenate([v for v in null_r.values()])
axs[0].hist(allnull, bins=60, density=True, color="0.7", label="circular-shift null")
axs[0].hist([v for v in obs_r.values()], bins=30, density=True, alpha=0.7,
            color="tab:red", label="observed")
axs[0].set_xlabel("spatial information (bits/spike)")
axs[0].set_ylabel("density")
axs[0].legend(fontsize=8, frameon=False)
axs[0].set_title("observed vs shuffled SI\n(rightward passes)", fontsize=10)

x = np.array([np.mean(null_r[u]) for u in units.keys()])
y = np.array([obs_r[u] for u in units.keys()])
sig = np.array([np.mean(null_r[u] >= obs_r[u]) < P_CRIT for u in units.keys()])
axs[1].plot(x[~sig], y[~sig], "o", ms=4, color="0.6", label="n.s.")
axs[1].plot(x[sig], y[sig], "o", ms=4, color="tab:red", label=f"p<{P_CRIT}")
lim = [0, max(y.max(), x.max()) * 1.05]
axs[1].plot(lim, lim, "k--", lw=1)
axs[1].set_xlim(lim); axs[1].set_ylim(lim)
axs[1].set_xlabel("mean shuffled SI (bits/spike)")
axs[1].set_ylabel("observed SI (bits/spike)")
axs[1].legend(fontsize=8, frameon=False)
axs[1].set_title(f"{sig.sum()}/{len(sig)} units exceed the null", fontsize=10)

axs[2].hist(m[m.is_place]["field_width"], bins=18, color="tab:red", alpha=0.85)
axs[2].set_xlabel("field width at 50% peak (m)")
axs[2].set_ylabel("number of fields")
axs[2].set_title(f"median width "
                 f"{m[m.is_place]['field_width'].median():.2f} m", fontsize=10)

axs[3].hist(m[m.is_place]["peak_rate"], bins=np.logspace(0, 2, 20), color="tab:red",
            alpha=0.85)
axs[3].set_xscale("log")
axs[3].set_xlabel("in-field peak rate (Hz)")
axs[3].set_ylabel("number of fields")
axs[3].set_title(f"median peak rate "
                 f"{m[m.is_place]['peak_rate'].median():.1f} Hz", fontsize=10)
fig.suptitle(f"Figure 4 — {S['label']}: spatial selectivity is far above chance",
             fontsize=12, y=1.04)
fig.savefig("fig04_spatial_information.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 5 — direction selectivity
#
# On a linear track, CA1 fields are typically direction-selective: the same cell
# fires at a different place, or not at all, depending on the running direction.

# %%
piv = m.pivot(index="unit", columns="direction",
              values=["peak_loc", "peak_rate", "is_place"])
both = piv[(piv[("is_place", "right")]) & (piv[("is_place", "left")])]
di = ((m.pivot(index="unit", columns="direction", values="peak_rate")["right"]
       - m.pivot(index="unit", columns="direction", values="peak_rate")["left"])
      / (m.pivot(index="unit", columns="direction", values="peak_rate")["right"]
         + m.pivot(index="unit", columns="direction", values="peak_rate")["left"]))
di = di[pc_units]

# correlation between the two directional rate maps, per place cell
corrs = []
for u in pc_units:
    a, b = S["tc"]["right"][u].values, S["tc"]["left"][u].values
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() > 5 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
        corrs.append(np.corrcoef(a[ok], b[ok])[0, 1])

fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"wspace": 0.34})
axs[0].plot(both[("peak_loc", "right")], both[("peak_loc", "left")], "o", ms=5,
            color="tab:purple")
axs[0].plot([0, S["track"]], [0, S["track"]], "k--", lw=1)
axs[0].set_xlabel("peak location, rightward (m)")
axs[0].set_ylabel("peak location, leftward (m)")
axs[0].set_title(f"cells with a field in both\ndirections (n={len(both)})", fontsize=10)

axs[1].hist(di.dropna(), bins=np.linspace(-1, 1, 25), color="tab:purple", alpha=0.85)
axs[1].set_xlabel("(R - L) / (R + L) peak rate")
axs[1].set_ylabel("number of place cells")
axs[1].set_title("directionality index", fontsize=10)

axs[2].hist(corrs, bins=np.linspace(-1, 1, 25), color="tab:purple", alpha=0.85)
axs[2].axvline(np.median(corrs), color="k", ls="--")
axs[2].set_xlabel("r (rightward vs leftward map)")
axs[2].set_ylabel("number of place cells")
axs[2].set_title(f"median r = {np.median(corrs):.2f}", fontsize=10)
fig.suptitle(f"Figure 5 — {S['label']}: place fields are direction-selective",
             fontsize=12, y=1.04)
fig.savefig("fig05_directionality.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 6 — decoding position from the population
#
# A naive-Bayes (Poisson) decoder is fitted on the odd-numbered rightward passes
# and evaluated on the even-numbered ones. Because the held-out passes are short
# and separated by long stays in the reward zone, the posterior is plotted pass
# by pass rather than against absolute time.

# %%
D = S["dec"]
proba, dec, true, test = D["proba"], D["decoded"], D["true"], D["test"]
n_show = min(8, len(test))
fig = plt.figure(figsize=(15, 7))
gs = GridSpec(2, 4, height_ratios=[1.2, 1], hspace=0.42, wspace=0.34)

ax = fig.add_subplot(gs[0, :])
xoff, ticks = 0, []
for k in range(n_show):
    one = nap.IntervalSet(test.start[k], test.end[k])
    pr = proba.restrict(one)
    if len(pr) == 0:
        continue
    n = len(pr)
    ax.imshow(pr.values.T, aspect="auto", origin="lower", cmap="viridis",
              extent=[xoff, xoff + n, 0, S["track"]], vmin=0,
              vmax=np.nanpercentile(proba.values, 99), interpolation="nearest")
    tr = true.restrict(one)
    ax.plot(xoff + 0.5 + np.arange(len(tr)), tr.values, "w-", lw=1.6)
    dd = dec.restrict(one)
    ax.plot(xoff + 0.5 + np.arange(len(dd)), dd.values, "r.", ms=5)
    ticks.append(xoff + n / 2)
    xoff += n
    ax.axvline(xoff, color="w", lw=1.2)
ax.set_xlim(0, xoff)
ax.set_ylim(0, S["track"])
ax.set_xticks(ticks)
ax.set_xticklabels([f"pass {i+1}" for i in range(len(ticks))])
ax.set_xlabel("held-out rightward passes (concatenated; "
              f"{DECODE_BIN} s decoding bins)")
ax.set_ylabel("position (m)")
ax.set_title("posterior P(position | spikes) on held-out rightward passes; "
             "white = true position, red = decoded", fontsize=10)

err = np.abs(dec.values - true.values)
err = err[np.isfinite(err)]
ax = fig.add_subplot(gs[1, 0])
ax.hist(err, bins=25, color="tab:green", alpha=0.85)
ax.axvline(np.median(err), color="k", ls="--")
ax.set_xlabel("|decoding error| (m)")
ax.set_ylabel(f"{DECODE_BIN}s bins")
ax.set_title(f"median error {np.median(err):.3f} m", fontsize=10)

# error as a function of ensemble size
ax = fig.add_subplot(gs[1, 1])
sizes = [2, 4, 8, 16, 24, len(D["grp"])]
sizes = sorted({s for s in sizes if s <= len(D["grp"])})
curve = []
allu = np.array(list(D["grp"].keys()))
for nsz in sizes:
    e = []
    for rep in range(8):
        sub = np.sort(RNG.choice(allu, size=nsz, replace=False))
        g = D["grp"][[int(x) for x in sub]]
        tcs = D["tc"][list(g.keys())]
        dd, _ = nap.decode_1d(tuning_curves=tcs, group=g, ep=test,
                              bin_size=DECODE_BIN, feature=pos.restrict(test))
        tt = pos.restrict(test).interpolate(dd)
        ee = np.abs(dd.values - tt.values)
        e.append(np.median(ee[np.isfinite(ee)]))
    curve.append((np.mean(e), np.std(e)))
curve = np.array(curve)
ax.errorbar(sizes, curve[:, 0], yerr=curve[:, 1], marker="o", color="tab:green")
ax.axhline(S["decoding"].query("direction=='right'")["chance_err"].iloc[0],
           color="k", ls="--", label="chance")
ax.set_xlabel("number of place cells")
ax.set_ylabel("median error (m)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("decoding improves with ensemble size", fontsize=10)

ax = fig.add_subplot(gs[1, 2:])
dd = S["decoding"]
xs = np.arange(len(dd))
ax.bar(xs - 0.18, dd["median_err"], width=0.36, color="tab:green", label="decoder")
ax.bar(xs + 0.18, dd["chance_err"], width=0.36, color="0.7", label="chance")
ax.set_xticks(xs)
ax.set_xticklabels([f"{r.direction}\n({r.n_cells} cells)" for _, r in dd.iterrows()])
ax.set_ylabel("median error (m)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("cross-validated decoding error vs chance", fontsize=10)
fig.suptitle(f"Figure 6 — {S['label']}: Bayesian position decoding", fontsize=12,
             y=0.98)
fig.savefig("fig06_decoding.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 7 — theta phase precession
#
# Spike times are assigned the instantaneous phase of the 6-10 Hz filtered LFP on
# the channel with the strongest theta. For each well-isolated field away from
# the reward zones, spike phase is regressed on the fraction of the field already
# traversed using a circular-linear correlation.

# %%
ph, filt, lfp = S["theta_ph"], S["theta_filt"], S["lfp"]
pp = S["precession"]
fig = plt.figure(figsize=(15, 8.5))
gs = GridSpec(3, 4, height_ratios=[0.8, 1.15, 1.15], hspace=0.55, wspace=0.33)

ax = fig.add_subplot(gs[0, :2])
tt0 = float(passes["right"].start[3])
sl = nap.IntervalSet(tt0, tt0 + 1.6)
ax.plot(lfp.restrict(sl).t, lfp.restrict(sl).values * 1e3, color="0.4", lw=0.8,
        label="raw LFP")
ax.plot(filt.restrict(sl).t, filt.restrict(sl).values * 1e3, color="tab:orange",
        lw=2, label=f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz")
ax.set_xlabel("time (s)"); ax.set_ylabel("LFP (mV)")
ax.legend(fontsize=8, frameon=False, ncol=2)
ax.set_title(f"CA1 theta during running (channel {S['theta']['channel']})",
             fontsize=10)

ax = fig.add_subplot(gs[0, 2])
f, pw = welch(np.asarray(lfp.restrict(passes["right"].union(passes["left"])).values,
                         float), fs=LFP_RATE, nperseg=int(4 * LFP_RATE))
ax.semilogy(f, pw, "k", lw=1.2)
ax.axvspan(*THETA_BAND, color="tab:orange", alpha=0.3)
ax.set_xlim(0, 30)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (V²/Hz)")
ax.set_title("LFP spectrum while running", fontsize=10)

ax = fig.add_subplot(gs[0, 3])
ax.hist(pp["rho"], bins=np.linspace(-0.8, 0.8, 21), color="tab:orange", alpha=0.9)
ax.axvline(0, color="k", lw=1)
ax.axvline(pp["rho"].median(), color="tab:red", ls="--", lw=1.5)
ax.set_xlabel("circular-linear rho")
ax.set_ylabel("number of fields")
ax.set_title(f"median rho = {pp['rho'].median():.2f} (n={len(pp)})", fontsize=10)

sel = pp.sort_values("rho").head(8)
for j, (_, r) in enumerate(sel.iterrows()):
    ax = fig.add_subplot(gs[1 + j // 4, j % 4])
    frac, phs, slope, phi0 = S["pp_detail"][(int(r.unit), r.direction)]
    ax.plot(frac, phs, ".", ms=3, alpha=0.45, color="0.25")
    ax.plot(frac, phs + 2 * np.pi, ".", ms=3, alpha=0.45, color="0.25")
    xx = np.linspace(0, 1, 200)
    for off in (0, 2 * np.pi, 4 * np.pi):
        yy = slope * xx + phi0 + off
        keep = (yy >= 0) & (yy <= 4 * np.pi)
        ax.plot(xx[keep], yy[keep], "-", color="tab:red", lw=1.8)
    ax.set_ylim(0, 4 * np.pi)
    ax.set_yticks([0, 2 * np.pi, 4 * np.pi])
    ax.set_yticklabels(["0", "2π", "4π"])
    ax.set_title(f"unit {int(r.unit)} ({r.direction})\n"
                 f"rho={r.rho:.2f}, {r.slope_cycles:.2f} cycles/field", fontsize=8.5)
    if j % 4 == 0:
        ax.set_ylabel("theta phase")
    if j // 4 == 1:
        ax.set_xlabel("fraction of field traversed")
fig.suptitle(f"Figure 7 — {S['label']}: theta phase precession", fontsize=12, y=0.97)
fig.savefig("fig07_theta_precession.png", bbox_inches="tight")
plt.close(fig)

nsig = int(((pp.rho < 0) & (pp.p < 0.05)).sum())
print(f"phase precession: {nsig}/{len(pp)} fields with a significant negative "
      f"phase-position correlation; median rho = {pp['rho'].median():.3f}, "
      f"median slope = {pp['slope_cycles'].median():.2f} theta cycles per field")

# %% [markdown]
# ## All linear-track sessions
#
# The prototype pipeline is now run on every linear-track session in the
# dandiset. The three circular-maze sessions are skipped, since the two-ended
# directional-pass logic does not apply to them.

# %%
S["io"].close()
results = {}
skipped = []
for asset, label in tqdm(list(SESSIONS.items()), desc="sessions"):
    if asset == PROTOTYPE:
        results[label] = {k: v for k, v in S.items()
                          if k not in ("h5f", "io", "units", "pos", "speed",
                                       "passes", "pos2d", "dec", "pp_detail",
                                       "lfp", "theta_ph", "theta_filt", "maze_ep")}
        continue
    nwbfile, io, h5f = open_session(asset)
    mt, _ = maze_type(nwbfile)
    io.close()
    if mt != "linear":
        skipped.append((label, mt))
        continue
    results[label] = analyse_session(asset)
print("analysed:", list(results))
print("skipped (circular maze):", skipped)

# %%
ALL = pd.concat([r["metrics"] for r in results.values()], ignore_index=True)
DEC = pd.concat([r["decoding"] for r in results.values()], ignore_index=True)
PP = pd.concat([r["precession"] for r in results.values()], ignore_index=True)
ALL.to_csv("place_cell_metrics_all_sessions.csv", index=False)
DEC.to_csv("decoding_all_sessions.csv", index=False)
PP.to_csv("phase_precession_all_sessions.csv", index=False)

summary = []
for lab, r in results.items():
    mm = r["metrics"]
    per_unit = mm.groupby("unit")["is_place"].any()
    summary.append(dict(
        session=lab, subject=r["subject"], track_m=r["track"],
        exc_units=r["n_exc"], analysed=r["n_analysed"],
        place_cells=int(per_unit.sum()),
        pct_place=100 * float(per_unit.mean()),
        median_si=float(mm[mm.is_place]["si"].median()),
        median_width_m=float(mm[mm.is_place]["field_width"].median()),
        median_peak_hz=float(mm[mm.is_place]["peak_rate"].median()),
        median_spikes=float(mm["n_spikes"].median()),
        pct_place_testable=100 * float(
            mm[mm.testable].groupby("unit")["is_place"].any().mean())
        if mm.testable.any() else np.nan,
        run_s=r["run_time"],
        decode_err_m=float(r["decoding"]["median_err"].mean())
        if len(r["decoding"]) else np.nan,
        chance_err_m=float(r["decoding"]["chance_err"].mean())
        if len(r["decoding"]) else np.nan,
        pp_fields=len(r["precession"]),
        pp_median_rho=float(r["precession"]["rho"].median())
        if len(r["precession"]) else np.nan))
SUM = pd.DataFrame(summary)
SUM.to_csv("session_summary.csv", index=False)
print(SUM.to_string(index=False))

# %% [markdown]
# ### Figure 8 — the result reproduces across sessions and animals

# %%
fig, axs = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"hspace": 0.55,
                                                           "wspace": 0.3})
labels = SUM["session"].str.replace("_", "\n")

ax = axs[0, 0]
ax.bar(range(len(SUM)), SUM["pct_place"], color="tab:red", alpha=0.85)
ax.set_xticks(range(len(SUM))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("% of analysed units")
ax.set_title("place cells (>=1 direction)", fontsize=10)
for i, (n, tot) in enumerate(zip(SUM["place_cells"], SUM["analysed"])):
    ax.text(i, SUM["pct_place"].iloc[i] + 1.5, f"{n}/{tot}", ha="center", fontsize=7.5)

ax = axs[0, 1]
data = [ALL[(ALL.session == s) & ALL.is_place]["si"].values for s in SUM["session"]]
ax.boxplot(data, showfliers=False)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("spatial information (bits/spike)")
ax.set_title("place-cell spatial information", fontsize=10)

ax = axs[0, 2]
data = [ALL[(ALL.session == s) & ALL.is_place]["field_width"].values
        for s in SUM["session"]]
ax.boxplot(data, showfliers=False)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("field width at 50% peak (m)")
ax.set_title("place-field width", fontsize=10)

ax = axs[1, 0]
w = 0.36
xs = np.arange(len(SUM))
ax.bar(xs - w / 2, SUM["decode_err_m"], width=w, color="tab:green", label="decoder")
ax.bar(xs + w / 2, SUM["chance_err_m"], width=w, color="0.7", label="chance")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("median error (m)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("cross-validated position decoding", fontsize=10)
for i, v in enumerate(SUM["decode_err_m"]):
    if not np.isfinite(v):
        ax.text(i, 0.02, "too few\nplace cells", ha="center", fontsize=7.5,
                color="0.3")

ax = axs[1, 1]
pooled_null = ALL["si_null"].values
ax.hist(pooled_null, bins=50, density=True, color="0.7", label="shuffled")
ax.hist(ALL[ALL.is_place]["si"], bins=40, density=True, alpha=0.7, color="tab:red",
        label="place cells")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("density")
ax.legend(fontsize=8, frameon=False)
ax.set_title(f"pooled: {int(ALL.is_place.sum())} significant fields\n"
             f"of {len(ALL)} unit-directions tested", fontsize=10)

ax = axs[1, 2]
ax.hist(PP["rho"], bins=np.linspace(-0.8, 0.8, 25), color="tab:orange", alpha=0.9)
ax.axvline(0, color="k", lw=1)
ax.axvline(PP["rho"].median(), color="tab:red", ls="--", lw=1.5)
ax.set_xlabel("circular-linear rho (phase vs position in field)")
ax.set_ylabel("number of fields")
ax.set_title(f"phase precession, all sessions\nmedian rho = {PP['rho'].median():.2f}"
             f" (n={len(PP)})", fontsize=10)
fig.suptitle("Figure 8 — place coding across all five linear-track sessions "
             "(four rats)", fontsize=13, y=0.98)
fig.savefig("fig08_across_sessions.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Summary statistics

# %%
per_unit_all = ALL.groupby(["session", "unit"])["is_place"].any()
w_stat = wilcoxon(PP["rho"], alternative="less")
print("=" * 72)
print(f"Sessions analysed        : {len(results)} "
      f"({SUM['subject'].nunique()} rats); {len(skipped)} circular-maze sessions "
      f"excluded")
print(f"Units analysed           : {int(SUM['analysed'].sum())} excitatory CA1 units")
print(f"Place cells              : {int(per_unit_all.sum())} "
      f"({100 * per_unit_all.mean():.0f}% of analysed units)")
testable_unit = ALL[ALL.testable].groupby(["session", "unit"])["is_place"].any()
print(f"  restricted to units with >= {MIN_SPIKES} in-run spikes in a direction: "
      f"{int(testable_unit.sum())}/{len(testable_unit)} "
      f"({100 * testable_unit.mean():.0f}%)")
print(f"Significant fields       : {int(ALL.is_place.sum())} of "
      f"{len(ALL)} unit-direction pairs")
print(f"Spatial information      : median {ALL[ALL.is_place]['si'].median():.2f} "
      f"bits/spike (shuffled median {np.median(ALL['si_null']):.2f})")
print(f"Field width (50% peak)   : median "
      f"{ALL[ALL.is_place]['field_width'].median():.2f} m")
print(f"In-field peak rate       : median "
      f"{ALL[ALL.is_place]['peak_rate'].median():.1f} Hz")
print(f"Sparsity                 : median "
      f"{ALL[ALL.is_place]['sparsity'].median():.2f}")
print(f"Decoding error           : median {DEC['median_err'].median():.3f} m "
      f"vs chance {DEC['chance_err'].median():.3f} m")
print(f"Phase precession         : {len(PP)} isolated fields; median rho "
      f"{PP['rho'].median():.3f}, median slope "
      f"{PP['slope_cycles'].median():.2f} cycles/field; "
      f"{int(((PP.rho < 0) & (PP.p < 0.05)).sum())} significantly negative, "
      f"{int(((PP.rho > 0) & (PP.p < 0.05)).sum())} significantly positive")
print(f"                           Wilcoxon signed-rank rho<0: "
      f"W={w_stat.statistic:.0f}, p={w_stat.pvalue:.2e}")
print("=" * 72)

with open("results_all_sessions.pkl", "wb") as fh:
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "tc"}
                 for k, v in results.items()}, fh)

# %% [markdown]
# ## Conclusion
#
# Streaming five linear-track sessions from four rats out of DANDI:000044 and
# analysing them with Pynapple reproduces the defining properties of hippocampal
# place cells:
#
# * A majority of the excitatory CA1 units that are active while the animal runs
#   have a spatially restricted field on the track, with spatial information far
#   above a circular-shift null. Fields are compact (tens of centimetres) and
#   their peaks tile the whole track.
# * Fields are largely direction-selective, as expected on a linear track where
#   the two running directions are behaviourally distinct.
# * The population code is accurate and redundant: a naive-Bayes decoder trained
#   on half the passes localises the animal on held-out passes to within a few
#   centimetres, roughly an order of magnitude better than chance, and the error
#   falls steadily as more cells are added to the ensemble.
# * Spikes precess through theta phase across the field, with a median slope
#   close to one theta cycle per field traversal, which is the signature that
#   distinguishes hippocampal place coding from a static rate code.
