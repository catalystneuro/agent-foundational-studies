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
# # Hippocampal place cells in DANDI:000044
#
# This notebook demonstrates hippocampal place cells using real extracellular
# recordings streamed from the DANDI Archive. It is self-contained: every figure
# it refers to is produced by the cells below, and no data are downloaded in
# full.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044),
# *"Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences"* (Grosmark & Buzsáki, *Science* 2016; the `hc-11`
# dataset). Eight sessions from four Long-Evans rats, each with bilateral
# silicon-probe recordings from dorsal CA1, spike-sorted units labelled
# excitatory or inhibitory, 128-channel LFP, and video tracking of the animal on
# a maze. Each session is a PRE sleep epoch, a maze-running epoch, and a POST
# sleep epoch. Files are 5-9 GB each, so they are read by streaming only the
# objects the analysis needs.
#
# **What a place cell is.** A hippocampal pyramidal cell is a place cell if it
# fires selectively when the animal occupies a particular part of the
# environment, its place field (O'Keefe & Dostrovsky 1971). The claim this
# notebook tests has three parts, in increasing strength:
#
# 1. Individual CA1 pyramidal cells fire in a spatially restricted, reproducible
#    part of the track, and carry more spatial information than a
#    temporally-matched shuffle of the same spike train.
# 2. Across the population, place fields tile the whole track, and they are
#    direction-specific on a linear maze.
# 3. Position can be read back out of the population spike counts on traversals
#    that were never used to fit the encoding model, both by Bayesian decoding
#    and by a Poisson GLM.
#
# **Tools.** Streaming with `remfile` + `h5py` + `pynwb`, analysis with
# `pynapple`, and a model-based check with `NeMoS`.

# %% [markdown]
# ## 1. Setup

# %%
import warnings

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from tqdm.auto import tqdm

nap.nap_config.suppress_conversion_warnings = True
mpl.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 160,
})

# %% [markdown]
# ### Streaming access
#
# Each asset is opened over HTTP with `remfile`, which serves HDF5 chunk reads
# from S3 and keeps a local disk cache. Only the units table, the behaviour
# module, and short windows of LFP are ever touched, so a 9 GB file costs a few
# tens of megabytes of transfer. The S3 URLs below were resolved once from the
# DANDI API (`GET /api/dandisets/000044/versions/draft/assets/`).

# %%
CACHE_DIR = "/tmp/remfile_cache_000044"

SESSIONS = {
    "Achilles_10252013": "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae",
    "Achilles_11012013": "https://dandiarchive.s3.amazonaws.com/blobs/75a/14b/75a14b2b-05f2-4bb1-abcb-e2af85579af2",
    "Buddy_06272013":    "https://dandiarchive.s3.amazonaws.com/blobs/98b/25c/98b25cb1-310c-45f7-97cc-669fce2057b7",
    "Cicero_09012014":   "https://dandiarchive.s3.amazonaws.com/blobs/ced/326/ced32609-16ab-4c4c-991f-8fa9f40e9401",
    "Cicero_09102014":   "https://dandiarchive.s3.amazonaws.com/blobs/6b8/206/6b8206bd-2829-48d9-95fc-e713cb3d363a",
    "Cicero_09172014":   "https://dandiarchive.s3.amazonaws.com/blobs/12f/dc6/12fdc641-306b-4c83-8112-8244e4a846e1",
    "Gatsby_08022013":   "https://dandiarchive.s3.amazonaws.com/blobs/810/e9d/810e9d83-a8f3-475a-9a71-634659b1c690",
    "Gatsby_08282013":   "https://dandiarchive.s3.amazonaws.com/blobs/560/a77/560a7710-50a3-4a16-9a5b-d079310eaa1f",
}


def open_session(name):
    """Stream one NWB file. Returns (pynapple view, pynwb file, io handle)."""
    rem_file = remfile.File(SESSIONS[name], disk_cache=remfile.DiskCache(CACHE_DIR))
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


# %% [markdown]
# ## 2. A first look at one session
#
# `Achilles_10252013` is used throughout as the prototype session. Printing the
# pynapple view shows what the file contains.

# %%
SESSION = "Achilles_10252013"
_nwb, _nwbfile, _io = open_session(SESSION)
print(_nwbfile.session_description[:200])
print()
print("subject:", _nwbfile.subject.subject_id, "|", _nwbfile.subject.species)
print()
print(_nwb)
print()
print("units columns:", _nwbfile.units.colnames)
print()
print(_nwbfile.epochs.to_dataframe())

# %% [markdown]
# So the file gives spike times for 137 sorted units with an excitatory /
# inhibitory label, a 128-channel LFP, sleep-state intervals, the animal's
# (x, y) tracking, and a linearized position along a 1.6 m linear maze.
#
# ### Two things in this file need care before it can be trusted
#
# **The position time base is mislabelled.** The `SpatialSeries` objects store
# `rate = 0.0256`, but the hc-11 tracking runs at 39.06 Hz, so what is in the
# `rate` field is actually the sampling *period*. Taking it at face value would
# spread 80,762 position samples over 36 days. Rebuilding the time base as
# `starting_time + i * rate` gives a duration that matches the MazeEpoch to
# within a second, which is the assertion used below.
#
# **One unit per session can be corrupt.** In three of the eight sessions, the
# row with unit id 2 holds two full-session spike trains concatenated end to
# end, so its spike times jump backwards to zero part way through. Pynapple
# sorts such a train silently, which would turn it into a plausible-looking but
# meaningless unit, so the check is done on the raw ragged array and offending
# rows are dropped.

# %%
def load_behavior_and_spikes(session_name):
    """Load one session's spikes and position, with the two fixes described above."""
    nwb, nwbfile, io = open_session(session_name)

    behavior = nwbfile.processing["behavior"]
    lin_name = [k for k in behavior.data_interfaces if "Linearized" in k][0]
    pos_name = [k for k in behavior.data_interfaces
                if k.endswith("Position") and "Linearized" not in k][0]
    maze_type = "circular" if "Circular" in lin_name else "linear"
    lin_ss = list(behavior[lin_name].spatial_series.values())[0]
    pos_ss = list(behavior[pos_name].spatial_series.values())[0]

    dt = float(lin_ss.rate)          # the sampling period, despite the field name
    lin = np.asarray(lin_ss.data[:]).ravel()
    xy = np.asarray(pos_ss.data[:])
    t = float(lin_ss.starting_time) + np.arange(lin.size) * dt

    epochs = nwb["epochs"]
    maze_ep = epochs[np.asarray(epochs.label) == "MazeEpoch"]
    mismatch = abs(lin.size * dt - (maze_ep.end[0] - maze_ep.start[0]))
    assert mismatch < 1.0, f"reconstructed time base is {mismatch:.2f} s off the MazeEpoch"

    good = np.isfinite(lin)
    position = nap.Tsd(t=t[good], d=lin[good] * 100.0, time_support=maze_ep)   # m -> cm
    xy_good = np.isfinite(xy).all(axis=1)
    position_2d = nap.TsdFrame(t=t[xy_good], d=xy[xy_good] * 100.0,
                               columns=["x", "y"], time_support=maze_ep)

    corrupt = []
    for row, uid in enumerate(np.asarray(nwbfile.units.id[:])):
        st = np.asarray(nwbfile.units["spike_times"][row])
        if st.size and np.any(np.diff(st) < 0):
            corrupt.append(int(uid))
    spikes = nwb["units"]
    if corrupt:
        spikes = spikes[[k for k in spikes.keys() if k not in corrupt]]

    meta = dict(
        session=session_name, subject=nwbfile.subject.subject_id,
        maze=lin_name.replace("LinearizedPosition", ""), maze_type=maze_type,
        track_length_cm=float(np.ceil(np.nanmax(lin) * 100.0 / 10.0) * 10.0),
        n_units_total=len(spikes), n_units_excluded_corrupt=len(corrupt),
        corrupt_unit_ids=tuple(corrupt),
        maze_duration_s=float(maze_ep.end[0] - maze_ep.start[0]),
        pos_fs_hz=1.0 / dt,
    )
    return spikes, position, position_2d, maze_ep, meta, (nwbfile, io)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name in SESSIONS:
        *_, m, handles = load_behavior_and_spikes(name)
        handles[1].close()
        print(f"{name:<20s} {m['maze']:<15s} L = {m['track_length_cm']:5.0f} cm   "
              f"units = {m['n_units_total']:3d}   corrupt dropped = "
              f"{m['n_units_excluded_corrupt']} {m['corrupt_unit_ids']}")

# %% [markdown]
# Three of the eight sessions were recorded on a closed **circular** maze, where
# a lap wraps around and the two-direction traversal logic used here does not
# apply. Those three are analysed nowhere below; the five linear-track sessions
# (four rats, 1.6 m and 2 m tracks) are used instead. The exclusion is by maze
# geometry, decided before looking at any spiking.

# %% [markdown]
# ## 3. Behavioural preprocessing
#
# The linearized position is only defined while the animal is on the track, so
# contiguous blocks of valid samples are candidate traversals. Brief LED
# dropouts fragment single traversals, so blocks separated by less than a second
# are merged first. A merged block counts as a traversal if it lasts at least
# 0.5 s, spans at least half the track, and is close to monotonic, which rejects
# back-and-forth wandering. Each traversal is then labelled rightward or
# leftward by the sign of its net displacement.
#
# For rate maps the traversals are further masked by a 5 cm/s speed threshold,
# so that pauses at the reward ends do not inflate occupancy. The unmasked
# traversals are kept as trials for raster plots and for the decoding folds.

# %%
POS_BIN_CM = 4.0             # spatial bin width for rate maps
SMOOTH_BINS = 1.0            # gaussian sigma, in bins
MIN_RUN_DURATION = 0.5       # s
MIN_RUN_COVERAGE_FRAC = 0.5  # fraction of the track a traversal must span
MAX_TRACKING_GAP = 1.0       # s; brief LED dropouts inside a single traversal
MIN_MONOTONICITY = 0.8       # |net displacement| / range
SPEED_THRESHOLD_CMS = 5.0
MIN_MEAN_RATE_HZ = 0.1       # exclude near-silent units
MAX_MEAN_RATE_HZ = 10.0      # exclude fast-spiking units from the pyramidal set
PEAK_RATE_CRITERION_HZ = 1.0
STABILITY_CRITERION = 0.5
N_SHUFFLES = 1000
SHUFFLE_ALPHA = 0.01

DIRECTIONS = ("rightward", "leftward")
DIR_COLORS = {"rightward": "#1f77b4", "leftward": "#d62728"}


def find_run_epochs(position, track_length):
    """Split the linearized position into individual, direction-labelled traversals."""
    t, d = position.t, position.values
    gap = np.diff(t) > MAX_TRACKING_GAP
    seg_start = np.concatenate([[0], np.flatnonzero(gap) + 1])
    seg_end = np.concatenate([np.flatnonzero(gap), [t.size - 1]])

    starts, ends, directions = [], [], []
    for i0, i1 in zip(seg_start, seg_end):
        seg = d[i0:i1 + 1]
        span = seg.max() - seg.min()
        net = seg[-1] - seg[0]
        if (t[i1] - t[i0] < MIN_RUN_DURATION
                or span < MIN_RUN_COVERAGE_FRAC * track_length
                or abs(net) < MIN_MONOTONICITY * span):
            continue
        starts.append(t[i0])
        ends.append(t[i1])
        directions.append(1 if net > 0 else -1)
    return nap.IntervalSet(start=np.array(starts), end=np.array(ends)), np.array(directions)


def compute_speed(position, run_ep):
    p = position.restrict(run_ep)
    return nap.Tsd(t=p.t, d=np.abs(np.gradient(p.values, p.t)), time_support=run_ep)


def direction_epochs(run_ep, directions, speed):
    """Whole traversals (trials) and their speed-masked versions (for rate maps)."""
    moving = speed.threshold(SPEED_THRESHOLD_CMS, "above").time_support
    trials, masked = {}, {}
    for label, sign in [("rightward", 1), ("leftward", -1)]:
        sel = np.flatnonzero(directions == sign)
        ep = nap.IntervalSet(start=run_ep.start[sel], end=run_ep.end[sel])
        trials[label] = ep
        masked[label] = ep.intersect(moving).drop_short_intervals(0.1)
    return trials, masked


def select_pyramidal(spikes, run_ep):
    """Putative pyramidal cells with a usable firing rate during track running."""
    sub = spikes.restrict(run_ep)
    rate = np.array([len(sub[k]) / run_ep.tot_length() for k in sub.keys()])
    keep = ((np.asarray(spikes.metadata["cell_type"]) == "excitatory")
            & (rate >= MIN_MEAN_RATE_HZ) & (rate <= MAX_MEAN_RATE_HZ))
    return spikes[list(np.array(list(spikes.keys()))[keep])]


# %% [markdown]
# ## 4. Rate maps, spatial information, and the shuffle test
#
# Rate maps are spike counts per spatial bin divided by occupancy, lightly
# smoothed. Three quantities are computed per cell and direction:
#
# - **Skaggs spatial information** in bits per spike,
#   $I = \sum_x P(x)\,\frac{\lambda(x)}{\bar\lambda}\log_2\frac{\lambda(x)}{\bar\lambda}$.
# - **A circular-shift null.** Spike times and position samples are projected
#   onto a virtual time axis in which the selected traversals are concatenated,
#   and each spike train is circularly shifted on that axis by a random offset.
#   This preserves the spike train's own temporal structure (burstiness, refractory
#   period, overall rate) while destroying its alignment to position, so it is a
#   much stricter null than shuffling spike times outright.
# - **Split-half reliability**, the correlation between rate maps built from
#   odd and even traversals.
#
# A cell counts as a place cell in a given direction if its information exceeds
# the 99th percentile of its own null, its peak rate is at least 1 Hz, and its
# odd/even map correlation exceeds 0.5.

# %%
def _virtual_time(x, ep):
    """Map times inside `ep` onto a concatenated ('virtual') time axis."""
    starts, ends = ep.start, ep.end
    cum = np.concatenate([[0.0], np.cumsum(ends - starts)])
    idx = np.clip(np.searchsorted(starts, x, side="right") - 1, 0, len(starts) - 1)
    return cum[idx] + (x - starts[idx]), cum[-1]


def _smooth(maps, sigma=SMOOTH_BINS):
    return gaussian_filter1d(maps, sigma, axis=-1, mode="nearest") if sigma > 0 else maps


class RateMapEngine:
    """Occupancy-normalised rate maps plus a circular-shift shuffle test."""

    def __init__(self, spikes, position, ep, n_bins, track_length, pos_fs):
        self.ep, self.n_bins = ep, n_bins
        self.bin_edges = np.linspace(0.0, track_length, n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.unit_ids = np.array(list(spikes.keys()))

        pos = position.restrict(ep)
        vt, self.total_time = _virtual_time(pos.t, ep)
        order = np.argsort(vt)
        self.pos_vt, self.pos_val = vt[order], pos.values[order]

        self.pos_dt = 1.0 / pos_fs
        self.occupancy = np.histogram(self.pos_val, bins=self.bin_edges)[0] * self.pos_dt
        self.occupancy_valid = self.occupancy > 0

        self.spike_vt = [np.sort(_virtual_time(spikes[k].restrict(ep).t, ep)[0])
                         for k in self.unit_ids]
        self.mean_rates = np.array([len(v) / self.total_time for v in self.spike_vt])

    def _maps_from_vt(self, spike_vt_list):
        maps = np.zeros((len(spike_vt_list), self.n_bins))
        for i, vt in enumerate(spike_vt_list):
            p = np.interp(vt, self.pos_vt, self.pos_val)
            counts = np.histogram(p, bins=self.bin_edges)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                maps[i] = np.where(self.occupancy_valid,
                                   counts / np.where(self.occupancy_valid, self.occupancy, 1),
                                   0.0)
        return maps

    def rate_maps(self, smooth=True):
        maps = self._maps_from_vt(self.spike_vt)
        return _smooth(maps) if smooth else maps

    def spatial_information(self, maps):
        p_x = self.occupancy / self.occupancy.sum()
        mean_rate = (maps * p_x[None, :]).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = maps / np.where(mean_rate > 0, mean_rate, 1.0)[:, None]
            term = np.where(maps > 0, maps * np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
        bits_per_sec = (term * p_x[None, :]).sum(axis=1)
        bits_per_spike = np.where(mean_rate > 0,
                                  bits_per_sec / np.where(mean_rate > 0, mean_rate, 1.0), 0.0)
        return bits_per_spike, bits_per_sec

    def shuffle_information(self, n_shuffles=N_SHUFFLES, seed=0, progress=True):
        rng = np.random.default_rng(seed)
        lo, hi = 0.05 * self.total_time, 0.95 * self.total_time
        out = np.zeros((n_shuffles, len(self.unit_ids)))
        it = tqdm(range(n_shuffles), desc="shuffles", leave=False) if progress else range(n_shuffles)
        for s in it:
            offsets = rng.uniform(lo, hi, size=len(self.unit_ids))
            shifted = [np.sort((vt + o) % self.total_time)
                       for vt, o in zip(self.spike_vt, offsets)]
            out[s] = self.spatial_information(_smooth(self._maps_from_vt(shifted)))[0]
        return out

    def split_half_maps(self):
        """Rate maps built separately from odd and even traversals."""
        cum = np.concatenate([[0.0], np.cumsum(self.ep.end - self.ep.start)])
        halves = []
        for parity in (0, 1):
            sel = np.arange(len(self.ep)) % 2 == parity
            occ_mask = np.zeros(self.pos_vt.size, bool)
            spk_masks = [np.zeros(v.size, bool) for v in self.spike_vt]
            for a, b in zip(cum[:-1][sel], cum[1:][sel]):
                occ_mask |= (self.pos_vt >= a) & (self.pos_vt < b)
                for j, v in enumerate(self.spike_vt):
                    spk_masks[j] |= (v >= a) & (v < b)
            occ = np.histogram(self.pos_val[occ_mask], bins=self.bin_edges)[0] * self.pos_dt
            maps = np.zeros((len(self.spike_vt), self.n_bins))
            for j, v in enumerate(self.spike_vt):
                p = np.interp(v[spk_masks[j]], self.pos_vt, self.pos_val)
                counts = np.histogram(p, bins=self.bin_edges)[0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    maps[j] = np.where(occ > 0, counts / np.where(occ > 0, occ, 1), 0.0)
            halves.append(_smooth(maps))
        return halves


def split_half_correlation(a_maps, b_maps):
    out = np.full(a_maps.shape[0], np.nan)
    for i, (a, b) in enumerate(zip(a_maps, b_maps)):
        if a.std() > 0 and b.std() > 0:
            out[i] = np.corrcoef(a, b)[0, 1]
    return out


def field_properties(maps, bin_centers):
    """Peak rate, peak location, and field width at 50 % of peak."""
    peak_rate = maps.max(axis=1)
    peak_bin = maps.argmax(axis=1)
    bin_w = bin_centers[1] - bin_centers[0]
    widths = np.zeros(maps.shape[0])
    for i in range(maps.shape[0]):
        if peak_rate[i] <= 0:
            continue
        above = maps[i] >= 0.5 * peak_rate[i]
        lo = hi = peak_bin[i]
        while lo - 1 >= 0 and above[lo - 1]:
            lo -= 1
        while hi + 1 < maps.shape[1] and above[hi + 1]:
            hi += 1
        widths[i] = (hi - lo + 1) * bin_w
    return peak_rate, bin_centers[peak_bin], widths


def classify_place_cells(engine, n_shuffles=N_SHUFFLES, seed=0, progress=True):
    maps = engine.rate_maps()
    info_bits, info_rate = engine.spatial_information(maps)
    null = engine.shuffle_information(n_shuffles=n_shuffles, seed=seed, progress=progress)
    p_val = (null >= info_bits[None, :]).sum(axis=0) / n_shuffles
    sd = null.std(axis=0)
    z = (info_bits - null.mean(axis=0)) / np.where(sd > 0, sd, np.nan)

    stability = split_half_correlation(*engine.split_half_maps())
    peak_rate, peak_pos, width = field_properties(maps, engine.bin_centers)

    df = pd.DataFrame({
        "unit": engine.unit_ids, "mean_rate": engine.mean_rates, "peak_rate": peak_rate,
        "peak_pos_cm": peak_pos, "field_width_cm": width,
        "info_bits_per_spike": info_bits, "info_bits_per_sec": info_rate,
        "shuffle_p": p_val, "info_z": z, "stability": stability,
    })
    df["is_place_cell"] = ((df.shuffle_p < SHUFFLE_ALPHA)
                           & (df.peak_rate >= PEAK_RATE_CRITERION_HZ)
                           & (df.stability > STABILITY_CRITERION))
    return df, maps, null


def analyze_session(session_name, n_shuffles=N_SHUFFLES, seed=0, progress=True,
                    keep_handles=False):
    """Full single-session place-cell pipeline for a linear track."""
    spikes, position, pos2d, maze_ep, meta, handles = load_behavior_and_spikes(session_name)
    if meta["maze_type"] != "linear":
        raise ValueError(f"{session_name} uses a {meta['maze_type']} maze")
    n_bins = int(round(meta["track_length_cm"] / POS_BIN_CM))
    meta["n_pos_bins"] = n_bins

    run_ep, directions = find_run_epochs(position, meta["track_length_cm"])
    speed = compute_speed(position, run_ep)
    trial_eps, dir_eps = direction_epochs(run_ep, directions, speed)
    all_run_ep = dir_eps["rightward"].union(dir_eps["leftward"])
    pyr = select_pyramidal(spikes, all_run_ep)

    per_dir = {}
    for label in DIRECTIONS:
        engine = RateMapEngine(pyr, position, dir_eps[label], n_bins,
                               meta["track_length_cm"], meta["pos_fs_hz"])
        table, maps, null = classify_place_cells(engine, n_shuffles, seed, progress)
        table.insert(0, "direction", label)
        table.insert(0, "session", session_name)
        per_dir[label] = dict(engine=engine, table=table, maps=maps, null=null)

    meta.update(n_runs=len(run_ep), n_runs_right=int((directions == 1).sum()),
                n_runs_left=int((directions == -1).sum()),
                run_time_s=float(run_ep.tot_length()), n_pyramidal=len(pyr),
                median_speed_cms=float(np.median(speed.values)))
    out = dict(meta=meta, spikes=spikes, pyr=pyr, position=position, position_2d=pos2d,
               maze_ep=maze_ep, run_ep=run_ep, directions=directions, speed=speed,
               trial_eps=trial_eps, dir_eps=dir_eps, per_dir=per_dir)
    if keep_handles:
        out["handles"] = handles
    else:
        handles[1].close()
    return out


# %%
res = analyze_session(SESSION, keep_handles=True)
meta = res["meta"]
nwbfile = res["handles"][0]
position, pos2d, pyr = res["position"], res["position_2d"], res["pyr"]
run_ep, directions, speed = res["run_ep"], res["directions"], res["speed"]
trial_eps, dir_eps = res["trial_eps"], res["dir_eps"]
tables = {d: res["per_dir"][d]["table"] for d in DIRECTIONS}
maps = {d: res["per_dir"][d]["maps"] for d in DIRECTIONS}
nulls = {d: res["per_dir"][d]["null"] for d in DIRECTIONS}
bin_centers = res["per_dir"]["rightward"]["engine"].bin_centers
TRACK_LEN, N_BINS = meta["track_length_cm"], meta["n_pos_bins"]

print(f"{SESSION}: {meta['n_units_total']} units -> {meta['n_pyramidal']} putative pyramidal")
print(f"{meta['n_runs']} traversals ({meta['n_runs_right']} rightward, "
      f"{meta['n_runs_left']} leftward), {meta['run_time_s']:.0f} s of running")
for d in DIRECTIONS:
    t = tables[d]
    print(f"  {d:<10s} place cells: {t.is_place_cell.sum():3d} / {len(t)} "
          f"({100 * t.is_place_cell.mean():.0f} %)")

# %% [markdown]
# ### Figure 1. Behaviour on the track
#
# The animal shuttles end to end along a 1.6 m track. Occupancy is close to
# uniform apart from the reward ends, and running speed peaks around 60-80 cm/s,
# so the rate maps below are not dominated by a few over-sampled locations.

# %%
fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                      left=0.07, right=0.98, top=0.90, bottom=0.09)

ax = fig.add_subplot(gs[0, 0])
p2 = pos2d.restrict(res["maze_ep"])
ax.plot(p2["x"].values, p2["y"].values, lw=0.3, color="0.75", label="all tracking")
pr = pos2d.restrict(run_ep)
ax.plot(pr["x"].values, pr["y"].values, ".", ms=0.7, color="#1f77b4", label="scored traversals")
ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=6)
ax.set_title("A  2-D tracking, maze epoch", loc="left", fontweight="bold")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[0, 1:])
win = nap.IntervalSet(start=run_ep.start[10] - 5, end=run_ep.start[10] + 115)
for lab in DIRECTIONS:
    for s, e in zip(*[getattr(dir_eps[lab].intersect(win), a) for a in ("start", "end")]):
        ax.axvspan(s - win.start[0], e - win.start[0], color=DIR_COLORS[lab], alpha=0.16, lw=0)
pw = position.restrict(win)
ax.plot(pw.t - win.start[0], pw.values, ".", ms=1.6, color="k")
ax.set_xlabel("time from window start (s)"); ax.set_ylabel("linearized position (cm)")
ax.set_title("B  Track traversals (blue = rightward, red = leftward)",
             loc="left", fontweight="bold")
ax.set_xlim(0, 120)

ax = fig.add_subplot(gs[1, 0])
ax.hist(speed.values, bins=60, color="0.4")
ax.axvline(SPEED_THRESHOLD_CMS, color="crimson", ls="--", lw=1.2)
ax.set_xlabel("running speed (cm/s)"); ax.set_ylabel("position samples")
ax.set_title("C  Speed during traversals", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 1])
w = bin_centers[1] - bin_centers[0]
for lab, shift in [("rightward", -w / 4), ("leftward", w / 4)]:
    ax.bar(bin_centers + shift, res["per_dir"][lab]["engine"].occupancy,
           width=w / 2, color=DIR_COLORS[lab], label=lab)
ax.set_xlabel("position (cm)"); ax.set_ylabel("occupancy (s)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("D  Spatial occupancy", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 2])
all_run_ep = dir_eps["rightward"].union(dir_eps["leftward"])
sub = res["spikes"].restrict(all_run_ep)
rates_on_track = np.array([len(sub[k]) / all_run_ep.tot_length() for k in sub.keys()])
ctype = np.asarray(res["spikes"].metadata["cell_type"])
bins = np.logspace(-2, 1.8, 30)
ax.hist(rates_on_track[ctype == "excitatory"], bins=bins, color="#4c72b0", alpha=0.85,
        label="excitatory")
ax.hist(rates_on_track[ctype == "inhibitory"], bins=bins, color="#dd8452", alpha=0.85,
        label="inhibitory")
ax.set_xscale("log")
ax.set_xlabel("firing rate on track (Hz)"); ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("E  Unit firing rates", loc="left", fontweight="bold")

fig.suptitle(f"DANDI:000044  {SESSION}  (rat {meta['subject']}, 1.6 m linear track)",
             fontweight="bold")
fig.savefig("fig01_behavior_overview.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Figure 2. Raw neural activity
#
# Before any tuning curve is computed, the effect is already visible in the raw
# spike raster. Units are ordered by where their *rightward* place field peaks,
# and on the rightward traversal the population sweeps once through that
# ordering as a clear diagonal. The leftward traversal does not simply run the
# diagonal backwards, because the fields are direction-specific; Figure 4 makes
# that point quantitatively. The LFP shows the strong 6-10 Hz theta rhythm
# characteristic of hippocampal locomotion.
#
# The electrode table in this dandiset carries no anatomical location, so the
# CA1 channel is chosen functionally, as the channel with the largest
# theta-to-delta power ratio during running.

# %%
lfp_es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
fs_lfp = float(lfp_es.rate)
i0 = int(run_ep.start[0] * fs_lfp)
block = np.asarray(lfp_es.data[i0:i0 + int(60 * fs_lfp), :]) * lfp_es.conversion
freqs, pxx = welch(block, fs=fs_lfp, nperseg=int(4 * fs_lfp), axis=0)
td_ratio = (pxx[(freqs >= 6) & (freqs <= 10)].mean(axis=0)
            / pxx[(freqs >= 2) & (freqs <= 4)].mean(axis=0))
ca1_ch = int(np.argmax(td_ratio))
print(f"LFP channel {ca1_ch} chosen (theta/delta = {td_ratio[ca1_ch]:.2f})")

t_start, t_stop = run_ep.start[10] - 2.0, run_ep.end[11] + 2.0
j0, j1 = int(t_start * fs_lfp), int(t_stop * fs_lfp)
lfp = nap.Tsd(t=np.arange(j0, j1) / fs_lfp,
              d=np.asarray(lfp_es.data[j0:j1, ca1_ch]) * lfp_es.conversion * 1e3)
theta = nap.apply_bandpass_filter(lfp, (6.0, 10.0), fs=fs_lfp)

win = nap.IntervalSet(start=t_start, end=t_stop)
order = np.argsort(bin_centers[np.nanargmax(maps["rightward"], axis=1)])
ordered_keys = np.array(list(pyr.keys()))[order]

fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.0, 3.4, 1.0], "hspace": 0.30})
axes[0].plot(lfp.t - t_start, lfp.values, lw=0.5, color="0.6", label="raw LFP")
axes[0].plot(theta.t - t_start, theta.values, lw=1.1, color="#c44e52", label="6-10 Hz theta")
axes[0].set_ylabel("CA1 LFP (mV)")
lim = np.percentile(np.abs(lfp.values), 99.8)
axes[0].set_ylim(-lim, lim)
axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
axes[0].set_title("A  Hippocampal LFP (theta-dominated during running)",
                  loc="left", fontweight="bold", pad=6)

for row, k in enumerate(ordered_keys):
    st = pyr[k].restrict(win).t - t_start
    axes[1].plot(st, np.full_like(st, row), "|", ms=3.2, color="k", mew=0.7)
axes[1].set_ylabel("unit (ordered by place-field peak)")
axes[1].set_ylim(-1, len(ordered_keys))
axes[1].set_title(f"B  Spike raster of {len(pyr)} putative pyramidal cells, ordered by "
                  "rightward place-field peak",
                  loc="left", fontweight="bold", pad=6)

pw = position.restrict(win)
axes[2].plot(pw.t - t_start, pw.values, ".", ms=2.5, color="k")
for lab in DIRECTIONS:
    ep = trial_eps[lab].intersect(win)
    for s, e in zip(ep.start, ep.end):
        for a in axes:
            a.axvspan(s - t_start, e - t_start, color=DIR_COLORS[lab], alpha=0.14, lw=0)
axes[2].set_ylabel("position (cm)"); axes[2].set_xlabel("time (s)")
axes[2].set_title("C  Linearized position (blue = rightward, red = leftward)",
                  loc="left", fontweight="bold", pad=6)
axes[2].set_xlim(0, t_stop - t_start)
fig.suptitle(f"Raw data, {SESSION}", fontweight="bold", y=0.955)
fig.savefig("fig02_raw_activity.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Figure 3. Individual place cells
#
# For six cells spanning the track, the top panel of each pair plots the
# animal's position at the time of every spike, one row per traversal. Spikes
# cluster in a narrow band of positions that is the same on traversal after
# traversal, which is the defining signature of a place field. The bottom panel
# is the occupancy-normalised rate map that band produces.

# %%
def spike_positions_by_trial(unit_ts, position, trial_ep):
    out = []
    for i in range(len(trial_ep)):
        st = unit_ts.restrict(trial_ep[i]).t
        p = position.restrict(trial_ep[i])
        out.append(np.interp(st, p.t, p.values) if p.t.size > 1 else np.array([]))
    return out


tr = tables["rightward"]
good = (tr.is_place_cell.values & (tr.field_width_cm.values < 55)
        & (tr.peak_rate.values > 3.0) & (tr.stability.values > 0.7))
cand = np.flatnonzero(good)
examples = list(dict.fromkeys(
    [cand[np.argmin(np.abs(tr.peak_pos_cm.values[cand] - x))]
     for x in np.linspace(0.05 * TRACK_LEN, 0.95 * TRACK_LEN, 6)]))

fig = plt.figure(figsize=(12, 7.0))
outer = fig.add_gridspec(2, 1, hspace=0.42, left=0.07, right=0.98, top=0.86, bottom=0.08)
for block in range(2):
    inner = outer[block].subgridspec(2, 3, height_ratios=[2.3, 1.0], hspace=0.10, wspace=0.24)
    for c in range(3):
        n = block * 3 + c
        if n >= len(examples):
            continue
        idx = examples[n]
        unit = int(tr.unit.values[idx])
        ax_r = fig.add_subplot(inner[0, c])
        ax_t = fig.add_subplot(inner[1, c], sharex=ax_r)

        offset = 0
        for d in DIRECTIONS:
            for j, p in enumerate(spike_positions_by_trial(pyr[unit], position, trial_eps[d])):
                ax_r.plot(p, np.full_like(p, offset + j), "|", ms=3.0,
                          color=DIR_COLORS[d], mew=0.8)
            offset += len(trial_eps[d]) + 3
        ax_r.set_ylim(-1, offset - 2)
        ax_r.set_title(f"unit {unit}   ({tr.info_bits_per_spike.values[idx]:.2f} bits/spike)",
                       fontsize=8.5, pad=4)
        ax_r.tick_params(labelbottom=False)
        if c == 0:
            ax_r.set_ylabel("traversal")

        for d in DIRECTIONS:
            ax_t.plot(bin_centers, maps[d][idx], color=DIR_COLORS[d], lw=1.6,
                      label=d if n == 0 else None)
        ax_t.set_xlim(0, TRACK_LEN)
        ax_t.set_xlabel("position (cm)")
        if c == 0:
            ax_t.set_ylabel("rate (Hz)")
        if n == 0:
            ax_t.legend(frameon=False, fontsize=7.5, loc="upper right")

fig.suptitle("Individual CA1 place cells fire at a reproducible track location\n"
             "top of each pair: spike positions on every traversal (blue = rightward, "
             "red = leftward);   bottom: occupancy-normalised rate map",
             fontweight="bold", fontsize=10)
fig.savefig("fig03_example_place_cells.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Figure 4. The population tiles the track, and the code is directional
#
# Sorting the place cells by where their rightward field peaks produces a clean
# diagonal: every point on the track is covered by a distinct subset of cells.
# Applying that same sorting to the leftward rate maps scrambles it, which is
# the classic result that place fields on a linear track are direction-specific
# rather than a pure map of allocentric space.

# %%
fig = plt.figure(figsize=(11.5, 5.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.06, 1.05], wspace=0.42,
                      left=0.065, right=0.98, top=0.82, bottom=0.13)
pc_r = tables["rightward"].is_place_cell.values
sort_ref = np.argsort(tables["rightward"].peak_pos_cm.values[pc_r])

for i, d in enumerate(DIRECTIONS):
    ax = fig.add_subplot(gs[0, i])
    m = maps[d][pc_r][sort_ref]
    m = m / np.maximum(m.max(axis=1, keepdims=True), 1e-9)
    im = ax.imshow(m, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, TRACK_LEN, 0, m.shape[0]], vmin=0, vmax=1)
    ax.set_xlabel("position (cm)")
    ax.set_title(f"{'AB'[i]}  {d} runs", loc="left", fontweight="bold")
    if i == 0:
        ax.set_ylabel("place cell (sorted by rightward peak)")
fig.colorbar(im, cax=fig.add_subplot(gs[0, 2])).set_label("normalised firing rate")

ax = fig.add_subplot(gs[0, 3])
for d in DIRECTIONS:
    ax.hist(tables[d].peak_pos_cm.values[tables[d].is_place_cell.values],
            bins=np.linspace(0, TRACK_LEN, 17), histtype="step", lw=1.8,
            color=DIR_COLORS[d], label=d)
ax.set_xlabel("place-field peak position (cm)"); ax.set_ylabel("number of place cells")
ax.legend(frameon=False, fontsize=8)
ax.set_title("C  Fields tile the track", loc="left", fontweight="bold")

fig.suptitle(f"Population place-field map, {SESSION}.  Sorting by rightward peak (A) produces "
             "a continuous diagonal;\nthe same sorting applied to leftward runs (B) is "
             "scrambled, so the fields are direction-specific.",
             fontweight="bold", fontsize=9.5)
fig.savefig("fig04_population_place_fields.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Figure 5. Spatial information against the circular-shift null
#
# Panel B is the important one: each point compares a unit's observed spatial
# information against the 99th percentile of that unit's *own* shuffle
# distribution, so low-rate units, which have inflated information by chance,
# are held to a correspondingly higher bar.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
fig.subplots_adjust(wspace=0.34, top=0.80, bottom=0.19, left=0.07, right=0.98)

d = "rightward"
t = tables[d]
ax = axes[0]
ax.hist(nulls[d].ravel(), bins=60, density=True, color="0.75", label="circular-shift null")
ax.hist(t.info_bits_per_spike.values, bins=30, density=True, histtype="step", lw=1.8,
        color=DIR_COLORS[d], label="observed")
ax.set_xlabel("spatial information (bits/spike)"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.set_title("A  Observed vs null, rightward", loc="left", fontweight="bold")

ax = axes[1]
p99 = np.percentile(nulls[d], 100 * (1 - SHUFFLE_ALPHA), axis=0)
sig = t.shuffle_p.values < SHUFFLE_ALPHA
ax.scatter(p99[~sig], t.info_bits_per_spike.values[~sig], s=16, color="0.6",
           label=f"n.s. (n={(~sig).sum()})")
ax.scatter(p99[sig], t.info_bits_per_spike.values[sig], s=16, color=DIR_COLORS[d],
           label=f"p < {SHUFFLE_ALPHA} (n={sig.sum()})")
lim = [0, max(p99.max(), t.info_bits_per_spike.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("99th percentile of that unit's null (bits/spike)")
ax.set_ylabel("observed (bits/spike)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("B  Per-unit significance", loc="left", fontweight="bold")

ax = axes[2]
for dd in DIRECTIONS:
    ax.hist(tables[dd].stability.values, bins=np.linspace(-1, 1, 25), histtype="step",
            lw=1.8, color=DIR_COLORS[dd], label=dd)
ax.axvline(STABILITY_CRITERION, color="k", ls="--", lw=1.0)
ax.set_xlabel("odd/even traversal map correlation"); ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("C  Within-session stability", loc="left", fontweight="bold")

n_pc = int((tables["rightward"].is_place_cell | tables["leftward"].is_place_cell).sum())
fig.suptitle(f"Place-cell criteria, {SESSION}: {n_pc}/{len(t)} pyramidal cells qualify in at "
             "least one direction\n(spatial information above the 99th shuffle percentile, "
             "peak > 1 Hz, odd/even map correlation > 0.5)", fontweight="bold", fontsize=10)
fig.savefig("fig05_spatial_information.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Figure 6. Place-field properties
#
# Field widths and peak rates fall where the CA1 literature puts them for a
# track of this length. The directionality index in panel D is bimodal: most
# cells fire substantially more in one travel direction than the other.

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
fig.subplots_adjust(wspace=0.38, top=0.79, bottom=0.20, left=0.055, right=0.99)
pc_mask = {d: tables[d].is_place_cell.values for d in DIRECTIONS}

ax = axes[0]
w = np.concatenate([tables[d].field_width_cm.values[pc_mask[d]] for d in DIRECTIONS])
ax.hist(w, bins=np.arange(0, 100, 6), color="#4c72b0")
ax.axvline(np.median(w), color="crimson", ls="--")
ax.set_xlabel("field width at 50 % of peak (cm)"); ax.set_ylabel("place fields")
ax.set_title(f"A  Width, median {np.median(w):.0f} cm", loc="left", fontweight="bold")

ax = axes[1]
pk = np.concatenate([tables[d].peak_rate.values[pc_mask[d]] for d in DIRECTIONS])
ax.hist(pk, bins=np.logspace(0, 2, 24), color="#4c72b0")
ax.set_xscale("log")
ax.axvline(np.median(pk), color="crimson", ls="--")
ax.set_xlabel("in-field peak rate (Hz)"); ax.set_ylabel("place fields")
ax.set_title(f"B  Peak rate, median {np.median(pk):.1f} Hz", loc="left", fontweight="bold")

ax = axes[2]
both_pc = pc_mask["rightward"] & pc_mask["leftward"]
xr = tables["rightward"].peak_pos_cm.values[both_pc]
yl = tables["leftward"].peak_pos_cm.values[both_pc]
ax.scatter(xr, yl, s=16, color="0.3")
ax.plot([0, TRACK_LEN], [0, TRACK_LEN], "k--", lw=1)
ax.set_xlabel("peak position, rightward (cm)"); ax.set_ylabel("peak position, leftward (cm)")
ax.set_title(f"C  Peak location, r = {np.corrcoef(xr, yl)[0, 1]:.2f}",
             loc="left", fontweight="bold")

ax = axes[3]
rr, ll = tables["rightward"].peak_rate.values, tables["leftward"].peak_rate.values
sel = pc_mask["rightward"] | pc_mask["leftward"]
di = (rr - ll) / (rr + ll)
ax.hist(di[sel], bins=np.linspace(-1, 1, 25), color="#4c72b0")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("directionality index\n(right - left) / (right + left)")
ax.set_ylabel("place cells")
ax.set_title(f"D  {100 * np.mean(np.abs(di[sel]) > 0.3):.0f} % strongly directional",
             loc="left", fontweight="bold")

fig.suptitle(f"Place-field properties, {SESSION}", fontweight="bold", fontsize=10)
fig.savefig("fig06_field_properties.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Reading position back out of the population
#
# Tuning curves fitted to the data they are evaluated on will always look
# informative. The stronger test is whether the population code generalises:
# fit rate maps on half the traversals, then ask what a Bayesian observer
# infers about position from spike counts in the other half.
#
# Decoding uses `pynapple.decode_bayes`, which assumes conditionally
# independent Poisson firing,
# $P(x \mid n) \propto P(x) \prod_i \frac{\lambda_i(x)^{n_i} e^{-\lambda_i(x)}}{n_i!}$.
# Two folds are run per direction (train on even traversals, test on odd, then
# the reverse), and **no place-cell selection is applied**: every putative
# pyramidal cell is handed to the decoder.
#
# The null is the same decoder with each unit's tuning curve circularly rolled
# by a random amount, which preserves the shape and amplitude of every tuning
# curve but destroys the agreement between cells about where the animal is.

# %%
DECODE_BIN_SIZE = 0.2  # s
rng = np.random.default_rng(1)


def decode_fold(res, direction, parity, units=None, shuffle_tc=False, rng=None,
                bin_size=DECODE_BIN_SIZE):
    units = res["pyr"] if units is None else units
    pos = res["position"]
    trials = res["trial_eps"][direction]
    idx = np.arange(len(trials))
    train_ep = res["dir_eps"][direction].intersect(trials[idx[idx % 2 == parity]])
    test_trials = trials[idx[idx % 2 != parity]]

    tc = nap.compute_tuning_curves(units, pos, bins=res["meta"]["n_pos_bins"],
                                   range=[(0, res["meta"]["track_length_cm"])],
                                   epochs=train_ep, feature_names=["position"])
    tc.data = gaussian_filter1d(np.nan_to_num(tc.data), SMOOTH_BINS, axis=-1, mode="nearest")
    if shuffle_tc:
        rng = np.random.default_rng() if rng is None else rng
        tc.data = np.stack([np.roll(r, rng.integers(tc.data.shape[-1])) for r in tc.data])

    decoded, prob = nap.decode_bayes(tc, units, test_trials, bin_size=bin_size)
    true = np.interp(decoded.t, pos.t, pos.values)
    return decoded, prob, true, test_trials, tc


def decoding_error(res, units=None, shuffle_tc=False, rng=None, bin_size=DECODE_BIN_SIZE):
    errs = []
    for direction in DIRECTIONS:
        for parity in (0, 1):
            dec, _, true, _, _ = decode_fold(res, direction, parity, units, shuffle_tc,
                                             rng, bin_size)
            ok = np.isfinite(true)
            errs.append(np.abs(dec.values[ok] - true[ok]))
    return np.concatenate(errs)


store, dec_list, true_list = {}, [], []
for direction in DIRECTIONS:
    for parity in (0, 1):
        dec, prob, true, test_trials, tc = decode_fold(res, direction, parity)
        store[(direction, parity)] = (dec, prob, true, test_trials, tc)
        ok = np.isfinite(true)
        dec_list.append(dec.values[ok]); true_list.append(true[ok])

dec_all, true_all = np.concatenate(dec_list), np.concatenate(true_list)
err = np.abs(dec_all - true_all)
err_shuf = decoding_error(res, shuffle_tc=True, rng=rng)
print(f"median decoding error: {np.median(err):.1f} cm "
      f"(rolled tuning curves: {np.median(err_shuf):.1f} cm; track is {TRACK_LEN:.0f} cm)")
print(f"fraction of 200 ms bins decoded within 20 cm: {np.mean(err < 20):.2f}")

# %%
sizes = sorted({min(s, len(pyr)) for s in [2, 5, 10, 20, 40, 60, 80, len(pyr)]})
keys = np.array(list(pyr.keys()))
curve_med, curve_lo, curve_hi = [], [], []
for n in tqdm(sizes, desc="population size"):
    meds = [np.median(decoding_error(res, units=pyr[list(rng.choice(keys, n, replace=False))]))
            for _ in range(8 if n < len(pyr) else 1)]
    curve_med.append(np.median(meds)); curve_lo.append(min(meds)); curve_hi.append(max(meds))

# %% [markdown]
# ### Figure 7. Bayesian decoding
#
# The posterior tracks the animal along each held-out traversal. Median error is
# a few centimetres on a 1.6 m track, roughly an order of magnitude better than
# the rolled-tuning-curve null, and accuracy improves monotonically as more
# cells are added, which is what a distributed population code predicts.

# %%
fig = plt.figure(figsize=(12.5, 7.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.46, wspace=0.32,
                      left=0.06, right=0.98, top=0.86, bottom=0.09)

ax = fig.add_subplot(gs[0, :])
dec, prob, true, test_trials, tc = store[("rightward", 0)]
sel = test_trials[:6]
P = np.asarray(prob.restrict(sel).values).T
tt = prob.restrict(sel).t
xt = np.arange(tt.size)
im = ax.pcolormesh(xt, tc.coords["position"].values, P, cmap="magma",
                   vmin=0, vmax=np.percentile(P, 99.5))
ax.plot(xt, np.interp(tt, position.t, position.values), color="#4fd1c5", lw=2.0,
        label="true position")
ax.plot(xt, dec.restrict(sel).values, ".", ms=3.5, color="w", label="decoded (MAP)")
for g in np.flatnonzero(np.diff(tt) > 3 * DECODE_BIN_SIZE):
    ax.axvline(g + 0.5, color="0.8", lw=1.0, ls=":")
ax.set_xlabel(f"{DECODE_BIN_SIZE * 1000:.0f} ms decoding bins (six consecutive held-out "
              "rightward traversals, separated by dotted lines)")
ax.set_ylabel("position (cm)")
ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor="w")
ax.set_title("A  Posterior probability over position, decoded from held-out traversals",
             loc="left", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("P(position | spikes)")

ax = fig.add_subplot(gs[1, 0])
edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
H = np.histogram2d(true_all, dec_all, bins=[edges, edges])[0]
H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
im = ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
               extent=[0, TRACK_LEN, 0, TRACK_LEN])
ax.plot([0, TRACK_LEN], [0, TRACK_LEN], "w--", lw=1)
ax.set_xlabel("true position (cm)"); ax.set_ylabel("decoded position (cm)")
ax.set_title("B  Confusion matrix", loc="left", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("P(decoded | true)")

ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, TRACK_LEN, 45)
ax.hist(err_shuf, bins=bins, density=True, color="0.75", label="rolled tuning curves")
ax.hist(err, bins=bins, density=True, histtype="step", lw=2.0, color="#1f77b4",
        label="observed")
ax.axvline(np.median(err), color="#1f77b4", ls="--", lw=1.2)
ax.set_xlabel("absolute decoding error (cm)"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"C  Median error {np.median(err):.1f} cm vs {np.median(err_shuf):.1f} cm",
             loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 2])
ax.fill_between(sizes, curve_lo, curve_hi, color="#1f77b4", alpha=0.25)
ax.plot(sizes, curve_med, "o-", color="#1f77b4", ms=4)
ax.axhline(np.median(err_shuf), color="0.5", ls="--", lw=1.2, label="chance")
ax.set_xscale("log")
ax.set_xlabel("number of cells used"); ax.set_ylabel("median error (cm)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("D  Accuracy grows with population", loc="left", fontweight="bold")

fig.suptitle(f"Bayesian decoding of position from CA1 spiking, {SESSION}\n"
             "tuning curves fitted on even traversals and tested on odd ones "
             "(and vice versa); no place-cell selection applied",
             fontweight="bold", fontsize=10)
fig.savefig("fig07_bayesian_decoding.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. A Poisson GLM of position tuning (NeMoS)
#
# The shuffle test asks whether a cell's spikes are more spatially concentrated
# than chance. A GLM asks something stronger and more direct: does knowing the
# animal's position let a Poisson model predict a cell's spike counts in
# traversals it was never fitted on?
#
# Position is expanded in 12 B-spline basis functions, spikes are binned at
# 40 ms, and one Poisson GLM per cell is fitted on half the traversals and
# scored on the other half with McFadden's pseudo-$R^2$. A pseudo-$R^2$ above
# zero on held-out data means position genuinely improves prediction relative to
# a constant-rate model; below zero means the position model overfits and does
# worse than knowing nothing.

# %%
import nemos as nmo  # noqa: E402

GLM_BIN_SIZE = 0.04
N_BASIS = 12

basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS, bounds=(0.0, TRACK_LEN), label="position")
grid, basis_grid = basis.evaluate_on_grid(200)


def glm_design(epochs):
    counts = pyr.count(GLM_BIN_SIZE, ep=epochs)
    pos = position.interpolate(counts, ep=counts.time_support)
    X = np.asarray(basis.compute_features(pos))
    ok = np.isfinite(X).all(axis=1)
    return X[ok], np.asarray(counts)[ok]


unit_keys = np.array(list(pyr.keys()))
glm_scores = {d: np.full((len(unit_keys), 2), np.nan) for d in DIRECTIONS}
glm_tuning = {d: np.zeros((len(unit_keys), grid.size)) for d in DIRECTIONS}

for direction in DIRECTIONS:
    for parity in (0, 1):
        trials = trial_eps[direction]
        idx = np.arange(len(trials))
        Xtr, Ytr = glm_design(dir_eps[direction].intersect(trials[idx[idx % 2 == parity]]))
        Xte, Yte = glm_design(dir_eps[direction].intersect(trials[idx[idx % 2 != parity]]))
        for i in tqdm(range(len(unit_keys)), desc=f"GLM {direction} fold {parity}", leave=False):
            # A unit silent in one fold has no rate to initialise the intercept
            # from and no null model to score against; leave it NaN.
            if Ytr[:, i].sum() == 0 or Yte[:, i].sum() == 0:
                glm_tuning[direction][i] = np.nan
                continue
            model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                                solver_name="LBFGS").fit(Xtr, Ytr[:, i])
            glm_scores[direction][i, parity] = model.score(
                Xte, Yte[:, i], score_type="pseudo-r2-McFadden")
            rate = np.exp(basis_grid @ model.coef_ + model.intercept_)
            glm_tuning[direction][i] += 0.5 * np.ravel(rate) / GLM_BIN_SIZE

cv_r2 = {}
for d in DIRECTIONS:
    ok = np.isfinite(glm_scores[d])
    n_ok = ok.sum(axis=1)
    cv_r2[d] = np.where(n_ok > 0,
                        np.where(ok, glm_scores[d], 0.0).sum(axis=1) / np.maximum(n_ok, 1),
                        np.nan)
print("unit-fold combinations skipped (silent in a fold):",
      sum(int((~np.isfinite(glm_scores[d])).any(axis=1).sum()) for d in DIRECTIONS))
for d in DIRECTIONS:
    pc = tables[d].is_place_cell.values
    print(f"{d}: median held-out pseudo-R2  place cells {np.nanmedian(cv_r2[d][pc]):+.3f}, "
          f"other units {np.nanmedian(cv_r2[d][~pc]):+.3f}")
    print(f"{' ':>{len(d)}}  fraction with pseudo-R2 > 0.01: place cells "
          f"{np.nanmean(cv_r2[d][pc] > 0.01):.2f}, other units "
          f"{np.nanmean(cv_r2[d][~pc] > 0.01):.2f}")

glm_table = pd.DataFrame({
    "session": SESSION,
    "unit": np.tile(unit_keys, len(DIRECTIONS)),
    "direction": np.repeat(DIRECTIONS, len(unit_keys)),
    "cv_pseudo_r2": np.concatenate([cv_r2[d] for d in DIRECTIONS]),
    "is_place_cell": np.concatenate([tables[d].is_place_cell.values for d in DIRECTIONS]),
    "info_bits_per_spike": np.concatenate(
        [tables[d].info_bits_per_spike.values for d in DIRECTIONS]),
})
glm_table.to_csv("glm_scores_Achilles_10252013.csv", index=False)

# %% [markdown]
# ### Figure 8. GLM position tuning
#
# The smooth GLM tuning curves reproduce the binned rate maps closely, and the
# held-out pseudo-$R^2$ separates the two groups cleanly: nearly all place cells
# are predicted better than chance by position alone, while most other units are
# not. The GLM score and the Skaggs information agree in ranking cells, so two
# quite different criteria are picking out the same population.

# %%
fig = plt.figure(figsize=(12, 6.4))
gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.30,
                      left=0.07, right=0.98, top=0.85, bottom=0.10)

good = tr.is_place_cell.values & (tr.peak_rate.values > 3.0)
cand = np.flatnonzero(good)
glm_examples = list(dict.fromkeys(
    [cand[np.argmin(np.abs(tr.peak_pos_cm.values[cand] - x))]
     for x in np.array([0.15, 0.47, 0.8]) * TRACK_LEN]))

for n, idx in enumerate(glm_examples):
    ax = fig.add_subplot(gs[0, n])
    for d in DIRECTIONS:
        ax.plot(bin_centers, maps[d][idx], color=DIR_COLORS[d], lw=1.0, alpha=0.5,
                label=f"{d}, binned" if n == 0 else None)
        ax.plot(grid, glm_tuning[d][idx], color=DIR_COLORS[d], lw=2.0,
                label=f"{d}, GLM" if n == 0 else None)
    ax.set_xlim(0, TRACK_LEN)
    ax.set_xlabel("position (cm)")
    if n == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"{'ABC'[n]}  unit {int(tr.unit.values[idx])}  "
                 f"(CV pseudo-$R^2$ = {cv_r2['rightward'][idx]:.2f})",
                 loc="left", fontweight="bold", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
d = "rightward"
pc = tables[d].is_place_cell.values
fin = np.isfinite(cv_r2[d])
bins = np.linspace(min(-0.02, np.nanmin(cv_r2[d]) * 1.05),
                   max(0.25, np.nanmax(cv_r2[d]) * 1.05), 36)
ax.hist(cv_r2[d][~pc & fin], bins=bins, color="0.7",
        label=f"not a place cell (n={(~pc).sum()})")
ax.hist(cv_r2[d][pc & fin], bins=bins, histtype="step", lw=2.0, color=DIR_COLORS[d],
        label=f"place cell (n={pc.sum()})")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("cross-validated pseudo-$R^2$"); ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("D  Held-out model fit", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 1])
for d in DIRECTIONS:
    pc = tables[d].is_place_cell.values
    ax.scatter(tables[d].info_bits_per_spike.values[pc], cv_r2[d][pc], s=14,
               color=DIR_COLORS[d], label=f"{d}, place cell")
    ax.scatter(tables[d].info_bits_per_spike.values[~pc], cv_r2[d][~pc], s=14,
               facecolors="none", edgecolors=DIR_COLORS[d], linewidths=0.7, alpha=0.6)
ai = np.concatenate([tables[d].info_bits_per_spike.values for d in DIRECTIONS])
ar = np.concatenate([cv_r2[d] for d in DIRECTIONS])
ok = np.isfinite(ai) & np.isfinite(ar)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Skaggs information (bits/spike)"); ax.set_ylabel("cross-validated pseudo-$R^2$")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title(f"E  Agreement of the two measures, r = {np.corrcoef(ai[ok], ar[ok])[0, 1]:.2f}",
             loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 2])
frac = np.array([[np.nanmean(cv_r2[d][tables[d].is_place_cell.values] > 0.01),
                  np.nanmean(cv_r2[d][~tables[d].is_place_cell.values] > 0.01)]
                 for d in DIRECTIONS])
xx = np.arange(2)
for k, d in enumerate(DIRECTIONS):
    ax.bar(xx + (k - 0.5) * 0.38, 100 * frac[k], width=0.38, color=DIR_COLORS[d], label=d)
ax.set_xticks(xx); ax.set_xticklabels(["place cells", "other units"])
ax.set_ylabel("% of units with pseudo-$R^2$ > 0.01"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("F  Position improves held-out prediction", loc="left", fontweight="bold")

fig.suptitle(f"Poisson GLM of position tuning (NeMoS), {SESSION}\n"
             f"{N_BASIS} B-spline basis functions over position, "
             f"{GLM_BIN_SIZE * 1000:.0f} ms bins, fitted on half the traversals and "
             "scored on the other half", fontweight="bold", fontsize=10)
fig.savefig("fig08_glm_position_tuning.png", bbox_inches="tight")
plt.show()

res["handles"][1].close()

# %% [markdown]
# ## 7. All five linear-track sessions
#
# The whole pipeline is now rerun unchanged on every linear-track session in the
# dandiset: four rats, tracks of 1.6 m and 2 m, and a wide range of running
# speeds and numbers of traversals.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    linear_sessions = []
    for name in SESSIONS:
        *_, m, handles = load_behavior_and_spikes(name)
        handles[1].close()
        if m["maze_type"] == "linear":
            linear_sessions.append(name)
print("linear-track sessions:", linear_sessions)

rng = np.random.default_rng(7)
rows, summaries = [], []
for name in tqdm(linear_sessions, desc="sessions"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = analyze_session(name, n_shuffles=500, progress=False)
    m = r["meta"]
    tab = pd.concat([r["per_dir"][d]["table"] for d in DIRECTIONS], ignore_index=True)
    rows.append(tab)
    e = decoding_error(r)
    e_shuf = decoding_error(r, shuffle_tc=True, rng=rng)
    any_pc = tab.groupby("unit").is_place_cell.any()
    summaries.append(dict(
        session=name, subject=m["subject"], maze=m["maze"],
        track_length_cm=m["track_length_cm"], n_units=m["n_units_total"],
        n_pyramidal=m["n_pyramidal"], n_runs=m["n_runs"], run_time_s=m["run_time_s"],
        median_speed_cms=m["median_speed_cms"], n_place_cells=int(any_pc.sum()),
        frac_place_cells=float(any_pc.mean()),
        median_info=float(tab.loc[tab.is_place_cell, "info_bits_per_spike"].median()),
        median_width_cm=float(tab.loc[tab.is_place_cell, "field_width_cm"].median()),
        median_peak_hz=float(tab.loc[tab.is_place_cell, "peak_rate"].median()),
        median_decode_err_cm=float(np.median(e)),
        median_decode_err_shuffled_cm=float(np.median(e_shuf))))

all_cells = pd.concat(rows, ignore_index=True)
summary = pd.DataFrame(summaries)
all_cells.to_csv("place_cell_table_all_sessions.csv", index=False)
summary.to_csv("session_summary.csv", index=False)
summary

# %% [markdown]
# ### Figure 9. Every session shows the same effect

# %%
labels = [s.replace("_", "\n") for s in summary.session]
x = np.arange(len(summary))
colors = plt.get_cmap("tab10")(np.arange(len(summary)) % 10)

fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
fig.subplots_adjust(wspace=0.32, top=0.76, bottom=0.28, left=0.05, right=0.99)

ax = axes[0]
ax.bar(x, 100 * summary.frac_place_cells, color=colors)
for xi, (f, n) in enumerate(zip(summary.frac_place_cells, summary.n_place_cells)):
    ax.text(xi, 100 * f + 1.5, str(n), ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
ax.set_ylabel("place cells (% of pyramidal cells)"); ax.set_ylim(0, 100)
ax.set_title("A  Place-cell yield", loc="left", fontweight="bold")

for ax, col, lab, title in [
        (axes[1], "info_bits_per_spike", "spatial information (bits/spike)",
         "B  Spatial information"),
        (axes[2], "field_width_cm", "field width at 50 % of peak (cm)", "C  Field width")]:
    data = [all_cells.loc[(all_cells.session == s) & all_cells.is_place_cell, col].values
            for s in summary.session]
    bp = ax.boxplot(data, positions=x, widths=0.65, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
    ax.set_ylabel(lab)
    ax.set_title(title, loc="left", fontweight="bold")

ax = axes[3]
ax.bar(x - 0.19, summary.median_decode_err_cm, width=0.38, color="#1f77b4", label="observed")
ax.bar(x + 0.19, summary.median_decode_err_shuffled_cm, width=0.38, color="0.7",
       label="rolled tuning curves")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
ax.set_ylabel("median decoding error (cm)")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("D  Cross-validated decoding", loc="left", fontweight="bold")

fig.suptitle(f"All {len(summary)} linear-track sessions of DANDI:000044 "
             f"({summary.subject.nunique()} rats): the place code is present in every session\n"
             f"pooled: {int(summary.n_place_cells.sum())} place cells of "
             f"{int(summary.n_pyramidal.sum())} pyramidal cells "
             f"({100 * summary.n_place_cells.sum() / summary.n_pyramidal.sum():.0f} %), "
             f"median decoding error {summary.median_decode_err_cm.median():.1f} cm",
             fontweight="bold", fontsize=10)
fig.savefig("fig09_all_sessions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Summary
#
# Streaming five linear-track sessions from DANDI:000044 reproduces the defining
# properties of hippocampal place cells:
#
# - **Spatial selectivity.** Pooling across sessions, a large majority of
#   putative CA1 pyramidal cells qualify as place cells in at least one travel
#   direction, carrying spatial information well above a circular-shift null
#   that preserves each spike train's own temporal statistics, with stable
#   odd/even maps.
# - **Field geometry.** Median field width is around 30 cm on a 1.6-2 m track
#   with in-field peak rates of roughly 5-7 Hz, matching the classical CA1
#   description.
# - **Population tiling and directionality.** Fields cover the whole track, and
#   sorting cells by their rightward field peak scrambles the leftward maps, so
#   the code is a map of position-plus-direction rather than of position alone.
# - **Generalisation.** Position is decoded from held-out traversals to within a
#   few centimetres, an order of magnitude better than the rolled-tuning-curve
#   null, in every session and without selecting for place cells. A Poisson GLM
#   fitted on half the traversals predicts held-out spike counts better than a
#   constant-rate model for nearly all place cells and for few other units.
#
# Two data-quality caveats are worth restating. The `SpatialSeries.rate` field
# in this dandiset holds the sampling period rather than the frequency, and one
# unit in each of three sessions contains two concatenated spike trains. Both
# are handled explicitly above; neither is detectable without checking, and
# either would corrupt the analysis silently.
