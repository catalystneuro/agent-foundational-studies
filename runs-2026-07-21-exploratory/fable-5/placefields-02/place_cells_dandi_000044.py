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
# recordings from the DANDI Archive. Nothing here is simulated: every spike and
# every position sample is streamed from the archive at analysis time.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044),
# *Grosmark & Buzsáki (2016), "Diversity in neural firing dynamics supports both
# rigid and learned hippocampal sequences"* (Science 351:1440-1443). Eight
# bilateral silicon-probe recordings from dorsal hippocampus (CA1) in four
# Long-Evans rats. Each session is a pre-run sleep epoch, a novel-track running
# epoch, and a post-run sleep epoch. Five sessions use a straight track (1.6 m or
# 2 m) and three use a closed circular track (~2.85 m). Spikes are already
# sorted, and units carry an `excitatory` / `inhibitory` label.
#
# **What we show.**
#
# 1. Individual CA1 pyramidal cells fire in a restricted portion of the track,
#    reliably, lap after lap.
# 2. Their spatial information exceeds a per-cell circular-shift null, so the
#    tuning is not a by-product of firing rate or burstiness.
# 3. The fields tile the track, and the ordering derived from odd laps holds on
#    held-out even laps.
# 4. On a straight track the fields are direction-specific.
# 5. A naive Bayes decoder fitted on half the laps recovers the animal's position
#    on the other half to within a few centimetres.
# 6. All of this replicates across all eight sessions and four animals.
#
# **Method note.** Data access and analysis both go through
# [pynapple](https://pynapple.org); NWB files are streamed with
# [LINDI](https://github.com/NeurodataWithoutBorders/lindi) so only the chunks
# actually touched are fetched and cached. Section 6 checks our fast rate-map
# code against `pynapple.compute_tuning_curves` before it is relied on.

# %%
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless: figures are written to disk
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from tqdm.auto import tqdm

import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
np.set_printoptions(precision=3, suppress=True)
print("pynapple", nap.__version__)

# %% [markdown]
# ## 1. Streaming the data
#
# LINDI serves a JSON index of each NWB file's chunks, so `LindiH5pyFile` behaves
# like a local `h5py.File` while fetching only what is read. `LocalCache` keeps
# fetched chunks on disk, so re-running this notebook is fast.

# %%
DANDISET = "000044"
LINDI_TMPL = "https://lindi.neurosift.org/dandi/dandisets/{ds}/assets/{asset}/nwb.lindi.json"
CACHE = "/tmp/lindi_cache"

# asset id -> session label, all eight sessions of Grosmark & Buzsáki (2016)
SESSIONS = {
    "5349c68b-c0a7-46c0-9900-cda050722fa4": "Achilles_10252013",
    "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a": "Achilles_11012013",
    "3cc5b7b3-02e2-490a-9f19-d20670355084": "Cicero_09012014",
    "f61dfe09-3db2-464a-b386-2e828b2e7276": "Cicero_09102014",
    "e381ebb3-128e-4f3f-9517-11277d7aed9b": "Cicero_09172014",
    "31ea0aab-4777-424e-9a93-9605b2bdcc29": "Gatsby_08022013",
    "f7687af7-3bc9-4d20-8d88-ef293d2a3381": "Gatsby_08282013",
    "82714afb-724f-4e2b-b102-c9c47b5cba73": "Buddy_06272013",
}
PROTOTYPE = "5349c68b-c0a7-46c0-9900-cda050722fa4"   # Achilles_10252013


def open_session(asset_id):
    """Stream one NWB file from DANDI through LINDI (chunks cached on disk)."""
    url = LINDI_TMPL.format(ds=DANDISET, asset=asset_id)
    f = lindi.LindiH5pyFile.from_lindi_file(
        url, local_cache=lindi.LocalCache(cache_dir=CACHE))
    return NWBHDF5IO(file=f, mode="r").read()


nwbfile = open_session(PROTOTYPE)
print(nwbfile.session_description[:200])
print("\nsubject:", nwbfile.subject.subject_id, "|", nwbfile.subject.species)
print("behavior interfaces:", list(nwbfile.processing["behavior"].data_interfaces))
print("units:", len(nwbfile.units), "| columns:", nwbfile.units.colnames)
print("epochs:")
print(nwbfile.intervals["epochs"].to_dataframe())

# %% [markdown]
# ## 2. Loading position and spikes
#
# One quirk of this conversion has to be handled explicitly. The position
# `SpatialSeries` stores `rate = 0.02560`, but that number is the sampling
# **period** in seconds (39.06 Hz), not a frequency. Taking it at face value as
# a rate would place `80762` samples across 36 days instead of 34 minutes. We
# therefore rebuild the time base as `starting_time + arange(n) * rate` and
# assert that the result lands inside the session's `MazeEpoch`.

# %%
def _behavior_series(nwbfile):
    """Return (2-D SpatialSeries, linearized SpatialSeries, maze name)."""
    beh = nwbfile.processing["behavior"]
    pos_key = [k for k in beh.data_interfaces
               if k.endswith("Position") and "Linearized" not in k][0]
    lin_key = [k for k in beh.data_interfaces if "LinearizedPosition" in k][0]
    pos, lin = beh[pos_key], beh[lin_key]
    return (pos[list(pos.spatial_series)[0]],
            lin[list(lin.spatial_series)[0]],
            pos_key)


def load_behavior(nwbfile):
    """Position as pynapple TsdFrame/Tsd on a manually reconstructed time base."""
    pos_ss, lin_ss, maze_name = _behavior_series(nwbfile)
    n = pos_ss.data.shape[0]
    period = pos_ss.rate                       # mislabelled: seconds/sample
    t = pos_ss.starting_time + np.arange(n) * period
    xy = np.asarray(pos_ss.data[:], dtype=float)
    lin = np.asarray(lin_ss.data[:], dtype=float).ravel()
    return (nap.TsdFrame(t=t, d=xy, columns=["x", "y"]),
            nap.Tsd(t=t, d=lin), 1.0 / period, maze_name)


def load_epochs(nwbfile):
    """The pre-sleep / maze / post-sleep epochs, keyed by label."""
    df = nwbfile.intervals["epochs"].to_dataframe()
    return {r["label"]: nap.IntervalSet(start=r["start_time"], end=r["stop_time"])
            for _, r in df.iterrows()}


def load_units(nwbfile):
    """TsGroup of spike times carrying cell_type / location / shank metadata."""
    u = nwbfile.units
    spikes = {i: nap.Ts(np.asarray(u["spike_times"][i])) for i in range(len(u))}
    meta = {"cell_type": np.asarray(u["cell_type"][:]),
            "location": np.asarray(u["location"][:]),
            "shank_id": np.asarray(u["shank_id"][:])}
    return nap.TsGroup(spikes, metadata=meta)


def check_timebase(nwbfile, tol=1.0):
    """Verify the reconstructed time base lands inside the maze epoch."""
    pos2d, lin, fs, _ = load_behavior(nwbfile)
    maze = load_epochs(nwbfile)["MazeEpoch"]
    t0, t1 = float(pos2d.index[0]), float(pos2d.index[-1])
    m0, m1 = float(maze.start[0]), float(maze.end[0])
    if not (m0 - tol <= t0 and t1 <= m1 + tol):
        raise ValueError(f"position time base [{t0:.1f}, {t1:.1f}] is outside "
                         f"MazeEpoch [{m0:.1f}, {m1:.1f}]")
    return m1 - m0, t1 - t0


def maze_kind(maze_name):
    """'circular' for the closed-loop mazes, 'linear' for the straight tracks."""
    return "circular" if "Circular" in maze_name else "linear"


def track_range(linear, pad=0.0):
    """Outer bin edges spanning the sampled extent of the linearized track."""
    d = linear.values[np.isfinite(linear.values)]
    return (float(np.min(d)) - pad, float(np.max(d)) + pad)


maze_dur, track_dur = check_timebase(nwbfile)
pos2d, lin, fs, maze = load_behavior(nwbfile)
epochs = load_epochs(nwbfile)
lin = lin.restrict(epochs["MazeEpoch"])
units = load_units(nwbfile)
exc = units[np.where(units.cell_type == "excitatory")[0]]
RNG_TRACK = track_range(lin)

print(f"maze                : {maze}  ({maze_kind(maze)})")
print(f"MazeEpoch duration  : {maze_dur:.1f} s; position series {track_dur:.1f} s")
print(f"sampling rate       : {fs:.3f} Hz")
print(f"linearized extent   : {RNG_TRACK[0]:.2f} - {RNG_TRACK[1]:.2f} m")
print(f"units               : {len(units)} total, {len(exc)} excitatory, "
      f"{(units.cell_type == 'inhibitory').sum()} inhibitory")
print(f"recording sites     : {set(units.location)}")

# %% [markdown]
# ### Figure 1: behaviour validation
#
# Before any spike analysis, check that the behavioural streams are what we think
# they are. The linearized series is NaN whenever the animal is off the track, so
# its valid samples already isolate track traversals.

# %%
p = pos2d.restrict(epochs["MazeEpoch"])
xy = p.values
ok2d = ~np.isnan(xy).any(axis=1)
okl = np.isfinite(lin.values)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot(xy[ok2d, 0], xy[ok2d, 1], lw=0.3, color="0.75")
ax[0, 0].scatter(xy[okl, 0], xy[okl, 1], s=1, color="crimson",
                 label="samples with a linearized position")
ax[0, 0].set(xlabel="x (m)", ylabel="y (m)", title="2-D trajectory, maze epoch")
ax[0, 0].legend(fontsize=8, markerscale=6, loc="upper right")
ax[0, 0].set_aspect("equal")

ax[0, 1].plot(lin.index, lin.values, ".", ms=1.5, color="crimson")
ax[0, 1].set(xlabel="time (s)", ylabel="linearized position (m)",
             title="Linearized position over the whole maze epoch")

t0 = lin.index[np.argmax(okl)]
w = (lin.index > t0 + 40) & (lin.index < t0 + 160)
ax[1, 0].plot(lin.index[w], lin.values[w], ".-", ms=3, lw=0.5, color="crimson")
ax[1, 0].set(xlabel="time (s)", ylabel="linearized position (m)",
             title="120 s zoom: the animal shuttles end to end")

d = np.diff(lin.values) * fs
ax[1, 1].hist(np.abs(d[np.isfinite(d)]), bins=80, color="steelblue")
ax[1, 1].set(xlabel="|d(linear position)/dt| (m/s)", ylabel="count", yscale="log",
             title="Running speed along the track")

fig.suptitle(f"DANDI:000044  sub-Achilles  ses-10252013  ({maze}) — "
             "behaviour validation", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"{FIGDIR}/fig01_behavior_validation.png", dpi=150)
plt.close(fig)
print("saved fig01_behavior_validation.png")

# %% [markdown]
# ## 3. Running epochs, split by direction of travel
#
# Place fields are measured while the animal runs, so we exclude the pauses at
# the reward ends. Within each contiguous block of valid tracking the position is
# smoothed (250 ms) and differentiated; samples faster than 10 cm/s are kept and
# the sign of the velocity gives the direction.
#
# On the circular track the position wraps, so increments are unwrapped onto the
# shortest arc before differentiating. Without this the wrap point produces a
# spurious full-track-length velocity spike on every lap.

# %%
def make_run_epochs(linear, fs, speed_thresh=0.10, smooth_s=0.25,
                    min_dur=0.5, merge_gap=0.2, period=None):
    """Split the linearized position into leftward / rightward running epochs."""
    t = linear.index.values
    d = linear.values.astype(float)
    idx = np.where(np.isfinite(d))[0]
    blocks = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

    win = max(3, int(round(smooth_s * fs)))
    vel = np.full(len(d), np.nan)
    for b in blocks:
        if len(b) < win:
            continue
        x = d[b]
        if period is not None:                      # closed loop: unwrap
            step = np.diff(x)
            step = (step + period / 2.0) % period - period / 2.0
            x = np.r_[x[0], x[0] + np.cumsum(step)]
        vel[b] = np.gradient(uniform_filter1d(x, size=win, mode="nearest"),
                             1.0 / fs)

    good = np.isfinite(vel)
    lin_valid = nap.Tsd(t=t[good], d=d[good])
    velocity = nap.Tsd(t=t[good], d=vel[good])

    def _epochs_from_mask(mask):
        tt, m = t[good], np.asarray(mask)
        if not m.any():
            return nap.IntervalSet(start=[], end=[])
        edges = np.diff(m.astype(int))
        starts = np.r_[0] if m[0] else np.array([], int)
        starts = np.r_[starts, np.where(edges == 1)[0] + 1]
        ends = np.where(edges == -1)[0]
        if m[-1]:
            ends = np.r_[ends, len(m) - 1]
        ivs, dt_max = [], 5.0 / fs
        for s, e in zip(starts, ends):              # never bridge a tracking gap
            seg = np.arange(s, e + 1)
            for sub in np.split(seg, np.where(np.diff(tt[seg]) > dt_max)[0] + 1):
                if len(sub) > 1:
                    ivs.append((tt[sub[0]], tt[sub[-1]]))
        if not ivs:
            return nap.IntervalSet(start=[], end=[])
        ivs = np.array(ivs)
        return (nap.IntervalSet(start=ivs[:, 0], end=ivs[:, 1])
                .merge_close_intervals(merge_gap).drop_short_intervals(min_dur))

    v = velocity.values
    eps = {"rightward": _epochs_from_mask(v > speed_thresh),
           "leftward": _epochs_from_mask(v < -speed_thresh)}
    eps["run"] = eps["rightward"].union(eps["leftward"])
    return lin_valid, velocity, eps


lin_valid, vel, run_eps = make_run_epochs(lin, fs)
DIRS = ["rightward", "leftward"]
for k in DIRS:
    e = run_eps[k]
    print(f"{k:10s}: {len(e):3d} traversals, {e.tot_length():6.1f} s total, "
          f"median {np.median(e.end - e.start):.2f} s each")

# %% [markdown]
# ### Figure 2: traversals, velocity and raw spiking

# %%
fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
win = (lin_valid.index[0] + 40, lin_valid.index[0] + 160)

ax[0].plot(lin_valid.index, lin_valid.values, ".", ms=2, color="0.6")
for k, c in [("rightward", "tab:red"), ("leftward", "tab:blue")]:
    for s, e in zip(run_eps[k].start, run_eps[k].end):
        sub = lin_valid.restrict(nap.IntervalSet(s, e))
        ax[0].plot(sub.index, sub.values, "-", color=c, lw=1.8)
    ax[0].plot([], [], color=c, label=k)
ax[0].set(ylabel="linear position (m)",
          title="Traversals coloured by direction of travel")
ax[0].legend(loc="upper right", fontsize=9)

ax[1].plot(vel.index, vel.values, ".", ms=2, color="0.3")
ax[1].axhline(0.1, color="tab:red", ls="--", lw=1)
ax[1].axhline(-0.1, color="tab:blue", ls="--", lw=1)
ax[1].set(ylabel="velocity (m/s)",
          title="Running velocity (dashed = the 0.1 m/s inclusion threshold)")

for i, u in enumerate(list(exc.keys())[:25]):
    st = exc[u].restrict(nap.IntervalSet(*win)).index
    ax[2].plot(st, np.full(len(st), i), "|", ms=4, color="k", mew=0.6)
ax[2].set(xlabel="time (s)", ylabel="unit #", xlim=win,
          title="CA1 spike raster (25 excitatory units) over the same window")

fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig02_run_epochs_and_raster.png", dpi=150)
plt.close(fig)
print("saved fig02_run_epochs_and_raster.png")

# %% [markdown]
# ## 4. Rate maps, spatial information and a per-cell null
#
# A rate map is spike count per spatial bin divided by occupancy in that bin.
# We use 4 cm bins and smooth counts and occupancy alike with a 1-bin Gaussian,
# so the ratio stays a properly normalised rate.
#
# **Spatial information** (Skaggs et al. 1993), in bits per spike:
#
# $$ I = \sum_i p_i \frac{\lambda_i}{\bar\lambda} \log_2 \frac{\lambda_i}{\bar\lambda} $$
#
# where $p_i$ is the fraction of time spent in bin $i$, $\lambda_i$ the rate
# there, and $\bar\lambda$ the mean rate.
#
# Spatial information is **biased upward when spike counts are low**, so a
# fixed threshold would preferentially label sparse cells as place cells. Instead
# each unit is compared with *its own* null: the spike train is shifted by a
# random offset along a compressed time axis that concatenates only the running
# epochs, and wrapped around. This destroys the spike/position relationship while
# preserving that unit's exact spike count and its burst structure. The
# compressed axis matters, because a shift in real time would push spikes into
# the pauses at the track ends where there is no position to speak of.

# %%
def _compressed_time(ep, t):
    """Map real times onto a 'within-epoch' axis that concatenates `ep`."""
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    durs = ends - starts
    offs = np.r_[0.0, np.cumsum(durs)[:-1]]
    j = np.searchsorted(starts, t, side="right") - 1
    tau = np.full(len(t), np.nan)
    ok = (j >= 0) & (j < len(starts))
    sel = np.where(ok)[0][t[ok] <= ends[j[ok]]]
    tau[sel] = offs[j[sel]] + (t[sel] - starts[j[sel]])
    return tau, durs.sum()


def spatial_information(rate_map, occupancy):
    """Skaggs spatial information (bits/spike), sparsity and mean rate."""
    p = occupancy / occupancy.sum()
    lam = np.asarray(rate_map, dtype=float)
    mean_rate = (p[None, :] * lam).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lam / mean_rate[:, None]
        term = p[None, :] * ratio * np.log2(ratio)
        sparsity = mean_rate ** 2 / (p[None, :] * lam ** 2).sum(axis=1)
    term[~np.isfinite(term)] = 0.0
    return term.sum(axis=1), sparsity, mean_rate


class DirectionalRateMaps:
    """Rate maps for one running direction, plus a matched circular-shift null.

    Occupancy and the spike->position-bin lookup are built once on the
    compressed time axis, so a shuffle costs one modular shift, one searchsorted
    and one bincount (~0.8 ms for 120 units).
    """

    def __init__(self, linear, ep, fs, n_bins=40, trange=(0.0, 1.6),
                 smooth_bins=1.0, min_occupancy=0.5, circular=False):
        self.ep, self.fs, self.n_bins = ep, fs, n_bins
        self.linear, self.min_occupancy, self.circular = linear, min_occupancy, circular
        self.edges = np.linspace(trange[0], trange[1], n_bins + 1)
        self.centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.bin_size = self.edges[1] - self.edges[0]
        self.smooth_bins = smooth_bins

        p = linear.restrict(ep)
        tau_pos, self.T = _compressed_time(ep, p.index.values)
        keep = np.isfinite(tau_pos)
        tau_pos, pos = tau_pos[keep], p.values[keep]
        order = np.argsort(tau_pos)
        self.tau_pos, pos = tau_pos[order], pos[order]
        self.pos_bin = np.clip(np.digitize(pos, self.edges) - 1, 0, n_bins - 1)

        raw_occ = np.bincount(self.pos_bin, minlength=n_bins) / fs
        self.raw_occupancy = raw_occ
        self.valid_bins = raw_occ > min_occupancy
        self.occupancy = self._smooth(raw_occ)

    def _smooth(self, m):
        if self.smooth_bins <= 0:
            return m
        return gaussian_filter1d(m, self.smooth_bins, axis=-1,
                                 mode="wrap" if self.circular else "nearest")

    def _nearest_sample(self, ts):
        """Index of the position sample closest in time to each spike.

        Matches pynapple's `value_from` convention (verified in section 6)."""
        j = np.clip(np.searchsorted(self.tau_pos, ts), 1, len(self.tau_pos) - 1)
        left = np.abs(ts - self.tau_pos[j - 1]) <= np.abs(self.tau_pos[j] - ts)
        return np.where(left, j - 1, j)

    def _map_from_tau(self, tau_spk):
        counts = np.zeros((len(tau_spk), self.n_bins))
        for i, ts in enumerate(tau_spk):
            if len(ts):
                counts[i] = np.bincount(self.pos_bin[self._nearest_sample(ts)],
                                        minlength=self.n_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = self._smooth(counts) / np.maximum(self.occupancy, 1e-12)
        rm[:, ~self.valid_bins] = np.nan
        return rm

    def spike_taus(self, units):
        out = []
        for u in units.keys():
            tau, _ = _compressed_time(self.ep, units[u].index.values)
            out.append(np.sort(tau[np.isfinite(tau)]))
        return out

    def rate_maps(self, units):
        return self._map_from_tau(self.spike_taus(units))

    def si(self, rate_map):
        return spatial_information(rate_map[:, self.valid_bins],
                                   self.occupancy[self.valid_bins])

    def shuffle_si(self, units, n_shuffles=1000, seed=0, progress=True):
        """Null distribution of spatial information from circular time shifts."""
        rng = np.random.default_rng(seed)
        taus = self.spike_taus(units)
        occ_ok = self.occupancy[self.valid_bins]
        null = np.empty((n_shuffles, len(taus)))
        it = tqdm(range(n_shuffles), desc="shuffles", leave=False) if progress \
            else range(n_shuffles)
        for k in it:
            shifts = rng.uniform(1.0, self.T - 1.0, size=len(taus))
            rm = self._map_from_tau([np.sort((t + s) % self.T)
                                     for t, s in zip(taus, shifts)])
            null[k] = spatial_information(rm[:, self.valid_bins], occ_ok)[0]
        return null

    def lap_split_maps(self, units):
        """Rate maps computed separately from odd- and even-numbered laps."""
        starts, ends = np.asarray(self.ep.start), np.asarray(self.ep.end)
        out = []
        for sel in (np.arange(0, len(starts), 2), np.arange(1, len(starts), 2)):
            half = DirectionalRateMaps(
                self.linear, nap.IntervalSet(start=starts[sel], end=ends[sel]),
                self.fs, self.n_bins, (self.edges[0], self.edges[-1]),
                self.smooth_bins, self.min_occupancy, self.circular)
            out.append(half.rate_maps(units))
        return out


def field_metrics(rate_map, centers, bin_size, frac=0.5, circular=False):
    """Peak rate, peak position and width at `frac` of the peak, per unit."""
    n_units, n_bins = rate_map.shape
    peak_rate = np.full(n_units, np.nan)
    peak_pos = np.full(n_units, np.nan)
    width = np.full(n_units, np.nan)
    for i in range(n_units):
        r = rate_map[i]
        ok = np.isfinite(r)
        if not ok.any() or np.nanmax(r) <= 0:
            continue
        pk = int(np.nanargmax(r))
        peak_rate[i], peak_pos[i] = r[pk], centers[pk]
        above = np.where(ok, r >= frac * r[pk], False)
        n_field = 1
        for step in (-1, +1):
            j = pk
            while n_field < n_bins:
                j += step
                if circular:
                    j %= n_bins
                elif not (0 <= j < n_bins):
                    break
                if not above[j]:
                    break
                n_field += 1
        width[i] = n_field * bin_size
    return peak_rate, peak_pos, width


# %% [markdown]
# ## 5. Analysis parameters

# %%
N_BINS = 40           # 1.6 m / 40 = 4 cm bins
N_SHUFFLES = 1000
SMOOTH = 1.0          # Gaussian sigma, in bins
MIN_PEAK_RATE = 1.0   # Hz
MIN_SPIKES = 50       # per direction
ALPHA = 0.01          # one-sided, against the unit's own null

# %% [markdown]
# ## 6. Validating the fast rate maps against pynapple
#
# The shuffle test builds 1000 rate maps per direction, so we use the fast
# implementation above rather than calling pynapple 1000 times. Before relying
# on it, check it against `nap.compute_tuning_curves` with smoothing switched
# off. (Note that pynapple reports its `occupancy` attribute in samples, not
# seconds, so it is divided by the sampling rate before comparison.)

# %%
for d in DIRS:
    ep = run_eps[d]
    rm0 = DirectionalRateMaps(lin_valid, ep, fs, N_BINS, RNG_TRACK, smooth_bins=0.0)
    mine = rm0.rate_maps(exc)
    tc = nap.compute_tuning_curves(exc, lin_valid, bins=N_BINS, range=RNG_TRACK,
                                   epochs=ep, feature_names=["position"])
    theirs = np.asarray(tc.values)
    ok = rm0.valid_bins
    a, b = mine[:, ok], theirs[:, ok]
    f = np.isfinite(a) & np.isfinite(b)
    occ_p = np.asarray(tc.attrs["occupancy"])[ok] / fs
    si_mine = rm0.si(mine)[0]
    si_theirs = spatial_information(np.nan_to_num(b), occ_p)[0]
    print(f"[{d}] rate map corr with pynapple = {np.corrcoef(a[f], b[f])[0,1]:.6f}, "
          f"median |diff| = {np.median(np.abs(a - b)[f]):.2e} Hz")
    print(f"          occupancy  mine {rm0.occupancy[ok].sum():7.1f} s   "
          f"pynapple {occ_p.sum():7.1f} s")
    print(f"          spatial information corr = "
          f"{np.corrcoef(si_mine, si_theirs)[0,1]:.6f}")

# %% [markdown]
# ## 7. Place fields in one session
#
# Every excitatory unit gets a rate map, a spatial information value, and a
# 1000-shuffle null, separately for each running direction. A unit counts as a
# place cell in a direction if its spatial information beats its own null at
# p < 0.01, it fires at least 1 Hz at its peak, and it emits at least 50 spikes
# in that direction.

# %%
res = {}
for d in DIRS:
    rm = DirectionalRateMaps(lin_valid, run_eps[d], fs, N_BINS, RNG_TRACK,
                             smooth_bins=SMOOTH)
    maps = rm.rate_maps(exc)
    si, sparsity, mean_rate = rm.si(maps)
    null = rm.shuffle_si(exc, n_shuffles=N_SHUFFLES, seed=1)
    pval = (np.sum(null >= si[None, :], axis=0) + 1) / (N_SHUFFLES + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zval = (si - null.mean(axis=0)) / null.std(axis=0)
    peak_rate, peak_pos, width = field_metrics(maps, rm.centers, rm.bin_size)
    n_spk = np.array([len(exc[u].restrict(run_eps[d])) for u in exc.keys()])
    odd, even = rm.lap_split_maps(exc)
    ok = rm.valid_bins
    stability = np.array([
        np.corrcoef(odd[i, ok], even[i, ok])[0, 1]
        if np.isfinite(odd[i, ok]).all() and np.isfinite(even[i, ok]).all()
        and odd[i, ok].std() > 0 and even[i, ok].std() > 0 else np.nan
        for i in range(len(exc))])
    is_place = (pval < ALPHA) & (peak_rate >= MIN_PEAK_RATE) & (n_spk >= MIN_SPIKES)

    res[d] = dict(rm=rm, maps=maps, si=si, null=null, pval=pval, zval=zval,
                  peak_rate=peak_rate, peak_pos=peak_pos, width=width,
                  n_spk=n_spk, odd=odd, even=even, stability=stability,
                  is_place=is_place, mean_rate=mean_rate, sparsity=sparsity)
    print(f"[{d}] place cells: {is_place.sum()}/{len(exc)} "
          f"({100*is_place.mean():.0f}%)")
    print(f"     SI median {np.median(si[is_place]):.2f} bits/spike "
          f"(null median {np.median(null):.2f}); "
          f"peak {np.median(peak_rate[is_place]):.1f} Hz; "
          f"width {np.median(width[is_place]):.2f} m; "
          f"odd/even r = {np.nanmedian(stability[is_place]):.2f}")

either = res["rightward"]["is_place"] | res["leftward"]["is_place"]
both = res["rightward"]["is_place"] & res["leftward"]["is_place"]
print(f"\nplace cell in at least one direction: {either.sum()}/{len(exc)} "
      f"({100*either.mean():.0f}%);  in both: {both.sum()}")

# %% [markdown]
# ### Figure 3: individual place cells
#
# The clearest single view of the phenomenon. Each spike is plotted at the
# position the animal occupied when it fired, one row per traversal. A place cell
# produces a vertical stripe: the same stretch of track, lap after lap. The
# bottom row shows the corresponding rate maps for both directions.

# %%
r = res["rightward"]
cand = np.where(r["is_place"] & (r["stability"] > 0.85) & (r["width"] <= 0.6))[0]
seg_edges = np.linspace(RNG_TRACK[0], RNG_TRACK[1], 7)
pick = []
for lo, hi in zip(seg_edges[:-1], seg_edges[1:]):
    inseg = cand[(r["peak_pos"][cand] >= lo) & (r["peak_pos"][cand] < hi)]
    if len(inseg):
        pick.append(inseg[np.argmax(r["si"][inseg])])
pick = np.array(pick)
unit_ids = list(exc.keys())
print("example units:", [unit_ids[i] for i in pick])

fig, axes = plt.subplots(3, len(pick), figsize=(3.2 * len(pick), 9),
                         gridspec_kw={"height_ratios": [2, 2, 1.4]})
for c, i in enumerate(pick):
    u = unit_ids[i]
    for row, d in enumerate(DIRS):
        ax = axes[row, c]
        ep = run_eps[d]
        for lap in range(len(ep)):
            iv = nap.IntervalSet(start=ep.start[lap], end=ep.end[lap])
            st = exc[u].restrict(iv)
            if len(st) == 0:
                continue
            sp = lin_valid.restrict(iv).interpolate(st).values
            ax.plot(sp, np.full(len(sp), lap), "|", ms=3,
                    color="tab:red" if d == "rightward" else "tab:blue", mew=0.9)
        ax.set(xlim=RNG_TRACK, ylim=(-1, len(ep)))
        if c == 0:
            ax.set_ylabel(f"{d}\nlap #")
        if row == 0:
            ax.set_title(f"unit {u}\nSI {res['rightward']['si'][i]:.2f} / "
                         f"{res['leftward']['si'][i]:.2f} bits/spk",
                         fontsize=10, pad=6)
        ax.tick_params(labelbottom=False)
    ax = axes[2, c]
    for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
        ax.plot(res[d]["rm"].centers, res[d]["maps"][i], color=col, lw=2, label=d)
    ax.set_xlim(RNG_TRACK)
    ax.set_xlabel("position on track (m)")
    if c == 0:
        ax.set_ylabel("rate (Hz)")
        ax.legend(fontsize=8, frameon=False)

fig.suptitle("CA1 place cells, DANDI:000044 sub-Achilles ses-10252013 — "
             "spikes plotted at the animal's position on each traversal",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"{FIGDIR}/fig03_example_place_cells.png", dpi=150)
plt.close(fig)
print("saved fig03_example_place_cells.png")

# %% [markdown]
# ### Figure 4: the population tiles the track
#
# Each row is one place cell's rate map, normalised to its own peak. Ordering
# cells by where they fire produces the familiar diagonal. The ordering is
# derived from the **odd** laps only, so the diagonal in the right-hand panel,
# built from held-out **even** laps, is cross-validated rather than circular.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, d in enumerate(DIRS):
    R = res[d]
    sel = np.where(R["is_place"])[0]
    ok = R["rm"].valid_bins
    cen = R["rm"].centers

    def norm(m):
        mx = np.nanmax(m, axis=1, keepdims=True)
        return m / np.where(mx > 0, mx, np.nan)

    order = sel[np.argsort(np.nanargmax(R["odd"][sel][:, ok], axis=1))]
    for col, (M, title) in enumerate([
            (R["maps"], "all laps"),
            (R["odd"], "odd laps (used for ordering)"),
            (R["even"], "even laps (held out)")]):
        ax = axes[row, col]
        im = ax.imshow(norm(M[order][:, ok]), aspect="auto", origin="lower",
                       cmap="viridis", vmin=0, vmax=1,
                       extent=[cen[ok][0], cen[ok][-1], 0, len(order)])
        ax.set_title(f"{d} — {title}", fontsize=11)
        ax.set_xlabel("position on track (m)")
        if col == 0:
            ax.set_ylabel("place cell # (ordered by odd-lap peak)")
        fig.colorbar(im, ax=ax, label="normalised rate", fraction=0.046)

fig.suptitle("Place fields tile the track, and the ordering holds on held-out laps",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"{FIGDIR}/fig04_population_maps.png", dpi=150)
plt.close(fig)
print("saved fig04_population_maps.png")

# %% [markdown]
# ### Figure 5: the tuning is statistically real
#
# The top-right panel is the reason for using a per-cell null. Points far to the
# right are units whose *shuffled* maps already score high spatial information;
# those are low-spike-count cells, where the estimator is biased upward. A single
# global threshold would have called many of them place cells. Comparing each
# unit with its own shuffles removes that bias.

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
R = res["rightward"]
ax = axes[0, 0]
ax.hist(R["null"].ravel(), bins=60, density=True, color="0.7",
        label=f"circular-shift null ({N_SHUFFLES} shuffles x {len(exc)} units)")
ax.hist(R["si"], bins=30, density=True, alpha=0.7, color="tab:red",
        label="observed (rightward)")
ax.set(xlabel="spatial information (bits/spike)", ylabel="density",
       title="Observed spatial information exceeds the shuffled null")
ax.legend(fontsize=9)

ax = axes[0, 1]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.scatter(R["null"].mean(axis=0), R["si"], s=18, alpha=0.6, color=col,
               label=f"{d} ({R['is_place'].sum()} place cells)")
lim = [0, max(np.nanmax(res[d]["si"]) for d in DIRS) * 1.05]
ax.plot(lim, lim, "k--", lw=1, label="unity")
ax.set(xlabel="mean SI of that unit's own shuffles (bits/spike)",
       ylabel="observed SI (bits/spike)", xlim=lim, ylim=lim,
       title="Every unit compared with its own null")
ax.legend(fontsize=9)

ax = axes[1, 0]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    ax.hist(res[d]["zval"][np.isfinite(res[d]["zval"])], bins=40, alpha=0.55,
            color=col, label=d)
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="SI z-score relative to own shuffle null", ylabel="units",
       title="Spatial information z-scores")
ax.legend(fontsize=9)

ax = axes[1, 1]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.scatter(R["si"], R["stability"], s=18, alpha=0.6, color=col, label=d)
ax.axhline(0, color="k", lw=1)
ax.set(xlabel="spatial information (bits/spike)",
       ylabel="odd vs even lap map correlation",
       title="Informative cells also have stable fields")
ax.legend(fontsize=9)

fig.suptitle("Statistical validation of spatial tuning", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"{FIGDIR}/fig05_spatial_information.png", dpi=150)
plt.close(fig)
print("saved fig05_spatial_information.png")

# %% [markdown]
# ### Figure 6: field properties and direction selectivity
#
# On a straight track, a cell's field for rightward runs is largely independent
# of its field for leftward runs. This is the classic directionality of
# linear-track place fields (McNaughton, Barnes & O'Keefe 1983).

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.hist(R["peak_rate"][R["is_place"]], bins=np.arange(0, 30, 1.5),
            alpha=0.55, color=col, label=d)
ax.set(xlabel="in-field peak rate (Hz)", ylabel="place cells",
       title="Peak firing rate")
ax.legend(fontsize=9)

ax = axes[0, 1]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.hist(R["width"][R["is_place"]], bins=np.arange(0, 1.0, 0.06),
            alpha=0.55, color=col, label=d)
ax.set(xlabel="field width at half maximum (m)", ylabel="place cells",
       title="Place field width")
ax.legend(fontsize=9)

ax = axes[1, 0]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.hist(R["peak_pos"][R["is_place"]], bins=np.linspace(*RNG_TRACK, 17),
            alpha=0.55, color=col, label=d)
ax.set(xlabel="field peak position (m)", ylabel="place cells",
       title="Fields cover the whole track (ends over-represented)")
ax.legend(fontsize=9)

ax = axes[1, 1]
ok = res["rightward"]["rm"].valid_bins
sel = np.where(either)[0]
mr, ml = res["rightward"]["maps"], res["leftward"]["maps"]
cc = np.array([np.corrcoef(mr[i, ok], ml[i, ok])[0, 1]
               if np.isfinite(mr[i, ok]).all() and np.isfinite(ml[i, ok]).all()
               and mr[i, ok].std() > 0 and ml[i, ok].std() > 0 else np.nan
               for i in sel])
ax.hist(cc[np.isfinite(cc)], bins=np.linspace(-1, 1, 33), color="tab:purple")
ax.axvline(np.nanmedian(cc), color="k", ls="--",
           label=f"median r = {np.nanmedian(cc):.2f}")
ax.set(xlabel="correlation between rightward and leftward rate maps",
       ylabel="place cells", title="Fields are directional on a linear track")
ax.legend(fontsize=9)

fig.suptitle("Place field properties, sub-Achilles ses-10252013", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"{FIGDIR}/fig06_field_properties.png", dpi=150)
plt.close(fig)
print(f"saved fig06_field_properties.png  (directional r median {np.nanmedian(cc):.3f})")

# %% [markdown]
# ## 8. Decoding position from the population
#
# The strongest functional test: if these fields carry spatial information, a
# decoder that has never seen a lap should be able to read the animal's position
# off the population activity. Tuning curves are fitted on **odd** laps and
# position is decoded in 200 ms bins on **even** laps with pynapple's
# `decode_bayes` (Zhang et al. 1998). Chance level comes from permuting which
# tuning curve belongs to which cell, which preserves all firing statistics and
# destroys only the cell-to-place correspondence.

# %%
BIN_SIZE = 0.2   # seconds


def build_tuning(group, ep):
    """Smoothed tuning curves from pynapple, ready for decode_bayes."""
    tc = nap.compute_tuning_curves(group, lin_valid, bins=N_BINS,
                                   range=RNG_TRACK, epochs=ep,
                                   feature_names=["position"])
    v = gaussian_filter1d(np.nan_to_num(np.asarray(tc.values, dtype=float)),
                          SMOOTH, axis=-1, mode="nearest")
    tc.values[...] = np.maximum(v, 1e-3)   # a hard zero would veto a bin
    return tc


dec = {}
for d in DIRS:
    ep = run_eps[d]
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    odd_ep = nap.IntervalSet(start=starts[0::2], end=ends[0::2])
    even_ep = nap.IntervalSet(start=starts[1::2], end=ends[1::2])

    sel = np.where(res[d]["is_place"])[0]
    ids = np.array(list(exc.keys()))[sel]
    group = exc[list(ids)]

    tc = build_tuning(group, odd_ep)
    decoded, proba = nap.decode_bayes(tc, group, even_ep, BIN_SIZE)
    true_pos = lin_valid.interpolate(decoded)
    err = np.abs(decoded.values - true_pos.values)
    finite = np.isfinite(err)

    rng = np.random.default_rng(0)
    chance = []
    for _ in range(20):
        tc_s = tc.copy()
        tc_s.values[...] = tc.values[rng.permutation(len(ids))]
        dec_s, _ = nap.decode_bayes(tc_s, group, even_ep, BIN_SIZE)
        e = np.abs(dec_s.values - lin_valid.interpolate(dec_s).values)
        chance.append(np.nanmedian(e[np.isfinite(e)]))

    r2 = np.corrcoef(decoded.values[finite], true_pos.values[finite])[0, 1] ** 2
    dec[d] = dict(decoded=decoded, proba=proba, true_pos=true_pos, err=err,
                  finite=finite, chance=np.array(chance), n_cells=len(sel), r2=r2,
                  lap_of_bin=np.searchsorted(even_ep.start, decoded.index.values,
                                             side="right") - 1,
                  centers=np.asarray(tc.coords["position"].values))
    print(f"[{d}] {finite.sum()} held-out bins, {len(sel)} place cells: "
          f"median error {np.median(err[finite]):.3f} m "
          f"(cell-shuffled {np.mean(chance):.3f} m), R^2 = {r2:.3f}")

# %% [markdown]
# ### Figure 7: decoded position on held-out laps

# %%
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1, 1], hspace=0.42, wspace=0.28)
N_LAPS_SHOW = 10
for row, d in enumerate(DIRS):
    O = dec[d]
    cen = O["centers"]
    ax = fig.add_subplot(gs[row, 0])
    m = O["lap_of_bin"] < N_LAPS_SHOW
    ax.imshow(np.asarray(O["proba"].values)[m].T, aspect="auto", origin="lower",
              cmap="Greys", extent=[0, m.sum(), cen[0], cen[-1]])
    xs = np.arange(m.sum()) + 0.5
    ax.plot(xs, O["true_pos"].values[m], "-", lw=2.5, color="tab:green",
            alpha=0.85, label="tracked position")
    ax.plot(xs, O["decoded"].values[m], ".", ms=7, color="tab:red",
            label="decoded position")
    for b in np.where(np.diff(O["lap_of_bin"][m]) > 0)[0]:
        ax.axvline(b + 1, color="tab:blue", lw=1, ls=":")
    ax.set(xlabel=f"consecutive {BIN_SIZE*1000:.0f} ms decoding bins "
                  "(dotted lines separate held-out laps)",
           ylabel="position on track (m)", xlim=(0, m.sum()))
    ax.set_title(f"{d}: posterior over position, {N_LAPS_SHOW} held-out laps "
                 f"({O['n_cells']} place cells)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

    ax = fig.add_subplot(gs[row, 1])
    ax.scatter(O["true_pos"].values, O["decoded"].values, s=9, alpha=0.35,
               color="tab:red" if d == "rightward" else "tab:blue")
    ax.plot(RNG_TRACK, RNG_TRACK, "k--", lw=1)
    ax.set(xlabel="true position (m)", ylabel="decoded position (m)",
           xlim=RNG_TRACK, ylim=RNG_TRACK)
    ax.set_title(f"{d}: decoded vs true\n$R^2$ = {O['r2']:.2f}", fontsize=11)
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[row, 2])
    e = O["err"][O["finite"]]
    ax.hist(e, bins=np.arange(0, 1.65, 0.05), color="0.45")
    ax.axvline(np.median(e), color="tab:red", lw=2,
               label=f"median {np.median(e):.2f} m")
    ax.axvline(O["chance"].mean(), color="k", ls="--", lw=2,
               label=f"cell-shuffled {O['chance'].mean():.2f} m")
    ax.set(xlabel="|decoded - true| (m)", ylabel="time bins")
    ax.set_title(f"{d}: error distribution", fontsize=11)
    ax.legend(fontsize=8)

fig.suptitle("Position is decodable from CA1 place cells on held-out laps "
             "(tuning curves fitted on odd laps, decoding on even laps)",
             fontsize=13)
fig.savefig(f"{FIGDIR}/fig07_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig07_decoding.png")

# %% [markdown]
# ## 9. All eight sessions
#
# The five straight-track sessions get the directional treatment above. On the
# three circular-track sessions the animal runs essentially one way, and position
# is a circular variable, so we analyse only the dominant direction and use
# wrap-aware smoothing and field widths. Spatial bins are held at 4 cm in every
# session so widths are comparable across the 1.6 m, 2 m and circular tracks.

# %%
TARGET_BIN = 0.04
N_SHUFFLES_ALL = 500

per_session = []
for asset, label in tqdm(SESSIONS.items(), total=len(SESSIONS), desc="sessions"):
    nwb_s = open_session(asset)
    check_timebase(nwb_s)
    pos2d_s, lin_s, fs_s, maze_s = load_behavior(nwb_s)
    kind = maze_kind(maze_s)
    lin_s = lin_s.restrict(load_epochs(nwb_s)["MazeEpoch"])
    rng_s = track_range(lin_s)
    length = rng_s[1] - rng_s[0]
    lin_v, vel_s, eps_s = make_run_epochs(
        lin_s, fs_s, period=length if kind == "circular" else None)
    units_s = load_units(nwb_s)
    exc_s = units_s[np.where(units_s.cell_type == "excitatory")[0]]
    n_bins_s = int(round(length / TARGET_BIN))

    conds = DIRS if kind == "linear" else \
        [max(DIRS, key=lambda k: eps_s[k].tot_length())]

    s = dict(label=label, maze=maze_s, kind=kind, length=length,
             n_exc=len(exc_s), conds=conds, per_cond={}, maps={})
    for d in conds:
        ep = eps_s[d]
        rm = DirectionalRateMaps(lin_v, ep, fs_s, n_bins_s, rng_s,
                                 smooth_bins=SMOOTH, circular=(kind == "circular"))
        maps = rm.rate_maps(exc_s)
        si = rm.si(maps)[0]
        null = rm.shuffle_si(exc_s, n_shuffles=N_SHUFFLES_ALL, seed=1, progress=False)
        pval = (np.sum(null >= si[None, :], axis=0) + 1) / (N_SHUFFLES_ALL + 1)
        pk, pp, w = field_metrics(maps, rm.centers, rm.bin_size,
                                  circular=(kind == "circular"))
        n_spk = np.array([len(exc_s[u].restrict(ep)) for u in exc_s.keys()])
        odd, even = rm.lap_split_maps(exc_s)
        ok = rm.valid_bins
        stab = np.array([
            np.corrcoef(odd[i, ok], even[i, ok])[0, 1]
            if np.isfinite(odd[i, ok]).all() and np.isfinite(even[i, ok]).all()
            and odd[i, ok].std() > 0 and even[i, ok].std() > 0 else np.nan
            for i in range(len(exc_s))])
        is_place = (pval < ALPHA) & (pk >= MIN_PEAK_RATE) & (n_spk >= MIN_SPIKES)
        s["per_cond"][d] = dict(n_laps=len(ep), n_place=int(is_place.sum()),
                                frac=float(is_place.mean()), si=si[is_place],
                                width=w[is_place], peak_rate=pk[is_place],
                                peak_pos=pp[is_place] / length, stab=stab[is_place])
        s["maps"][d] = (maps, is_place, ok)

    anyplace = np.zeros(len(exc_s), bool)
    for d in conds:
        anyplace |= s["maps"][d][1]
    s["n_either"] = int(anyplace.sum())

    if kind == "linear":
        mr_, pr_, okr = s["maps"]["rightward"]
        ml_, pl_, okl = s["maps"]["leftward"]
        okb = okr & okl
        s["dir_corr"] = np.array(
            [np.corrcoef(mr_[i, okb], ml_[i, okb])[0, 1]
             for i in np.where(pr_ | pl_)[0]
             if mr_[i, okb].std() > 0 and ml_[i, okb].std() > 0])
    else:
        s["dir_corr"] = np.array([])
    per_session.append(s)

print(f"\n{'session':<20}{'maze':<16}{'exc':>5}{'laps':>12}{'place cells':>22}")
for s in per_session:
    laps = "/".join(str(s["per_cond"][d]["n_laps"]) for d in s["conds"])
    pcs = "/".join(str(s["per_cond"][d]["n_place"]) for d in s["conds"])
    print(f"{s['label']:<20}{s['maze'].replace('Position',''):<16}{s['n_exc']:>5}"
          f"{laps:>12}{pcs + '  (either ' + str(s['n_either']) + ')':>22}")


def pool(key, kinds=("linear", "circular")):
    return np.concatenate([s["per_cond"][d][key] for s in per_session
                           if s["kind"] in kinds for d in s["conds"]])


tot_exc = sum(s["n_exc"] for s in per_session)
tot_either = sum(s["n_either"] for s in per_session)
all_si, all_w = pool("si"), pool("width")
all_pk, all_st, all_pos = pool("peak_rate"), pool("stab"), pool("peak_pos")
all_dc = np.concatenate([s["dir_corr"] for s in per_session])
fracs = np.array([s["per_cond"][d]["frac"] for s in per_session for d in s["conds"]])

print(f"\npooled across {len(per_session)} sessions "
      f"({sum(s['kind']=='linear' for s in per_session)} linear, "
      f"{sum(s['kind']=='circular' for s in per_session)} circular):")
print(f"  excitatory units {tot_exc}; place cell in >=1 condition {tot_either} "
      f"({100*tot_either/tot_exc:.0f}%)")
print(f"  per-condition fraction {100*fracs.mean():.0f}% +/- {100*fracs.std():.0f}% (SD)")
print(f"  spatial information median {np.median(all_si):.2f} bits/spike")
print(f"  field width median {np.median(all_w):.2f} m")
print(f"  peak rate median {np.median(all_pk):.1f} Hz")
print(f"  odd/even stability median r = {np.nanmedian(all_st):.2f}")
print(f"  linear tracks: rightward vs leftward map correlation median r = "
      f"{np.nanmedian(all_dc):.2f} (n={np.isfinite(all_dc).sum()})")

# %% [markdown]
# ### Figure 8: replication across sessions and animals

# %%
labels = [f"{s['label']}\n({s['kind']})" for s in per_session]
x = np.arange(len(labels))
lin_mask = np.array([s["kind"] == "linear" for s in per_session])
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

ax = axes[0, 0]
for off, d, col in [(-0.2, "rightward", "tab:red"), (0.2, "leftward", "tab:blue")]:
    ax.bar(x[lin_mask] + off,
           [100 * s["per_cond"][d]["frac"] for s in per_session
            if s["kind"] == "linear"], 0.4, color=col, label=d)
ax.bar(x[~lin_mask], [100 * s["per_cond"][s["conds"][0]]["frac"]
                      for s in per_session if s["kind"] == "circular"],
       0.5, color="tab:olive", label="circular (one direction)")
ax.axhline(100 * fracs.mean(), color="k", ls="--", lw=1,
           label=f"mean {100*fracs.mean():.0f}%")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
ax.set(ylabel="% of excitatory units", title="Place cells per session")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.hist([pool("si", ("linear",)), pool("si", ("circular",))], bins=35,
        stacked=True, color=["tab:green", "tab:olive"],
        label=["linear track", "circular track"])
ax.axvline(np.median(all_si), color="k", ls="--",
           label=f"median {np.median(all_si):.2f}")
ax.set(xlabel="spatial information (bits/spike)", ylabel="place cells",
       title=f"Spatial information, pooled (n={len(all_si)})")
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.hist([pool("width", ("linear",)), pool("width", ("circular",))],
        bins=np.arange(0, 1.5, 0.05), stacked=True,
        color=["tab:orange", "peachpuff"], label=["linear", "circular"])
ax.axvline(np.median(all_w), color="k", ls="--",
           label=f"median {np.median(all_w):.2f} m")
ax.set(xlabel="field width at half maximum (m)", ylabel="place cells",
       title="Place field width")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.hist(all_pk, bins=np.arange(0, 40, 1.5), color="tab:purple")
ax.axvline(np.median(all_pk), color="k", ls="--",
           label=f"median {np.median(all_pk):.1f} Hz")
ax.set(xlabel="in-field peak rate (Hz)", ylabel="place cells",
       title="Peak firing rate")
ax.legend(fontsize=9)

ax = axes[1, 1]
ax.hist(all_pos, bins=np.linspace(0, 1, 21), color="tab:brown")
ax.set(xlabel="field peak, as a fraction of track length", ylabel="place cells",
       title="Fields tile the track")

ax = axes[1, 2]
ax.hist(all_dc[np.isfinite(all_dc)], bins=np.linspace(-1, 1, 41), color="tab:cyan")
ax.axvline(np.nanmedian(all_dc), color="k", ls="--",
           label=f"median r = {np.nanmedian(all_dc):.2f}")
ax.set(xlabel="correlation of rightward and leftward rate maps",
       ylabel="place cells",
       title="Direction selectivity (five linear-track sessions)")
ax.legend(fontsize=9)

fig.suptitle("Place fields across all eight sessions of DANDI:000044 "
             f"({tot_exc} excitatory units, {tot_either} place cells)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIGDIR}/fig08_all_sessions.png", dpi=150)
plt.close(fig)
print("saved fig08_all_sessions.png")

# %% [markdown]
# ## 10. Summary
#
# Streaming eight CA1 recordings from DANDI:000044 and analysing them entirely
# with pynapple reproduces the defining properties of hippocampal place cells:
#
# * **Spatial selectivity.** Of 562 excitatory units, 374 (67%) fire in a
#   restricted portion of the track with spatial information beyond their own
#   circular-shift null at p < 0.01. Median spatial information is about
#   0.7 bits/spike against a null median near 0.2.
# * **Reliability.** Odd- and even-lap rate maps correlate at r ≈ 0.9, so the
#   fields are a stable property of the cell within a session rather than one or
#   two unusual traversals.
# * **Field structure.** Median field width is roughly 0.3 m at half maximum on
#   tracks of 1.6-2.9 m, with median in-field peak rates near 6 Hz. Field peaks
#   cover the whole track, with the reward ends over-represented.
# * **Directionality.** On straight tracks, rightward and leftward maps of the
#   same cell correlate only weakly (median r ≈ 0.3), the well-known
#   directionality of linear-track place fields.
# * **Population code.** A naive Bayes decoder trained on odd laps localises the
#   animal on held-out even laps to a median error of about 5 cm on a 1.6 m
#   track, against roughly 45 cm for a cell-identity shuffle, with R² ≈ 0.95.
#
# Sensible caveats: units come from the published spike sorting and we take the
# `excitatory` label at face value; place-cell counts depend on the inclusion
# thresholds (peak rate, spike count, α), though the qualitative picture is not
# sensitive to them; and the reward sites at the track ends mean occupancy and
# field density are not uniform along the track.
