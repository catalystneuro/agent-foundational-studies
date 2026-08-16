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
# # Hippocampal place cells in rat CA1 (DANDI:000044)
#
# This notebook demonstrates hippocampal place cells using real extracellular
# recordings streamed from the DANDI Archive. Nothing here is simulated.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044), *"Diversity
# in neural firing dynamics supports both rigid and learned hippocampal sequences"*
# (Grosmark & Buzsáki, Science 2016). Eight sessions from four Long-Evans rats,
# each with bilateral silicon-probe recordings from dorsal CA1 and simultaneous
# video tracking. Every session is a sleep / maze / sleep sandwich; we use the maze
# epoch, during which the animal shuttles back and forth on a linear track for
# water reward at the ends. Spike sorting and a pyramidal-cell / interneuron label
# are provided with the dataset, as is a linearized position signal sampled at
# 39 Hz.
#
# Five of the eight sessions use a linear track (four 1.6 m, one 2 m); the other
# three use a circular maze, whose different topology would need separate
# treatment, so they are left out here. Files are 5-9 GB each, dominated by the raw
# LFP, so we stream them with `remfile` and a local chunk cache and only ever touch
# the spike times and the behavior module.
#
# **What is shown.**
#
# 1. Individual CA1 pyramidal cells fire in a restricted portion of the track and
#    are nearly silent elsewhere, reliably from lap to lap.
# 2. The fields of the recorded population tile the whole track.
# 3. Spatial information exceeds a circular-shift null for roughly two thirds of
#    cell-by-direction pairs, and about 80% of active pyramidal cells have a field
#    in at least one running direction.
# 4. Position can be decoded from the population with a median error of 7-12 cm on
#    tracks 1.6-2 m long, from held-out laps.
# 5. A Poisson GLM with a spline basis over position (NeMoS) reproduces the
#    measured rate maps and predicts held-out spike counts well above a
#    rate-matched null.
# 6. All of the above holds in each of the five linear-track sessions.

# %% [markdown]
# ## Setup

# %%
import json
import os
import re
import time

import h5py
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln
from tqdm.auto import tqdm

np.random.seed(0)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": False, "font.size": 10})

DANDISET = "000044"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000044")
ASSET_FILE = "assets_000044.json"

NBINS = 40           # spatial bins across the track
SPEED_THRESH = 0.05  # m/s; excludes pauses on the track
POS_FS = 39.0626     # position sampling rate of this dandiset (Hz)
SMOOTH_BINS = 1.0    # s.d. of the Gaussian smoothing of rate maps, in bins
N_SHUFFLES = 500     # circular shifts for the spatial-information null

print("pynapple", nap.__version__, "| nemos", nmo.__version__)


# %% [markdown]
# ## Finding the data on DANDI
#
# The asset list is fetched once from the DANDI API and cached next to the
# notebook, so re-running is offline apart from the NWB chunks themselves.

# %%
def get_assets():
    """Return {path: download_url} for every NWB asset in the dandiset."""
    if os.path.exists(ASSET_FILE):
        return json.load(open(ASSET_FILE))
    import urllib.request

    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           "/versions/draft/assets/?page_size=100")
    with urllib.request.urlopen(url) as r:
        payload = json.load(r)
    assets = {
        a["path"]: (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
                    f"/versions/draft/assets/{a['asset_id']}/download/")
        for a in sorted(payload["results"], key=lambda a: a["path"])
    }
    json.dump(assets, open(ASSET_FILE, "w"), indent=1)
    return assets


assets = get_assets()
for p in assets:
    print(p)


# %% [markdown]
# ## Streaming an NWB file and converting it to pynapple objects
#
# `load_session` pulls out four things: the spike trains, the linearized position,
# the running speed derived from it, and the set of individual track traversals.
#
# The linearized position trace is only defined while the animal is actually on the
# track; the time it spends in the end zones is NaN. Every contiguous run of valid
# samples that covers at least half the track is therefore one lap, and the sign of
# the displacement across that lap gives the running direction. Place fields on a
# linear track are strongly direction-selective, so the two directions are always
# treated as separate conditions.

# %%
def open_session(url):
    """Stream one NWB file from S3 with an on-disk chunk cache."""
    f = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(f, "r"), load_namespaces=True)
    return io.read(), io


def maze_type(url):
    """Name of the maze used in a session, without loading the bulk data."""
    nwbfile, io = open_session(url)
    key = [k for k in nwbfile.processing["behavior"].data_interfaces
           if k.endswith("LinearizedPosition")][0]
    io.close()
    return key.replace("LinearizedPosition", "")


def load_session(url):
    """Load spikes, position, speed and laps of one session into pynapple."""
    nwbfile, io = open_session(url)

    epochs = nwbfile.intervals["epochs"].to_dataframe()
    maze_row = epochs[epochs["label"].str.contains("Maze")].iloc[0]
    maze = nap.IntervalSet(start=maze_row["start_time"], end=maze_row["stop_time"])

    beh = nwbfile.processing["behavior"]
    lin_key = [k for k in beh.data_interfaces if k.endswith("LinearizedPosition")][0]
    maze_name = lin_key.replace("LinearizedPosition", "")
    lin_ss = list(beh[lin_key].spatial_series.values())[0]
    pos_ss = list(beh[maze_name + "Position"].spatial_series.values())[0]

    lin_raw = np.asarray(lin_ss.data[:]).ravel()
    m = re.match(r"([\d.]+)m", maze_name)
    track_len = float(m.group(1)) if m else float(np.nanmax(lin_raw))

    t = lin_ss.starting_time + np.arange(lin_raw.size) / lin_ss.rate
    dt = 1.0 / lin_ss.rate
    pos2d = nap.TsdFrame(t=t, d=np.asarray(pos_ss.data[:]), columns=["x", "y"])

    valid = np.isfinite(lin_raw)
    idx = np.flatnonzero(valid)
    segments = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    lin = nap.Tsd(t=t[valid], d=lin_raw[valid], time_support=maze)

    starts, ends, dirs = [], [], []
    for seg in segments:
        if seg.size < 10:  # drop tracking fragments shorter than ~0.25 s
            continue
        travel = lin_raw[seg[-1]] - lin_raw[seg[0]]
        if abs(travel) < 0.5 * track_len:  # require crossing half the track
            continue
        starts.append(t[seg[0]])
        ends.append(t[seg[-1]] + dt)
        dirs.append(1 if travel > 0 else -1)
    laps = nap.IntervalSet(start=np.array(starts), end=np.array(ends))
    lap_dir = np.array(dirs)

    # speed computed lap by lap so the off-track gaps create no spurious jumps
    sp_t, sp_v = [], []
    for i in range(len(laps)):
        seg = lin.restrict(laps[i])
        if len(seg) < 3:
            continue
        sp_t.append(seg.times())
        sp_v.append(np.abs(np.gradient(seg.values, seg.times())))
    speed = nap.Tsd(t=np.concatenate(sp_t), d=np.concatenate(sp_v), time_support=maze)

    udf = nwbfile.units.to_dataframe()
    spikes = {int(i): nap.Ts(np.asarray(row["spike_times"]))
              for i, row in udf.iterrows()}
    meta = pd.DataFrame({"cell_type": udf["cell_type"].values,
                         "location": udf["location"].values,
                         "shank_id": udf["shank_id"].values},
                        index=[int(i) for i in udf.index])
    units = nap.TsGroup(spikes, metadata=meta)

    return dict(units=units, lin=lin, pos2d=pos2d, speed=speed, laps=laps,
                lap_dir=lap_dir, maze=maze, track_len=track_len,
                maze_name=maze_name, session=nwbfile.session_id or "",
                subject=nwbfile.subject.subject_id, io=io)


SESSION = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
s = load_session(assets[SESSION])
name = f"{s['subject']} {s['session']}"

print(name, "|", s["maze_name"], "| track", s["track_len"], "m")
print(len(s["units"]), "sorted units:",
      dict(s["units"].metadata["cell_type"].value_counts()))
print(dict(s["units"].metadata["location"].value_counts()))
print(len(s["laps"]), "laps;",
      f"{(s['lap_dir'] > 0).sum()} rightward, {(s['lap_dir'] < 0).sum()} leftward")
print("maze epoch:", np.round(s["maze"].tot_length()), "s;",
      "time on track:", np.round(s["laps"].tot_length()), "s")
print(s["units"])


# %% [markdown]
# ## Checking the behavior before touching the spikes
#
# The 2D tracking shows the track itself plus a side box the animal is carried to
# between blocks; the laps we extracted (blue) are confined to the track. The
# linearized trace is a clean sawtooth, one ramp per traversal, at roughly
# 0.7 m/s.

# %%
def direction_epochs(laps, lap_dir):
    """Split laps into two IntervalSets by running direction."""
    return {name: nap.IntervalSet(start=laps.start[lap_dir == d],
                                  end=laps.end[lap_dir == d])
            for d, name in ((1, "rightward"), (-1, "leftward"))}


def run_epochs(speed, ep, thresh=SPEED_THRESH):
    """Portions of `ep` in which the animal runs faster than `thresh`."""
    return speed.restrict(ep).threshold(thresh, "above").time_support


def occupancy(lin, ep, track_len, nbins=NBINS):
    """Seconds spent in each spatial bin."""
    counts, edges = np.histogram(lin.restrict(ep).values,
                                 bins=np.linspace(0, track_len, nbins + 1))
    return counts / POS_FS, edges


deps = direction_epochs(s["laps"], s["lap_dir"])
run = {k: run_epochs(s["speed"], v) for k, v in deps.items()}
print({k: round(v.tot_length(), 1) for k, v in run.items()}, "s of running")

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
xy = s["pos2d"].restrict(s["maze"])
ax[0, 0].plot(xy["x"].values, xy["y"].values, ".", ms=1, alpha=0.15, color="0.5")
onlap = s["pos2d"].restrict(s["laps"])
ax[0, 0].plot(onlap["x"].values, onlap["y"].values, ".", ms=1, alpha=0.3, color="C0")
ax[0, 0].set(xlabel="x (m)", ylabel="y (m)",
             title="2D tracking (blue = extracted laps)")
ax[0, 0].set_aspect("equal")

t0 = s["laps"].start[0]
w = s["lin"].restrict(nap.IntervalSet(t0 - 5, t0 + 115))
ax[0, 1].plot(w.times() - t0, w.values, ".", ms=2, color="C0")
for i in range(len(s["laps"])):
    a, b = s["laps"].start[i] - t0, s["laps"].end[i] - t0
    if b >= -5 and a <= 115:
        ax[0, 1].axvspan(a, b, color="C1" if s["lap_dir"][i] > 0 else "C2", alpha=0.2)
ax[0, 1].set(xlabel="time from first lap (s)", ylabel="linearized position (m)",
             title="Laps shaded (orange = rightward, green = leftward)")

ax[1, 0].hist(s["speed"].restrict(s["laps"]).values, bins=50, color="0.35")
ax[1, 0].axvline(SPEED_THRESH, color="r", ls="--", label=f"{SPEED_THRESH} m/s cutoff")
ax[1, 0].set(xlabel="running speed (m/s)", ylabel="position samples",
             title="Speed during traversals")
ax[1, 0].legend()

for k, c in (("rightward", "C1"), ("leftward", "C2")):
    occ, edges = occupancy(s["lin"], run[k], s["track_len"])
    ax[1, 1].step(edges[:-1] + np.diff(edges) / 2, occ, where="mid", color=c, label=k)
ax[1, 1].set(xlabel="position (m)", ylabel="occupancy (s)",
             title="Occupancy is roughly uniform along the track", ylim=(0, None))
ax[1, 1].legend()
fig.suptitle(f"Behavior on the {s['track_len']} m linear track - {name}", fontsize=13)
fig.tight_layout()
fig.savefig("fig01_behavior.png", dpi=130)
plt.show()


# %% [markdown]
# ## Rate maps, spatial information and the shuffle test
#
# Rate maps are occupancy-normalized firing rates in 4-5 cm bins, computed
# separately for each running direction and lightly smoothed (Gaussian, s.d. one
# bin). Spatial information is the Skaggs measure in bits per spike,
#
# $$\mathrm{SI} = \sum_i p_i \frac{r_i}{\bar r}\log_2\frac{r_i}{\bar r},$$
#
# with $p_i$ the fraction of running time in bin $i$ and $\bar r$ the
# occupancy-weighted mean rate. Sparse, low-rate cells can reach high SI by
# chance, so each unit is compared against **its own** null: the spike train is
# circularly shifted by a random offset within the running epochs 500 times, which
# destroys the relationship to position while preserving spike count and burst
# structure. A unit counts as a place cell in a direction when its SI exceeds the
# 99th percentile of that null and its peak rate is at least 1 Hz. Split-half
# stability (odd versus even laps) is computed as an independent check and is not
# part of the criterion.

# %%
def tuning_curves(units, lin, ep, track_len, nbins=NBINS, as_xarray=False,
                  smooth=SMOOTH_BINS):
    """Occupancy-normalized 1D rate maps (Hz). `smooth` is in bins."""
    tc = nap.compute_tuning_curves(units, lin, bins=nbins, range=[(0.0, track_len)],
                                   epochs=ep, fs=POS_FS, return_pandas=not as_xarray)
    if smooth:
        vals = gaussian_filter1d(np.nan_to_num(np.asarray(tc)), smooth,
                                 axis=-1 if as_xarray else 0, mode="nearest")
        tc = tc.copy(data=vals) if as_xarray else pd.DataFrame(
            vals, index=tc.index, columns=tc.columns)
    return tc


def spatial_information(tc, lin, ep, track_len, nbins=NBINS):
    """Skaggs spatial information (bits/spike) for each column of `tc`."""
    occ, _ = occupancy(lin, ep, track_len, nbins)
    p = occ / occ.sum()
    r = tc.values
    ok = np.isfinite(r) & (p[:, None] > 0)
    rbar = np.nansum(np.where(ok, r * p[:, None], 0.0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r / rbar
        term = np.where(ok & (r > 0), p[:, None] * ratio * np.log2(ratio), 0.0)
    si = np.nansum(term, axis=0)
    si[rbar <= 0] = np.nan
    return pd.Series(si, index=tc.columns)


def field_stats(tc):
    """Peak rate, peak position, mean rate and sparsity for every unit."""
    r = np.nan_to_num(tc.values)
    mean_r, mean_r2 = r.mean(axis=0), (r ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sparsity = mean_r ** 2 / mean_r2
    return pd.DataFrame({"peak_rate": r.max(axis=0),
                         "peak_pos": tc.index.values[r.argmax(axis=0)],
                         "mean_rate": mean_r, "sparsity": sparsity},
                        index=tc.columns)


def field_width(tc, frac=0.5):
    """Width (m) of the contiguous region around the peak above frac * peak."""
    binw = tc.index.values[1] - tc.index.values[0]
    widths = {}
    for c in tc.columns:
        r = np.nan_to_num(tc[c].values)
        pk = int(r.argmax())
        thr = frac * r[pk]
        lo = hi = pk
        while lo > 0 and r[lo - 1] >= thr:
            lo -= 1
        while hi < len(r) - 1 and r[hi + 1] >= thr:
            hi += 1
        widths[c] = (hi - lo + 1) * binw
    return pd.Series(widths)


def half_tuning_curves(s, units, k, parity, nbins=NBINS):
    sel = np.flatnonzero(s["lap_dir"] == (1 if k == "rightward" else -1))[parity::2]
    ep = run_epochs(s["speed"], nap.IntervalSet(s["laps"].start[sel],
                                                s["laps"].end[sel]))
    return tuning_curves(units, s["lin"], ep, s["track_len"], nbins)


def split_half_stability(s, units, k, nbins=NBINS):
    """Correlation between rate maps built from odd and from even laps."""
    a = half_tuning_curves(s, units, k, 0, nbins)
    b = half_tuning_curves(s, units, k, 1, nbins)
    return pd.Series({u: np.corrcoef(np.nan_to_num(a[u]), np.nan_to_num(b[u]))[0, 1]
                      for u in a.columns})


def shuffle_si(units, lin, ep, track_len, n_shuffles=N_SHUFFLES, nbins=NBINS, seed=0):
    """Null distribution of SI from spike trains circularly shifted within `ep`."""
    rng = np.random.default_rng(seed)
    starts, durs = ep.start, ep.end - ep.start
    cum = np.concatenate([[0], np.cumsum(durs)])
    total = cum[-1]

    def to_flat(times):
        i = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(durs) - 1)
        return cum[i] + (times - starts[i])

    def from_flat(u):
        i = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(durs) - 1)
        return starts[i] + (u - cum[i])

    flat = {k: to_flat(units[k].restrict(ep).times()) for k in units.keys()}
    null = np.full((n_shuffles, len(units)), np.nan)
    for sh in range(n_shuffles):
        shifted = {}
        for k in units.keys():
            if flat[k].size == 0:
                shifted[k] = nap.Ts(np.array([starts[0]]))
                continue
            off = rng.uniform(20.0, max(total - 20.0, 21.0))
            shifted[k] = nap.Ts(np.sort(from_flat(np.mod(flat[k] + off, total))))
        tc = tuning_curves(nap.TsGroup(shifted, time_support=ep), lin, ep,
                           track_len, nbins)
        null[sh] = spatial_information(tc, lin, ep, track_len, nbins).values
    return null


def track_units(s, min_rate=0.1):
    """Putative pyramidal cells that fire at all while the animal is on the track."""
    exc = s["units"].getby_category("cell_type")["excitatory"]
    deps = direction_epochs(s["laps"], s["lap_dir"])
    on_track = run_epochs(s["speed"],
                          deps["rightward"].union(deps["leftward"]))
    return exc[np.asarray(exc.restrict(on_track).rates > min_rate)]


def classify_place_cells(s, units, n_shuffles=N_SHUFFLES, nbins=NBINS, seed=0):
    """Rate maps, spatial information, shuffle test and field statistics."""
    deps = direction_epochs(s["laps"], s["lap_dir"])
    run = {k: run_epochs(s["speed"], v) for k, v in deps.items()}
    tc, si, null, stats = {}, {}, {}, {}
    for i, k in enumerate(run):
        tc[k] = tuning_curves(units, s["lin"], run[k], s["track_len"], nbins)
        si[k] = spatial_information(tc[k], s["lin"], run[k], s["track_len"], nbins)
        null[k] = shuffle_si(units, s["lin"], run[k], s["track_len"], n_shuffles,
                             nbins, seed + i)
        fs = field_stats(tc[k])
        fs["si"] = si[k].values
        fs["si_thresh"] = np.nanpercentile(null[k], 99, axis=0)
        fs["si_p"] = (null[k] >= si[k].values[None, :]).mean(axis=0)
        fs["width"] = field_width(tc[k]).values
        fs["stability"] = split_half_stability(s, units, k, nbins).reindex(fs.index).values
        fs["is_place"] = (fs["si"] > fs["si_thresh"]) & (fs["peak_rate"] >= 1.0)
        fs["is_place_strict"] = fs["is_place"] & (fs["stability"] > 0.3)
        stats[k] = fs
    return dict(tc=tc, si=si, null=null, stats=stats, run=run)


units = track_units(s)
print(f"{len(units)} putative pyramidal cells active on the track "
      f"(of {len(s['units'])} sorted units)")

t_start = time.time()
res = classify_place_cells(s, units)
print("classification took %.0f s" % (time.time() - t_start))

stats = pd.concat(res["stats"], names=["direction", "unit"])
for k, v in res["stats"].items():
    print(f"{k}: {int(v['is_place'].sum())}/{len(v)} place fields")
print("units with a field in at least one direction: "
      f"{stats.groupby('unit')['is_place'].any().mean():.0%}")
stats.to_csv("single_session_unit_stats.csv")
stats.head()


# %% [markdown]
# ## Individual place cells
#
# For each example: spikes (red) superimposed on the position trace, a per-lap
# raster in track coordinates with the two running directions separated, and the
# rate map. The cells fire in one restricted stretch of the track, on essentially
# every lap, and mostly in one direction of travel.

# %%
tc, si = res["tc"], res["si"]

def pick_examples(cand, track_len, n=4):
    """Strong, well-isolated fields spread along the track."""
    for spacing in (0.15, 0.12, 0.09, 0.06, 0.0):
        chosen, used = [], []
        for (direction, unit), row in cand.iterrows():
            if unit in [c[0] for c in chosen]:
                continue
            if any(abs(row["peak_pos"] - q) < spacing * track_len for q in used):
                continue
            chosen.append((unit, direction))
            used.append(row["peak_pos"])
            if len(chosen) == n:
                return chosen
    return chosen


# fields away from the ends, so the whole field is inside the recorded range
margin = 0.1 * s["track_len"]
cand = stats[stats["is_place"] & (stats["peak_rate"] >= 5) &
             (stats["stability"] >= 0.5) &
             stats["peak_pos"].between(margin, s["track_len"] - margin)]
chosen = pick_examples(cand.sort_values("si", ascending=False), s["track_len"])
n_ex = len(chosen)
print("examples (unit, preferred direction):", chosen)

fig, axes = plt.subplots(3, n_ex, figsize=(3.75 * n_ex, 9.5),
                         gridspec_kw=dict(height_ratios=[1.1, 1.5, 1.0]))
lap_idx = {k: np.flatnonzero(s["lap_dir"] == (1 if k == "rightward" else -1))
           for k in ("rightward", "leftward")}
t0 = s["laps"].start[0]
window = nap.IntervalSet(t0, t0 + 90)

for j, (u, best_dir) in enumerate(chosen):
    axa, axb, axc = axes[0, j], axes[1, j], axes[2, j]

    w = s["lin"].restrict(window)
    axa.plot(w.times() - t0, w.values, ".", ms=1.5, color="0.75")
    sp = units[u].restrict(window.intersect(s["laps"]))
    axa.plot(sp.times() - t0,
             np.interp(sp.times(), s["lin"].times(), s["lin"].values),
             "|", ms=7, color="C3", mew=1.2)
    axa.set(title=f"unit {u}", xlabel="time (s)", ylim=(0, s["track_len"]))
    if j == 0:
        axa.set_ylabel("position (m)")

    y = 0
    for k, c in (("rightward", "C1"), ("leftward", "C2")):
        for li in lap_idx[k]:
            sp = units[u].restrict(s["laps"][li])
            if len(sp):
                pp = np.interp(sp.times(), s["lin"].times(), s["lin"].values)
                axb.plot(pp, np.full(len(pp), y), ".", ms=2.5, color=c)
            y += 1
        axb.axhline(y - 0.5, color="k", lw=0.6)
    axb.set(xlim=(0, s["track_len"]), ylim=(-1, y), xlabel="position (m)")
    if j == 0:
        axb.set_ylabel("lap (rightward, then leftward)")

    for k, c in (("rightward", "C1"), ("leftward", "C2")):
        axc.plot(tc[k].index, tc[k][u], color=c, lw=2, label=k)
    r = stats.loc[(best_dir, u)]
    axc.set(xlabel="position (m)", xlim=(0, s["track_len"]), ylim=(0, None))
    axc.text(0.03, 0.96,
             f"{best_dir}\nSI {r['si']:.2f} b/spk\npeak {r['peak_rate']:.1f} Hz\n"
             f"width {r['width']:.2f} m\nstability r={r['stability']:.2f}",
             transform=axc.transAxes, va="top", fontsize=8,
             bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.5))
    if j == 0:
        axc.set_ylabel("firing rate (Hz)")
        axc.legend(fontsize=8, loc="upper right")

fig.suptitle(f"Example CA1 place cells - {name}", fontsize=13)
fig.tight_layout()
fig.savefig("fig02_example_cells.png", dpi=130)
plt.show()


# %% [markdown]
# ## The population tiles the track
#
# Sorting the peak-normalized rate maps by the location of their peak gives the
# usual diagonal band: for every position on the track there are cells whose
# firing is concentrated there. Sorting is done within each direction separately,
# and the same units appear in both panels in a different order, which is itself a
# demonstration of directionality.

# %%
stability_all = np.concatenate([res["stats"][k]["stability"].values for k in tc])

fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.5),
                       gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
for i, k in enumerate(["rightward", "leftward"]):
    m = np.nan_to_num(tc[k].values.T)
    norm = m / np.maximum(m.max(axis=1, keepdims=True), 1e-9)
    idx = np.argsort(np.argmax(m, axis=1))
    im = ax[i].imshow(norm[idx], aspect="auto", origin="lower", cmap="viridis",
                      extent=[0, s["track_len"], 0, m.shape[0]])
    ax[i].set(title=f"{k} runs", xlabel="position (m)",
              ylabel="unit (sorted by peak)" if i == 0 else "")
    plt.colorbar(im, ax=ax[i], label="rate / peak rate")
ax[2].hist(stability_all, bins=25, color="0.35")
ax[2].axvline(0, color="r", ls="--")
ax[2].set(xlabel="odd-vs-even lap map correlation", ylabel="unit x direction",
          title=f"Within-session stability\n(median r = {np.nanmedian(stability_all):.2f})")
fig.suptitle(f"Place fields of the recorded population tile the track - {name}",
             fontsize=13)
fig.tight_layout()
fig.savefig("fig03_population_maps.png", dpi=130)
plt.show()


# %% [markdown]
# ## Spatial information against the circular-shift null
#
# The first panel compares each unit's observed SI against the 99th percentile of
# its own shuffle distribution. Note how much the per-unit threshold varies: a
# pooled threshold would be badly misleading, because sparse cells reach high SI
# by chance. The remaining panels summarize the fields that pass.

# %%
fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.5))

obs = stats["si"].values
thr = stats["si_thresh"].values
sig = stats["is_place"].values
ax[0, 0].scatter(thr[~sig], obs[~sig], s=12, color="0.7", label="not significant")
ax[0, 0].scatter(thr[sig], obs[sig], s=12, color="C0", label="place field")
lim = [0, np.nanmax([obs.max(), thr.max()]) * 1.05]
ax[0, 0].plot(lim, lim, "r--", lw=1)
ax[0, 0].set(xlim=lim, ylim=lim, xlabel="99th percentile of that unit's null (bits/spike)",
             ylabel="observed SI (bits/spike)",
             title="Spatial information vs per-unit null")
ax[0, 0].legend(fontsize=8, loc="lower right")

nul = np.concatenate([res["null"][k].ravel() for k in tc])
ax[0, 1].hist(nul, bins=60, density=True, alpha=0.6, color="0.6", label="shuffled")
ax[0, 1].hist(obs[np.isfinite(obs)], bins=30, density=True, alpha=0.7, color="C0",
              label="observed")
ax[0, 1].set(xlabel="spatial information (bits/spike)", ylabel="density",
             title="Pooled SI distributions")
ax[0, 1].legend(fontsize=8)

pc = stats[stats["is_place"]]
ax[0, 2].hist(pc["peak_rate"], bins=25, color="C0")
ax[0, 2].set(xlabel="peak in-field rate (Hz)", ylabel="place fields",
             title=f"Peak rates (median {pc['peak_rate'].median():.1f} Hz)")
ax[1, 0].hist(pc["width"], bins=25, color="C0")
ax[1, 0].set(xlabel="field width at half maximum (m)", ylabel="place fields",
             title=f"Field widths (median {pc['width'].median():.2f} m)")
ax[1, 1].hist(pc["peak_pos"], bins=20, color="C0")
ax[1, 1].set(xlabel="field peak position (m)", ylabel="place fields",
             title="Peak positions (track ends over-represented)")
ax[1, 2].scatter(stats.loc[~stats["is_place"], "si"],
                 stats.loc[~stats["is_place"], "stability"], s=12, color="0.7")
ax[1, 2].scatter(pc["si"], pc["stability"], s=12, color="C0")
ax[1, 2].set(xlabel="spatial information (bits/spike)",
             ylabel="odd-vs-even map correlation",
             title="Cells that pass the test are also stable")
fig.suptitle(f"Place-cell statistics - {name} "
             f"({stats['is_place'].mean():.0%} of unit x direction pairs pass)",
             fontsize=13)
fig.tight_layout()
fig.savefig("fig04_spatial_information.png", dpi=130)
plt.show()


# %% [markdown]
# ## Decoding position from the population
#
# A tuning-curve-based description is only convincing if the code can be read
# backwards. We use pynapple's Bayesian decoder with 250 ms bins under the usual
# Poisson independence assumption. Folds are split over laps: rate maps come from
# the training laps only and are used to decode the held-out laps, so no
# information about the test data enters the encoding model. Unsmoothed rate maps
# are used here, since smoothing the likelihood would leak information across
# bins.

# %%
DECODE_BIN = 0.25
NFOLD = 5


def decode_cv(s, units, nfold=NFOLD, bin_size=DECODE_BIN, nbins=NBINS):
    """Cross-validated Bayesian decoding, folds split over laps per direction."""
    out, folds = [], []
    for k, d in (("rightward", 1), ("leftward", -1)):
        laps = np.flatnonzero(s["lap_dir"] == d)
        fold_id = np.arange(len(laps)) % nfold
        for f in range(nfold):
            tr, te = laps[fold_id != f], laps[fold_id == f]
            if len(tr) < 2 or len(te) == 0:
                continue
            ep_tr = run_epochs(s["speed"], nap.IntervalSet(s["laps"].start[tr],
                                                           s["laps"].end[tr]))
            ep_te = run_epochs(s["speed"], nap.IntervalSet(s["laps"].start[te],
                                                           s["laps"].end[te]))
            tcx = tuning_curves(units, s["lin"], ep_tr, s["track_len"], nbins,
                                as_xarray=True, smooth=0.0)
            decoded, post = nap.decode_bayes(tcx, units, ep_te, bin_size)
            true = np.interp(decoded.times(), s["lin"].times(), s["lin"].values)
            out.append(np.column_stack([true, decoded.values,
                                        np.full(len(true), d), decoded.times()]))
            folds.append((k, f, ep_te, decoded, post))
    return np.concatenate(out), folds


dec, folds = decode_cv(s, units)
err = np.abs(dec[:, 0] - dec[:, 1])
rng = np.random.default_rng(0)
chance = np.abs(dec[:, 0] - rng.permutation(dec[:, 1]))
print("median |error| = %.3f m (chance %.3f m), %d time bins"
      % (np.median(err), np.median(chance), len(err)))

sizes = sorted({n for n in [2, 5, 10, 20, 40, 80, len(units)] if n <= len(units)})
curve = []
for n in tqdm(sizes, desc="population size"):
    e = []
    for rep in range(3):
        sub = units[list(rng.choice(list(units.keys()), size=n, replace=False))]
        r, _ = decode_cv(s, sub)
        e.append(np.median(np.abs(r[:, 0] - r[:, 1])))
    curve.append(np.mean(e))
print(dict(zip(sizes, np.round(curve, 3))))

# %%
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.28)

axp = fig.add_subplot(gs[0, :])
k, f, ep_te, decoded, post = folds[0]
cols, dd_all, tt_all, bounds = [], [], [], [0]
for i in range(min(8, len(ep_te))):
    pp, ddi = post.restrict(ep_te[i]), decoded.restrict(ep_te[i])
    if len(ddi) == 0:
        continue
    cols.append(np.asarray(pp))
    dd_all.append(ddi.values)
    tt_all.append(np.interp(ddi.times(), s["lin"].times(), s["lin"].values))
    bounds.append(bounds[-1] + len(ddi))
M = np.concatenate(cols)
dd_all, tt_all = np.concatenate(dd_all), np.concatenate(tt_all)
x = (np.arange(M.shape[0]) + 0.5) * DECODE_BIN
axp.imshow(M.T, aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
           extent=[0, M.shape[0] * DECODE_BIN, 0, s["track_len"]])
axp.plot(x, tt_all, "w.-", lw=1.5, ms=5, label="true position")
axp.plot(x, dd_all, "c.", ms=6, label="decoded (posterior max)")
for b in bounds[1:-1]:
    axp.axvline(b * DECODE_BIN, color="w", lw=1.5, ls=":")
axp.set(xlabel=f"time (s); {len(bounds) - 1} held-out laps concatenated, "
                "dotted lines mark lap boundaries",
        ylabel="position (m)", xlim=(0, M.shape[0] * DECODE_BIN),
        title=f"Posterior P(position | spikes) on held-out laps, "
              f"{DECODE_BIN * 1e3:.0f} ms bins")
axp.legend(loc="upper left", bbox_to_anchor=(1.002, 1.0), fontsize=9)

ax = fig.add_subplot(gs[1, 0])
edges = np.linspace(0, s["track_len"], NBINS + 1)
H, _, _ = np.histogram2d(dec[:, 0], dec[:, 1], bins=[edges, edges])
H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
im = ax.imshow(H.T, origin="lower", cmap="viridis", aspect="equal",
               extent=[0, s["track_len"], 0, s["track_len"]])
ax.plot([0, s["track_len"]], [0, s["track_len"]], "w--", lw=1)
ax.set(xlabel="true position (m)", ylabel="decoded position (m)",
       title="Confusion matrix")
plt.colorbar(im, ax=ax, label="P(decoded | true)", fraction=0.046)

ax = fig.add_subplot(gs[1, 1])
ax.hist(chance, bins=40, color="0.7", density=True, label="position shuffled")
ax.hist(err, bins=40, color="C0", density=True, alpha=0.8, label="decoded")
ax.axvline(np.median(err), color="C3", ls="--", label=f"median {np.median(err):.2f} m")
ax.set(xlabel="absolute decoding error (m)", ylabel="density",
       title="Error distribution")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 2])
ax.plot(sizes, curve, "o-", color="C0")
ax.axhline(np.median(chance), color="0.5", ls="--", label="chance")
ax.set(xscale="log", xlabel="number of CA1 cells (random subsets)",
       ylabel="median error (m)", title="Accuracy grows with population size")
ax.legend(fontsize=9)
fig.suptitle(f"Position is recoverable from CA1 population activity - {name}",
             fontsize=13)
fig.savefig("fig05_decoding.png", dpi=130, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## A GLM encoding model of position (NeMoS)
#
# The histogram-based rate map is a nonparametric estimate. Fitting a Poisson GLM
# whose only covariate is position, expanded in a 12-element B-spline basis, gives
# a parametric version of the same thing and, more usefully, a likelihood we can
# evaluate on held-out laps. We report McFadden's pseudo-$R^2$ against a
# homogeneous-Poisson null whose rate is the training-set mean, so a positive value
# means knowing where the animal is genuinely predicts spike counts in laps the
# model never saw.

# %%
GLM_BIN = 0.05
N_BASIS = 12


def glm_design(s, units, ep, basis, bin_size=GLM_BIN):
    counts = units.count(bin_size, ep=ep)
    pos = s["lin"].interpolate(counts, ep=ep)
    X = basis.compute_features(pos)
    ok = np.all(np.isfinite(np.asarray(X)), axis=1) & np.isfinite(np.asarray(pos))
    return np.asarray(X)[ok], np.asarray(counts)[ok]


def poisson_ll(y, mu):
    """Mean Poisson log-likelihood per bin (mu in counts/bin)."""
    mu = np.maximum(mu, 1e-10)
    return (y * np.log(mu) - mu - gammaln(y + 1)).mean(axis=0)


def fit_population_glm(X, y, n_basis):
    """Ridge-regularized Poisson population GLM with a safe intercept init.

    Units that are silent in a training fold would make the default
    log(mean rate) initialization non-finite, so the intercept is floored.
    """
    init = (np.zeros((X.shape[1], y.shape[1])),
            np.log(np.maximum(y.mean(axis=0), 1e-3)))
    model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-4,
                                  solver_name="LBFGS")
    return model.fit(X, y, init_params=init)


def fit_glm_cv(s, units, direction="rightward", nfold=NFOLD, n_basis=N_BASIS,
               bin_size=GLM_BIN, min_spikes=20):
    basis = nmo.basis.BSplineEval(n_basis_funcs=n_basis, label="position")
    laps = np.flatnonzero(s["lap_dir"] == (1 if direction == "rightward" else -1))
    fold_id = np.arange(len(laps)) % nfold
    ep_all = run_epochs(s["speed"], nap.IntervalSet(s["laps"].start[laps],
                                                    s["laps"].end[laps]))
    keys = [k for k in units.keys() if len(units[k].restrict(ep_all)) >= min_spikes]
    units = units[keys]

    ll_model = np.zeros((nfold, len(keys)))
    ll_null = np.zeros((nfold, len(keys)))
    for f in range(nfold):
        tr, te = laps[fold_id != f], laps[fold_id == f]
        ep_tr = run_epochs(s["speed"], nap.IntervalSet(s["laps"].start[tr],
                                                       s["laps"].end[tr]))
        ep_te = run_epochs(s["speed"], nap.IntervalSet(s["laps"].start[te],
                                                       s["laps"].end[te]))
        Xtr, ytr = glm_design(s, units, ep_tr, basis, bin_size)
        Xte, yte = glm_design(s, units, ep_te, basis, bin_size)
        model = fit_population_glm(Xtr, ytr, n_basis)
        ll_model[f] = poisson_ll(yte, np.asarray(model.predict(Xte)))
        ll_null[f] = poisson_ll(yte, np.maximum(ytr.mean(axis=0), 1e-6)[None, :])
    pr2 = 1.0 - ll_model.mean(0) / ll_null.mean(0)

    Xa, ya = glm_design(s, units, ep_all, basis, bin_size)
    model = fit_population_glm(Xa, ya, n_basis)
    grid = np.linspace(0, s["track_len"], 100)
    Xg = np.asarray(basis.compute_features(
        nap.Tsd(t=np.arange(len(grid)) * 1.0, d=grid)))
    return dict(pseudo_r2=pd.Series(pr2, index=keys), grid=grid, keys=keys,
                glm_tc=np.asarray(model.predict(Xg)) / bin_size, basis=basis)


glm = {k: fit_glm_cv(s, units, direction=k) for k in ("rightward", "leftward")}
for k, g in glm.items():
    print(k, f"{len(g['keys'])} units with >= 20 spikes;",
          "held-out pseudo-R2 percentiles [10, 50, 90]:",
          np.round(np.percentile(g["pseudo_r2"], [10, 50, 90]), 3))

# %%
fig, axes = plt.subplots(2, max(n_ex, 4), figsize=(3.75 * max(n_ex, 4), 7.5))
for j in range(n_ex, axes.shape[1]):
    axes[0, j].axis("off")
for j, (u, best_dir) in enumerate(chosen):
    ax = axes[0, j]
    g = glm[best_dir]
    ax.plot(tc[best_dir].index, tc[best_dir][u], color="0.4", lw=1.5,
            label="rate map")
    ax.plot(g["grid"], g["glm_tc"][:, g["keys"].index(u)], color="C3", lw=2,
            label="GLM")
    ax.set(title=f"unit {u} ({best_dir})", xlabel="position (m)",
           xlim=(0, s["track_len"]), ylim=(0, None))
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(fontsize=8)

ax = axes[1, 0]
_, bfun = glm["rightward"]["basis"].evaluate_on_grid(200)
ax.plot(np.linspace(0, s["track_len"], 200), bfun, lw=1)
ax.set(xlabel="position (m)", ylabel="basis value",
       title=f"{N_BASIS} B-spline bases over position")

pr2 = pd.concat({k: glm[k]["pseudo_r2"] for k in glm}, names=["direction", "unit"])
is_place = stats["is_place"].reindex(pr2.index)
ax = axes[1, 1]
ax.hist([pr2[is_place].values, pr2[~is_place].values], bins=25, stacked=True,
        color=["C0", "0.75"], label=["place field", "not significant"])
ax.axvline(0, color="r", ls="--")
ax.set(xlabel="held-out pseudo-$R^2$", ylabel="unit x direction",
       title="Position predicts held-out spikes")
ax.legend(fontsize=8)

ax = axes[1, 2]
ax.scatter(stats["si"].reindex(pr2.index), pr2, s=12,
           c=np.where(is_place, "C0", "0.75"))
ax.set(xlabel="spatial information (bits/spike)", ylabel="held-out pseudo-$R^2$",
       title="GLM fit tracks spatial information")

ax = axes[1, 3]
emp = np.concatenate([tc[k][u].values for k in glm for u in glm[k]["keys"]])
mod = np.concatenate([np.interp(tc[k].index.values, glm[k]["grid"],
                                glm[k]["glm_tc"][:, glm[k]["keys"].index(u)])
                      for k in glm for u in glm[k]["keys"]])
ax.scatter(emp, mod, s=3, alpha=0.2, color="C0")
lim = [0, max(emp.max(), mod.max()) * 1.02]
ax.plot(lim, lim, "r--", lw=1)
ax.set(xlabel="rate map (Hz)", ylabel="GLM prediction (Hz)", xlim=lim, ylim=lim,
       title=f"r = {np.corrcoef(emp, mod)[0, 1]:.3f}")
fig.suptitle(f"Poisson GLM encoding model of position - {name}", fontsize=13)
fig.tight_layout()
fig.savefig("fig06_glm.png", dpi=130)
plt.show()


# %% [markdown]
# ## The same analysis on every linear-track session
#
# Everything above runs unchanged on the other four linear-track sessions, which
# come from four different rats. The three circular-maze sessions are skipped, as
# noted at the top.

# %%
linear = {p: u for p, u in assets.items() if "Circular" not in maze_type(u)}
print(f"{len(linear)} linear-track sessions of {len(assets)}")

summaries, all_stats, all_tc = [], [], {}
for path, url in tqdm(sorted(linear.items()), desc="sessions"):
    t = time.time()
    ss = load_session(url)
    uu = track_units(ss)
    rr = classify_place_cells(ss, uu)
    dd, _ = decode_cv(ss, uu)
    e = np.abs(dd[:, 0] - dd[:, 1])
    ch = np.abs(dd[:, 0] - np.random.default_rng(0).permutation(dd[:, 1]))
    st = pd.concat(rr["stats"], names=["direction", "unit"]).reset_index()
    st["session"], st["subject"] = ss["session"], ss["subject"]
    summaries.append(dict(
        session=ss["session"], subject=ss["subject"], maze=ss["maze_name"],
        track_len=ss["track_len"], n_units_total=len(ss["units"]),
        n_units_used=len(uu), n_laps=len(ss["laps"]),
        run_time=sum(v.tot_length() for v in rr["run"].values()),
        n_place=int(st["is_place"].sum()), n_pairs=len(st),
        frac_place=float(st["is_place"].mean()),
        frac_place_strict=float(st["is_place_strict"].mean()),
        frac_place_any=float(st.groupby("unit")["is_place"].any().mean()),
        median_err=float(np.median(e)), chance_err=float(np.median(ch)),
        seconds=time.time() - t))
    all_stats.append(st)
    all_tc[ss["session"]] = rr["tc"]
    ss["io"].close()
    print(summaries[-1]["session"],
          "%d/%d place fields, decode %.3f m (chance %.3f)"
          % (summaries[-1]["n_place"], summaries[-1]["n_pairs"],
             summaries[-1]["median_err"], summaries[-1]["chance_err"]))

summary = pd.DataFrame(summaries)
pooled = pd.concat(all_stats, ignore_index=True)
summary.to_csv("session_summary.csv", index=False)
pooled.to_csv("all_unit_stats.csv", index=False)
summary[["session", "subject", "track_len", "n_units_used", "n_laps",
         "frac_place", "frac_place_any", "median_err", "chance_err"]]

# %%
fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.5))
sess = summary["session"].values
xpos = np.arange(len(sess))
short = [x.replace("_", "\n") for x in sess]

ax[0, 0].bar(xpos - 0.2, summary["frac_place"], 0.4, color="C0",
             label="per direction")
ax[0, 0].bar(xpos + 0.2, summary["frac_place_any"], 0.4, color="C1",
             label="either direction")
ax[0, 0].set_xticks(xpos, short, fontsize=7)
ax[0, 0].set(ylabel="fraction of pyramidal cells", ylim=(0, 1),
             title="Place-cell yield per session")
ax[0, 0].legend(fontsize=8)

for i, (sn, gdf) in enumerate(pooled.groupby("session")):
    ax[0, 1].hist(gdf.loc[gdf["is_place"], "si"], bins=np.linspace(0, 3.5, 30),
                  histtype="step", lw=1.6, label=sn)
ax[0, 1].set(xlabel="spatial information (bits/spike)", ylabel="place fields",
             title="SI of significant fields")
ax[0, 1].legend(fontsize=6)

pcp = pooled[pooled["is_place"]]
ax[0, 2].hist(pcp["width"], bins=25, color="C0")
ax[0, 2].set(xlabel="field width at half maximum (m)", ylabel="place fields",
             title=f"Pooled field widths (median {pcp['width'].median():.2f} m, "
                   f"n = {len(pcp)})")

ax[1, 0].hist(pcp["peak_rate"], bins=np.linspace(0, 40, 30), color="C0")
ax[1, 0].set(xlabel="peak in-field rate (Hz)", ylabel="place fields",
             title=f"Pooled peak rates (median {pcp['peak_rate'].median():.1f} Hz)")

for sn, gdf in pooled.groupby("session"):
    tl = summary.loc[summary["session"] == sn, "track_len"].iloc[0]
    h, e = np.histogram(gdf.loc[gdf["is_place"], "peak_pos"] / tl,
                        bins=np.linspace(0, 1, 21))
    ax[1, 1].step(e[:-1] + np.diff(e) / 2, h / h.sum(), where="mid", lw=1.5, label=sn)
ax[1, 1].set(xlabel="field peak, fraction of track", ylabel="fraction of fields",
             title="Coverage of the track", ylim=(0, None))
ax[1, 1].legend(fontsize=6)

ax[1, 2].bar(xpos - 0.2, summary["median_err"], 0.4, color="C0", label="decoded")
ax[1, 2].bar(xpos + 0.2, summary["chance_err"], 0.4, color="0.7", label="chance")
ax[1, 2].set_xticks(xpos, short, fontsize=7)
ax[1, 2].set(ylabel="median |decoding error| (m)",
             title="Cross-validated decoding, all sessions")
ax[1, 2].legend(fontsize=8)
fig.suptitle("Place coding in five linear-track sessions from four rats "
             "(DANDI:000044)", fontsize=13)
fig.tight_layout()
fig.savefig("fig07_multisession.png", dpi=130)
plt.show()

# %%
fig, axes = plt.subplots(2, len(sess), figsize=(3.0 * len(sess), 6.4))
for j, sn in enumerate(sess):
    for i, k in enumerate(["rightward", "leftward"]):
        m = np.nan_to_num(all_tc[sn][k].values.T)
        norm = m / np.maximum(m.max(axis=1, keepdims=True), 1e-9)
        tl = summary.loc[summary["session"] == sn, "track_len"].iloc[0]
        axes[i, j].imshow(norm[np.argsort(np.argmax(m, axis=1))], aspect="auto",
                          origin="lower", cmap="viridis",
                          extent=[0, tl, 0, m.shape[0]])
        axes[i, j].set(xlabel="position (m)" if i == 1 else "",
                       ylabel=f"{k}\nunit (sorted)" if j == 0 else "",
                       title=sn if i == 0 else "")
        axes[i, j].title.set_size(9)
fig.suptitle("Peak-normalized rate maps, every linear-track session", fontsize=13)
fig.tight_layout()
fig.savefig("fig08_all_session_maps.png", dpi=130)
plt.show()


# %% [markdown]
# ## Summary
#
# Across five linear-track sessions from four rats in DANDI:000044:
#
# * 60-72% of the putative pyramidal cells active on the track have a
#   significant, direction-specific place field (SI above the 99th percentile of
#   that cell's own circular-shift null, peak rate at least 1 Hz), and 73-90% have
#   one in at least one running direction.
# * The fields are compact and stable: median width at half maximum around 0.3 m
#   on tracks of 1.6-2 m, median peak rate around 5 Hz, and a median correlation
#   near 0.85 between rate maps built from odd and from even laps.
# * Peak positions cover the whole track, with the usual over-representation of
#   the two ends where the animal pauses and drinks.
# * The population code is decodable: Bayesian decoding of held-out laps recovers
#   position to a median of 7-12 cm, against a shuffled baseline of 44-58 cm, and
#   the error falls monotonically as more cells are included.
# * A Poisson GLM with position as its only covariate reproduces the measured rate
#   maps (r = 0.99 between the two) and gives clearly positive held-out
#   pseudo-$R^2$ for the cells classified as place cells.
#
# Two caveats worth stating. First, spike sorting and the pyramidal/interneuron
# labels are taken as given from the dataset; no re-curation was done, so some
# units may be merges or splits. Second, the over-representation of the track ends
# means positional coverage is not uniform, and the reward-zone fields there could
# partly reflect reward or immobility rather than position alone. Neither affects
# the main claim, which is that individual CA1 pyramidal cells fire in restricted,
# reproducible portions of the environment and that the population jointly encodes
# the animal's location.

# %%
print(summary.to_string(index=False))
