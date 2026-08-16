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
# # Head-direction cells in the mouse postsubiculum (DANDI:000939)
#
# A head-direction cell fires when the animal's head points in a particular
# direction in the horizontal plane, independently of where the animal is or what
# it is doing. This notebook demonstrates the phenomenon end to end on real data
# streamed from the DANDI Archive:
#
# 1. load head direction, position and spike times from one session and check the
#    raw streams,
# 2. compute directional tuning curves and decide which cells are genuinely
#    direction-modulated using a circular-shift null distribution,
# 3. describe the population code (preferred directions, pairwise tuning
#    similarity),
# 4. show that the tuning survives a change of environment, rotating coherently
#    across the population,
# 5. decode head direction from the population on held-out data,
# 6. fit a Poisson GLM (NeMoS) with a cyclic B-spline basis and compare it with a
#    spike-history model,
# 7. repeat the whole pipeline over all 31 sessions of the dandiset.
#
# ## Dataset
#
# **DANDI:000939**, "Large-scale recordings of head direction cells in mouse
# postsubiculum" (Peyrache Lab, McGill University; Duszkiewicz et al., *Nature
# Neuroscience* 2024, doi:10.1038/s41593-024-01588-5). Silicon-probe recordings
# from the postsubiculum of freely moving mice foraging in an open field, with
# head direction tracked at 100 Hz from head-mounted LEDs. 31 sessions, 2691
# sorted units. The files are 18-30 GB each because they contain the raw
# broadband traces; we stream only the spike times and the behavioural series, so
# a session costs a few tens of megabytes of transfer.
#
# Nothing here is simulated. Every number and every figure comes from these
# recordings.

# %% [markdown]
# ## Setup

# %%
import h5py
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from tqdm.auto import tqdm

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

DANDISET = "000939"
CACHE_DIR = "/tmp/remfile_cache_000939"
API = "https://api.dandiarchive.org/api/dandisets"

NB_BINS = 120        # head-direction bins for tuning curves
P_THRESH = 0.01      # circular-shift test
STAB_THRESH = 0.5    # split-half tuning-curve correlation
DECODE_BIN = 0.2     # s

# %% [markdown]
# ### Streaming access
#
# DANDI redirects an asset download to a presigned S3 URL. The bucket is public,
# so we strip the signature and hand the plain object URL to `remfile`, which
# serves HDF5 range requests from a local disk cache.

# %%
def list_assets(dandiset=DANDISET, version="draft"):
    """[(path, asset_id, size_bytes), ...] for every NWB asset in the dandiset."""
    r = requests.get(f"{API}/{dandiset}/versions/{version}/assets/",
                     params={"page_size": 200, "order": "path"})
    r.raise_for_status()
    return [(a["path"], a["asset_id"], a["size"]) for a in r.json()["results"]
            if a["path"].endswith(".nwb")]


def open_session(asset_id, dandiset=DANDISET, version="draft"):
    """Stream one NWB file; returns (pynapple NWBFile, pynwb NWBFile, io)."""
    url = f"{API}/{dandiset}/versions/{version}/assets/{asset_id}/download/"
    s3_url = requests.head(url, allow_redirects=False).headers["Location"].split("?")[0]
    rem_file = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rem_file, "r"), load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


def get_epochs(nwbfile):
    """{tag: IntervalSet} for the labelled behavioural epochs."""
    df = nwbfile.intervals["epochs"].to_dataframe()
    return {row["tags"][0]: nap.IntervalSet(start=row["start_time"],
                                            end=row["stop_time"])
            for _, row in df.iterrows()}


def clean_head_direction(hd, ep):
    """Restrict head direction to `ep`, drop NaN samples, wrap to [0, 2*pi)."""
    hd = hd.restrict(ep)
    good = ~np.isnan(hd.values)
    return nap.Tsd(t=hd.index.values[good], d=np.mod(hd.values[good], 2 * np.pi),
                   time_support=ep)


# %% [markdown]
# ### Analysis helpers
#
# Tuning curves come from `pynapple.compute_tuning_curves` (spike counts per
# direction bin divided by occupancy). Two summary numbers describe a curve: the
# **mean vector length** (MVL), which is 0 for a flat curve and 1 for a curve
# concentrated in a single bin, and the **preferred direction**, the angle of that
# mean vector. Directional information follows Skaggs et al. (1993).

# %%
def tuning_curves(units, hd, ep, nb_bins=NB_BINS):
    tc = nap.compute_tuning_curves(units, hd, bins=nb_bins, range=[(0, 2 * np.pi)],
                                   epochs=ep, return_pandas=True)
    tc.index = tc.index.astype(float)
    return tc


def mvl_and_pref(tc):
    theta = tc.index.values
    r = tc.values
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (r * np.exp(1j * theta)[:, None]).sum(0) / np.nansum(r, axis=0)
    return np.abs(z), np.mod(np.angle(z), 2 * np.pi)


def occupancy_hist(hd, nb_bins=NB_BINS):
    edges = np.linspace(0, 2 * np.pi, nb_bins + 1)
    dt = np.median(np.diff(hd.index.values))
    counts, _ = np.histogram(hd.values, bins=edges)
    return counts * dt, edges


def hd_information(tc, occupancy):
    """Directional information in bits/spike (Skaggs et al., 1993)."""
    p = occupancy / occupancy.sum()
    r = tc.values
    rbar = (p[:, None] * r).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        term = p[:, None] * (r / rbar) * np.log2(r / rbar)
    return np.nansum(np.where(np.isfinite(term), term, 0.0), axis=0)


def circ_diff(a, b):
    """Signed angular difference wrapped to (-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def rotation_coherence(a, b):
    """(R, offset): how rigidly preferred directions rotate between conditions.

    R = 1 means every cell shifted by exactly the same angle, R = 0 that the
    shifts are unrelated.
    """
    z = np.exp(1j * circ_diff(b, a)).mean()
    return np.abs(z), np.mod(np.angle(z), 2 * np.pi)


# %% [markdown]
# ### The significance test
#
# A cell counts as direction-modulated if its MVL is larger than expected when the
# pairing between spikes and head direction is destroyed. The standard control is
# to shift the spike train circularly in time, which preserves both the
# autocorrelation of the spike train and the statistics of the behaviour.
#
# The MVL at shift $k$ can be written as a ratio of two circular
# cross-correlations between the binned spike train and per-sample weights derived
# from head direction, so all $T$ possible shifts follow from one FFT per unit
# instead of a loop over shuffles. Concretely, with $c_t$ the spike count in time
# bin $t$, $b(t)$ the direction bin the animal occupied and $o_b$ the occupancy,
#
# $$\mathrm{MVL}(k)=\frac{\left|\sum_t c_{t+k}\, e^{i\theta_{b(t)}}/o_{b(t)}\right|}
#                        {\sum_t c_{t+k}/o_{b(t)}}.$$
#
# Shifts smaller than 20 s are excluded so that the null is not contaminated by
# the slow autocorrelation of head direction. The zero-shift value of this
# expression was checked against a direct recomputation of the tuning curve.

# %%
def shift_null_mvl(units, hd, ep, nb_bins=NB_BINS, n_draws=1000, min_shift_s=20.0,
                   seed=0):
    """Returns (mvl_observed, mvl_null) with mvl_null of shape (n_draws, n_units)."""
    dt = float(np.median(np.diff(hd.index.values)))
    t_edges = np.concatenate([hd.index.values - dt / 2,
                              [hd.index.values[-1] + dt / 2]])
    counts = np.stack([np.histogram(units[u].restrict(ep).index.values, bins=t_edges)[0]
                       for u in units.index], axis=1).astype(float)  # (T, N)

    edges = np.linspace(0, 2 * np.pi, nb_bins + 1)
    bin_idx = np.clip(np.digitize(hd.values, edges) - 1, 0, nb_bins - 1)
    occ = np.bincount(bin_idx, minlength=nb_bins) * dt
    theta = (edges[:-1] + edges[1:]) / 2

    v = np.exp(1j * theta)[bin_idx] / occ[bin_idx]
    u = 1.0 / occ[bin_idx]

    T, n_units = counts.shape
    lo = int(min_shift_s / dt)
    draws = np.random.default_rng(seed).integers(lo, T - lo, size=n_draws)
    Vr, Vi, U = (np.conj(np.fft.rfft(x)) for x in (v.real, v.imag, u))

    mvl_obs = np.empty(n_units)
    mvl_null = np.empty((n_draws, n_units))
    for a in range(0, n_units, 16):   # chunked to bound memory
        b = min(a + 16, n_units)
        F = np.fft.rfft(counts[:, a:b], axis=0)
        num = (np.fft.irfft(F * Vr[:, None], n=T, axis=0)
               + 1j * np.fft.irfft(F * Vi[:, None], n=T, axis=0))
        den = np.fft.irfft(F * U[:, None], n=T, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mvl_all = np.abs(num) / den
        mvl_obs[a:b] = mvl_all[0]
        mvl_null[:, a:b] = mvl_all[draws]
    return mvl_obs, mvl_null


def session_stats(units, hd, ep, nb_bins=NB_BINS, n_draws=1000, seed=0):
    """Per-unit directional statistics for one epoch."""
    tc = tuning_curves(units, hd, ep, nb_bins=nb_bins)
    occ, _ = occupancy_hist(hd, nb_bins=nb_bins)
    mvl, pref = mvl_and_pref(tc)
    info = hd_information(tc, occ)
    mvl_obs, mvl_null = shift_null_mvl(units, hd, ep, nb_bins=nb_bins,
                                       n_draws=n_draws, seed=seed)
    pval = (mvl_null >= mvl_obs[None, :]).mean(0)

    mid = ep.start[0] + (ep.end[-1] - ep.start[0]) / 2
    ep1 = nap.IntervalSet(start=ep.start[0], end=mid)
    ep2 = nap.IntervalSet(start=mid, end=ep.end[-1])
    tc1 = tuning_curves(units, hd.restrict(ep1), ep1, nb_bins=nb_bins)
    tc2 = tuning_curves(units, hd.restrict(ep2), ep2, nb_bins=nb_bins)
    stab = np.array([pd.Series(tc1[c]).corr(pd.Series(tc2[c])) for c in tc.columns])

    stats = pd.DataFrame({
        "mvl": mvl_obs,          # same binning as the null, used for the test
        "mvl_tc": mvl,           # from the pynapple tuning curve, for display
        "pref_dir": pref,
        "peak_rate": tc.max(0).values,
        "mean_rate": np.array([len(units[u].restrict(ep)) / ep.tot_length()
                               for u in units.index]),
        "hd_info": info,
        "p_shift": pval,
        "stability": stab,
    }, index=tc.columns)
    return tc, stats, (tc1, tc2)


# %% [markdown]
# ## 1. What is in the dandiset

# %%
assets = list_assets()
print(f"{len(assets)} NWB files, "
      f"{sum(a[2] for a in assets)/1e12:.2f} TB in total")
for path, _, size in assets[:5]:
    print(f"  {path}   {size/1e9:.1f} GB")

SESSION = "A3707"   # the session used for the walk-through
path, aid, size = [a for a in assets if SESSION in a[0]][0]
nwb, nwbfile, io = open_session(aid)
print()
print(nwb)

# %%
print("subject      :", nwbfile.subject.subject_id, "|", nwbfile.subject.description)
print("publication  :", nwbfile.related_publications)
print("units        :", len(nwb["units"]), "on", nwbfile.units.colnames)
print("head direction:", nwbfile.processing["behavior"]["CompassDirection"]
      ["head-direction"].unit, "|",
      nwbfile.processing["behavior"]["CompassDirection"]["head-direction"].reference_frame)
epochs = get_epochs(nwbfile)
for tag, e in epochs.items():
    print(f"epoch {tag:15s} {float(e.tot_length()):8.1f} s")

# %% [markdown]
# The recording alternates home-cage rest with foraging in an open arena. We use
# `wake_square` (foraging in a square arena) for the main analysis and
# `wake_triangle` later as an independent environment. The units table carries
# the authors' own classification (`is_head_direction`, `is_excitatory`,
# `is_fast_spiking`), which we deliberately do not use for our own analysis but
# do compare against at the end.

# %%
units = nwb["units"]
ep = epochs["wake_square"]
hd = clean_head_direction(nwb["head-direction"], ep)
pos = nwb["position"].restrict(ep)
meta = units.metadata

print(f"{len(units)} units | {int(meta.is_head_direction.sum())} flagged HD by the "
      f"authors | {int(meta.is_excitatory.sum())} excitatory, "
      f"{int(meta.is_fast_spiking.sum())} fast-spiking")
print(f"head direction: {len(hd)} samples at {hd.rate:.1f} Hz over "
      f"{float(ep.tot_length()):.0f} s")

# %% [markdown]
# ## 2. Raw data check
#
# Before computing anything, look at the streams: the head-direction trace, the
# trajectory, how much time the animal spends facing each direction, and where in
# the trace individual cells put their spikes.

# %%
tc = tuning_curves(units, hd, ep)
mvl, pref = mvl_and_pref(tc)
rate = np.array([len(units[u].restrict(ep)) / ep.tot_length() for u in units.index])

# three tuned cells with clearly different preferred directions, plus an untuned one
ok = np.where((mvl > 0.5) & (rate > 2))[0]
examples = []
for target in np.radians([30, 150, 270]):
    cand = [c for c in ok[np.argsort(np.abs(circ_diff(pref[ok], target)))]
            if c not in examples]
    examples.append(cand[0])
untuned = np.where((mvl < 0.05) & (rate > 2))[0]
examples.append(untuned[np.argmax(rate[untuned])])
ids = [units.index[i] for i in examples]

# the 60 s window that samples the widest range of directions
cands = np.arange(ep.start[0], ep.end[-1] - 60, 20.0)
spread = []
for c in cands:
    v = hd.restrict(nap.IntervalSet(start=c, end=c + 60)).values
    p = np.histogram(v, bins=np.linspace(0, 2 * np.pi, 25))[0].astype(float)
    p /= max(p.sum(), 1)
    spread.append(-np.nansum(np.where(p > 0, p * np.log(p), 0)))
t0 = cands[int(np.argmax(spread))]
win = nap.IntervalSet(start=t0, end=t0 + 60)
h = hd.restrict(win)

fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.45, height_ratios=[1, 1.15, 1.1])

ax = fig.add_subplot(gs[0, :3])
ax.plot(h.index.values - t0, np.degrees(h.values), ".", ms=1.5, color="k")
ax.set(xlabel="time (s)", ylabel="head direction (deg)", ylim=(0, 360),
       title="Head direction during open-field foraging (60 s excerpt)")

ax = fig.add_subplot(gs[0, 3])
p = pos.restrict(win)
ax.plot(pos["x"].values, pos["y"].values, lw=0.3, color="0.8")
ax.plot(p["x"].values, p["y"].values, lw=1.2, color="crimson")
ax.set(xlabel="x (cm)", ylabel="y (cm)", title="Trajectory\n(red = same 60 s)")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[1, :3])
trace = np.degrees(h.values).copy()
trace[np.abs(np.diff(trace, prepend=trace[0])) > 180] = np.nan   # hide 0/360 wraps
ax.plot(h.index.values - t0, trace, "-", lw=0.8, color="0.75", zorder=1)
colors = ["#1b6ca8", "#c0392b", "#2e8b57"]
unwrapped = np.unwrap(h.values)
for uid, c in zip(ids[:3], colors):
    s = units[uid].restrict(win).index.values
    sd = np.degrees(np.mod(np.interp(s, h.index.values, unwrapped), 2 * np.pi))
    ax.plot(s - t0, sd, ".", ms=4, color=c, zorder=2, label=f"unit {uid}")
ax.set(xlabel="time (s)", ylabel="head direction (deg)", ylim=(0, 360),
       title="Each tuned cell fires only when the head points in its own direction")
ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.9)

occ, edges = occupancy_hist(hd)
centres = (edges[:-1] + edges[1:]) / 2
ax = fig.add_subplot(gs[1, 3], projection="polar")
ax.bar(centres, occ, width=np.diff(edges), color="0.5")
ax.set_title("Occupancy\n(s per direction)", pad=22, fontsize=10)
ax.set_yticklabels([])
ax.tick_params(labelsize=7)

labels = ["tuned", "tuned", "tuned", "untuned"]
for k, (i, uid) in enumerate(zip(examples, ids)):
    ax = fig.add_subplot(gs[2, k], projection="polar")
    col = colors[k] if k < 3 else "0.4"
    th = np.append(tc.index.values, tc.index.values[0])
    r = np.append(tc[uid].values, tc[uid].values[0])
    ax.plot(th, r, color=col)
    ax.fill(th, r, alpha=0.25, color=col)
    ax.set_title(f"unit {uid} ({labels[k]})\nMVL={mvl[i]:.2f}, peak {r.max():.1f} Hz",
                 pad=22, fontsize=9)
    ax.set_rticks(np.round(np.linspace(0, r.max(), 3)[1:], 0))
    ax.tick_params(labelsize=7)

fig.suptitle(f"DANDI:000939  {path.split('/')[-1]}  postsubiculum", y=0.97)
fig.savefig("fig01_raw_data_validation.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ![raw data](fig01_raw_data_validation.png)
#
# The three coloured cells fire in different, narrow bands of head direction, and
# each band matches the peak of the corresponding polar tuning curve underneath.
# The fourth cell, a high-rate unit, fires everywhere. The animal samples all
# directions but not uniformly, which is why tuning curves must be normalised by
# occupancy.

# %% [markdown]
# ## 3. Which cells are direction-modulated?
#
# The circular-shift test asks whether the observed MVL is larger than the MVL of
# the same spike train paired with head direction at other time lags. We require
# `p < 0.01` and a split-half tuning-curve correlation above 0.5, so a cell must
# be both unlikely under the null and reproducible within the session.

# %%
tc, stats, (tc1, tc2) = session_stats(units, hd, ep, n_draws=1000)
stats["author_hd"] = meta["is_head_direction"].values.astype(bool)
stats["is_fs"] = meta["is_fast_spiking"].values.astype(bool)
stats["is_exc"] = meta["is_excitatory"].values.astype(bool)
stats["hd_cell"] = (stats.p_shift < P_THRESH) & (stats.stability > STAB_THRESH)
stats.to_csv("stats_example_session.csv")

n_hd = int(stats.hd_cell.sum())
print(f"{len(stats)} units, {n_hd} direction-modulated "
      f"({100*n_hd/len(stats):.0f}%); the authors flag {int(stats.author_hd.sum())}")
print(stats.groupby("hd_cell")[["mvl", "hd_info", "stability", "mean_rate"]].median())

# %%
mvl_obs, mvl_null = shift_null_mvl(units, hd, ep, n_draws=1000)
example = stats.mvl.idxmax()
ex_i = list(stats.index).index(example)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

ax = axes[0, 0]
ax.hist(mvl_null[:, ex_i], bins=40, color="0.7", label="circular-shift null")
ax.axvline(mvl_obs[ex_i], color="crimson", lw=2, label="observed")
ax.set(xlabel="mean vector length", ylabel="shifts",
       title=f"Shift test, unit {example}\n(p < {1/len(mvl_null):.3f})")
ax.legend(fontsize=8)

ax = axes[0, 1]
bins = np.linspace(0, 1, 41)
ax.hist(stats.mvl, bins=bins, color="#1b6ca8", alpha=0.85, label="observed cells")
ax.hist(mvl_null.ravel(), bins=bins,
        weights=np.full(mvl_null.size, len(stats) / mvl_null.size),
        color="0.6", alpha=0.6, label="null (all shifts pooled)")
ax.set(xlabel="mean vector length", ylabel="units",
       title="Directional tuning strength vs chance")
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.scatter(stats.mvl[~stats.hd_cell], stats.stability[~stats.hd_cell], s=18,
           color="0.6", label="not classified")
ax.scatter(stats.mvl[stats.hd_cell], stats.stability[stats.hd_cell], s=18,
           color="crimson", label="HD cell")
ax.axhline(STAB_THRESH, ls="--", color="k", lw=0.8)
ax.set(xlabel="mean vector length", ylabel="split-half tuning correlation",
       title="Classification criteria")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1, 0]
ax.scatter(stats.mvl, stats.hd_info, s=18, c=np.where(stats.hd_cell, "crimson", "0.6"))
ax.set(xlabel="mean vector length", ylabel="directional information (bits/spike)",
       title="Tuning strength and information")

ax = axes[1, 1]
kind = np.where(stats.is_fs, "fast-spiking",
                np.where(stats.is_exc, "excitatory", "unclassified"))
for i, lbl in enumerate(["excitatory", "fast-spiking", "unclassified"]):
    m = kind == lbl
    ax.scatter(np.random.default_rng(i).normal(i, 0.07, m.sum()), stats.mvl[m], s=16,
               c=np.where(stats.hd_cell[m], "crimson", "0.6"))
ax.set(xticks=[0, 1, 2], xticklabels=["excitatory", "fast-spiking", "unclassified"],
       ylabel="mean vector length", title="Tuning by cell type (red = HD cell here)")

ax = axes[1, 2]
exc = stats[stats.is_exc]
cm = pd.crosstab(exc.hd_cell, exc.author_hd).reindex(
    index=[False, True], columns=[False, True]).fillna(0).values
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13,
                color="k" if cm[i, j] < cm.max() / 2 else "w")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"],
       yticklabels=["no", "yes"], xlabel="author label: HD cell",
       ylabel="this analysis: HD cell",
       title=f"Excitatory cells only\nagreement {100*(cm[0,0]+cm[1,1])/cm.sum():.0f}%")

fig.suptitle(f"Identifying head-direction cells  ({path.split('/')[-1]})", y=1.0)
fig.tight_layout()
fig.savefig("fig02_hd_cell_classification.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ![classification](fig02_hd_cell_classification.png)
#
# The observed MVL distribution is strongly bimodal, with a mode near zero that
# overlaps the null and a second mode above 0.8 far outside it. Directional
# information rises with MVL up to about 2.7 bits/spike. Restricted to excitatory
# cells, our classification recovers the authors' labels with 94% agreement and
# without a single false positive; the cells we miss sit at the threshold. The
# extra cells we do call direction-modulated are fast-spiking or unclassified
# units with weak but reliable tuning, which the authors' HD label does not cover.
# This is consistent with the paper's own point that fast-spiking interneurons in
# the postsubiculum carry a directional signal too, but a much broader one.

# %% [markdown]
# ## 4. The population code
#
# Sorting the tuning curves by preferred direction shows that the population
# covers the whole circle, and that the similarity between two cells' tuning
# curves falls off monotonically with the angle between their preferred
# directions, crossing zero near 70-80 degrees.

# %%
hd_ids = stats.index[stats.hd_cell]
pref_s = stats.pref_dir[hd_ids].values
order = hd_ids[np.argsort(pref_s)]
norm = tc[order] / tc[order].max(0)

fig = plt.figure(figsize=(14, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1.2], wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(norm.values.T, aspect="auto", origin="lower", cmap="magma",
               extent=[0, 360, 0, len(order)])
ax.set(xlabel="head direction (deg)",
       ylabel="HD cell (sorted by preferred direction)",
       title=f"Tuning curves of all {len(order)} HD cells")
fig.colorbar(im, ax=ax, label="normalised rate")

ax = fig.add_subplot(gs[0, 1], projection="polar")
ax.hist(pref_s, bins=np.linspace(0, 2 * np.pi, 25), color="#1b6ca8")
ax.set_title("Preferred directions", pad=20, fontsize=10)
ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[0, 2])
tch = tc[hd_ids].values
sd = tch.std(0)
keep = sd > 0
tch = (tch[:, keep] - tch[:, keep].mean(0)) / sd[keep]
corr = (tch.T @ tch) / tch.shape[0]
pref_k = pref_s[keep]
d = np.abs(circ_diff(pref_k[:, None], pref_k[None, :]))
iu = np.triu_indices(len(pref_k), 1)
bins = np.linspace(0, np.pi, 19)
bi = np.digitize(d[iu], bins) - 1
m = np.array([corr[iu][bi == k].mean() for k in range(len(bins) - 1)])
se = np.array([corr[iu][bi == k].std() / max(np.sqrt((bi == k).sum()), 1)
               for k in range(len(bins) - 1)])
ax.scatter(np.degrees(d[iu]), corr[iu], s=3, color="0.8")
ax.errorbar(np.degrees((bins[:-1] + bins[1:]) / 2), m, yerr=se, color="crimson", lw=2)
ax.axhline(0, ls="--", color="k", lw=0.8)
ax.set(xlabel="difference in preferred direction (deg)",
       ylabel="tuning-curve correlation", title="Pairwise tuning similarity")

fig.suptitle("Population structure of the head-direction code", y=1.03)
fig.savefig("fig03_population_structure.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ![population](fig03_population_structure.png)

# %% [markdown]
# ## 5. The same cells in a different arena
#
# A head-direction cell is not a place cell or a view cell: its tuning should
# follow the animal's internal compass, which is anchored to the room and to the
# available cues. The session includes a second foraging epoch in a triangular
# arena. We compare preferred directions between the two, and summarise the
# comparison with the coherence $R$ of the per-cell rotations: $R = 1$ means every
# cell shifted by the same angle, that is, the whole map rotated rigidly.

# %%
ep2 = epochs["wake_triangle"]
hd2 = clean_head_direction(nwb["head-direction"], ep2)
tc_t = tuning_curves(units, hd2, ep2)
mvl_t, pref_t = mvl_and_pref(tc_t)
mvl_t = pd.Series(mvl_t, index=tc_t.columns)
pref_t = pd.Series(pref_t, index=tc_t.columns)

# a preferred direction is only meaningful for cells tuned in both arenas
both_ids = [i for i in hd_ids if mvl_t[i] > 0.3 and stats.mvl[i] > 0.3]
delta = circ_diff(pref_t[both_ids].values, stats.pref_dir[both_ids].values)
R, offset = rotation_coherence(stats.pref_dir[both_ids].values, pref_t[both_ids].values)
resid = np.degrees(np.abs(circ_diff(delta, offset)))
print(f"{len(both_ids)} of {len(hd_ids)} HD cells tuned (MVL > 0.3) in both arenas")
print(f"common rotation {np.degrees(offset):.0f} deg, coherence R = {R:.2f}, "
      f"median residual shift {np.median(resid):.1f} deg")

fig = plt.figure(figsize=(14, 4.4))
gs = fig.add_gridspec(1, 4, wspace=0.45)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(np.degrees(stats.pref_dir[both_ids]), np.degrees(pref_t[both_ids]), s=16,
           color="#1b6ca8")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set(xlabel="preferred direction, square (deg)",
       ylabel="preferred direction, triangle (deg)",
       title=f"Preferred directions rotate together\ncoherence R = {R:.2f} "
             f"(n = {len(both_ids)})")

ax = fig.add_subplot(gs[0, 1])
ax.hist(np.degrees(circ_diff(delta, offset)), bins=np.arange(-180, 185, 10), color="0.5")
ax.axvline(0, color="crimson", lw=1)
ax.set(xlabel="shift between environments (deg)", ylabel="HD cells",
       title=f"Shift after removing the\ncommon rotation ({np.degrees(offset):.0f} deg)")

ax = fig.add_subplot(gs[0, 2])
ax.scatter(stats.mvl[both_ids], mvl_t[both_ids], s=16, color="#1b6ca8")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="MVL, square", ylabel="MVL, triangle", title="Tuning strength")

ax = fig.add_subplot(gs[0, 3], projection="polar")
exu = (stats.mvl[both_ids] + mvl_t[both_ids]).idxmax()
for curve, lbl, c in [(tc, "square", "#1b6ca8"), (tc_t, "triangle", "#c0392b")]:
    th = np.append(curve.index.values, curve.index.values[0])
    r = np.append(curve[exu].values, curve[exu].values[0])
    ax.plot(th, r, color=c, label=lbl)
ax.set_title(f"unit {exu}", pad=20, fontsize=10)
ax.set_rticks(np.round(np.linspace(0, tc[exu].max(), 3)[1:], 0))
ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.15))
ax.tick_params(labelsize=7)

fig.suptitle("The same cells keep their tuning in a different arena", y=1.05)
fig.savefig("fig04_cross_environment.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ![cross environment](fig04_cross_environment.png)
#
# Every cell's preferred direction shifts by roughly the same angle between the
# two arenas, so the internal compass rotated as a whole rather than the cells
# re-tuning independently. Tuning strength is unchanged.

# %% [markdown]
# ## 6. Decoding head direction from the population
#
# If these cells carry the animal's heading, a naive Bayesian decoder built from
# their tuning curves should recover it. Tuning curves are estimated on the first
# half of the foraging epoch, and cells are selected using only that half; the
# second half is untouched test data.

# %%
mid = ep.start[0] + ep.tot_length() / 2
ep_train = nap.IntervalSet(start=ep.start[0], end=mid)
ep_test = nap.IntervalSet(start=mid, end=ep.end[-1])
hd_train, hd_test = hd.restrict(ep_train), hd.restrict(ep_test)

_, stats_tr, _ = session_stats(units, hd_train, ep_train, nb_bins=60, n_draws=400)
train_hd_ids = list(stats_tr.index[(stats_tr.p_shift < P_THRESH)
                                   & (stats_tr.stability > 0)])
other_ids = [i for i in stats_tr.index if i not in set(train_hd_ids)]
print(f"{len(train_hd_ids)} cells selected on the training half")


def decode(unit_ids):
    tcx = nap.compute_tuning_curves(units[list(unit_ids)], hd_train, bins=60,
                                    range=[(0, 2 * np.pi)], epochs=ep_train)
    dec, post = nap.decode_bayes(tcx, units[list(unit_ids)], ep_test, DECODE_BIN)
    true = hd_test.bin_average(DECODE_BIN, ep_test)
    ok = ~np.isnan(true.values)
    return dec, post, true, ok, np.abs(circ_diff(dec.values[ok], true.values[ok]))


dec, post, true, ok, err = decode(train_hd_ids)
*_, err_other = decode(other_ids)
err_shuf = np.abs(circ_diff(dec.values[ok],
                            np.random.default_rng(0).permutation(true.values[ok])))
print(f"median error: HD cells {np.degrees(np.median(err)):.1f} deg, "
      f"other cells {np.degrees(np.median(err_other)):.1f} deg, "
      f"time-shuffled {np.degrees(np.median(err_shuf)):.1f} deg")

sizes = sorted({s for s in [1, 2, 4, 8, 16, 32, 64, len(train_hd_ids)]
                if s <= len(train_hd_ids)})
curve = []
for n in tqdm(sizes, desc="decoding vs population size"):
    reps = [np.degrees(np.median(decode(
        np.random.default_rng(r).choice(train_hd_ids, n, replace=False))[4]))
        for r in range(8 if n < len(train_hd_ids) else 1)]
    curve.append((np.mean(reps), np.std(reps)))
curve = np.array(curve)

# %%
fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

ax = fig.add_subplot(gs[0, :])
w = nap.IntervalSet(start=ep_test.start[0] + 200, end=ep_test.start[0] + 320)
pw = post.restrict(w)
ax.imshow(pw.values.T, aspect="auto", origin="lower", cmap="Greys",
          extent=[0, w.tot_length(), 0, 360])
t = hd_test.restrict(w)
ax.plot(t.index.values - w.start[0], np.degrees(t.values), ".", ms=2, color="crimson",
        label="true head direction")
ax.set(xlabel="time (s)", ylabel="head direction (deg)",
       title="Bayesian decoding from the HD-cell population (posterior in grey)")
ax.legend(fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[1, 0])
bins = np.arange(0, 181, 5)
for e, lbl, c in [(err, "HD cells", "crimson"), (err_other, "other cells", "#1b6ca8"),
                  (err_shuf, "shuffled time", "0.6")]:
    ax.hist(np.degrees(e), bins=bins, histtype="step", lw=2, density=True, color=c,
            label=f"{lbl} (median {np.degrees(np.median(e)):.0f}$\\degree$)")
ax.set(xlabel="absolute decoding error (deg)", ylabel="density",
       title=f"Decoding error, {DECODE_BIN*1000:.0f} ms bins")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.errorbar(sizes, curve[:, 0], yerr=curve[:, 1], marker="o", color="crimson")
ax.axhline(90, ls="--", color="0.5", lw=1)
ax.text(sizes[-1], 92, "chance", ha="right", fontsize=8, color="0.4")
ax.set(xscale="log", xlabel="number of HD cells", ylabel="median error (deg)",
       title="Decoding accuracy vs population size")

ax = fig.add_subplot(gs[1, 2])
ax.scatter(np.degrees(true.values[ok]), np.degrees(dec.values[ok]), s=2, alpha=0.3,
           color="k")
ax.plot([0, 360], [0, 360], "--", color="crimson", lw=1)
ax.set(xlabel="true head direction (deg)", ylabel="decoded (deg)",
       title="Decoded vs true")

fig.suptitle("Head direction can be read out from the population", y=0.97)
fig.savefig("fig05_decoding.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ![decoding](fig05_decoding.png)
#
# The posterior follows the animal's heading closely. The median error over the
# held-out half is about 12 degrees in 200 ms bins, against 87 degrees for the same
# decoder applied to time-shuffled data (chance is 90 degrees). Sixteen cells
# already get within about 16 degrees; beyond about 30 cells the curve flattens.
# The cells that failed our criterion still decode better than chance, which is
# expected: the criterion is a significance threshold, not a statement that the
# remaining cells carry no directional signal at all.

# %% [markdown]
# ## 7. A GLM description of the tuning (NeMoS)
#
# Tuning curves are descriptive. A Poisson GLM with a cyclic B-spline basis over
# head direction fits the same relationship as a smooth parametric model and lets
# us ask a question the tuning curve cannot answer: how much of the spiking is
# explained by head direction rather than by the cell's own recent spiking? We fit
# three models per cell (head direction, spike history, both) on the first half of
# the epoch and score McFadden's pseudo-$R^2$ on the second half.

# %%
GLM_BIN = 0.02
counts = units.count(GLM_BIN, ep=ep)
angle = hd.interpolate(counts, ep=ep)
t_counts = counts.t

basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=10, label="head_direction")
X_hd = np.asarray(basis.compute_features(angle))
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5,
                                           window_size=int(0.25 / GLM_BIN),
                                           label="spike_history")
is_train = t_counts < mid
valid_hd = ~np.isnan(angle.values) & ~np.isnan(X_hd).any(1)

grid = np.linspace(0, 2 * np.pi, 180, endpoint=False)
X_grid = np.asarray(basis.compute_features(
    nap.Tsd(t=np.arange(len(grid)) * 1.0, d=grid)))


def fit_score(X, y, mask):
    m = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-5)
    m.fit(X[mask & is_train], y[mask & is_train])
    return m, float(m.score(X[mask & ~is_train], y[mask & ~is_train],
                            score_type="pseudo-r2-McFadden"))


rows, glm_tuning = [], {}
for uid in tqdm(units.index, desc="GLM per unit"):
    cu = counts.loc[uid]
    y = np.asarray(cu).astype(float)
    X_hist = np.asarray(hist_basis.compute_features(cu))
    valid = valid_hd & ~np.isnan(X_hist).any(1)
    m_hd, r2_hd = fit_score(X_hd, y, valid)
    _, r2_hist = fit_score(X_hist, y, valid)
    _, r2_both = fit_score(np.concatenate([X_hd, X_hist], 1), y, valid)
    glm_tuning[uid] = np.asarray(m_hd.predict(X_grid)) / GLM_BIN
    rows.append(dict(unit=uid, r2_hd=r2_hd, r2_hist=r2_hist, r2_both=r2_both))

glm = pd.DataFrame(rows).set_index("unit").join(
    stats[["mvl", "pref_dir", "hd_cell", "mean_rate", "is_fs"]])
glm.to_csv("glm_example_session.csv")
print(glm.groupby("hd_cell")[["r2_hd", "r2_hist", "r2_both"]].median())

# %%
tc60 = tuning_curves(units, hd, ep, nb_bins=60)
ex = []
best = glm.loc[list(hd_ids)].sort_values("r2_hd", ascending=False)
for target in np.radians([60, 180, 300]):
    cand = [i for i in best.index[:40] if i not in ex]
    ex.append(min(cand, key=lambda i: abs(circ_diff(glm.pref_dir[i], target))))
ex.append(glm[~glm.hd_cell].sort_values("mean_rate").index[-1])

fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)

for k, uid in enumerate(ex):
    ax = fig.add_subplot(gs[0, k])
    ax.plot(np.degrees(tc60.index.values), tc60[uid].values, color="0.6", lw=1,
            label="empirical")
    ax.plot(np.degrees(grid), glm_tuning[uid], color="crimson", lw=2, label="GLM")
    ax.set(xlabel="head direction (deg)", ylabel="rate (Hz)" if k == 0 else "",
           xticks=[0, 180, 360],
           title=f"unit {uid}  {'HD' if glm.hd_cell[uid] else 'non-HD'}\n"
                 f"pseudo-$R^2$ = {glm.r2_hd[uid]:.3f}")
    if k == 0:
        ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(glm.mvl, glm.r2_hd, s=16, c=np.where(glm.hd_cell, "crimson", "0.6"))
ax.set(xlabel="mean vector length", ylabel="held-out pseudo-$R^2$",
       title="Direction explains spiking")

lim = [-0.2, 0.55]
n_off = int(((glm.r2_hist < lim[0]) | (glm.r2_hd < lim[0])).sum())
ax = fig.add_subplot(gs[1, 1])
ax.scatter(glm.r2_hist, glm.r2_hd, s=16, c=np.where(glm.hd_cell, "crimson", "0.6"))
ax.plot(lim, lim, "k--", lw=0.8)
ax.set(xlabel="pseudo-$R^2$, history only", ylabel="pseudo-$R^2$, direction only",
       xlim=lim, ylim=lim, title=f"Direction beats spike history\n({n_off} off scale)")

ax = fig.add_subplot(gs[1, 2])
ax.scatter(glm.r2_hd, glm.r2_both, s=16, c=np.where(glm.hd_cell, "crimson", "0.6"))
ax.plot(lim, lim, "k--", lw=0.8)
ax.set(xlabel="pseudo-$R^2$, direction", ylabel="pseudo-$R^2$, direction + history",
       xlim=lim, ylim=lim, title="Adding history adds a little")

ax = fig.add_subplot(gs[1, 3])
width = np.array([(glm_tuning[u] >= glm_tuning[u].max() / 2).mean() * 360
                  for u in hd_ids])
ax.hist(width, bins=20, color="#1b6ca8")
ax.set(xlabel="width at half maximum (deg)", ylabel="HD cells",
       title=f"Tuning width\n(median {np.median(width):.0f}$\\degree$)")

fig.suptitle("Poisson GLM with a cyclic B-spline basis over head direction (NeMoS)",
             y=0.98)
fig.savefig("fig06_glm.png", dpi=150)
plt.close(fig)
io.close()

# %% [markdown]
# ![glm](fig06_glm.png)
#
# The GLM reproduces the empirical tuning curves and gives a held-out
# pseudo-$R^2$ that grows with MVL, reaching about 0.47. For the direction-modulated
# cells the head-direction model beats a spike-history model of the same cell, and
# adding history on top of direction improves the fit only modestly, so the
# directional signal is not a by-product of burst structure. The fitted curves put
# the median tuning width at half maximum near 40 degrees.

# %% [markdown]
# ## 8. All 31 sessions
#
# The walk-through above is one session. The same pipeline now runs over the whole
# dandiset: classification, held-out decoding, and, where the session contains a
# second arena, the cross-environment rotation. This takes a few minutes.

# %%
unit_rows, session_rows = [], []

for spath, said, ssize in tqdm(assets, desc="sessions"):
    subject = spath.split("/")[0].replace("sub-", "")
    s_nwb, s_nwbfile, s_io = open_session(said)
    s_eps = get_epochs(s_nwbfile)
    s_units = s_nwb["units"]
    s_meta = s_units.metadata

    s_ep = s_eps["wake_square"]
    s_hd = clean_head_direction(s_nwb["head-direction"], s_ep)
    _, s_stats, _ = session_stats(s_units, s_hd, s_ep, n_draws=500)
    s_stats["hd_cell"] = (s_stats.p_shift < P_THRESH) & (s_stats.stability > STAB_THRESH)
    s_stats["author_hd"] = s_meta["is_head_direction"].values.astype(bool)
    s_stats["is_fs"] = s_meta["is_fast_spiking"].values.astype(bool)
    s_stats["is_exc"] = s_meta["is_excitatory"].values.astype(bool)
    s_stats["session"] = subject
    unit_rows.append(s_stats.reset_index().rename(columns={"index": "unit"}))

    s_mid = s_ep.start[0] + s_ep.tot_length() / 2
    ep_tr = nap.IntervalSet(start=s_ep.start[0], end=s_mid)
    ep_te = nap.IntervalSet(start=s_mid, end=s_ep.end[-1])
    hd_tr, hd_te = s_hd.restrict(ep_tr), s_hd.restrict(ep_te)
    _, st_tr, _ = session_stats(s_units, hd_tr, ep_tr, nb_bins=60, n_draws=300)
    sel = list(st_tr.index[(st_tr.p_shift < P_THRESH) & (st_tr.stability > 0)])

    med_err = np.nan
    if len(sel) >= 2:
        tcx = nap.compute_tuning_curves(s_units[sel], hd_tr, bins=60,
                                        range=[(0, 2 * np.pi)], epochs=ep_tr)
        s_dec, _ = nap.decode_bayes(tcx, s_units[sel], ep_te, DECODE_BIN)
        s_true = hd_te.bin_average(DECODE_BIN, ep_te)
        m_ok = ~np.isnan(s_true.values)
        med_err = float(np.degrees(np.median(
            np.abs(circ_diff(s_dec.values[m_ok], s_true.values[m_ok])))))

    s_R = s_offset = s_resid = np.nan
    n_both = 0
    if "wake_triangle" in s_eps:      # the opto sessions have no second arena
        s_ep2 = s_eps["wake_triangle"]
        s_hd2 = clean_head_direction(s_nwb["head-direction"], s_ep2)
        s_tc2 = tuning_curves(s_units, s_hd2, s_ep2)
        s_mvl2, s_pref2 = mvl_and_pref(s_tc2)
        s_mvl2 = pd.Series(s_mvl2, index=s_tc2.columns)
        s_pref2 = pd.Series(s_pref2, index=s_tc2.columns)
        both = [i for i in s_stats.index[s_stats.hd_cell]
                if s_mvl2[i] > 0.3 and s_stats.mvl[i] > 0.3]
        n_both = len(both)
        if n_both >= 5:
            s_R, s_offset = rotation_coherence(s_stats.pref_dir[both].values,
                                               s_pref2[both].values)
            dd = circ_diff(s_pref2[both].values, s_stats.pref_dir[both].values)
            s_resid = float(np.degrees(np.median(np.abs(circ_diff(dd, s_offset)))))

    s_exc = s_stats[s_stats.is_exc]
    session_rows.append(dict(
        session=subject, file=spath.split("/")[-1],
        duration_s=float(s_ep.tot_length()), n_units=len(s_stats),
        n_hd=int(s_stats.hd_cell.sum()), frac_hd=float(s_stats.hd_cell.mean()),
        n_author_hd=int(s_stats.author_hd.sum()),
        agree_exc=float((s_exc.hd_cell == s_exc.author_hd).mean()) if len(s_exc) else np.nan,
        n_exc=int(s_stats.is_exc.sum()), n_fs=int(s_stats.is_fs.sum()),
        n_fs_hd=int((s_stats.is_fs & s_stats.hd_cell).sum()),
        median_mvl_hd=float(s_stats.mvl[s_stats.hd_cell].median()),
        decode_err_deg=med_err, n_decode_cells=len(sel),
        rot_coherence=s_R, rot_resid_deg=s_resid, n_cross_env=n_both,
        rot_offset_deg=float(np.degrees(s_offset)) if np.isfinite(s_offset) else np.nan,
    ))
    s_io.close()

units_df = pd.concat(unit_rows, ignore_index=True)
sess = pd.DataFrame(session_rows)
units_df.to_csv("all_units.csv", index=False)
sess.to_csv("session_summary.csv", index=False)
print(sess[["session", "n_units", "n_hd", "frac_hd", "agree_exc", "decode_err_deg",
            "rot_coherence"]].to_string(index=False))

# %%
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
o = sess.sort_values("frac_hd")
y = np.arange(len(o))
ax.barh(y, 100 * o.frac_hd, color="crimson", height=0.75, label="direction-modulated")
ax.plot(100 * o.n_author_hd / o.n_units, y, "o", ms=3.5, color="k",
        label="author HD label")
ax.set(yticks=y, yticklabels=o.session, ylabel="session",
       xlabel="% of recorded units", title="Head-direction cells per session")
ax.tick_params(axis="y", labelsize=6)
ax.legend(fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, 1, 41)
ax.hist(units_df.mvl[~units_df.hd_cell], bins=bins, color="0.6", label="not significant")
ax.hist(units_df.mvl[units_df.hd_cell], bins=bins, color="crimson", alpha=0.85,
        label="direction-modulated")
ax.axvline(0.3, ls="--", color="k", lw=0.8)
ax.text(0.31, ax.get_ylim()[1] * 0.55, "MVL = 0.3", fontsize=7)
ax.set(xlabel="mean vector length", ylabel="units",
       title=f"All {len(units_df)} units, {len(sess)} sessions")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2], projection="polar")
sel_u = units_df[units_df.hd_cell & (units_df.mvl > 0.3)]
ax.hist(sel_u.pref_dir, bins=np.linspace(0, 2 * np.pi, 37), color="#1b6ca8")
ax.set_title(f"Preferred directions\n({len(sel_u)} cells, all sessions)", pad=20,
             fontsize=10)
ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(sess.n_decode_cells, sess.decode_err_deg, s=26, color="crimson")
ax.axhline(90, ls="--", color="0.5", lw=1)
ax.text(sess.n_decode_cells.max(), 86, "chance", ha="right", va="top", fontsize=8,
        color="0.4")
ax.set(xlabel="cells used for decoding", ylabel="median decoding error (deg)",
       ylim=(0, 100),
       title=f"Held-out decoding\n(median {sess.decode_err_deg.median():.1f}$\\degree$)")

ax = fig.add_subplot(gs[1, 1])
cs = sess.dropna(subset=["rot_coherence"])
ax.scatter(cs.rot_coherence, cs.rot_resid_deg, s=26, color="#1b6ca8")
ax.set(xlabel="rotation coherence R", ylabel="median residual shift (deg)",
       xlim=(0.5, 1.02), ylim=(0, None),
       title=f"Square vs triangle arena\n({len(cs)} sessions with both)")

ax = fig.add_subplot(gs[1, 2])
ax.hist(100 * sess.agree_exc, bins=np.arange(80, 102, 2), color="0.5")
ax.axvline(100 * sess.agree_exc.median(), color="crimson", lw=2)
ax.set(xlabel="agreement with author labels (%)", ylabel="sessions",
       title=f"Median agreement {100*sess.agree_exc.median():.0f}%\n"
             "(excitatory cells only)")

fig.suptitle("Head-direction coding across all 31 sessions of DANDI:000939", y=0.97)
fig.savefig("fig07_cross_session.png", dpi=150)
plt.close(fig)

print(f"units: {len(units_df)}, direction-modulated: {int(units_df.hd_cell.sum())} "
      f"({100*units_df.hd_cell.mean():.0f}%), of which MVL > 0.3: "
      f"{int((units_df.hd_cell & (units_df.mvl > 0.3)).sum())}")
print(f"decoding error across sessions: median {sess.decode_err_deg.median():.1f} deg "
      f"(range {sess.decode_err_deg.min():.1f}-{sess.decode_err_deg.max():.1f})")
print(f"rotation coherence: median {cs.rot_coherence.median():.2f}, "
      f"median residual {cs.rot_resid_deg.median():.1f} deg")

# %% [markdown]
# ![cross session](fig07_cross_session.png)

# %% [markdown]
# ## Summary
#
# Across all 31 sessions of DANDI:000939, 1989 of 2691 postsubicular units
# (74%) are significantly direction-modulated by the circular-shift test, and 1493
# of them have a mean vector length above 0.3, the range usually described as a
# head-direction cell. The proportion is high because the test asks whether tuning
# is present at all, not whether it is strong: with 30 to 45 minutes of foraging,
# even a broadly tuned interneuron reaches significance. Restricted to excitatory
# cells, our classification agrees with the labels published with the dataset in
# 94% of cases (median across sessions).
#
# The tuning is not an artefact of the animal's trajectory or of the cells' burst
# structure. It survives circular shifts of the spike train, it reproduces across
# halves of a session, it holds up in a GLM that competes head direction against
# the cell's own spike history, and it transfers to a different arena, where the
# whole population rotates by a common angle (median coherence R = 0.98, median
# residual shift 4.5 degrees). Preferred directions tile the circle, and a
# Bayesian decoder trained on one half of a session recovers head direction in the
# other half to a median error of 9.4 degrees across sessions, against 90 degrees
# at chance.
#
# ### Caveats
#
# - Head direction is measured from head-mounted LEDs, so it is the direction of
#   the headstage, not of the gaze; small tracking errors broaden tuning curves
#   and set a floor on the decoding error.
# - Our classification uses a significance threshold plus a stability threshold.
#   Changing either moves the counts; the MVL distribution in figure 7 is the more
#   informative summary.
# - Angular head velocity is not controlled for. Postsubicular cells can be
#   modulated by turning speed as well as direction, and part of the residual
#   decoding error is likely systematic lag rather than noise.
# - The cross-environment comparison uses the square and triangle epochs of the
#   same session. Sessions with optogenetic manipulation contribute only their
#   pre-stimulation foraging epoch.
