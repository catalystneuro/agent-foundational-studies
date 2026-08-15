# %% [markdown]
# # Head-Direction Cells in the Mouse Anterior Thalamus
#
# This notebook demonstrates **head-direction (HD) cells** using extracellular
# recordings from the mouse anterodorsal thalamic nucleus (ADn) and post-subiculum
# (PoS), streamed directly from the DANDI Archive.
#
# **Dataset:** [DANDI:000056](https://dandiarchive.org/dandiset/000056) —
# *"Internally organized mechanisms of the head direction sense"* (Peyrache, Lacroix,
# Petersen & Buzsáki, 2015). Freely moving mice foraged in an open arena while
# neurons were recorded from anterior thalamus and post-subiculum, brain regions
# rich in head-direction cells.
#
# **What is a head-direction cell?** An HD cell fires selectively when the animal's
# head points in a particular *allocentric* direction, regardless of the animal's
# location in the environment. Each cell has a "preferred direction" and its firing
# rate falls off smoothly as the head turns away from it, giving a bell-shaped
# *tuning curve* on the circle. Together the HD population forms the brain's internal
# compass.
#
# **Approach.** Head direction is not stored directly in these files; it is recovered
# from two head-mounted LEDs (red and blue) tracked on video. We compute the heading
# as the angle of the vector between the two LEDs, restrict to the open-field foraging
# epoch, and then:
#
# 1. Build directional **tuning curves** for every unit.
# 2. Classify HD cells with a **circular shuffle test** (occupancy-corrected mean
#    vector length against a null distribution).
# 3. Show **polar tuning curves**, preferred-direction and tuning-strength summaries.
# 4. Test **within-session stability** (first vs. second half of the session).
# 5. **Decode** head direction from the HD population and compare to the animal's
#    actual heading (Bayesian population decoding).
# 6. **Scale** the HD-cell classification across several sessions and mice.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # silence pynapple deprecation / sort notices for readability
np.random.seed(0)

DANDISET = "000056"
VERSION = "draft"
CACHE_DIR = "/tmp/remfile_cache"
FIGDIR = "."

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})


# %% [markdown]
# ## Data access helpers
#
# We stream NWB files from S3 with `remfile` plus an on-disk cache, so only the
# byte ranges we actually read (spike times, LED tracking, epoch tables) are
# fetched. The raw broadband ecephys in these multi-GB files is never downloaded.

# %%
def list_assets():
    """Return {filename: asset_id} for the dandiset."""
    url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/"
    r = requests.get(url, params={"page_size": 100}, timeout=120).json()
    return {a["path"].split("/")[-1]: a["asset_id"] for a in r["results"]}


def asset_url(asset_id):
    return (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/"
        f"{VERSION}/assets/{asset_id}/download/"
    )


def open_nwb(asset_id):
    """Open a streamed NWB file; returns (pynwb NWBFile, pynapple NWBFile)."""
    rem = remfile.File(asset_url(asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    nwbfile = NWBHDF5IO(file=h5).read()
    return nwbfile, nap.NWBFile(nwbfile)


# %% [markdown]
# ## Recover head direction from the two head LEDs
#
# The two LEDs are mounted a fixed distance apart on the animal's head, so the
# vector from one to the other points along the head's facing direction. Frames
# where either LED is untracked are flagged with coordinate value `-1` and are
# dropped. We also pick the **foraging epoch**: the recordings interleave open-field
# exploration with sleep, and only the awake exploration epoch has good directional
# sampling. We take the longest "Awake" block as the foraging session.

# %%
def head_direction_tsd(nwbfile):
    """Compute allocentric head direction (radians, [0, 2*pi)) from red & blue LEDs.

    Returns a pynapple Tsd sampled at the video rate, with NaN where tracking is lost.
    """
    beh = nwbfile.processing["behavior"].data_interfaces["SubjectPosition"]
    blue = beh.spatial_series["BlueLED"]
    red = beh.spatial_series["RedLED"]

    B = blue.data[:]
    R = red.data[:]
    if blue.timestamps is not None:
        t = blue.timestamps[:]
    else:
        t = blue.starting_time + np.arange(B.shape[0]) / blue.rate

    valid = (B[:, 0] > -1) & (B[:, 1] > -1) & (R[:, 0] > -1) & (R[:, 1] > -1)
    ang = np.arctan2(R[:, 1] - B[:, 1], R[:, 0] - B[:, 0]) % (2 * np.pi)
    ang = np.where(valid, ang, np.nan)
    return nap.Tsd(t=t, d=ang)


def foraging_epoch(nwbfile):
    """Longest 'Awake' block = the open-field foraging session."""
    st = nwbfile.processing["behavior"].data_interfaces["states"].to_dataframe()
    aw = st[st["label"] == "Awake"].copy()
    aw["dur"] = aw["stop_time"] - aw["start_time"]
    top = aw.sort_values("dur", ascending=False).iloc[0]
    return nap.IntervalSet(start=top["start_time"], end=top["stop_time"])


# %% [markdown]
# ## Load the primary session
#
# We prototype on `Mouse12-120806`, which has 75 sorted units.

# %%
assets = list_assets()
PRIMARY = "sub-Mouse12_ses-Mouse12-120806_behavior+ecephys.nwb"
nwbfile, nwb = open_nwb(assets[PRIMARY])

spikes = nwb["units"]                       # TsGroup of spike times
hd = head_direction_tsd(nwbfile)            # head direction, NaN where untracked
epoch = foraging_epoch(nwbfile)             # foraging IntervalSet
hd_forage = hd.restrict(epoch).dropna()     # clean head direction during foraging

print(f"Session: {nwbfile.session_id}")
print(f"Units: {len(spikes)}")
print(f"Foraging epoch: {float(epoch.start[0]):.0f}-{float(epoch.end[0]):.0f} s "
      f"({epoch.tot_length():.0f} s)")
print(f"Head-direction samples during foraging: {len(hd_forage)}")

# %% [markdown]
# ## Inspect the raw behavioral data
#
# Before any analysis we verify that the recovered head direction is sensible: the
# animal should sample the full arena and (at least most of) all directions, and the
# heading should evolve smoothly over time.

# %%
red_ss = nwbfile.processing["behavior"].data_interfaces["SubjectPosition"].spatial_series["RedLED"]
red_t = (red_ss.timestamps[:] if red_ss.timestamps is not None
         else red_ss.starting_time + np.arange(red_ss.data.shape[0]) / red_ss.rate)
red_d = red_ss.data[:]
# keep frames inside the foraging epoch with valid tracking
in_ep = (red_t >= float(epoch.start[0])) & (red_t <= float(epoch.end[0]))
valid_xy = in_ep & (red_d[:, 0] > -1) & (red_d[:, 1] > -1)
rx, ry, rt = red_d[valid_xy, 0], red_d[valid_xy, 1], red_t[valid_xy]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) trajectory colored by head direction (nearest HD sample per frame)
sub = slice(None, None, 5)
_idx = np.clip(np.searchsorted(hd_forage.index.values, rt[sub]),
               0, len(hd_forage) - 1)
sc = axes[0].scatter(
    rx[sub], ry[sub], c=hd_forage.values[_idx],
    cmap="hsv", s=2, vmin=0, vmax=2 * np.pi,
)
axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Foraging path\n(color = head direction)")
axes[0].set_aspect("equal")
cb = fig.colorbar(sc, ax=axes[0], fraction=0.046)
cb.set_label("head direction (rad)")

# (b) head direction over a 60 s window
w = nap.IntervalSet(start=epoch.start[0] + 200, end=epoch.start[0] + 260)
hw = hd.restrict(w)
axes[1].plot(hw.index.values, hw.values, ".", ms=2, color="k")
axes[1].set(xlabel="time (s)", ylabel="head direction (rad)",
            title="Head direction over 60 s", ylim=(0, 2 * np.pi))

# (c) occupancy: time spent facing each direction
axes[2].hist(hd_forage.values, bins=36, range=(0, 2 * np.pi), color="slateblue")
axes[2].set(xlabel="head direction (rad)", ylabel="samples",
            title="Directional occupancy")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig1_behavior_overview.png", bbox_inches="tight")
plt.close()
print("saved fig1_behavior_overview.png")


# %% [markdown]
# ## Directional tuning curves and HD-cell classification
#
# For each unit we build an **occupancy-corrected** tuning curve: spikes are binned by
# the animal's head direction at spike time and divided by the time spent facing each
# direction. Tuning strength is summarized by the **mean vector length (MVL)** of the
# tuning curve, the length of the resultant vector when each direction bin is weighted
# by its firing rate. MVL is 0 for a flat (non-directional) cell and approaches 1 for a
# perfectly concentrated cell.
#
# To decide which cells are *significantly* directional we run a **circular shuffle
# test**: we rigidly time-shift each spike train relative to the head-direction signal
# many times and recompute MVL, building a null distribution that preserves each cell's
# firing statistics but destroys its true relationship to heading. A cell is called an
# HD cell if its real MVL exceeds the 99th percentile of its own null and clears
# modest rate/strength floors.

# %%
N_BINS = 60
BIN_EDGES = np.linspace(0, 2 * np.pi, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def mean_vector_length(rates):
    rates = np.asarray(rates, float)
    s = rates.sum()
    if s == 0:
        return 0.0, 0.0
    v = np.sum(rates * np.exp(1j * BIN_CENTERS)) / s
    return np.abs(v), np.angle(v) % (2 * np.pi)


def fast_tuning(spike_times, hd_t, hd_v, occ):
    """Occupancy-corrected tuning curve from spike times (nearest-sample HD lookup)."""
    if len(spike_times) == 0:
        return np.zeros(N_BINS)
    idx = np.searchsorted(hd_t, spike_times)
    idx = np.clip(idx, 0, len(hd_v) - 1)
    hd_at_spike = hd_v[idx]
    counts, _ = np.histogram(hd_at_spike, bins=BIN_EDGES)
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = np.where(occ > 0, counts / occ, 0.0)
    return tc


def classify_hd_cells(spikes, hd_forage, epoch, n_shuffle=200,
                      mvl_floor=0.3, rate_floor=1.0, pct=99):
    """Return a DataFrame of per-unit HD metrics + boolean HD-cell label."""
    hd_t = hd_forage.index.values
    hd_v = hd_forage.values
    dt = np.median(np.diff(hd_t))
    occ, _ = np.histogram(hd_v, bins=BIN_EDGES)
    occ = occ * dt  # seconds per direction bin

    t0 = float(epoch.start[0])
    dur = float(epoch.end[0]) - t0

    rows = []
    tuning = {}
    for u in tqdm(spikes.keys(), desc="shuffle test"):
        st = spikes[u].restrict(epoch).index.values
        tc = fast_tuning(st, hd_t, hd_v, occ)
        tuning[u] = tc
        mvl, pref = mean_vector_length(tc)
        peak = tc.max()

        # circular time-shift null within the foraging epoch
        null = np.empty(n_shuffle)
        rel = st - t0
        shifts = np.random.uniform(dur * 0.05, dur * 0.95, n_shuffle)
        for i, sh in enumerate(shifts):
            sh_st = t0 + np.mod(rel + sh, dur)
            null[i] = mean_vector_length(fast_tuning(sh_st, hd_t, hd_v, occ))[0]
        thr = np.percentile(null, pct)
        p = float(np.mean(null >= mvl))
        rows.append(dict(unit=u, n_spikes=len(st), mean_rate=len(st) / dur,
                         mvl=mvl, pref_dir=pref, peak_rate=peak,
                         mvl_null99=thr, p_value=p,
                         is_hd=(mvl > thr) and (mvl > mvl_floor) and (peak > rate_floor)))
    df = pd.DataFrame(rows).set_index("unit")
    return df, tuning, occ, dt


metrics, tuning, occ, dt = classify_hd_cells(spikes, hd_forage, epoch)
n_hd = int(metrics["is_hd"].sum())
print(f"\nHead-direction cells: {n_hd} / {len(metrics)} units "
      f"({100 * n_hd / len(metrics):.0f}%)")
print(metrics.sort_values("mvl", ascending=False).head(12)[
    ["n_spikes", "mean_rate", "peak_rate", "mvl", "p_value", "is_hd"]].round(3).to_string())

# %% [markdown]
# ## Polar tuning curves of example HD cells
#
# The signature of an HD cell is a single, sharp lobe in polar coordinates. Below are
# the twelve most strongly tuned cells; note how each points to a different preferred
# direction, tiling the circle.

# %%
hd_units = metrics[metrics["is_hd"]].sort_values("mvl", ascending=False)
show = hd_units.head(12).index.tolist()

fig, axes = plt.subplots(3, 4, figsize=(14, 11), subplot_kw={"projection": "polar"})
for ax, u in zip(axes.ravel(), show):
    tc = tuning[u]
    theta = np.concatenate([BIN_CENTERS, BIN_CENTERS[:1]])
    r = np.concatenate([tc, tc[:1]])
    ax.plot(theta, r, color="crimson", lw=2)
    ax.fill(theta, r, color="crimson", alpha=0.25)
    m = metrics.loc[u]
    ax.set_title(f"unit {u}\nMVL={m.mvl:.2f}, peak={m.peak_rate:.0f} Hz",
                 fontsize=9, pad=14)
    ax.set_theta_zero_location("E")
    ax.set_xticklabels([])
    ax.tick_params(labelsize=7)
plt.suptitle("Head-direction tuning curves (12 strongest HD cells, Mouse12-120806)",
             fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_polar_tuning.png", bbox_inches="tight")
plt.close()
print("saved fig2_polar_tuning.png")

# %% [markdown]
# ## Population summary
#
# HD cells should (a) show clearly higher MVL than the rest of the population and
# (b) have preferred directions that span the whole circle, since the internal compass
# must represent every heading.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

# (a) MVL distribution, HD vs non-HD
axes[0].hist(metrics.loc[~metrics.is_hd, "mvl"], bins=20, range=(0, 1),
             color="gray", alpha=0.7, label="non-HD")
axes[0].hist(metrics.loc[metrics.is_hd, "mvl"], bins=20, range=(0, 1),
             color="crimson", alpha=0.8, label="HD cell")
axes[0].set(xlabel="mean vector length", ylabel="# units",
            title="Tuning strength")
axes[0].legend()

# (b) preferred directions of HD cells around the circle
axp = fig.add_subplot(1, 3, 2, projection="polar")
axes[1].remove()
prefs = metrics.loc[metrics.is_hd, "pref_dir"].values
axp.hist(prefs, bins=16, range=(0, 2 * np.pi), color="crimson", alpha=0.7)
axp.set_title("Preferred directions of HD cells", pad=18)
axp.set_theta_zero_location("E")

# (c) peak rate vs MVL
axes[2].scatter(metrics.loc[~metrics.is_hd, "mvl"],
                metrics.loc[~metrics.is_hd, "peak_rate"],
                c="gray", s=25, alpha=0.6, label="non-HD")
axes[2].scatter(metrics.loc[metrics.is_hd, "mvl"],
                metrics.loc[metrics.is_hd, "peak_rate"],
                c="crimson", s=30, label="HD cell")
axes[2].set(xlabel="mean vector length", ylabel="peak firing rate (Hz)",
            title="Rate vs. tuning strength", yscale="log")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig3_population_summary.png", bbox_inches="tight")
plt.close()
print("saved fig3_population_summary.png")

# %% [markdown]
# ## Stability within the session
#
# A genuine directional code should be stable over time. We split the foraging epoch
# into first and second halves, rebuild each HD cell's tuning curve independently, and
# compare. Stable cells fall on the diagonal (same preferred direction) and their
# tuning curves are highly correlated across halves.

# %%
mid = float(epoch.start[0]) + 0.5 * (float(epoch.end[0]) - float(epoch.start[0]))
ep1 = nap.IntervalSet(start=epoch.start[0], end=mid)
ep2 = nap.IntervalSet(start=mid, end=epoch.end[0])


def half_tuning(spikes, hd_forage, ep):
    hf = hd_forage.restrict(ep)
    hd_t, hd_v = hf.index.values, hf.values
    o, _ = np.histogram(hd_v, bins=BIN_EDGES)
    o = o * dt
    out = {}
    for u in hd_units.index:
        st = spikes[u].restrict(ep).index.values
        out[u] = fast_tuning(st, hd_t, hd_v, o)
    return out


t1 = half_tuning(spikes, hd_forage, ep1)
t2 = half_tuning(spikes, hd_forage, ep2)

pref1 = np.array([mean_vector_length(t1[u])[1] for u in hd_units.index])
pref2 = np.array([mean_vector_length(t2[u])[1] for u in hd_units.index])
corrs = [np.corrcoef(t1[u], t2[u])[0, 1] for u in hd_units.index]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].scatter(pref1, pref2, c="crimson", s=35)
axes[0].plot([0, 2 * np.pi], [0, 2 * np.pi], "k--", lw=1)
axes[0].set(xlabel="preferred dir, 1st half (rad)",
            ylabel="preferred dir, 2nd half (rad)",
            title="Preferred direction is stable", xlim=(0, 2 * np.pi), ylim=(0, 2 * np.pi))
axes[1].hist(corrs, bins=15, range=(-1, 1), color="crimson", alpha=0.8)
axes[1].axvline(np.median(corrs), color="k", ls="--",
                label=f"median r = {np.median(corrs):.2f}")
axes[1].set(xlabel="tuning-curve correlation (half vs. half)",
            ylabel="# HD cells", title="Tuning is reproducible")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig4_stability.png", bbox_inches="tight")
plt.close()
print(f"saved fig4_stability.png  (median half-half r = {np.median(corrs):.2f})")

# %% [markdown]
# ## Decoding head direction from the HD population
#
# If these cells form a coherent internal compass, we should be able to read the
# animal's heading straight out of their joint spiking. We use pynapple's Bayesian
# 1-D decoder: the HD tuning curves define each cell's likelihood, and from spike
# counts in short (200 ms) time bins we decode the most probable heading. We then
# compare the decoded angle to the animal's actual head direction.

# %%
# build TsGroup and tuning curves with identical, matching column/key order
hd_subset = nap.TsGroup({int(u): spikes[u].restrict(epoch) for u in hd_units.index})
order = list(hd_subset.keys())
tc_df = pd.DataFrame({u: tuning[u] for u in order}, index=BIN_CENTERS)

decoded, proba = nap.decode_1d(
    tuning_curves=tc_df,
    group=hd_subset,
    ep=epoch,
    bin_size=0.2,
    feature=hd_forage,
)

# angular error between decoded and actual heading
actual = decoded.value_from(hd_forage)   # actual HD sampled at each decoded time bin
err = np.angle(np.exp(1j * (decoded.values - actual.values)))
median_err = np.degrees(np.median(np.abs(err)))
print(f"Median absolute decoding error: {median_err:.1f} deg  "
      f"(chance ~ 90 deg)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                         gridspec_kw={"width_ratios": [2, 1]})
w2 = nap.IntervalSet(start=float(epoch.start[0]) + 300,
                     end=float(epoch.start[0]) + 360)
d_w = decoded.restrict(w2)
a_w = hd_forage.restrict(w2)
axes[0].plot(a_w.index.values, a_w.values, "k.", ms=3, label="actual")
axes[0].plot(d_w.index.values, d_w.values, "crimson", lw=1.4, label="decoded")
axes[0].set(xlabel="time (s)", ylabel="head direction (rad)",
            title=f"Decoded vs. actual heading ({len(hd_subset)} HD cells)",
            ylim=(0, 2 * np.pi))
axes[0].legend(loc="upper right")

axes[1].hist(np.degrees(np.abs(err)), bins=30, range=(0, 180),
             color="crimson", alpha=0.8)
axes[1].axvline(median_err, color="k", ls="--",
                label=f"median = {median_err:.0f} deg")
axes[1].set(xlabel="absolute decoding error (deg)", ylabel="# time bins",
            title="Decoding error")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig5_decoding.png", bbox_inches="tight")
plt.close()
print("saved fig5_decoding.png")

# %% [markdown]
# ## Scaling across sessions and mice
#
# Finally we confirm the phenomenon is not a quirk of one session by running the same
# HD-cell classification on several sessions from different animals and pooling the
# results.

# %%
SESSIONS = [
    "sub-Mouse12_ses-Mouse12-120806_behavior+ecephys.nwb",
    "sub-Mouse17_ses-Mouse17-130130_behavior+ecephys.nwb",
    "sub-Mouse28_ses-Mouse28-140313_behavior+ecephys.nwb",
    "sub-Mouse25_ses-Mouse25-140124_behavior+ecephys.nwb",
]

session_summ = []
pooled = []
for fn in SESSIONS:
    nf, nb = open_nwb(assets[fn])
    sp = nb["units"]
    hdi = head_direction_tsd(nf)
    ep = foraging_epoch(nf)
    hdf = hdi.restrict(ep).dropna()
    m, _, _, _ = classify_hd_cells(sp, hdf, ep, n_shuffle=100)
    m["session"] = fn.split("ses-")[1].split("_")[0]
    pooled.append(m)
    session_summ.append(dict(
        session=m["session"].iloc[0], n_units=len(m),
        n_hd=int(m.is_hd.sum()), frac_hd=m.is_hd.mean(),
        median_mvl_hd=m.loc[m.is_hd, "mvl"].median()))
    print(f"{m['session'].iloc[0]:>16s}: {int(m.is_hd.sum()):2d}/{len(m):2d} HD cells")

summ = pd.DataFrame(session_summ)
pooled = pd.concat(pooled)
print("\n", summ.round(2).to_string(index=False))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
axes[0].bar(summ.session, summ.n_hd, color="crimson", alpha=0.8, label="HD cells")
axes[0].bar(summ.session, summ.n_units - summ.n_hd, bottom=summ.n_hd,
            color="gray", alpha=0.5, label="other")
axes[0].set(ylabel="# units", title="HD cells per session")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend()

axes[1].bar(summ.session, 100 * summ.frac_hd, color="crimson", alpha=0.8)
axes[1].set(ylabel="% units classified HD", title="HD-cell fraction")
axes[1].tick_params(axis="x", rotation=30)

for s, grp in pooled[pooled.is_hd].groupby("session"):
    axes[2].hist(grp["mvl"], bins=12, range=(0.3, 1), histtype="step", lw=2, label=s)
axes[2].set(xlabel="mean vector length", ylabel="# HD cells",
            title="HD tuning strength across sessions")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig6_cross_session.png", bbox_inches="tight")
plt.close()
print("saved fig6_cross_session.png")

# %% [markdown]
# ## Summary
#
# Across four sessions from four different mice, a substantial fraction of anterior-
# thalamic / post-subicular units are significantly tuned to head direction, with
# sharp single-peaked polar tuning curves whose preferred directions tile the circle.
# The tuning is stable across halves of a session, and the animal's moment-to-moment
# heading can be decoded from the HD population to within roughly ten degrees. These
# are the defining properties of head-direction cells: the neural substrate of the
# brain's internal compass.
print("Done.")
