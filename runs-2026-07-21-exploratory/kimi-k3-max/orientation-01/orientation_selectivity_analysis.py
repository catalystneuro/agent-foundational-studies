# %% [markdown]
# # Orientation Selectivity in Mouse Primary Visual Cortex
#
# **Data:** DANDI Archive dataset [000021](https://dandiarchive.org/dandiset/000021) —
# Allen Institute Visual Coding, Neuropixels (Brain Observatory 1.1 stimulus set).
# Extracellular spike-sorted units recorded with 6 Neuropixels probes while mice
# passively viewed, among other stimuli, **drifting gratings** (8 directions x 5
# temporal frequencies, 2 s presentations, 80% contrast) and **static gratings**
# (6 orientations x 5 spatial frequencies x 4 phases, 0.25 s presentations).
#
# **Question:** Are neurons in mouse visual cortex (VISp and higher visual areas)
# selective for grating orientation, the classic hallmark of visual cortex first
# described by Hubel and Wiesel?
#
# **Approach:**
# 1. Stream three sessions from DANDI with LINDI (no full downloads).
# 2. Keep good-quality units whose peak channel maps to a visual cortical area
#    (VISp, VISl, VISpm, VISam, VISrl) via the electrodes table.
# 3. Compute the mean firing rate of every unit for every grating direction.
# 4. Quantify selectivity with the vector-based global orientation selectivity
#    index (gOSI) and global direction selectivity index (gDSI), and assess
#    significance with a permutation test on gOSI.
# 5. Replicate the orientation preference with the independent static-grating
#    stimulus set (30 deg resolution).
#
# Everything is computed with Pynapple/NWB data structures plus numpy; figures
# are saved as PNG files next to this script.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm

import lindi
from pynwb import NWBHDF5IO
import pynapple as nap

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})

# Three sessions from DANDI 000021, addressed through their Neurosift LINDI
# references so only the required chunks are fetched (and cached locally).
SESSIONS = {
    "715093703": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json"),
    "719161530": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "02291b99-e583-498b-9929-b68bba2c50e2/nwb.lindi.json"),
    "721123822": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "224b57e5-c9a3-46ef-85db-966713f3ccbe/nwb.lindi.json"),
}

VISUAL_AREAS = ["VISp", "VISl", "VISpm", "VISam", "VISrl"]
DIRECTIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
DIR_COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:8]
N_SHUFFLES = 500
RNG = np.random.default_rng(42)

# %% [markdown]
# ## Helper functions
#
# The stimulus tables in these NWB files contain a compound `timeseries` column
# that cannot be converted wholesale to a DataFrame, so the needed columns are
# read directly. Selectivity indices follow the circular-vector definitions:
# gOSI sums response vectors at twice the stimulus angle (orientation is
# 180-degree periodic), gDSI at the angle itself (direction is 360-degree
# periodic).

# %%
def intervals_to_df(tab, cols):
    """Read selected columns of an NWB TimeIntervals table into a DataFrame."""
    return pd.DataFrame({c: np.asarray(tab[c].data[:]) for c in cols})


def rates_for_sweeps(spike_trains, starts, stops):
    """Mean firing rate of every unit within every [start, stop) sweep."""
    counts = np.zeros((len(spike_trains), len(starts)))
    for i, st in enumerate(spike_trains):
        counts[i] = np.searchsorted(st, stops) - np.searchsorted(st, starts)
    return counts / (stops - starts)[None, :]


def gosi_matrix(mean_rates, angs_deg):
    """Global orientation selectivity index for a (units x angles) matrix."""
    th = np.deg2rad(angs_deg)
    # errstate: Apple Accelerate BLAS can raise spurious FP warnings in matmul
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        vec = mean_rates @ np.exp(2j * th)
    denom = mean_rates.sum(axis=1)
    out = np.zeros(len(mean_rates))
    np.divide(np.abs(vec), denom, out=out, where=denom > 0)
    return out


def gdsi_matrix(mean_rates, angs_deg):
    """Global direction selectivity index for a (units x angles) matrix."""
    th = np.deg2rad(angs_deg)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        vec = mean_rates @ np.exp(1j * th)
    denom = mean_rates.sum(axis=1)
    out = np.zeros(len(mean_rates))
    np.divide(np.abs(vec), denom, out=out, where=denom > 0)
    return out


def condition_mean_rates(rates, labels, conditions):
    """Average per-sweep rates over the presentations of each condition."""
    oh = np.zeros((len(labels), len(conditions)))
    for j, c in enumerate(conditions):
        oh[labels == c, j] = 1.0
    n_per = oh.sum(axis=0)
    # NOTE: numpy linked against Apple Accelerate can raise spurious
    # overflow/divide-by-zero warnings inside matmul even for clean inputs.
    # The results are verified explicitly with isfinite checks below.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_rates = (rates @ oh) / n_per[None, :]
    assert np.isfinite(mean_rates).all(), "non-finite values in tuning curves"
    return mean_rates, oh, n_per

# %% [markdown]
# ## Load sessions and compute tuning
#
# For each session we stream the NWB file, select good-quality units in visual
# cortical areas (unit quality from the units table, brain structure from the
# electrodes table via `peak_channel_id`), and compute per-sweep firing rates
# for the drifting-grating and static-grating blocks. A permutation test
# (shuffling direction labels across sweeps) yields a p-value for each unit's
# gOSI.

# %%
session_data = {}  # session id -> dict with everything figures need
metrics_frames = []

for sid, url in SESSIONS.items():
    print(f"\n=== session {sid} ===")
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)

    # ---- unit selection: good quality + visual cortical structure
    units = nwb["units"]
    elec = nwbfile.electrodes.to_dataframe()
    meta = units.metadata.copy()
    meta["structure"] = meta["peak_channel_id"].map(elec["location"])
    keep = meta[(meta["quality"] == "good") & (meta["structure"].isin(VISUAL_AREAS))]
    unit_ids = list(keep.index)
    print(f"good visual-cortex units: {len(unit_ids)} "
          f"{keep['structure'].value_counts().to_dict()}")

    spike_trains = [np.asarray(units[uid].times())
                    for uid in tqdm(unit_ids, desc="reading spike trains")]

    # ---- drifting gratings
    dg = intervals_to_df(nwbfile.intervals["drifting_gratings_presentations"],
                         ["start_time", "stop_time", "orientation", "temporal_frequency"])
    dg.columns = ["start", "stop", "orientation", "temporal_frequency"]
    dg = dg.dropna(subset=["orientation"]).reset_index(drop=True)
    dg["orientation"] = dg["orientation"].astype(float)

    rates = rates_for_sweeps(spike_trains, dg["start"].to_numpy(), dg["stop"].to_numpy())
    assert np.isfinite(rates).all()

    directions = np.sort(dg["orientation"].unique())
    dir_labels = dg["orientation"].to_numpy()
    mean_rates, oh, n_per = condition_mean_rates(rates, dir_labels, directions)

    gOSI = gosi_matrix(mean_rates, directions)
    gDSI = gdsi_matrix(mean_rates, directions)

    # permutation test: shuffle direction labels, recompute gOSI
    exceed = np.zeros(len(unit_ids), dtype=int)
    for _ in range(N_SHUFFLES):
        perm = RNG.permutation(len(dg))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            m = (rates @ oh[perm]) / n_per[None, :]
        exceed += gosi_matrix(m, directions) >= gOSI
    p_gOSI = (1 + exceed) / (1 + N_SHUFFLES)

    pref_dir = directions[np.argmax(mean_rates, axis=1)]
    metrics_frames.append(pd.DataFrame({
        "unit_id": unit_ids, "session": int(sid),
        "structure": keep["structure"].to_numpy(),
        "mean_rate": rates.mean(axis=1), "max_rate": mean_rates.max(axis=1),
        "pref_dir": pref_dir, "pref_ori": pref_dir % 180,
        "gOSI": gOSI, "gDSI": gDSI, "p_gOSI": p_gOSI,
    }))

    # ---- static gratings (finer orientation resolution)
    sg = intervals_to_df(nwbfile.intervals["static_gratings_presentations"],
                         ["start_time", "stop_time", "orientation", "spatial_frequency", "phase"])
    sg.columns = ["start", "stop", "orientation", "spatial_frequency", "phase"]
    sg = sg.dropna(subset=["orientation"]).reset_index(drop=True)
    sg["orientation"] = sg["orientation"].astype(float)
    srates = rates_for_sweeps(spike_trains, sg["start"].to_numpy(), sg["stop"].to_numpy())
    assert np.isfinite(srates).all()

    soris = np.sort(sg["orientation"].unique())
    sori_labels = sg["orientation"].to_numpy()
    static_tc, soh, sn_per = condition_mean_rates(srates, sori_labels, soris)
    static_gOSI = gosi_matrix(static_tc, soris)
    static_pref = soris[np.argmax(static_tc, axis=1)]

    exceed = np.zeros(len(unit_ids), dtype=int)
    for _ in range(300):
        perm = RNG.permutation(len(sg))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            m = (srates @ soh[perm]) / sn_per[None, :]
        exceed += gosi_matrix(m, soris) >= static_gOSI
    static_p = (1 + exceed) / 301

    session_data[sid] = dict(
        nwb=nwb, io=io, unit_ids=unit_ids, keep=keep,
        dg=dg, rates=rates, directions=directions,
        sg=sg, srates=srates, soris=soris, static_tc=static_tc,
        static_gOSI=static_gOSI, static_pref=static_pref, static_p=static_p,
    )

metrics = pd.concat(metrics_frames, ignore_index=True)
metrics.to_csv("metrics_all_sessions.csv", index=False)

sig = metrics["p_gOSI"] < 0.01
print(f"\nTotal visual-cortex units: {len(metrics)}")
print(f"Significantly orientation-selective (permutation p<0.01): "
      f"{sig.sum()} ({sig.mean() * 100:.0f}%)")
print(metrics.groupby("structure").agg(
    n=("unit_id", "count"),
    frac_sig=("p_gOSI", lambda x: (x < 0.01).mean()),
    median_gOSI=("gOSI", "median")))

# %% [markdown]
# ## Figure 1: raw data
#
# Raster of the 30 most orientation-selective VISp units over a 24 s excerpt of
# the drifting-grating block of session 715093703. Background color encodes the
# grating direction of each 2 s presentation. Spiking is clearly locked to the
# gratings, and different units prefer different directions.

# %%
SID1 = "715093703"
d1 = session_data[SID1]
m1 = metrics[metrics["session"] == int(SID1)]
all_units = d1["nwb"]["units"]

visp_metrics = m1[m1["structure"] == "VISp"].sort_values("gOSI", ascending=False)
show_ids = visp_metrics["unit_id"].to_numpy()[:30]

sweeps = d1["dg"]
t0 = sweeps["start"].iloc[0] - 2.0
t1 = t0 + 24.0
win = sweeps[(sweeps["start"] >= t0) & (sweeps["stop"] <= t1)]

fig, ax = plt.subplots(figsize=(11, 6))
for _, row in win.iterrows():
    di = int(np.where(d1["directions"] == row["orientation"])[0][0])
    ax.axvspan(row["start"], row["stop"], color=DIR_COLORS[di], alpha=0.35, lw=0)
for i, uid in enumerate(tqdm(show_ids, desc="fig1 rasters")):
    st = np.asarray(all_units[uid].times())
    st = st[(st >= t0) & (st <= t1)]
    ax.vlines(st, i + 0.6, i + 1.4, color="black", lw=0.5)
ax.set_xlim(t0, t1)
ax.set_ylim(0.4, len(show_ids) + 0.6)
ax.set_xlabel("time (s)")
ax.set_ylabel("unit (sorted by orientation selectivity)")
ax.set_title(f"VISp population raster during drifting gratings (session {SID1})\n"
             "background color = grating direction (2 s presentations)")
handles = [plt.Rectangle((0, 0), 1, 1, color=DIR_COLORS[i], alpha=0.5) for i in range(8)]
ax.legend(handles, [f"{int(d)}°" for d in DIRECTIONS], ncol=9, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), frameon=False, title="grating direction")
fig.tight_layout()
fig.savefig("fig1_raw_data_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig1_raw_data_overview.png")

# %% [markdown]
# ## Figure 2: an example orientation-selective unit
#
# The most selective VISp unit of session 715093703. Left: spike raster and
# PSTH aligned to the onset of its preferred direction. Right: PSTHs for all
# eight directions. The unit fires vigorously at 135 deg and 315 deg (the same
# orientation, opposite drift directions) and is nearly silent for the
# orthogonal orientations: orientation selectivity.

# %%
example_uid = int(visp_metrics.iloc[0]["unit_id"])
spk = all_units[example_uid]
pref_dir = visp_metrics.iloc[0]["pref_dir"]
bins = np.arange(-0.5, 2.55, 0.05)

fig = plt.figure(figsize=(13, 5.5))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.3], height_ratios=[2, 1],
                       hspace=0.05, wspace=0.25)

pref_sweeps = sweeps[sweeps["orientation"] == pref_dir]
pe = nap.compute_perievent(spk, nap.Ts(t=pref_sweeps["start"].to_numpy()),
                           window=(-0.5, 2.5))
ax_r = fig.add_subplot(gs[0, 0])
for tr, ts in enumerate(pe.values()):
    ax_r.vlines(np.asarray(ts.t), tr + 0.5, tr + 1.5, color="black", lw=0.4)
ax_r.axvspan(0, 2, color=DIR_COLORS[int(np.where(DIRECTIONS == pref_dir)[0])], alpha=0.2)
ax_r.set_xlim(-0.5, 2.5)
ax_r.set_ylim(0.5, len(pref_sweeps) + 0.5)
ax_r.invert_yaxis()
ax_r.set_ylabel("trial")
ax_r.set_xticklabels([])
ax_r.set_title(f"unit {example_uid} - preferred direction ({int(pref_dir)}°)")

ax_p = fig.add_subplot(gs[1, 0])
all_rel = np.concatenate([np.asarray(ts.t) for ts in pe.values()])
h, _ = np.histogram(all_rel, bins=bins)
ax_p.bar(bins[:-1], h / (len(pref_sweeps) * 0.05), width=0.05, color="black")
ax_p.axvspan(0, 2, color=DIR_COLORS[int(np.where(DIRECTIONS == pref_dir)[0])], alpha=0.2)
ax_p.set_xlim(-0.5, 2.5)
ax_p.set_xlabel("time from grating onset (s)")
ax_p.set_ylabel("rate (Hz)")

ax_all = fig.add_subplot(gs[:, 1])
for di, d in enumerate(DIRECTIONS):
    sw = sweeps[sweeps["orientation"] == d]
    pe_d = nap.compute_perievent(spk, nap.Ts(t=sw["start"].to_numpy()),
                                 window=(-0.5, 2.5))
    rel = (np.concatenate([np.asarray(ts.t) for ts in pe_d.values()])
           if len(sw) else np.array([]))
    h, _ = np.histogram(rel, bins=bins)
    ax_all.plot(bins[:-1] + 0.025, h / (len(sw) * 0.05), color=DIR_COLORS[di],
                label=f"{int(d)}°", lw=1.5)
ax_all.axvspan(0, 2, color="grey", alpha=0.12)
ax_all.set_xlim(-0.5, 2.5)
ax_all.set_xlabel("time from grating onset (s)")
ax_all.set_ylabel("rate (Hz)")
ax_all.set_title(f"unit {example_uid} - responses to all 8 directions")
ax_all.legend(title="direction", ncol=2, frameon=False)
fig.suptitle("Example orientation-selective VISp unit "
             f"(gOSI = {visp_metrics.iloc[0]['gOSI']:.2f})", y=0.98)
fig.savefig("fig2_example_unit_psth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig2_example_unit_psth.png")

# %% [markdown]
# ## Figure 3: direction tuning curves across the selectivity range
#
# Six VISp units from strongly selective to flat. Cartesian tuning curves with
# SEM error bars on the left of each pair, the same curve in polar coordinates
# on the right. Selective units show the classic 180-degree-periodic tuning;
# the p-values come from the permutation test.

# %%
visp_sorted = m1[(m1["structure"] == "VISp") & (m1["mean_rate"] > 1)].sort_values(
    "gOSI", ascending=False)
picks = [int(visp_sorted.iloc[i]["unit_id"]) for i in [0, 2, 10, 40, 80, -1]]

uid_to_row = {int(u): i for i, u in enumerate(d1["unit_ids"])}
ori_labels = d1["dg"]["orientation"].to_numpy()
rates1 = d1["rates"]

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
for k, uid in enumerate(picks):
    r, c = divmod(k, 2)
    row = rates1[uid_to_row[uid]]
    tc = np.array([row[ori_labels == d].mean() for d in DIRECTIONS])
    se = np.array([row[ori_labels == d].std(ddof=1) / np.sqrt((ori_labels == d).sum())
                   for d in DIRECTIONS])
    g = m1[m1["unit_id"] == uid].iloc[0]

    ax = fig.add_subplot(gs[r, c * 2])
    ax.errorbar(DIRECTIONS, tc, yerr=se, marker="o", ms=4, color="black",
                ecolor="steelblue", capsize=2)
    ax.set_xticks(DIRECTIONS)
    ax.set_xlabel("direction (deg)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(f"unit {uid}  gOSI={g['gOSI']:.2f}  p={g['p_gOSI']:.3f}", fontsize=9)

    axp = fig.add_subplot(gs[r, c * 2 + 1], projection="polar")
    th = np.deg2rad(np.append(DIRECTIONS, DIRECTIONS[0]))
    axp.plot(th, np.append(tc, tc[0]), color="darkred", lw=2)
    axp.fill(th, np.append(tc, tc[0]), color="darkred", alpha=0.25)
    axp.set_theta_zero_location("N")
    axp.set_title(f"pref {int(g['pref_dir'])}°", fontsize=9, pad=24)

fig.suptitle(f"Direction tuning of example VISp units, ordered by selectivity (session {SID1})")
fig.savefig("fig3_example_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig3_example_tuning_curves.png")

# %% [markdown]
# ## Figure 4: population summary across three sessions
#
# Pooling all 1389 good-quality visual-cortex units from the three sessions:
# (a) gOSI distribution split by permutation-test significance, (b) preferred
# orientation distribution of the selective units, (c) direction selectivity of
# the orientation-selective units (most are orientation- but not
# direction-selective), (d) fraction of selective units per visual area.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
bins = np.linspace(0, 1, 41)
ax.hist(metrics.loc[~sig, "gOSI"], bins=bins, alpha=0.7,
        label=f"not significant (n={(~sig).sum()})", color="grey")
ax.hist(metrics.loc[sig, "gOSI"], bins=bins, alpha=0.7,
        label=f"orientation-selective, p<0.01 (n={sig.sum()})", color="darkred")
ax.set_xlabel("global orientation selectivity index (gOSI)")
ax.set_ylabel("number of units")
ax.legend(frameon=False)
ax.set_title("Orientation selectivity across the population")

ax = axes[0, 1]
# drifting gratings sample 4 orientations (directions folded mod 180)
ori_vals = [0, 45, 90, 135]
counts_ori = [(metrics.loc[sig, "pref_ori"] == o).sum() for o in ori_vals]
ax.bar([str(o) for o in ori_vals], counts_ori, color="steelblue", edgecolor="white")
for i, c in enumerate(counts_ori):
    ax.text(i, c + 3, str(c), ha="center", fontsize=9)
ax.set_xlabel("preferred orientation (deg)")
ax.set_ylabel("number of units")
ax.set_title("Preferred orientation distribution (selective units)")

ax = axes[1, 0]
ax.hist(metrics.loc[sig, "gDSI"], bins=np.linspace(0, 1, 41), color="seagreen", alpha=0.8)
ax.axvline(0.5, color="black", ls="--", lw=1, label="gDSI = 0.5")
ax.set_xlabel("global direction selectivity index (gDSI)")
ax.set_ylabel("number of units")
ax.set_title("Direction selectivity of orientation-selective units")
ax.legend(frameon=False)

ax = axes[1, 1]
area_stats = metrics.groupby("structure").agg(
    n=("unit_id", "count"), frac=("p_gOSI", lambda x: (x < 0.01).mean()),
    ).loc[["VISp", "VISl", "VISpm", "VISam", "VISrl"]]
bars = ax.bar(area_stats.index, area_stats["frac"], color="slateblue", alpha=0.85)
for b, n in zip(bars, area_stats["n"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={n}",
            ha="center", fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("fraction orientation-selective (p<0.01)")
ax.set_title("Selective fraction by visual cortical area")

fig.suptitle(f"Population summary: {len(metrics)} units, 3 sessions, "
             f"{sig.mean() * 100:.0f}% significantly orientation-selective")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig4_population_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig4_population_summary.png")

# %% [markdown]
# ## Figure 5: replication with static gratings
#
# Static gratings sample 6 orientations at 30 deg resolution and are an
# independent stimulus block, so they provide a within-session replication.
# (a) The example unit's normalized tuning to drifting vs static gratings.
# (b) Preferred orientation from drifting vs static gratings for units
# significantly tuned in both; most lie on the identity line.
# (c) Population distribution of static-grating gOSI.

# %%
stc = d1["static_tc"]
SORIS = d1["soris"]
static_gOSI = d1["static_gOSI"]
static_pref = d1["static_pref"]
static_p = d1["static_p"]
m1_idx = m1.set_index("unit_id")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

ax = axes[0]
uid = picks[0]
i = uid_to_row[uid]
drift_ori_curve = np.array([rates1[i][ori_labels == d].mean() for d in [0, 45, 90, 135]])
ax.plot([0, 45, 90, 135], drift_ori_curve / drift_ori_curve.max(), "o-",
        color="darkred", label="drifting gratings (folded to orientation)")
ax.plot(SORIS, stc[i] / stc[i].max(), "s-", color="steelblue", label="static gratings")
ax.set_xlabel("orientation (deg)")
ax.set_ylabel("normalized rate")
ax.set_title(f"unit {uid}: consistent preference across stimuli")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
both_sig = [(uid_to_row[u], u) for u in m1[m1["p_gOSI"] < 0.01]["unit_id"]
            if u in uid_to_row and static_p[uid_to_row[u]] < 0.05]
x = np.array([m1_idx.loc[u, "pref_ori"] for _, u in both_sig])
y = np.array([static_pref[i] for i, _ in both_sig])
circ_diff = np.abs((x - y + 90) % 180 - 90)
frac_close = (circ_diff <= 30).mean()
jrng = np.random.default_rng(1)
ax.scatter(x + jrng.normal(0, 1.5, len(x)), y + jrng.normal(0, 1.5, len(y)),
           s=14, alpha=0.6, color="darkslateblue")
ax.plot([0, 180], [0, 180], "k--", lw=1)
ax.set_xlim(-8, 188)
ax.set_ylim(-8, 188)
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_yticks([0, 45, 90, 135, 180])
ax.set_xlabel("preferred orientation, drifting (deg)")
ax.set_ylabel("preferred orientation, static (deg)")
ax.set_title(f"Preference consistency (n={len(x)} units selective in both)\n"
             f"{frac_close * 100:.0f}% within ±30°", fontsize=9)

ax = axes[2]
bins = np.linspace(0, 1, 41)
ax.hist(static_gOSI[static_p >= 0.05], bins=bins, color="grey", alpha=0.7,
        label=f"not significant (n={(static_p >= 0.05).sum()})")
ax.hist(static_gOSI[static_p < 0.05], bins=bins, color="teal", alpha=0.75,
        label=f"selective, p<0.05 (n={(static_p < 0.05).sum()})")
ax.set_xlabel("static-grating gOSI")
ax.set_ylabel("number of units")
ax.set_title(f"Static grating orientation selectivity (session {SID1})")
ax.legend(frameon=False)

fig.suptitle("Orientation selectivity to static gratings (6 orientations, 30° resolution)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig5_static_gratings.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig5_static_gratings.png")

# close the NWB file handles
for d in session_data.values():
    d["io"].close()

# %% [markdown]
# ## Results
#
# Across three sessions of the Allen Visual Coding Neuropixels dataset
# (DANDI 000021), 1389 good-quality units were recorded from visual cortical
# areas. About two thirds of them (65%, 904 units) are significantly
# orientation-selective for drifting gratings (permutation test on gOSI,
# p < 0.01), with similar fractions across VISp (62%) and the higher visual
# areas VISl (68%), VISpm (72%), VISam (61%) and VISrl (65%). Preferred orientations cover all four sampled
# orientations with no strong bias. Most orientation-selective units respond
# similarly to both drift directions of their preferred orientation (low gDSI),
# i.e. they are orientation- rather than direction-selective, as expected for
# mouse V1. The orientation preference estimated from drifting gratings is
# reproduced by the independent static-grating stimulus block: for units
# selective in both, the preferred orientations agree within +/-30 deg in the
# large majority of cases. This is a clean demonstration of orientation
# selectivity, the canonical receptive-field property of visual cortex, in
# openly available data.
