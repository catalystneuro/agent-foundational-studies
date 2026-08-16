"""Generate all figures for the orientation selectivity analysis."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})

SID = "715093703"
LINDI_URL = ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
             "58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json")
DIRECTIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
DIR_COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:8]  # one per direction

metrics = pd.read_csv("metrics_all_sessions.csv")
m1 = metrics[metrics["session"] == int(SID)]
rates = np.load(f"rates_{SID}.npy")
sweeps = pd.read_csv(f"sweeps_{SID}.csv")
units1 = pd.read_csv(f"units_{SID}.csv")

# ================================================================ fig 1: raw data
print("fig1: raw data overview ...")
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
nwbfile = NWBHDF5IO(file=f).read()
nwb = nap.NWBFile(nwbfile)
all_units = nwb["units"]

# choose 30 VISp units spanning selectivity, sorted by gOSI
visp_metrics = m1[m1["structure"] == "VISp"].sort_values("gOSI", ascending=False)
show_ids = visp_metrics["unit_id"].to_numpy()[:30]

t0 = sweeps["start"].iloc[0] - 2.0
t1 = t0 + 24.0
win = sweeps[(sweeps["start"] >= t0) & (sweeps["stop"] <= t1)]

fig, ax = plt.subplots(figsize=(11, 6))
# stimulus bars
for _, row in win.iterrows():
    di = int(np.where(DIRECTIONS == row["orientation"])[0][0])
    ax.axvspan(row["start"], row["stop"], color=DIR_COLORS[di], alpha=0.35, lw=0)
for i, uid in enumerate(show_ids):
    st = np.asarray(all_units[uid].times())
    st = st[(st >= t0) & (st <= t1)]
    ax.vlines(st, i + 0.6, i + 1.4, color="black", lw=0.5)
ax.set_xlim(t0, t1)
ax.set_ylim(0.4, len(show_ids) + 0.6)
ax.set_xlabel("time (s)")
ax.set_ylabel("unit (sorted by orientation selectivity)")
ax.set_title(f"VISp population raster during drifting gratings (session {SID})\n"
             "background color = grating direction (2 s presentations)")

# legend for directions
handles = [plt.Rectangle((0, 0), 1, 1, color=DIR_COLORS[i], alpha=0.5) for i in range(8)]
ax.legend(handles, [f"{int(d)}°" for d in DIRECTIONS], ncol=9, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), frameon=False, title="grating direction")
fig.tight_layout()
fig.savefig("fig1_raw_data_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig1_raw_data_overview.png")

# ================================================================ fig 2: example unit PSTH
print("fig2: example unit perievent responses ...")
example_uid = int(visp_metrics.iloc[0]["unit_id"])
spk = all_units[example_uid]

fig = plt.figure(figsize=(13, 5.5))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.3], height_ratios=[2, 1],
                       hspace=0.05, wspace=0.25)

# left: raster + PSTH for preferred direction
pref_dir = visp_metrics.iloc[0]["pref_dir"]
pref_sweeps = sweeps[sweeps["orientation"] == pref_dir]
onsets = nap.Ts(t=pref_sweeps["start"].to_numpy())
pe = nap.compute_perievent(spk, onsets, window=(-0.5, 2.5))

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
bins = np.arange(-0.5, 2.55, 0.05)
h, edges = np.histogram(all_rel, bins=bins)
rate = h / (len(pref_sweeps) * 0.05)
ax_p.bar(bins[:-1], rate, width=0.05, color="black")
ax_p.axvspan(0, 2, color=DIR_COLORS[int(np.where(DIRECTIONS == pref_dir)[0])], alpha=0.2)
ax_p.set_xlim(-0.5, 2.5)
ax_p.set_xlabel("time from grating onset (s)")
ax_p.set_ylabel("rate (Hz)")

# right: PSTHs for all 8 directions
ax_all = fig.add_subplot(gs[:, 1])
for di, d in enumerate(DIRECTIONS):
    sw = sweeps[sweeps["orientation"] == d]
    ons = nap.Ts(t=sw["start"].to_numpy())
    pe_d = nap.compute_perievent(spk, ons, window=(-0.5, 2.5))
    rel = np.concatenate([np.asarray(ts.t) for ts in pe_d.values()]) if len(sw) else np.array([])
    h, _ = np.histogram(rel, bins=bins)
    ax_all.plot(bins[:-1] + 0.025, h / (len(sw) * 0.05), color=DIR_COLORS[di],
                label=f"{int(d)}°", lw=1.5)
ax_all.axvspan(0, 2, color="grey", alpha=0.12)
ax_all.set_xlim(-0.5, 2.5)
ax_all.set_xlabel("time from grating onset (s)")
ax_all.set_ylabel("rate (Hz)")
ax_all.set_title(f"unit {example_uid} - responses to all 8 directions")
ax_all.legend(title="direction", ncol=2, frameon=False)
fig.suptitle(f"Example orientation-selective VISp unit (gOSI = {visp_metrics.iloc[0]['gOSI']:.2f})",
             y=0.98)
fig.savefig("fig2_example_unit_psth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig2_example_unit_psth.png")

# ================================================================ fig 3: example tuning curves
print("fig3: example tuning curves ...")
# 6 example units from session 1 spanning selectivity
visp_sorted = m1[(m1["structure"] == "VISp") & (m1["mean_rate"] > 1)].sort_values("gOSI", ascending=False)
picks = [visp_sorted.iloc[0]["unit_id"], visp_sorted.iloc[2]["unit_id"],
         visp_sorted.iloc[10]["unit_id"], visp_sorted.iloc[40]["unit_id"],
         visp_sorted.iloc[80]["unit_id"], visp_sorted.iloc[-1]["unit_id"]]
picks = [int(p) for p in picks]

uid_to_row = {int(u): i for i, u in enumerate(units1["unit_id"])}
ori_labels = sweeps["orientation"].to_numpy()

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
for k, uid in enumerate(picks):
    r, c = divmod(k, 2)
    row = rates[uid_to_row[uid]]
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

fig.suptitle(f"Direction tuning of example VISp units, ordered by selectivity (session {SID})")
fig.savefig("fig3_example_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig3_example_tuning_curves.png")

# ================================================================ fig 4: population
print("fig4: population summary ...")
sig = metrics["p_gOSI"] < 0.01
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
bins = np.linspace(0, 1, 41)
ax.hist(metrics.loc[~sig, "gOSI"], bins=bins, alpha=0.7, label=f"not significant (n={(~sig).sum()})",
        color="grey")
ax.hist(metrics.loc[sig, "gOSI"], bins=bins, alpha=0.7,
        label=f"orientation-selective, p<0.01 (n={sig.sum()})", color="darkred")
ax.set_xlabel("global orientation selectivity index (gOSI)")
ax.set_ylabel("number of units")
ax.legend(frameon=False)
ax.set_title("Orientation selectivity across the population")

ax = axes[0, 1]
# drifting gratings sample 4 orientations (directions folded mod 180): use discrete bars
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
    med=("gOSI", "median")).loc[["VISp", "VISl", "VISpm", "VISam", "VISrl"]]
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

# ================================================================ fig 5: static gratings
print("fig5: static gratings ...")
srates = np.load(f"rates_static_{SID}.npy")
ssweeps = pd.read_csv(f"sweeps_static_{SID}.csv")
SORIS = np.sort(ssweeps["orientation"].unique())
sori_labels = ssweeps["orientation"].to_numpy()

# static tuning curves for all session-1 units
stc = np.array([[srates[i][sori_labels == o].mean() for o in SORIS]
                for i in range(srates.shape[0])])

def gosi_rows(M, angs):
    th = np.deg2rad(angs)
    vec = M @ np.exp(2j * th)
    den = M.sum(axis=1)
    out = np.zeros(len(M))
    np.divide(np.abs(vec), den, out=out, where=den > 0)
    return out

static_gOSI = gosi_rows(stc, SORIS)
static_pref = SORIS[np.argmax(stc, axis=1)]

# permutation test for static gOSI (session 1 units)
rng = np.random.default_rng(0)
n_sw = len(ssweeps)
oh = np.zeros((n_sw, len(SORIS)))
for j, o in enumerate(SORIS):
    oh[sori_labels == o, j] = 1.0
n_per = oh.sum(axis=0)
exceed = np.zeros(srates.shape[0], dtype=int)
for s in range(300):
    perm = rng.permutation(n_sw)
    m = (srates @ oh[perm]) / n_per[None, :]
    exceed += gosi_rows(m, SORIS) >= static_gOSI
static_p = (1 + exceed) / 301

uid_row = {int(u): i for i, u in enumerate(units1["unit_id"])}
m1_idx = m1.set_index("unit_id")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# (a) example unit: drifting vs static tuning (normalized)
ax = axes[0]
uid = picks[0]
i = uid_row[uid]
drift_ori_curve = np.array([rates[i][ori_labels == d].mean()
                            for d in [0, 45, 90, 135]])
drift_oris = np.array([0, 45, 90, 135])
ax.plot(drift_oris, drift_ori_curve / drift_ori_curve.max(), "o-", color="darkred",
        label="drifting gratings (folded to orientation)")
ax.plot(SORIS, stc[i] / stc[i].max(), "s-", color="steelblue",
        label="static gratings")
ax.set_xlabel("orientation (deg)")
ax.set_ylabel("normalized rate")
ax.set_title(f"unit {uid}: consistent preference across stimuli")
ax.legend(frameon=False, fontsize=8)

# (b) scatter of preferred orientation, drifting vs static
ax = axes[1]
both_sig = [(uid_row[u], u) for u in m1[(m1["p_gOSI"] < 0.01)]["unit_id"]
            if u in uid_row and static_p[uid_row[u]] < 0.05]
x = np.array([m1_idx.loc[u, "pref_ori"] for _, u in both_sig])
y = np.array([static_pref[i] for i, _ in both_sig])
# circular (180-deg) difference between the two estimates
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

# (c) static gOSI distribution
ax = axes[2]
bins = np.linspace(0, 1, 41)
ax.hist(static_gOSI[static_p >= 0.05], bins=bins, color="grey", alpha=0.7,
        label=f"not significant (n={(static_p >= 0.05).sum()})")
ax.hist(static_gOSI[static_p < 0.05], bins=bins, color="teal", alpha=0.75,
        label=f"selective, p<0.05 (n={(static_p < 0.05).sum()})")
ax.set_xlabel("static-grating gOSI")
ax.set_ylabel("number of units")
ax.set_title(f"Static grating orientation selectivity (session {SID})")
ax.legend(frameon=False)

fig.suptitle("Orientation selectivity to static gratings (6 orientations, 30° resolution)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig5_static_gratings.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig5_static_gratings.png")
print("all figures done")
