"""Stage 3: single-session figures - example units, population summaries,
cross-stimulus validation."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pynapple as nap
from scipy import stats

import orientation_lib as ol

SES = "715093703"
res = pd.read_csv(f"cache/res_{SES}.csv")
d = np.load(f"cache/session_{SES}.npz", allow_pickle=True)
t = np.load(f"cache/tc_{SES}.npz")
tc, tc_sem, dirs = t["tc"], t["tc_sem"], t["dirs"]
tc_b_aligned = t["tc_b_aligned"]
sg_tc, sg_sem, sg_dirs = t["sg_tc"], t["sg_sem"], t["sg_dirs"]
direction, tf = d["dg_direction"], d["dg_tf"]
dg_start, dg_stop = d["dg_start"], d["dg_stop"]

units = pd.read_csv(f"cache/units_{SES}.csv")
sessions = ol.list_session_assets()
row = sessions[sessions.session_id == SES].iloc[0]
f = ol.open_session(row["url"])
spikes, units = ol.load_spikes(f, units, progress=True)
assert list(units.unit_id) == list(res.unit_id)

# ===========================================================================
# FIGURE 2: example orientation-selective units (raster + PSTH + polar)
# ===========================================================================
cand = res[(res.region.isin(ol.VISUAL_CORTEX)) & res.sig_ori & (res.peak_rate > 3)]
# V1 first, then the higher visual areas, each ordered by selectivity
cand = pd.concat([cand[cand.region == "VISp"].sort_values("gosi", ascending=False),
                  cand[cand.region != "VISp"].sort_values("gosi", ascending=False)])
# pick 4 units with distinct preferred orientations
picks, used = [], []
for i, r in cand.iterrows():
    if all(ol.circ_dist_180(r.pref_ori, u) > 25 for u in used):
        picks.append(i)
        used.append(r.pref_ori)
    if len(picks) == 4:
        break
print("example units:", res.loc[picks, ["unit_id", "region", "gosi", "pref_ori"]].to_string())

WIN = (-0.5, 2.5)
fig = plt.figure(figsize=(15, 14))
gs = GridSpec(4, 3, figure=fig, width_ratios=[1.5, 1.1, 1.0], hspace=0.62, wspace=0.34,
              top=0.9, bottom=0.05)
colors = plt.cm.hsv(np.linspace(0, 1, len(dirs), endpoint=False))

for k, j in enumerate(picks):
    uid = res.unit_id.iloc[j]
    m_tf = tf == res.pref_tf.iloc[j]

    # ---- raster, trials grouped by direction ----
    ax = fig.add_subplot(gs[k, 0])
    ytick, ylab, y = [], [], 0
    for di, dd in enumerate(dirs):
        sel = np.where(m_tf & (direction == dd))[0]
        ev = nap.Ts(t=dg_start[sel])
        pe = nap.compute_perievent(spikes[uid], ev, WIN)
        y0 = y
        for e in pe.keys():
            tt = pe[e].t
            ax.plot(tt, np.full_like(tt, y), "|", ms=3, mew=0.8, color=colors[di])
            y += 1
        ytick.append((y0 + y) / 2)
        ylab.append("%d" % dd)
    ax.axvspan(0, 2, color="0.9", zorder=-2)
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab, fontsize=8)
    ax.set_ylim(-1, y)
    ax.set_xlim(*WIN)
    ax.set_ylabel("direction (deg)")
    ax.set_title("unit %d (%s)  gOSI=%.2f" % (uid, res.region.iloc[j], res.gosi.iloc[j]),
                 fontsize=10)
    if k == 3:
        ax.set_xlabel("time from grating onset (s)")

    # ---- PSTH per direction ----
    ax = fig.add_subplot(gs[k, 1])
    bins = np.arange(WIN[0], WIN[1] + 0.05, 0.05)
    for di, dd in enumerate(dirs):
        sel = np.where(m_tf & (direction == dd))[0]
        ev = nap.Ts(t=dg_start[sel])
        pe = nap.compute_perievent(spikes[uid], ev, WIN)
        allt = np.concatenate([pe[e].t for e in pe.keys()])
        h, _ = np.histogram(allt, bins=bins)
        ax.plot(bins[:-1] + 0.025, h / (len(sel) * 0.05), color=colors[di], lw=1.2,
                label="%d" % dd if k == 0 else None)
    ax.axvspan(0, 2, color="0.9", zorder=-2)
    ax.set_xlim(*WIN)
    ax.set_ylabel("rate (Hz)")
    if k == 0:
        ax.legend(fontsize=6.5, ncol=2, title="direction", title_fontsize=7,
                  loc="upper right", framealpha=0.85)
    if k == 3:
        ax.set_xlabel("time from onset (s)")

    # ---- polar tuning curve with von Mises fit ----
    ax = fig.add_subplot(gs[k, 2], projection="polar")
    th = np.deg2rad(dirs)
    r = tc[:, j]
    ax.errorbar(np.r_[th, th[0]], np.r_[r, r[0]], yerr=np.r_[tc_sem[:, j], tc_sem[0, j]],
                marker="o", ms=4, lw=1.4, color="k", capsize=2)
    popt, hwhm, r2 = ol.fit_von_mises(dirs, r)
    fine = np.linspace(0, 360, 361)
    ax.plot(np.deg2rad(fine), ol.double_von_mises(fine, *popt), color="C3", lw=1.6, alpha=0.85)
    ax.axhline(res.blank_rate.iloc[j], color="C0", ls="--", lw=1, alpha=0.7)
    ax.set_title("pref ori %.0f$\\degree$\nvon Mises HWHM %.0f$\\degree$"
                 % (res.pref_ori.iloc[j], hwhm), fontsize=9, pad=24)
    ax.tick_params(labelsize=7)

fig.suptitle("Orientation tuning of single units in mouse visual cortex "
             "(DANDI:000021, session %s)" % SES, fontsize=13, y=0.955)
plt.savefig("figures/fig02_example_units.png", dpi=140, bbox_inches="tight")
plt.close()

# ===========================================================================
# FIGURE 3: population summary by brain region
# ===========================================================================
order = ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "LGd", "LP", "CA1"]
order = [r for r in order if (res.region == r).sum() >= 10]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
data = [res[(res.region == r) & res.responsive].gosi.values for r in order]
parts = ax.violinplot(data, showmedians=True, widths=0.85)
for pc, r in zip(parts["bodies"], order):
    pc.set_facecolor("C0" if r.startswith("VIS") else ("C1" if r in ol.THALAMUS else "0.6"))
    pc.set_alpha(0.7)
for i, dd in enumerate(data):
    ax.plot(np.random.normal(i + 1, 0.06, len(dd)), dd, ".", ms=2, color="k", alpha=0.35)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("global OSI")
ax.set_title("Orientation selectivity by region\n(visually responsive units)", fontsize=10)

ax = axes[1]
frac = [res[res.region == r].sig_ori.mean() for r in order]
n = [(res.region == r).sum() for r in order]
cols = ["C0" if r.startswith("VIS") else ("C1" if r in ol.THALAMUS else "0.6") for r in order]
ax.bar(range(len(order)), frac, color=cols)
for i, (fr, nn) in enumerate(zip(frac, n)):
    ax.text(i, fr + 0.015, "%d" % nn, ha="center", fontsize=8)
ax.axhline(0.01, color="r", ls="--", lw=1, label="test alpha (0.01)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("fraction orientation-selective")
ax.set_title("Significantly tuned units\n(permutation test, p<0.01)", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
ctx = res[res.region.isin(ol.VISUAL_CORTEX) & res.responsive]
th = res[res.region.isin(ol.THALAMUS) & res.responsive]
ax.plot(th.gosi, th.gdsi, "o", ms=4, color="C1", alpha=0.6, label="thalamus (LGd, LP)")
ax.plot(ctx.gosi, ctx.gdsi, "o", ms=4, color="C0", alpha=0.6, label="visual cortex")
ax.plot([0, 0.8], [0, 0.8], "k--", lw=0.8)
ax.set_xlabel("global OSI")
ax.set_ylabel("global DSI")
ax.set_title("Orientation vs direction selectivity", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig03_population_by_region.png", dpi=140, bbox_inches="tight")
plt.close()

# ===========================================================================
# FIGURE 4: validation - split-half reliability and drifting vs static gratings
# ===========================================================================
sel = res[res.region.isin(ol.VISUAL_CORTEX) & res.sig_ori]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
ax.plot(sel.pref_a, sel.pref_b, "o", ms=4, alpha=0.6, color="C0")
ax.plot([0, 180], [0, 180], "k--", lw=0.8)
ax.set_xlabel("preferred orientation, trial half A (deg)")
ax.set_ylabel("half B (deg)")
cc = stats.circcorrcoef if hasattr(stats, "circcorrcoef") else None
ax.set_title("Split-half reliability\nmedian |$\\Delta$| = %.1f$\\degree$ (chance 45$\\degree$)"
             % sel.split_half_diff.median(), fontsize=10)

ax = axes[1]
ax.plot(sel.pref_ori, sel.sg_pref_ori, "o", ms=4, alpha=0.6, color="C2")
ax.plot([0, 180], [0, 180], "k--", lw=0.8)
ax.set_xlabel("preferred orientation, drifting gratings (deg)")
ax.set_ylabel("static gratings (deg)")
ax.set_title("Cross-stimulus agreement\nmedian |$\\Delta$| = %.1f$\\degree$"
             % sel.ori_diff_dg_sg.median(), fontsize=10)

ax = axes[2]
bins = np.arange(0, 95, 7.5)
ax.hist(sel.ori_diff_dg_sg, bins=bins, color="C2", alpha=0.75,
        label="drifting vs static", density=True)
ax.hist(sel.split_half_diff, bins=bins, histtype="step", lw=2, color="C0",
        label="split-half (drifting)", density=True)
ax.axhline(1 / 90, color="k", ls="--", lw=1, label="chance (uniform)")
ax.set_xlabel("|$\\Delta$ preferred orientation| (deg)")
ax.set_ylabel("density")
ax.set_title("Consistency of preferred orientation", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig04_validation.png", dpi=140, bbox_inches="tight")
plt.close()

# ===========================================================================
# FIGURE 5: cross-validated population tuning
# ===========================================================================
# Each unit's peak direction is estimated from one random half of the trials and
# the curve from the OTHER half is plotted, so a flat unit cannot manufacture a
# peak. Rates are divided by the unit's own mean so untuned cells stay near 1.
groups = [(res.region.isin(ol.VISUAL_CORTEX), "visual cortex"),
          (res.region.isin(ol.THALAMUS), "thalamus (LGd/LP)"),
          (res.region == "CA1", "hippocampus CA1 (control)")]
rel = np.r_[dirs[: len(dirs) // 2 + 1], dirs[len(dirs) // 2 + 1:] - 360]
rel_sorted = np.sort(rel)
order_rel = np.argsort(rel)

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5),
                         gridspec_kw=dict(height_ratios=[2, 1], hspace=0.35, wspace=0.42))
for c, (group, name) in enumerate(groups):
    m = (group & res.responsive).values
    z = tc_b_aligned[:, m]
    z = z / np.where(z.mean(0) == 0, np.nan, z.mean(0))
    z = z[order_rel]
    o = np.argsort(res.gosi.values[m])[::-1]
    ax = axes[0, c]
    im = ax.imshow(z[:, o].T, aspect="auto", cmap="magma", origin="lower",
                   vmin=0.2, vmax=2.0,
                   extent=[rel_sorted[0] - 22.5, rel_sorted[-1] + 22.5, 0, m.sum()])
    ax.set_xticks(rel_sorted)
    ax.set_ylabel("unit (sorted by gOSI)")
    ax.set_title("%s (n=%d responsive)" % (name, m.sum()), fontsize=10)
    ax.set_xlabel("direction relative to preferred (deg)", fontsize=8)
    if c == 2:
        plt.colorbar(im, ax=ax, label="rate / mean rate")

    ax = axes[1, c]
    mu = np.nanmean(z, axis=1)
    se = np.nanstd(z, axis=1) / np.sqrt(np.isfinite(z).sum(1))
    ax.errorbar(rel_sorted, mu, yerr=se, marker="o", ms=4, color="C3", capsize=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylim(0.3, 2.3)
    ax.set_xticks(rel_sorted)
    ax.set_xlabel("direction relative to preferred (deg)")
    if c == 0:
        ax.set_ylabel("population mean\n(rate / mean rate)")
fig.suptitle("Cross-validated tuning: peak direction from trial half A, "
             "response from half B", fontsize=12, y=0.97)
plt.savefig("figures/fig05_population_heatmap.png", dpi=140, bbox_inches="tight")
plt.close()
print("figures written")
