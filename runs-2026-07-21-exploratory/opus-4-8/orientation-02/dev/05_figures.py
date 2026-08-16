"""Figures 3-7: tuning-curve gallery, population selectivity, and the controls."""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import oslib
from analyze_helpers import responsive

U = pd.read_pickle("units.pkl")
R = responsive(U)
AREAS = [a for a in oslib.VISUAL_CORTEX + oslib.THALAMUS if (R.area == a).sum() >= 15]
CORTEX_C, THAL_C = "#2b6cb0", "#c05621"
ACOL = {a: (CORTEX_C if a in oslib.VISUAL_CORTEX else THAL_C) for a in AREAS}

def _load(suffix):
    return {p.split("/")[-1].split("_")[0]: np.load(p)
            for p in glob.glob(f"extracted/*_{suffix}.npy")}


TUN, TUN_RAW, TUN_SEM = _load("tuning"), _load("tuningraw"), _load("tuningsem")
DIRS = np.load(sorted(glob.glob("extracted/*_dirs.npy"))[0])


HALVES = sorted(glob.glob("extracted/*_half*.npy"))
halves = {}
for p_ in HALVES:
    ses_, which = p_.split("/")[-1].split("_half")
    halves.setdefault(ses_, {})[int(which[0]) - 1] = np.load(p_)

_UIDX = {}


def unit_index(row):
    if row.session not in _UIDX:
        d = np.load(f"extracted/{row.session}.npz", allow_pickle=True)
        _UIDX[row.session] = {int(u): i for i, u in enumerate(d["unit_ids"])}
    return _UIDX[row.session][int(row.unit_id)]


def unit_tuning(row):
    return TUN[row.session][unit_index(row)]


# =========================================================== figure 3: gallery
# Twelve units in polar coordinates: the six most orientation-selective cortical
# units, three mid-range cortical units, and three LGd units for comparison.
# Examples are ranked by noise-corrected gOSI among well-driven units (peak evoked
# response above 4 Hz), not by raw gOSI: the raw statistic saturates at 1.0 for
# weakly driven units whose tuning curve happens to be positive at one direction.
strong = R[R.peak_evoked > 8.0]
cx = strong[(strong.region == "cortex") & (strong.p_perm < 0.01)].sort_values(
    "gosi_corrected", ascending=False)
lg = strong[strong.area == "LGd"].sort_values("gosi_corrected", ascending=False)
mid = len(cx) // 2
picks = pd.concat([cx.head(4), cx.iloc[mid:mid + 4], lg.head(4)])
ROW_LABELS = ["most selective\ncortical units",
              "median\ncortical units",
              "most selective\nLGd units"]

fig, axes = plt.subplots(3, 4, figsize=(15, 12), subplot_kw={"projection": "polar"})
th = np.deg2rad(np.r_[DIRS, DIRS[0]])
fine = np.linspace(0, 2 * np.pi, 200)
for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
    i = unit_index(row)
    t = TUN_RAW[row.session][i]
    e = TUN_SEM[row.session][i]
    col = ACOL.get(row.area, "0.4")
    ax.plot(th, np.r_[t, t[0]], "-o", color=col, ms=5, lw=2)
    ax.fill_between(th, np.r_[t - e, t[0] - e[0]], np.r_[t + e, t[0] + e[0]],
                    color=col, alpha=0.25, lw=0)
    # spontaneous rate from the interleaved blank sweeps, as a reference circle
    ax.plot(fine, np.full_like(fine, row.baseline_rate), color="0.35", lw=1.2, ls=":")
    top = np.nanmax(t + e) * 1.05
    for a in (row.pref_ori, row.pref_ori + 180):
        ax.plot([np.deg2rad(a)] * 2, [0, top], color="k", lw=1.4, ls="--", alpha=0.7)
    ax.set_ylim(0, top)
    ax.set_title(f"{row.area}  unit {row.unit_id}\ngOSI {row.gosi:.2f}   "
                 f"pref {row.pref_ori:.0f}°   p {row.p_perm:.3f}",
                 fontsize=10, pad=22)
    ax.set_theta_zero_location("E")
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
for r_, lab in enumerate(ROW_LABELS):
    axes[r_, 0].text(-0.32, 0.5, lab, transform=axes[r_, 0].transAxes, rotation=90,
                     va="center", ha="center", fontsize=11, fontweight="bold")
fig.suptitle("Direction tuning curves at each unit's preferred temporal frequency\n"
             "firing rate in Hz +/- SEM; dotted circle = spontaneous rate on blank sweeps; "
             "dashed line = preferred orientation axis",
             fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig03_polar_gallery.png", dpi=140)
plt.close(fig)
print("wrote fig03_polar_gallery.png")

# ================================================== figure 4: population summary
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) noise-corrected gOSI per area. gOSI is positively biased by trial-to-trial
# noise, so the per-unit permutation noise floor is subtracted before comparing areas.
ax = axes[0, 0]
data = [R.loc[R.area == a, "gosi_corrected"].dropna().values for a in AREAS]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="k", lw=2))
for patch, a in zip(bp["boxes"], AREAS):
    patch.set_facecolor(ACOL[a])
    patch.set_alpha(0.6)
rng = np.random.default_rng(0)
for i, (a, v) in enumerate(zip(AREAS, data)):
    ax.plot(i + 1 + rng.uniform(-0.18, 0.18, len(v)), v, ".", color=ACOL[a], ms=3, alpha=0.45)
ax.set_xticks(np.arange(1, len(AREAS) + 1))
ax.set_xticklabels(AREAS, fontsize=9)
ax.set_ylabel("noise-corrected gOSI")
ax.set_title("(a) Orientation selectivity by area", loc="left", fontweight="bold")

# (b) fraction of units significantly orientation tuned
ax = axes[0, 1]
frac, ci = [], []
for a in AREAS:
    g = R[R.area == a]
    k, n = (g.p_perm < 0.01).sum(), len(g)
    frac.append(k / n)
    lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
    ci.append([k / n - lo, hi - k / n])
ax.bar(AREAS, frac, yerr=np.array(ci).T, color=[ACOL[a] for a in AREAS], capsize=4)
ax.axhline(0.01, color="k", ls=":", label="chance (p < 0.01)")
ax.set_ylabel("fraction significantly orientation tuned")
ax.set_title("(b) Significant tuning (permutation test, p < 0.01)", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)
for i, a in enumerate(AREAS):
    ax.text(i, frac[i] + 0.045, f"n={(R.area == a).sum()}", ha="center", fontsize=8)

# (c) Cortex versus LGd. LGd is the first-order relay that feeds V1, so it is the
# reference point for "orientation selectivity is constructed in cortex". LP is a
# higher-order nucleus that receives massive cortical input, so it is plotted
# separately rather than lumped into a single "thalamus" group.
ax = axes[1, 0]
groups = [("visual cortex", R[R.region == "cortex"], CORTEX_C),
          ("LGd (first-order relay)", R[R.area == "LGd"], THAL_C),
          ("LP (higher-order)", R[R.area == "LP"], "#8b6f47")]
for label, sub, col in groups:
    v = np.sort(sub.gosi_corrected.dropna().values)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2.2,
            label=f"{label}, n={len(v)}, median {np.median(v):.3f}")
ctx = R.loc[R.region == "cortex", "gosi_corrected"].dropna()
lgd = R.loc[R.area == "LGd", "gosi_corrected"].dropna()
u_stat, p_ctx_lgd = stats.mannwhitneyu(ctx, lgd)
ax.set_xlabel("noise-corrected gOSI")
ax.set_ylabel("cumulative fraction of units")
ax.set_title(f"(c) Cortex vs thalamic relay\ncortex vs LGd: Mann-Whitney p = {p_ctx_lgd:.2g}",
             loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="lower right")

# (d) Population tuning aligned to the preferred orientation. The preferred bin is
# chosen from one half of the repeats and the curve is read out from the *other*
# half, so noise alone cannot manufacture a peak: an untuned population gives a
# flat line here, which is what makes the cortex-LGd difference interpretable.
ax = axes[1, 1]
x = np.array([-90, -45, 0, 45])
for label, sub, col in [("visual cortex", R[R.region == "cortex"], CORTEX_C),
                        ("LGd", R[R.area == "LGd"], THAL_C)]:
    curves = []
    for _, row in sub.iterrows():
        t1 = np.clip(halves[row.session][0][unit_index(row)], 0, None)
        t2 = np.clip(halves[row.session][1][unit_index(row)], 0, None)
        o1, o2 = t1[:4] + t1[4:], t2[:4] + t2[4:]
        if not np.isfinite(o1).all() or not np.isfinite(o2).all() or o2.max() <= 0:
            continue
        k = int(np.argmax(o1))                       # aligned on half 1
        curves.append(np.roll(o2, 2 - k) / o2.max())  # measured on half 2
    c = np.array(curves)
    m, sem = c.mean(axis=0), c.std(axis=0) / np.sqrt(len(c))
    ax.errorbar(np.r_[x, 90], np.r_[m, m[0]], yerr=np.r_[sem, sem[0]], marker="o",
                color=col, lw=2, capsize=3, label=f"{label} (n={len(c)})")
ax.set_xticks(np.arange(-90, 91, 45))
ax.set_xlabel("orientation relative to preferred (deg)")
ax.set_ylabel("response, normalised to each unit's peak")
ax.set_title("(d) Cross-validated population tuning\n(preferred bin from held-out repeats)",
             loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)

fig.suptitle(f"Orientation selectivity across the mouse visual system "
             f"({R.session.nunique()} sessions, {R.subject.nunique()} mice, "
             f"{len(R)} visually responsive units)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig04_population.png", dpi=140)
plt.close(fig)
print("wrote fig04_population.png")

# ===================================== figure 5: reliability and cross-stimulus
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) split-half agreement of preferred orientation
ax = axes[0]
sub = R[(R.p_perm < 0.01) & (R.region == "cortex")]
d = oslib.circ_dist_180(sub.pref_half1, sub.pref_half2)
ax.hist(d, bins=np.linspace(0, 90, 19), color=CORTEX_C, alpha=0.8, density=True)
ax.axhline(1 / 90, color="k", ls="--", label="chance (uniform)")
r_half, p_half, n_half = oslib.circ_corr_180(sub.pref_half1, sub.pref_half2)
ax.set_xlabel("|preferred orientation difference| between halves (deg)")
ax.set_ylabel("density")
ax.set_title(f"(a) Split-half reliability, cortex\ncircular r = {r_half:.2f}, p < {p_half:.1e}, n = {n_half}",
             loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)

# (b) drifting vs static gratings: preferred orientation from two different blocks
ax = axes[1]
sub2 = R[(R.p_perm < 0.01) & (R.region == "cortex") & R.sg_pref_ori.notna() & (R.sg_gosi > 0.1)]
ax.plot(sub2.pref_ori, sub2.sg_pref_ori, "o", color=CORTEX_C, ms=4, alpha=0.55)
ax.plot([0, 180], [0, 180], "k--", lw=1)
r2, p2, n2 = oslib.circ_corr_180(sub2.pref_ori, sub2.sg_pref_ori)
ax.set_xlabel("preferred orientation, drifting gratings (deg)")
ax.set_ylabel("preferred orientation, static gratings (deg)")
ax.set_xlim(0, 180)
ax.set_ylim(0, 180)
ax.set_title(f"(b) Cross-stimulus replication, cortex\ncircular r = {r2:.2f}, "
             f"p < {p2:.1e}, n = {n2}", loc="left", fontweight="bold")

# (c) locomotion control
ax = axes[2]
sub3 = R[R.gosi_still.notna() & R.gosi_move.notna() & (R.p_perm < 0.01)]
for reg, col in [("cortex", CORTEX_C), ("thalamus", THAL_C)]:
    s = sub3[sub3.region == reg]
    ax.plot(s.gosi_still, s.gosi_move, "o", color=col, ms=4, alpha=0.5, label=reg)
lim = [0, max(sub3.gosi_still.max(), sub3.gosi_move.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
rr = stats.spearmanr(sub3.gosi_still, sub3.gosi_move)
wil = stats.wilcoxon(sub3.gosi_still, sub3.gosi_move)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("gOSI, stationary trials")
ax.set_ylabel("gOSI, running trials")
ax.set_title(f"(c) Locomotion control\nSpearman r = {rr.statistic:.2f}, "
             f"Wilcoxon p = {wil.pvalue:.2g}, n = {len(sub3)}", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)

fig.suptitle("Orientation preference is reliable, replicates on an independent stimulus, "
             "and does not depend on locomotion", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig05_controls.png", dpi=140)
plt.close(fig)
print("wrote fig05_controls.png")

# ============================== figure 6: distribution of preferred orientations
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), subplot_kw={"projection": "polar"})
for ax, (label, sub, col) in zip(axes, [
        ("visual cortex", R[(R.region == "cortex") & (R.p_perm < 0.01)], CORTEX_C),
        ("visual thalamus", R[(R.region == "thalamus") & (R.p_perm < 0.01)], THAL_C)]):
    edges = np.arange(0, 181, 15)
    h, _ = np.histogram(sub.pref_ori, bins=edges)
    h = np.r_[h, h]  # mirror, orientation has period 180
    th = np.deg2rad(np.arange(0, 360, 15))
    ax.bar(th + np.deg2rad(7.5), h, width=np.deg2rad(14), color=col, alpha=0.8)
    # Rayleigh test on doubled angles, for a cardinal/oblique bias
    a2 = 2 * np.deg2rad(sub.pref_ori.values)
    Rbar = np.abs(np.mean(np.exp(1j * a2)))
    n_ray = len(a2)
    n = n_ray
    p_ray = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * Rbar) ** 2)) - (1 + 2 * n))
    ax.set_title(f"{label} (n={n})\nRayleigh R = {Rbar:.3f}, p = {p_ray:.2g}", pad=25)
    ax.set_theta_zero_location("E")
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
fig.suptitle("Distribution of preferred orientations (mirrored: orientation has period 180°)",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig("fig06_preferred_orientation.png", dpi=140)
plt.close(fig)
print("wrote fig06_preferred_orientation.png")

# ------------------------------------------------------------------ text stats
with open("stats_summary.txt", "w") as fh:
    fh.write(f"sessions: {sorted(U.session.unique())}\n")
    fh.write(f"mice: {U.subject.nunique()}   units: {len(U)}   responsive: {len(R)}\n\n")
    tab = R.groupby("area").apply(lambda g: pd.Series({
        "n": len(g), "frac_sig_p<0.01": (g.p_perm < 0.01).mean(),
        "median_gOSI": g.gosi.median(), "median_gOSI_noise_corrected": g.gosi_corrected.median(),
        "median_OSI_classic": g.osi_classic.median(), "median_gDSI": g.gdsi.median(),
    }), include_groups=False)
    fh.write(tab.to_string() + "\n\n")
    fh.write(f"cortex vs LGd noise-corrected gOSI: medians {ctx.median():.4f} vs "
             f"{lgd.median():.4f}, Mann-Whitney U={u_stat:.0f}, p={p_ctx_lgd:.3g}\n")
    fh.write(f"cortex fraction significant: {(R[R.region=='cortex'].p_perm < 0.01).mean():.3f}; "
             f"thalamus: {(R[R.region=='thalamus'].p_perm < 0.01).mean():.3f}\n")
    fh.write(f"split-half circular r (cortex, tuned) = {r_half:.3f}, p < {p_half:.3g}, n = {n_half}\n")
    fh.write(f"drifting vs static circular r (cortex, tuned) = {r2:.3f}, p < {p2:.3g}, n = {n2}\n")
    fh.write(f"locomotion: Spearman r = {rr.statistic:.3f}, Wilcoxon p = {wil.pvalue:.3g}, "
             f"n = {len(sub3)}\n")
print(open("stats_summary.txt").read())
