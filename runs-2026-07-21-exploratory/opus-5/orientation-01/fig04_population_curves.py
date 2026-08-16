"""Figure 4: population tuning-curve structure and cross-stimulus consistency."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import plotting as pl

ALPHA = 0.01
with open("results_selectivity.pkl", "rb") as fh:
    res = pickle.load(fh)
p = pd.read_pickle("results_pooled_units.pkl")

# Pool the static-grating orientation tuning curves (6 orientations) across sessions.
angles = res[next(iter(res))]["sg_angles"]
curves = np.concatenate([r["sg_curves"] for r in res.values()], axis=1).T  # (units, angles)
assert len(curves) == len(p)


def sorted_block(mask):
    """Peak-normalised tuning curves, rows sorted by preferred orientation."""
    c = curves[mask]
    c = c - c.min(axis=1, keepdims=True)
    denom = c.max(axis=1, keepdims=True)
    c = np.divide(c, denom, out=np.zeros_like(c), where=denom > 0)
    order = np.argsort(p.loc[mask, "sg_pref_ori"].values)
    return c[order]


def aligned_mean(mask):
    """Tuning curves circularly shifted so each unit's peak sits at 0 deg."""
    c = curves[mask]
    base = c.min(axis=1, keepdims=True)
    rng = c.max(axis=1, keepdims=True) - base
    cn = np.divide(c - base, rng, out=np.zeros_like(c), where=rng > 0)
    shifted = np.array([np.roll(row, -np.argmax(row)) for row in cn])
    return shifted


fig = plt.figure(figsize=(13.5, 9))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.34, left=0.07, right=0.95, top=0.90, bottom=0.09)

# --- A/B: heat maps of normalised tuning curves ---------------------------
for k, (grp, letter) in enumerate([("visual cortex", "A"), ("hippocampus", "B")]):
    ax = fig.add_subplot(gs[0, k])
    mask = ((p.region_group == grp) & (p.sg_p_gOSI < ALPHA)).values
    block = sorted_block(mask)
    im = ax.imshow(block, aspect="auto", cmap="magma", origin="lower",
                   extent=[angles[0] - 15, angles[-1] + 15, 0, len(block)], vmin=0, vmax=1)
    ax.set_xticks(angles)
    ax.set_xlabel("grating orientation (deg)")
    ax.set_ylabel("unit (sorted by preferred orientation)")
    ax.set_title(f"{letter}  {grp}: significantly tuned units\n(n={mask.sum()} of "
                 f"{(p.region_group == grp).sum()}), peak-normalised", loc="left")
    plt.colorbar(im, ax=ax, label="normalised rate", fraction=0.046)

# --- C: split-half aligned average tuning curve ---------------------------
# Preferred orientation is chosen on half the trials and the curve is read out on
# the other half, so an untuned population stays flat instead of being forced
# into an apparent peak.
sh = pd.read_pickle("results_split_half.pkl")
ax = fig.add_subplot(gs[0, 2])
n_ang = sh["curves"].shape[1]
offs = np.concatenate([np.arange(n_ang) * 30, [180]])
offs = np.where(offs > 90, offs - 180, offs)
order = np.argsort(offs[:-1])
for grp in pl.REGION_ORDER:
    mask = (sh["meta"].region_group == grp).values
    c = sh["curves"][mask]
    mu = np.nanmean(c, axis=0)[order]
    se = (np.nanstd(c, axis=0) / np.sqrt(np.sum(~np.isnan(c), axis=0)))[order]
    x = offs[:-1][order]
    ax.plot(x, mu, "-o", ms=3, color=pl.REGION_COLORS[grp], lw=1.8,
            label=f"{grp} (n={mask.sum()})")
    ax.fill_between(x, mu - se, mu + se, color=pl.REGION_COLORS[grp], alpha=0.25, lw=0)
ax.axhline(1.0, color="k", ls=":", lw=0.8)
ax.set_xlabel("orientation relative to preferred (deg)")
ax.set_ylabel("rate / unit mean rate")
ax.legend(fontsize=7.5, loc="lower center")
ax.set_title("C  Split-half tuning curve (all units)\npreferred orientation from half the trials,\n"
             "curve measured on the held-out half", loc="left")

# --- D: drifting vs static gOSI -------------------------------------------
ax = fig.add_subplot(gs[1, 0])
for grp in ["hippocampus", "visual thalamus", "visual cortex"]:
    d = p[p.region_group == grp]
    ax.scatter(d.dg_gOSI, d.sg_gOSI, s=4, alpha=0.35, lw=0, color=pl.REGION_COLORS[grp], label=grp)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
cx = p[p.region_group == "visual cortex"].dropna(subset=["dg_gOSI", "sg_gOSI"])
r = stats.spearmanr(cx.dg_gOSI, cx.sg_gOSI)
ax.set_xlabel("gOSI, drifting gratings")
ax.set_ylabel("gOSI, static gratings")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=7.5, markerscale=2.5)
ax.set_title(f"D  Selectivity is consistent across stimuli\ncortex Spearman ρ={r.statistic:.2f} "
             f"(p={r.pvalue:.1e})", loc="left")

# --- E: preferred orientation agreement between stimuli -------------------
ax = fig.add_subplot(gs[1, 1])
both = p[(p.region_group == "visual cortex") & (p.dg_p_gOSI < ALPHA) & (p.sg_p_gOSI < ALPHA)]
ax.scatter(both.dg_pref_ori, both.sg_pref_ori, s=6, alpha=0.4, lw=0,
           color=pl.REGION_COLORS["visual cortex"])
ax.plot([0, 180], [0, 180], "k--", lw=0.8)
ax.set_xlabel("preferred orientation, drifting (deg)")
ax.set_ylabel("preferred orientation, static (deg)")
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_yticks([0, 45, 90, 135, 180])
delta = np.mod(both.sg_pref_ori - both.dg_pref_ori + 90, 180) - 90
med_abs = np.median(np.abs(delta))
# circular correlation of the doubled angles
a = np.deg2rad(2 * both.dg_pref_ori.values)
b = np.deg2rad(2 * both.sg_pref_ori.values)
am, bm = np.angle(np.mean(np.exp(1j * a))), np.angle(np.mean(np.exp(1j * b)))
rho = (np.sum(np.sin(a - am) * np.sin(b - bm))
       / np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2)))
ax.set_title(f"E  Preferred orientation agrees across stimuli\n"
             f"n={len(both)}, circular r={rho:.2f}, median |Δ|={med_abs:.0f}°", loc="left")

# --- F: |Δ preferred orientation| distribution vs chance ------------------
ax = fig.add_subplot(gs[1, 2])
ax.hist(np.abs(delta), bins=np.arange(0, 95, 7.5), color=pl.REGION_COLORS["visual cortex"],
        density=True, label="observed")
ax.axhline(1 / 90, color="k", ls="--", lw=1.2, label="uniform (chance)")
ax.set_xlabel("|Δ preferred orientation| between stimuli (deg)")
ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_title("F  Cross-stimulus reproducibility of\npreferred orientation", loc="left")

fig.suptitle("Population structure of orientation tuning (static gratings, "
             f"{p.session_id.nunique()} sessions)", y=0.965, fontsize=12)
fig.savefig("fig04_population_curves.png", bbox_inches="tight")
print("wrote fig04_population_curves.png")
