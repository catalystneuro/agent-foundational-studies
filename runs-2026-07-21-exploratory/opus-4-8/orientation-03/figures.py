"""Figures for the orientation-selectivity analysis of DANDI:000021.

Each function takes the per-unit results table (``df``) produced by
``orientation_lib.analyze_session`` and, where raw trial data are needed, the
``raw`` dict returned by the same function with ``keep_raw=True``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import orientation_lib as ol

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

GROUPS = ["VISp", "higher visual", "LGd"]
GCOL = {"VISp": "C0", "higher visual": "C2", "LGd": "C3"}


def _unpack(df, raw):
    """Common locals for the single-session figures."""
    unit_ids = raw["unit_ids"]
    col = {u: i for i, u in enumerate(unit_ids)}
    resp = df.responsive.values
    v1 = (df.area == "VISp").values
    cand = df[resp & v1].sort_values("gOSI_corrected", ascending=False)
    return col, resp, v1, cand

# --------------------------------------------------------------------------- #
# Figure 1 - raw data: population raster around grating onsets
# --------------------------------------------------------------------------- #
def fig_raw_activity(df, raw, fname="fig01_raw_activity.png"):
    col, resp, v1, cand = _unpack(df, raw)
    uniq, mean, sem = raw["uniq"], raw["mean"], raw["sem"]
    rates, ang, tf = raw["rates"], raw["ang"], raw["tf"]
    spikes, drive = raw["spikes"], raw["drive"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))
    t0 = drive.start_time.values[0] - 1.5
    t1 = drive.start_time.values[12]
    ep = nap.IntervalSet(start=t0, end=t1)

    area_order = {a: i for i, a in enumerate(["LGd"] + ol.VISUAL_AREAS)}
    ordering = df[resp].assign(k=lambda d: d.area.map(area_order))
    order = ordering.sort_values(["k", "unit_id"]).unit_id.values
    for row, uid in enumerate(order):
        t = spikes[int(uid)].restrict(ep).t
        axes[0].plot(t, np.full_like(t, row), "|", color="k", ms=1.6, mew=0.5)
    areas = df.set_index("unit_id").loc[order, "area"].values
    bounds = np.where(areas[1:] != areas[:-1])[0]
    for b in bounds:
        axes[0].axhline(b + 0.5, color="0.6", lw=0.6)
    for a in np.unique(areas):
        idx = np.where(areas == a)[0]
        axes[0].text(t1 + 0.4, idx.mean(), a, va="center", fontsize=8)
    axes[0].set_ylabel("unit (grouped by area)")
    axes[0].set_ylim(-1, len(order))
    axes[0].set_title("Spiking during 12 consecutive drifting-grating presentations "
                      "(session 715093703)\nshaded bars = grating on, number above = "
                      "drift direction (deg)", fontsize=10, pad=18)

    sub = drive[(drive.start_time >= t0) & (drive.stop_time <= t1)]
    cmap = plt.get_cmap("hsv")
    for _, s in sub.iterrows():
        c = cmap(s.orientation / 360)
        for ax in axes:
            ax.axvspan(s.start_time, s.stop_time, color=c, alpha=0.22, lw=0)
        axes[0].text(0.5 * (s.start_time + s.stop_time), len(order) + 3,
                     f"{int(s.orientation)}", ha="center", fontsize=7.5)
    axes[0].set_ylim(-1, len(order) + 12)

    pop = spikes.count(0.02, ep=ep).sum(axis=1) / (0.02 * len(spikes))
    axes[1].plot(pop.t, np.asarray(pop.values), color="C3", lw=0.8)
    axes[1].set_ylabel("population\nrate (Hz)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_xlim(t0, t1)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


# --------------------------------------------------------------------------- #
# Figure 2 - example direction tuning curves
# --------------------------------------------------------------------------- #
def fig_example_tuning(df, raw, fname="fig02_example_tuning_curves.png", n=12):
    col, resp, v1, cand = _unpack(df, raw)
    uniq, mean, sem = raw["uniq"], raw["mean"], raw["sem"]

    ex = cand.head(n)
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    x = np.append(uniq, 360)
    for ax, (_, u) in zip(axes.ravel(), ex.iterrows()):
        j = col[u.unit_id]
        m = np.append(mean[:, j], mean[0, j])
        s = np.append(sem[:, j], sem[0, j])
        ax.errorbar(x, m, yerr=s, marker="o", ms=3.5, lw=1.4, color="C0", capsize=2)
        ax.axhline(u.blank_rate, color="0.5", ls="--", lw=1)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlim(-15, 375)
        ax.set_title(f"unit {u.unit_id} ({u.area})\ngOSI={u.gOSI:.2f}  DSI={u.DSI:.2f}",
                     fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("drift direction (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("firing rate (Hz)")
    fig.suptitle("Direction tuning of the 12 most orientation-selective V1 units "
                 "(mean +/- SEM over 75 trials; dashed line = blank-screen rate)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


# --------------------------------------------------------------------------- #
# Figure 3 - rasters + polar tuning for three example units
# --------------------------------------------------------------------------- #
def fig_raster_polar(df, raw, fname="fig03_raster_and_polar.png"):
    col, resp, v1, cand = _unpack(df, raw)
    uniq, mean, sem = raw["uniq"], raw["mean"], raw["sem"]
    rates, ang, tf = raw["rates"], raw["ang"], raw["tf"]
    spikes, drive = raw["spikes"], raw["drive"]

    picks = list(cand.unit_id.values[:2]) + [
        df[resp & (df.glm_label == "direction") & v1].sort_values("gDSI_corrected",
                                                                  ascending=False).unit_id.values[0]
    ]
    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 1], hspace=0.45, wspace=0.35)
    cmap = plt.get_cmap("hsv")
    for r, uid in enumerate(picks):
        j = col[uid]
        row = df[df.unit_id == uid].iloc[0]
        axr = fig.add_subplot(gs[r, 0])
        y = 0
        keep = tf == row.pref_tf   # one temporal frequency, 15 trials per direction
        for a in uniq:
            idx = np.where((ang == a) & keep)[0]
            for i in idx:
                t = spikes[int(uid)].t
                st = drive.start_time.values[i]
                sel_t = t[(t > st - 0.5) & (t < st + 2.5)] - st
                axr.plot(sel_t, np.full_like(sel_t, y), "|", ms=4.0, mew=0.9,
                         color=cmap(a / 360))
                y += 1
            axr.axhline(y - 0.5, color="0.85", lw=0.5)
            axr.text(2.62, y - len(idx) / 2, f"{int(a)}", va="center", fontsize=8,
                     color=cmap(a / 360))
        axr.axvspan(0, 2, color="0.9", zorder=0)
        axr.set_xlim(-0.5, 2.55)
        axr.set_ylim(-1, y)
        axr.set_ylabel("trial (grouped by direction)")
        axr.set_title(f"unit {uid} ({row.area}) - raster at preferred temporal frequency "
                      f"({row.pref_tf:g} Hz)", fontsize=9)
        if r == 2:
            axr.set_xlabel("time from grating onset (s)")

        axp = fig.add_subplot(gs[r, 1], projection="polar")
        th = np.deg2rad(np.append(uniq, uniq[0]))
        axp.plot(th, np.append(mean[:, j], mean[0, j]), "o-", color="C0", ms=3)
        axp.fill(th, np.append(mean[:, j], mean[0, j]), color="C0", alpha=0.2)
        axp.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        axp.tick_params(labelsize=7, pad=1)
        axp.set_rlabel_position(112)
        axp.set_yticklabels([])
        axp.set_title(f"gOSI={row.gOSI:.2f}\nDSI={row.DSI:.2f}", fontsize=8.5, pad=18)

        axh = fig.add_subplot(gs[r, 2])
        tfs = np.unique(tf)
        mat = np.stack([[rates[(ang == a) & (tf == t), j].mean() for t in tfs] for a in uniq])
        im = axh.imshow(mat, aspect="auto", origin="lower", cmap="magma")
        axh.set_xticks(range(len(tfs)), [f"{t:g}" for t in tfs], fontsize=7)
        axh.set_yticks(range(len(uniq)), [f"{int(a)}" for a in uniq], fontsize=7)
        axh.set_xlabel("temporal freq (Hz)", fontsize=8)
        axh.set_ylabel("direction (deg)", fontsize=8)
        axh.set_title("rate (Hz)", fontsize=8.5)
        fig.colorbar(im, ax=axh, fraction=0.05)
    fig.suptitle("Orientation selectivity is visible in the raw spike trains", y=0.995)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)



def fig_population_selectivity(df, fname="fig05_population_selectivity.png"):
    r = df[df.responsive]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    v1 = r[r.group == "VISp"]
    bins = np.linspace(0, 1, 31)
    ax.hist(v1.gOSI, bins=bins, color="C0", alpha=0.8, label="observed")
    ax.hist(v1.gOSI_null, bins=bins, color="0.5", alpha=0.7,
            label="shuffled stimulus labels")
    ax.set_xlabel("gOSI"), ax.set_ylabel("V1 units")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"V1 (n={len(v1)}): observed vs chance selectivity", fontsize=9.5)

    ax = axes[0, 1]
    for g in GROUPS:
        x = np.sort(r[r.group == g].gOSI_corrected.dropna())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=GCOL[g],
                label=f"{g} (n={len(x)})")
    ax.set_xlabel("noise-corrected gOSI"), ax.set_ylabel("cumulative fraction")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Selectivity distribution by region", fontsize=9.5)

    ax = axes[1, 0]
    areas = [a for a in ["VISp"] + ol.VISUAL_AREAS[1:] + ol.CONTROL_AREAS
             if (r.area == a).sum() >= 10]
    data = [r.loc[r.area == a, "gOSI_corrected"].dropna().values for a in areas]
    bp = ax.boxplot(data, labels=areas, showfliers=False, patch_artist=True, widths=0.6)
    for patch, a in zip(bp["boxes"], areas):
        patch.set_facecolor("C3" if a in ol.CONTROL_AREAS else "C0")
        patch.set_alpha(0.55)
    for i, d in enumerate(data):
        ax.plot(np.random.default_rng(i).normal(i + 1, 0.07, len(d)), d, ".",
                color="0.25", ms=2.5, alpha=0.55)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_ylabel("noise-corrected gOSI")
    ax.set_title("Orientation selectivity by brain region", fontsize=9.5)

    ax = axes[1, 1]
    frac = r.groupby("area").apply(lambda x: (x.p_OSI < 0.05).mean())
    n = r.groupby("area").size()
    frac, n = frac.loc[areas], n.loc[areas]
    ax.bar(range(len(areas)), frac.values,
           color=["C3" if a in ol.CONTROL_AREAS else "C0" for a in areas], alpha=0.75)
    for i, (f, k) in enumerate(zip(frac.values, n.values)):
        ax.text(i, f + 0.02, f"{k}", ha="center", fontsize=7.5, color="0.35")
    ax.set_xticks(range(len(areas)), areas)
    ax.axhline(0.05, color="k", ls=":", lw=1)
    ax.text(len(areas) - 0.6, 0.075, "chance", fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction with p(gOSI) < 0.05")
    ax.set_title("Fraction with statistically reliable orientation tuning\n"
                 "(number above bar = n units)", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


def fig_reliability(df, fname="fig06_tuning_reliability.png"):
    r = df[df.responsive & df.sg_pref_ori.notna() & (df.sg_peak_rate > 1.0)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))

    ax = axes[0, 0]
    v1 = r[r.group == "VISp"]
    ax.plot(v1.half1_ori, v1.half2_ori, "o", ms=4, alpha=0.6, color="C0")
    ax.plot([0, 180], [0, 180], "k--", lw=0.8)
    ax.set_xlabel("preferred orientation, trial half 1 (deg)")
    ax.set_ylabel("preferred orientation, half 2 (deg)")
    ax.set_title(f"V1 split-half reliability (n={len(v1)})\n"
                 "orientation wraps at 180$\\degree$, so the opposite corners also agree",
                 fontsize=9.5)
    ax.set_xlim(0, 180), ax.set_ylim(0, 180)

    ax = axes[0, 1]
    bins = np.linspace(0, 90, 31)
    for g in GROUPS:
        e = ol.circular_distance(r.loc[r.group == g, "half1_ori"],
                                 r.loc[r.group == g, "half2_ori"], 180)
        ax.hist(e, bins=bins, histtype="step", lw=1.6, color=GCOL[g], density=True,
                label=f"{g} (median {np.median(e):.1f}°)")
    ax.axhline(1 / 90, color="k", ls=":", lw=1)
    ax.text(52, 1 / 90 + 0.001, "chance (uniform)", fontsize=7.5)
    ax.set_xlabel("split-half difference in preferred orientation (deg)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Within-stimulus reliability", fontsize=9.5)

    ax = axes[1, 0]
    ax.plot(v1.pref_ori, v1.sg_pref_ori, "o", ms=4, alpha=0.6, color="C0")
    ax.plot([0, 180], [0, 180], "k--", lw=0.8)
    ax.set_xlabel("preferred orientation, drifting gratings (deg)")
    ax.set_ylabel("preferred orientation, static gratings (deg)")
    err = ol.circular_distance(v1.pref_ori, v1.sg_pref_ori, 180)
    ax.set_title(f"V1 generalisation across stimuli (median error {np.median(err):.1f}°)",
                 fontsize=9.5)
    ax.set_xlim(0, 180), ax.set_ylim(0, 180)

    ax = axes[1, 1]
    for g in GROUPS:
        e = ol.circular_distance(r.loc[r.group == g, "pref_ori"],
                                 r.loc[r.group == g, "sg_pref_ori"], 180)
        ax.hist(e, bins=bins, histtype="step", lw=1.6, color=GCOL[g], density=True,
                label=f"{g} (median {np.median(e):.1f}°)")
    ax.axhline(1 / 90, color="k", ls=":", lw=1)
    ax.text(52, 1 / 90 + 0.001, "chance (uniform)", fontsize=7.5)
    ax.set_xlabel("drifting vs static difference in preferred orientation (deg)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cross-stimulus generalisation", fontsize=9.5)

    fig.suptitle("Preferred orientation is a stable property of the neuron, not noise",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


def fig_ori_vs_dir(df, fname="fig07_orientation_vs_direction.png"):
    r = df[df.responsive].copy()
    r["d_ori"] = r.ll_orientation - r.ll_constant
    r["d_dir"] = r.ll_direction - r.ll_orientation

    fig = plt.figure(figsize=(11.5, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for g in GROUPS[::-1]:
        sub = r[r.group == g]
        ax.plot(sub.gOSI_corrected, sub.gDSI_corrected, "o", ms=3, alpha=0.45,
                color=GCOL[g], label=g)
    lim = np.nanpercentile(r.gOSI_corrected, 99.5)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("noise-corrected gOSI"), ax.set_ylabel("noise-corrected gDSI")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Orientation selectivity exceeds direction selectivity", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1], projection="polar")
    v1 = r[(r.group == "VISp") & (r.p_OSI < 0.05)]
    edges = np.deg2rad(np.arange(0, 181, 15))
    h, _ = np.histogram(np.deg2rad(v1.pref_ori.values), bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.bar(centers, h, width=np.deg2rad(15), color="C0", alpha=0.8, align="center")
    ax.bar(centers + np.pi, h, width=np.deg2rad(15), color="C0", alpha=0.8, align="center")
    ax.set_thetagrids(np.arange(0, 360, 45))
    ax.set_yticklabels([])
    ax.grid(alpha=0.4)
    ax.set_title(f"Preferred orientations, tuned V1 units (n={len(v1)})",
                 fontsize=9.5, pad=22)

    ax = fig.add_subplot(gs[1, 0])
    for g in GROUPS[::-1]:
        sub = r[r.group == g]
        ax.plot(sub.d_ori.clip(1e-4), sub.d_dir.clip(1e-4), "o", ms=3, alpha=0.45,
                color=GCOL[g], label=g)
    ax.plot([1e-4, 10], [1e-4, 10], "k--", lw=0.8)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(8e-5, 20), ax.set_ylim(8e-5, 20)
    ax.set_xlabel("$\\Delta$LL orientation - constant (nats/trial)")
    ax.set_ylabel("$\\Delta$LL direction - orientation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("How much each model term buys, per unit\n"
                 "(values clipped at $10^{-4}$; points below the line are "
                 "orientation dominated)", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    tab = (pd.crosstab(r.group, r.glm_label, normalize="index")
             .reindex(index=GROUPS)
             .reindex(columns=["orientation", "direction", "untuned"], fill_value=0))
    shuf = (pd.crosstab(r.group, r.glm_label_shuffled, normalize="index")
              .reindex(index=GROUPS)
              .reindex(columns=["orientation", "direction", "untuned"], fill_value=0))
    x = np.arange(len(GROUPS))
    w = 0.26
    for i, lab in enumerate(["orientation", "direction", "untuned"]):
        ax.bar(x + (i - 1) * w, tab[lab].values, w, label=lab, color=f"C{i}", alpha=0.85)
        ax.hlines(shuf[lab].values, x + (i - 1.5) * w, x + (i - 0.5) * w,
                  color="k", lw=1.4, ls="--",
                  label="shuffled control" if i == 0 else None)
    ax.set_xticks(x, GROUPS)
    ax.set_ylabel("fraction of responsive units")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.set_ylim(0, 1.15)
    ax.set_title("Which model wins on held-out trials", fontsize=9.5)

    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


def fig_glm_fits(df, raw, fname="fig04_glm_example_fits.png"):
    import glm_lib as gl

    uniq, mean, sem = raw["uniq"], raw["mean"], raw["sem"]
    ang, drive = raw["ang"], raw["drive"]
    unit_ids = raw["unit_ids"]
    col = {u: i for i, u in enumerate(unit_ids)}
    dur = drive.stop_time.values - drive.start_time.values - ol.ONSET_LAG
    counts = np.rint(raw["rates"] * dur[:, None])

    r = df[df.responsive & (df.group == "VISp")]
    picks = list(r[r.glm_label == "orientation"].sort_values("ll_orientation",
                 ascending=False).unit_id.values[:3])
    picks += list(r[r.glm_label == "direction"]
                  .assign(gain=lambda d: d.ll_direction - d.ll_orientation)
                  .sort_values("gain", ascending=False).unit_id.values[:3])
    idx = [col[u] for u in picks]
    grid, pred_dir = gl.fitted_curves(counts[:, idx], ang, "direction")
    _, pred_ori = gl.fitted_curves(counts[:, idx], ang, "orientation")
    scale = 1.0 / dur.mean()

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
    for k, (ax, uid) in enumerate(zip(axes.ravel(), picks)):
        j = col[uid]
        row = df[df.unit_id == uid].iloc[0]
        ax.errorbar(uniq, mean[:, j], yerr=sem[:, j], fmt="o", ms=4, color="k",
                    capsize=2, lw=1, label="data")
        ax.plot(grid, pred_ori[:, k] * scale, color="C0", lw=1.8,
                label="orientation model (180$\\degree$)")
        ax.plot(grid, pred_dir[:, k] * scale, color="C1", lw=1.8,
                label="direction model (360$\\degree$)")
        ax.set_title(f"unit {uid} - GLM picks '{row.glm_label}'\n"
                     f"$\\Delta$LL(dir - ori) = {row.ll_direction - row.ll_orientation:+.3f} nats/trial",
                     fontsize=8.5)
        ax.set_xticks([0, 90, 180, 270, 360])
    for ax in axes[-1]:
        ax.set_xlabel("drift direction (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("firing rate (Hz)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Poisson GLMs with cyclic B-spline bases: 180$\\degree$-periodic model "
                 "suffices for orientation-selective units (top),\nwhile direction-selective "
                 "units need the 360$\\degree$ model (bottom)", y=1.08, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


def fig_multisession(df, fname="fig08_multisession_summary.png"):
    r = df[df.responsive]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    ax = axes[0]
    piv = r.pivot_table(index="session", columns="group", values="gOSI_corrected",
                        aggfunc="median").reindex(columns=GROUPS)
    for g in GROUPS:
        if g in piv:
            ax.plot(range(len(piv)), piv[g].values, "o-", color=GCOL[g], ms=5, label=g)
    ax.set_xticks(range(len(piv)), [str(s) for s in piv.index], rotation=45, fontsize=7.5)
    ax.set_ylabel("median noise-corrected gOSI")
    ax.set_xlabel("session")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Per-session medians", fontsize=9.5)

    ax = axes[1]
    for g in GROUPS:
        x = np.sort(r.loc[r.group == g, "gOSI_corrected"].dropna())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=GCOL[g],
                label=f"{g} (n={len(x)})")
    ax.set_xlabel("noise-corrected gOSI"), ax.set_ylabel("cumulative fraction")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Pooled across {r.session.nunique()} sessions", fontsize=9.5)

    ax = axes[2]
    err_split = ol.circular_distance(r.half1_ori, r.half2_ori, 180)
    sub = r[r.sg_pref_ori.notna() & (r.sg_peak_rate > 1.0)]
    err_cross = ol.circular_distance(sub.pref_ori, sub.sg_pref_ori, 180)
    bins = np.linspace(0, 90, 31)
    ax.hist(err_split, bins=bins, density=True, histtype="step", lw=1.8, color="C0",
            label=f"split half (median {np.median(err_split):.1f}°)")
    ax.hist(err_cross, bins=bins, density=True, histtype="step", lw=1.8, color="C1",
            label=f"drifting vs static (median {np.median(err_cross):.1f}°)")
    ax.axhline(1 / 90, color="k", ls=":", lw=1)
    ax.text(50, 1 / 90 + 0.002, "chance", fontsize=8)
    ax.set_xlabel("difference in preferred orientation (deg)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reliability, all responsive units", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


