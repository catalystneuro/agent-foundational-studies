"""Figures for the orientation-selectivity analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pynapple as nap

import ov_analysis as oa

AREA_COLORS = {
    "LGd": "#4C72B0", "LP": "#7BA4D0",
    "VISp": "#C44E52", "VISl": "#DD8452", "VISrl": "#E0A458",
    "VISal": "#8C8C3F", "VISpm": "#937860", "VISam": "#B07AA1",
    "HVA": "#DD8452",
}
AREA_ORDER = ["LGd", "LP", "VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
HVA = ["VISl", "VISrl", "VISal", "VISpm", "VISam"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.frameon": False, "savefig.bbox": "tight",
})


def group_label(area):
    if area in HVA:
        return "HVA"
    return area


# ------------------------------------------------------------------- figure 1

def fig_raw_overview(spikes, info, dg, running, out_path,
                     t_window=24.0, n_per_area=45, areas_shown=("LGd", "VISp")):
    """Session structure, a raw raster during drifting gratings, and running speed."""
    nwbfile = info["nwbfile"]
    all_areas = np.asarray(spikes.metadata["area"])
    all_uids = np.asarray(spikes.index)

    rng = np.random.default_rng(1)
    rows, row_area = [], []
    for a in areas_shown:
        idx = np.where(all_areas == a)[0]
        if len(idx) > n_per_area:
            idx = np.sort(rng.choice(idx, n_per_area, replace=False))
        rows.extend(all_uids[idx])
        row_area.extend([a] * len(idx))
    rows, row_area = np.array(rows), np.array(row_area)

    t0 = float(dg["start_time"].iloc[40])
    t1 = t0 + t_window
    ep = nap.IntervalSet(start=t0, end=t1)

    fig = plt.figure(figsize=(12, 9.0))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.7, 3.6, 1.0, 0.9], hspace=0.75)

    # -- A: whole-session stimulus timeline
    ax = fig.add_subplot(gs[0])
    stim_names = sorted(k for k in nwbfile.intervals if k.endswith("_presentations"))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(stim_names):
        tbl = nwbfile.intervals[name].to_dataframe()
        ax.barh(0, tbl["stop_time"].values - tbl["start_time"].values,
                left=tbl["start_time"].values, height=0.6,
                color=cmap(i % 10), label=name.replace("_presentations", ""))
    ax.axvline(t0, color="k", lw=1.2)
    ax.annotate("window below", xy=(t0, 0.45), xytext=(t0 + 350, 0.9),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(-0.4, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("time in session (s)", labelpad=1)
    ax.set_title("A   Stimulus blocks across the session", loc="left")
    ax.legend(ncol=1, fontsize=7, loc="center left", bbox_to_anchor=(1.005, 0.5),
              handlelength=1.1)

    # -- B: raster
    ax = fig.add_subplot(gs[1])
    dgw = dg[(dg["start_time"] < t1) & (dg["stop_time"] > t0)]
    ntop = len(rows) * 1.10
    for _, r in dgw.iterrows():
        ax.axvspan(r["start_time"], r["stop_time"], color="#FFEFC2", zorder=0)
        if np.isfinite(r["orientation"]):
            ax.text((r["start_time"] + r["stop_time"]) / 2, ntop * 1.005,
                    f"{int(r['orientation'])}°", ha="center", va="bottom", fontsize=7.5)
    for i, (u, a) in enumerate(zip(rows, row_area)):
        t = spikes[int(u)].restrict(ep).t
        ax.plot(t, np.full_like(t, i), "|", color=AREA_COLORS[a], ms=2.6, mew=0.65)
    for a in areas_shown[1:]:
        ax.axhline(np.where(row_area == a)[0][0] - 0.5, color="0.4", lw=0.7, ls="--")
    ax.set_xlim(t0, t1)
    ax.set_ylim(-2, ntop)
    ax.set_ylabel("unit")
    ax.set_title("B   Raw spike raster during drifting gratings "
                 "(shaded = 2 s grating; label = drift direction)", loc="left", pad=16)
    handles = [plt.Line2D([], [], color=AREA_COLORS[a], marker="|", ls="none",
                          ms=9, mew=2.2, label=f"{a} (n={np.sum(row_area == a)} shown)")
               for a in areas_shown]
    ax.legend(handles=handles, fontsize=8, loc="center left",
              bbox_to_anchor=(1.005, 0.5), handlelength=1.1)

    # -- C: population rate
    ax = fig.add_subplot(gs[2])
    for a in areas_shown:
        m = all_areas == a
        cnt = spikes[[int(u) for u in all_uids[m]]].count(0.02, ep=ep)
        rate = np.asarray(cnt.values).sum(axis=1) / (0.02 * m.sum())
        ax.plot(cnt.t, pd.Series(rate).rolling(5, center=True, min_periods=1).mean(),
                color=AREA_COLORS[a], lw=1.1, label=f"{a} (all {m.sum()})")
    for _, r in dgw.iterrows():
        ax.axvspan(r["start_time"], r["stop_time"], color="#FFEFC2", zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_ylabel("pop. rate\n(Hz / unit)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5),
              handlelength=1.1)
    ax.set_title("C   Population firing rate (20 ms bins, smoothed)", loc="left")

    # -- D: running speed
    ax = fig.add_subplot(gs[3])
    rs = running.restrict(ep)
    ax.plot(rs.t, rs.d, color="0.25", lw=0.9)
    for _, r in dgw.iterrows():
        ax.axvspan(r["start_time"], r["stop_time"], color="#FFEFC2", zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time in session (s)")
    ax.set_ylabel("running\n(cm/s)")
    ax.set_title("D   Locomotion", loc="left")

    fig.suptitle(f"Session {info['session']} — raw data overview", y=0.955, fontsize=13)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------- figure 2

def fig_example_unit(spikes, res, uid, area, out_path, gosi=None, dsi=None, pval=None,
                     note=""):
    """Classic orientation-tuning figure for one unit: rasters, PSTHs, polar plot."""
    tbl = res["trial_table"]
    ui = int(np.where(res["uids"] == uid)[0][0])
    pref_tf = res["pref_tf"][ui]
    sub = tbl[tbl["temporal_frequency"] == pref_tf]

    fig = plt.figure(figsize=(13, 6.6))
    gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 0.25, 1.5],
                           hspace=0.62, wspace=0.42)

    win = (-0.5, 2.5)
    bw = 0.05
    bins = np.arange(win[0], win[1] + bw, bw)

    psths, rasters = {}, {}
    psth_max = 0.0
    for d in oa.DG_DIRECTIONS:
        onsets = nap.Ts(sub[sub["orientation"] == d]["start_time"].values)
        pe = nap.compute_perievent(spikes[int(uid)], onsets, window=win)
        trials = [pe[i].t for i in pe.index]
        rasters[d] = trials
        allt = np.concatenate(trials) if len(trials) else np.array([])
        h = np.histogram(allt, bins)[0] / (len(trials) * bw)
        psths[d] = h
        psth_max = max(psth_max, h.max())

    for k, d in enumerate(oa.DG_DIRECTIONS):
        row, col = divmod(k, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.axvspan(0, 2, color="#FFEFC2", zorder=0)
        ntr = len(rasters[d])
        for j, t in enumerate(rasters[d]):
            ax.plot(t, np.full_like(t, j), "|", color="k", ms=3.2, mew=0.7)
        ax.set_ylim(-0.5, ntr - 0.5)
        ax.set_yticks([0, ntr - 1])
        ax.set_yticklabels(["1", str(ntr)], fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(bins[:-1] + bw / 2, psths[d], color=AREA_COLORS.get(area, "C3"), lw=1.1)
        ax2.set_ylim(0, psth_max * 1.08)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(labelsize=7)
        if col == 3:
            ax2.set_ylabel("rate (Hz)", fontsize=8)
        else:
            ax2.set_yticklabels([])
        ax.set_xlim(*win)
        ax.set_xticks([0, 1, 2])
        ax.set_title(f"{int(d)}°", fontsize=10, pad=3)
        if col == 0:
            ax.set_ylabel("trial", fontsize=8)
        if row == 1:
            ax.set_xlabel("time from onset (s)", fontsize=8)

    # polar tuning curve with von Mises fit
    ax = fig.add_subplot(gs[0, 5], projection="polar")
    tc = res["tc"][ui]
    rates, sel = res["rates"][ui], res["tfs"] == pref_tf
    sem = np.array([rates[sel & (res["dirs"] == d)].std(ddof=1)
                    / np.sqrt(np.sum(sel & (res["dirs"] == d))) for d in oa.DG_DIRECTIONS])
    th = np.deg2rad(oa.DG_DIRECTIONS)
    xs = np.linspace(0, 360, 361)
    popt = oa.fit_double_von_mises(tc)
    ax.plot(np.deg2rad(xs), oa.double_von_mises(xs, *popt), color="C3", lw=1.4,
            zorder=2, label="von Mises fit")
    ax.errorbar(np.append(th, th[0]), np.append(tc, tc[0]),
                yerr=np.append(sem, sem[0]), color="k", lw=1.4, marker="o", ms=4,
                capsize=2, zorder=3, label="mean ± SEM")
    ax.set_theta_zero_location("E")
    ax.set_title("direction tuning curve (Hz)", fontsize=9.5, pad=22)
    ax.tick_params(labelsize=7, pad=2)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)

    # direction x temporal-frequency matrix
    ax = fig.add_subplot(gs[1, 5])
    im = ax.imshow(res["cube"][ui], aspect="auto", origin="lower", cmap="magma",
                   extent=[-22.5, 337.5, -0.5, len(res["tf_levels"]) - 0.5])
    ax.set_xticks(oa.DG_DIRECTIONS)
    ax.set_xticklabels([int(d) for d in oa.DG_DIRECTIONS], fontsize=7, rotation=45)
    ax.set_yticks(range(len(res["tf_levels"])))
    ax.set_yticklabels([f"{t:g}" for t in res["tf_levels"]], fontsize=7)
    ax.set_xlabel("direction (deg)", fontsize=8)
    ax.set_ylabel("TF (Hz)", fontsize=8)
    ax.axhline(np.where(res["tf_levels"] == pref_tf)[0][0], color="w", lw=0.8, ls=":")
    ax.set_title("mean rate: direction × temporal frequency", fontsize=9.5, pad=4)
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Hz", fontsize=7)

    txt = f"unit {uid} · {area} · preferred TF {pref_tf:g} Hz"
    if gosi is not None:
        txt += (f" · gOSI {gosi:.2f} · DSI {dsi:.2f} · "
                f"tuning HWHM {oa.vm_hwhm(popt[3]):.0f}° · permutation p = {pval:.3g}")
    if note:
        txt += f"\n{note}"
    fig.suptitle(txt, y=1.0, fontsize=11)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------- figure 3

def fig_population_tuning(tc_by_group, out_path, angles=oa.DG_DIRECTIONS,
                          groups=("LGd", "LP", "VISp", "HVA"),
                          title="Drifting gratings (8 directions)"):
    """Heatmaps of normalised tuning curves plus preference-aligned mean curves."""
    groups = [g for g in groups if g in tc_by_group and len(tc_by_group[g]) > 5]
    n = len(groups)
    fig = plt.figure(figsize=(2.5 * n + 5.0, 4.6))
    gs = gridspec.GridSpec(1, n + 3, width_ratios=[1] * n + [0.09, 0.42, 1.9], wspace=0.32)

    step = angles[1] - angles[0]
    for i, g in enumerate(groups):
        tc = np.asarray(tc_by_group[g], dtype=float)
        # normalise by peak only: the depth of the trough then reflects tuning strength
        norm = tc / np.maximum(tc.max(axis=1, keepdims=True), 1e-9)
        _, pref = oa.circular_selectivity(tc, angles, harmonic=2)
        order = np.argsort(pref)
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(norm[order], aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[angles[0] - step / 2, angles[-1] + step / 2, len(tc), 0],
                       interpolation="nearest")
        ax.set_xticks(angles[::2])
        ax.set_xticklabels([int(x) for x in angles[::2]], fontsize=7.5)
        ax.set_title(f"{g}   (n = {len(tc)})", fontsize=10)
        ax.set_xlabel("stimulus angle (deg)", fontsize=8.5)
        ax.set_ylabel("unit (sorted by preferred orientation)" if i == 0 else "", fontsize=8.5)
    cax = fig.add_subplot(gs[n])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("response / peak response", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    # preference-aligned mean tuning curves
    ax = fig.add_subplot(gs[n + 2])
    k = len(angles)
    centre = k // 2
    rel = (np.arange(k) - centre) * step
    for g in groups:
        tc = np.asarray(tc_by_group[g], dtype=float)
        norm = tc / np.maximum(tc.max(axis=1, keepdims=True), 1e-9)
        aligned = np.stack([np.roll(r, centre - int(np.argmax(r))) for r in norm])
        m, s = aligned.mean(axis=0), aligned.std(axis=0) / np.sqrt(len(aligned))
        ax.plot(rel, m, color=AREA_COLORS.get(g, "k"), lw=1.7, marker="o", ms=4, label=g)
        ax.fill_between(rel, m - s, m + s, color=AREA_COLORS.get(g, "k"), alpha=0.22, lw=0)
    ax.set_xlabel("angle relative to each unit's preferred (deg)", fontsize=8.5)
    ax.set_ylabel("response / peak response", fontsize=8.5)
    ax.set_xticks(rel[::2])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower center", ncol=len(groups), columnspacing=1.0)
    ax.set_title("preference-aligned population mean ± SEM", fontsize=10)

    fig.suptitle(f"{title} — single-unit tuning across the visual hierarchy",
                 y=1.04, fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------- figure 4

def fig_selectivity_by_area(df, out_path, areas=None):
    """Distributions of gOSI / OSI / DSI and the fraction of significantly tuned units."""
    areas = areas or [a for a in AREA_ORDER if (df["area"] == a).sum() >= 20]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    fig.subplots_adjust(hspace=0.42, wspace=0.28)

    # A: gOSI cumulative distributions
    ax = axes[0, 0]
    for a in areas:
        v = np.sort(df.loc[df["area"] == a, "gosi"].dropna().values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=AREA_COLORS[a],
                lw=1.9 if a in ("LGd", "VISp") else 1.0,
                alpha=1.0 if a in ("LGd", "VISp") else 0.75, label=f"{a} ({len(v)})")
    ax.set_xlabel("global OSI (drifting gratings)")
    ax.set_ylabel("cumulative fraction of units")
    ax.set_title("A   Orientation selectivity by area", loc="left")
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.set_xlim(0, 1)

    # B: box/strip of gOSI
    ax = axes[0, 1]
    data = [df.loc[df["area"] == a, "gosi"].dropna().values for a in areas]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color="k", lw=1.4))
    for patch, a in zip(bp["boxes"], areas):
        patch.set_facecolor(AREA_COLORS[a])
        patch.set_alpha(0.65)
    rng = np.random.default_rng(0)
    for i, (v, a) in enumerate(zip(data, areas), start=1):
        ax.plot(i + rng.uniform(-0.16, 0.16, len(v)), v, ".", color="k", ms=1.6, alpha=0.35)
    ax.set_xticks(range(1, len(areas) + 1))
    ax.set_xticklabels(areas, rotation=45, fontsize=8)
    ax.set_ylabel("global OSI")
    ax.set_title("B   gOSI distribution (box = quartiles, line = median)", loc="left")

    # C: fraction significantly tuned
    ax = axes[1, 0]
    frac = [np.mean(df.loc[df["area"] == a, "p"] < 0.05) for a in areas]
    nn = [int((df["area"] == a).sum()) for a in areas]
    err = [np.sqrt(f * (1 - f) / n) for f, n in zip(frac, nn)]
    ax.bar(range(len(areas)), frac, yerr=err, capsize=3,
           color=[AREA_COLORS[a] for a in areas], alpha=0.85)
    ax.axhline(0.05, color="k", ls="--", lw=0.9)
    ax.text(len(areas) - 0.4, 0.075, "chance (α = 0.05)", ha="right", fontsize=7.5)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=45, fontsize=8)
    ax.set_ylabel("fraction of units")
    ax.set_ylim(0, 1)
    ax.set_title("C   Significantly orientation-tuned units (permutation test)", loc="left")

    # D: gOSI vs gDSI
    ax = axes[1, 1]
    for a in ["LGd", "VISp"]:
        m = df["area"] == a
        ax.plot(df.loc[m, "gosi"], df.loc[m, "gdsi"], ".", ms=4.5, alpha=0.6,
                color=AREA_COLORS[a], label=a)
    lim = [0, 1]
    ax.plot(lim, lim, color="0.6", lw=0.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("global OSI (orientation)")
    ax.set_ylabel("global DSI (direction)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("D   Orientation vs direction selectivity", loc="left")

    fig.suptitle("Orientation selectivity emerges between thalamus and cortex",
                 y=0.98, fontsize=12.5)
    fig.savefig(out_path)
    plt.close(fig)
