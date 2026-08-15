"""Figure generation for the orientation-selectivity analysis."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pynapple as nap

import orilib as O
from analysis import DIRECTIONS, STATIC_ORIS

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "legend.frameon": False,
    }
)

ORI_COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:8]


# --------------------------------------------------------------------------- helpers


def perievent_spikes(ts, starts, window=(-0.5, 2.5)):
    """List of relative spike-time arrays, one per trial."""
    t = ts.t
    out = []
    for s in starts:
        lo = np.searchsorted(t, s + window[0], "left")
        hi = np.searchsorted(t, s + window[1], "right")
        out.append(t[lo:hi] - s)
    return out


def psth(ts, starts, window=(-0.5, 2.5), bin_size=0.025):
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    t = ts.t
    counts = np.zeros(len(edges) - 1)
    for s in starts:
        lo = np.searchsorted(t, s + window[0], "left")
        hi = np.searchsorted(t, s + window[1], "right")
        counts += np.histogram(t[lo:hi] - s, edges)[0]
    return edges[:-1] + bin_size / 2, counts / (len(starts) * bin_size)


# ----------------------------------------------------------------------- figure 1


def fig_raw_data(res, running, fname="fig01_raw_data.png"):
    """Raw spiking during drifting gratings: population raster + example unit + behaviour."""
    u = res["units"]
    tsg = res["tsgroup"]
    dg = res["dg_table"]

    # a 30 s window at the start of the first drifting-gratings block
    t0 = dg["start_time"].iloc[0] - 2
    t1 = t0 + 32
    win = nap.IntervalSet(start=t0, end=t1)
    sub = dg[(dg.start_time >= t0) & (dg.stop_time <= t1)]

    # example unit: well driven, orientation-tuned, and with a legible preferred /
    # orthogonal rate contrast inside the specific window being plotted
    vis = u[u.area.isin(O.VISUAL_CORTEX) & (u.dg_max_rate > 15) & (u.dg_osi > 0.4)]
    best, best_ratio = None, -np.inf
    for uid in vis.index:
        d = O.circ_dist_ori(sub.orientation.values, u.loc[uid, "dg_pref_ori"])
        pref_ep = nap.IntervalSet(sub.start_time.values[d < 25], sub.stop_time.values[d < 25])
        orth_ep = nap.IntervalSet(sub.start_time.values[d > 65], sub.stop_time.values[d > 65])
        if len(pref_ep) == 0 or len(orth_ep) == 0:
            continue
        rp = len(tsg[uid].restrict(pref_ep)) / pref_ep.tot_length()
        ro = len(tsg[uid].restrict(orth_ep)) / orth_ep.tot_length()
        ratio = rp / (ro + 1.0)
        if rp > 5 and ratio > best_ratio:
            best, best_ratio = uid, ratio
    example = best

    order = u[u.area.isin(O.VISUAL_CORTEX + O.CONTROL)].sort_values("area")
    fig = plt.figure(figsize=(12, 8.5))
    gs = GridSpec(4, 1, height_ratios=[0.35, 2.0, 1.0, 0.8], hspace=0.45)

    # stimulus ribbon
    ax0 = fig.add_subplot(gs[0])
    for _, r in sub.iterrows():
        c = ORI_COLORS[int(np.where(DIRECTIONS == r.orientation)[0][0])]
        ax0.axvspan(r.start_time, r.stop_time, color=c, alpha=0.9)
        ax0.text(
            (r.start_time + r.stop_time) / 2,
            0.5,
            f"{int(r.orientation)}",
            ha="center",
            va="center",
            fontsize=6,
            rotation=90,
            color="w",
        )
    ax0.set_xlim(t0, t1)
    ax0.set_yticks([])
    ax0.set_title("Drifting-grating presentations (colour/number = drift direction, deg)")
    ax0.tick_params(labelbottom=False)

    # population raster
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ypos, ylabels, yticks = 0, [], []
    for area, grp in order.groupby("area", sort=False):
        start = ypos
        for uid in grp.index:
            t = tsg[uid].restrict(win).t
            ax1.plot(t, np.full_like(t, ypos), "|", ms=1.6, lw=0.3, alpha=0.85,
                     color="tab:blue" if area in O.VISUAL_CORTEX else "0.55")
            ypos += 1
        yticks.append((start + ypos) / 2)
        ylabels.append(area)
        ax1.axhline(ypos, color="k", lw=0.4, alpha=0.3)
    for _, r in sub.iterrows():
        ax1.axvspan(r.start_time, r.stop_time, color="k", alpha=0.045, lw=0)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels, fontsize=7)
    ax1.set_ylim(0, ypos)
    ax1.set_ylabel("unit (grouped by area)")
    ax1.set_title("Population raster: visual cortex (blue) vs hippocampus (grey)")
    ax1.tick_params(labelbottom=False)

    # example unit raster
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    t = tsg[example].restrict(win).t
    ax2.plot(t, np.zeros_like(t), "|", ms=14, color="k")
    for _, r in sub.iterrows():
        c = ORI_COLORS[int(np.where(DIRECTIONS == r.orientation)[0][0])]
        ax2.axvspan(r.start_time, r.stop_time, color=c, alpha=0.25, lw=0)
    ax2.set_yticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(
        f"Example unit {example} ({u.loc[example,'area']}), "
        f"OSI={u.loc[example,'dg_osi']:.2f}, preferred orientation "
        f"{u.loc[example,'dg_pref_ori']:.0f}°"
    )
    ax2.tick_params(labelbottom=False)

    # running speed
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    if running is not None:
        rs = running.restrict(win)
        ax3.plot(rs.t, rs.d, lw=0.7, color="tab:green")
        ax3.set_ylabel("running\n(cm/s)")
    ax3.set_xlabel("time in session (s)")
    ax3.set_xlim(t0, t1)
    ax3.tick_params(labelbottom=True)

    fig.suptitle(
        f"DANDI:000021 session {res['session_id']}: raw spiking during drifting gratings",
        y=0.985,
    )
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 2


def fig_example_units(res, n=6, fname="fig02_example_units.png"):
    """PSTH-by-direction and polar tuning curve for the most selective visual units."""
    u = res["units"]
    tsg = res["tsgroup"]
    dg = res["dg_table"]
    tc = res["tc_dg"]
    unit_pos = {uid: i for i, uid in enumerate(u.index)}

    vis = u[u.area.isin(O.VISUAL_CORTEX) & (u.dg_mean_rate > 1.0)]
    picks = vis.sort_values("dg_osi", ascending=False).index[:n]

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6.2),
                             subplot_kw=None, gridspec_kw=dict(height_ratios=[1.3, 1]))
    for k, uid in enumerate(picks):
        ax = axes[0, k]
        for i, d in enumerate(DIRECTIONS):
            starts = dg.loc[dg.orientation == d, "start_time"].values
            x, y = psth(tsg[uid], starts)
            ax.plot(x, y, color=ORI_COLORS[i], lw=1.0, label=f"{int(d)}°")
        ax.axvspan(0, 2, color="0.9", zorder=-5)
        ax.set_title(f"unit {uid}\n{u.loc[uid,'area']}  OSI={u.loc[uid,'dg_osi']:.2f}", pad=6)
        ax.set_xlabel("time from onset (s)")
        if k == 0:
            ax.set_ylabel("firing rate (Hz)")
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()

    for k, uid in enumerate(picks):
        axes[1, k].remove()
        ax = fig.add_subplot(2, n, n + k + 1, projection="polar")
        r = tc[:, unit_pos[uid]]
        th = np.deg2rad(np.append(DIRECTIONS, DIRECTIONS[0]))
        ax.plot(th, np.append(r, r[0]), "o-", color="tab:red", ms=3, lw=1.2)
        ax.fill(th, np.append(r, r[0]), color="tab:red", alpha=0.15)
        po = np.deg2rad(u.loc[uid, "dg_pref_ori"])
        ax.plot([po, po + np.pi], [r.max()] * 2, "--", color="k", lw=1.0)
        ax.set_theta_zero_location("E")
        ax.set_rlabel_position(105)
        ax.set_yticklabels([])
        ax.tick_params(labelsize=6, pad=0)
        ax.set_title(f"pref ori {u.loc[uid,'dg_pref_ori']:.0f}°\nDSI={u.loc[uid,'dg_dsi']:.2f}",
                     fontsize=8, pad=14)

    fig.legend(handles, labels, ncol=8, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), title="drift direction", title_fontsize=8)
    fig.suptitle(
        "Direction tuning of the most orientation-selective visual-cortex units "
        f"(session {res['session_id']})",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=3.0, w_pad=1.6)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 3


def fig_population_tuning(pool, fname="fig03_population_tuning.png"):
    """Normalised tuning curves for every unit, visual cortex vs hippocampal control."""
    u = pool["units"]
    tc = pool["tc_dg"]  # (8, n_units) aligned with u

    def block(mask, ax_h, ax_m, title):
        idx = np.where(mask)[0]
        M = tc[:, idx].T.copy()
        base = M.min(1, keepdims=True)
        rng_ = M.max(1, keepdims=True) - base
        rng_[rng_ == 0] = np.nan
        Mn = (M - base) / rng_
        peak = np.argmax(Mn, axis=1)
        # sort by peak direction, then by how sharply the response falls off, so the
        # preferred direction forms a diagonal and its 180 deg mirror is visible
        second = np.array([Mn[i, (peak[i] + 4) % 8] for i in range(len(idx))])
        order = np.lexsort((second, peak))
        im = ax_h.imshow(
            Mn[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
            extent=[-22.5, 337.5, len(idx), 0], interpolation="nearest",
        )
        ax_h.set_xticks(DIRECTIONS)
        ax_h.set_xlabel("drift direction (deg)")
        ax_h.set_ylabel("unit (sorted by preferred direction)")
        ax_h.set_title(f"{title}\nn = {len(idx)} units")
        # tuning curves rotated so each unit's preferred direction sits at 0
        rolled = np.stack([np.roll(Mn[i], -peak[i]) for i in range(len(idx))])
        rolled = np.concatenate([rolled, rolled[:, :1]], axis=1)
        xx = np.arange(9) * 45
        m = np.nanmean(rolled, 0)
        se = np.nanstd(rolled, 0) / np.sqrt(np.isfinite(rolled).sum(0))
        ax_m.plot(xx, m, "o-", color="tab:red", ms=4)
        ax_m.fill_between(xx, m - se, m + se, color="tab:red", alpha=0.3)
        ax_m.axvline(180, color="0.5", ls="--", lw=1)
        ax_m.set_xticks(np.arange(0, 361, 90))
        ax_m.set_xlabel("direction relative to preferred (deg)")
        ax_m.set_ylabel("normalised rate")
        ax_m.set_ylim(0, 1.05)
        ax_m.set_title("mean tuning curve aligned to preferred direction\n"
                       "(dashed line: same orientation, opposite direction)", fontsize=9)
        return im

    fig = plt.figure(figsize=(11.5, 8.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.15, 0.04, 1.0],
                  hspace=0.42, wspace=0.22)
    axes = [[fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 2])] for r in range(2)]
    cax = fig.add_subplot(gs[:, 1])

    vis = u["area"].isin(O.VISUAL_CORTEX).values & (u["dg_p_perm"].values < 0.05)
    ctl = u["area"].isin(O.CONTROL).values
    im = block(vis, axes[0][0], axes[0][1],
               "Visual cortex, orientation-selective units (p < 0.05)")
    block(ctl, axes[1][0], axes[1][1], "Hippocampus (CA1/CA3/DG), all units")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("normalised rate")
    fig.suptitle(
        "Population tuning: orientation structure is present in cortex, absent in hippocampus",
        y=0.955)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 4


def fig_osi_by_area(pool, fname="fig04_osi_by_area.png"):
    u = pool["units"]
    areas = [a for a in O.VISUAL_CORTEX + O.THALAMUS + O.CONTROL
             if (u["area"] == a).sum() >= 20]
    data = [u.loc[u.area == a, "dg_osi"].dropna().values for a in areas]
    null = [u.loc[u.area == a, "dg_osi_null_median"].dropna().values for a in areas]
    frac = np.array([(u.loc[u.area == a, "dg_p_perm"] < 0.05).mean() for a in areas])
    ns = np.array([len(d) for d in data])
    colors = ["tab:blue" if a in O.VISUAL_CORTEX else
              "tab:orange" if a in O.THALAMUS else "0.5" for a in areas]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    ax = axes[0]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    ax.boxplot(data, widths=0.16, showfliers=False,
               medianprops=dict(color="k", lw=1.4), whiskerprops=dict(lw=0.8))
    ax.plot(np.arange(1, len(areas) + 1), [np.median(n) for n in null], "kv", ms=5,
            label="median shuffled null")
    ax.set_xticks(np.arange(1, len(areas) + 1))
    ax.set_xticklabels([f"{a}\n(n={n})" for a, n in zip(areas, ns)], fontsize=7.5,
                       rotation=30, ha="right")
    ax.set_ylabel("orientation selectivity index (OSI)")
    ax.set_title("OSI by recorded structure")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.0)

    ax = axes[1]
    ax.bar(np.arange(len(areas)), frac, color=colors, alpha=0.8)
    ax.axhline(0.05, ls="--", color="k", lw=1, label="chance (α = 0.05)")
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of units")
    ax.set_title("Units with significant orientation tuning\n(within-block label shuffle, p < 0.05)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for a, c in [("VISp", "tab:blue"), ("VISl", "tab:cyan"), ("VISal", "tab:purple"),
                 ("LP", "tab:orange"), ("CA1", "0.4"), ("DG", "0.65")]:
        d = u.loc[u.area == a, "dg_osi"].dropna().values
        if len(d) < 20:
            continue
        ax.plot(np.sort(d), np.linspace(0, 1, len(d)), color=c, lw=1.8, label=f"{a} (n={len(d)})")
    nulld = u.loc[u.area.isin(O.VISUAL_CORTEX), "dg_osi_null_median"].dropna().values
    ax.plot(np.sort(nulld), np.linspace(0, 1, len(nulld)), color="k", ls=":", lw=1.5,
            label="shuffled null")
    ax.set_xlabel("OSI")
    ax.set_ylabel("cumulative fraction of units")
    ax.set_title("Cumulative OSI distributions")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 0.9)

    fig.suptitle(
        f"Orientation selectivity across {pool['n_sessions']} sessions "
        f"({len(u)} quality-passing units)", y=1.01)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 5


def fig_cross_stimulus(pool, fname="fig05_cross_stimulus.png", seed=0):
    """Preferred orientation measured with drifting gratings vs static gratings."""
    u = pool["units"]
    m = (u.area.isin(O.VISUAL_CORTEX) & (u.dg_p_perm < 0.05) & (u.sg_p_perm < 0.05))
    d = u[m]
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    ax = axes[0]
    ax.plot(d.dg_pref_ori, d.sg_pref_ori, "o", ms=4, alpha=0.45, color="tab:blue")
    for off in (-180, 0, 180):
        ax.plot([0, 180], [off, off + 180], "k--", lw=1)
    ax.set_xlabel("preferred orientation, drifting gratings (deg)")
    ax.set_ylabel("preferred orientation, static gratings (deg)")
    ax.set_title(f"Cross-stimulus agreement\nn = {len(d)} visual-cortex units")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_yticks(np.arange(0, 181, 45))

    ax = axes[1]
    obs = O.circ_dist_ori(d.dg_pref_ori.values, d.sg_pref_ori.values)
    shuf = O.circ_dist_ori(d.dg_pref_ori.values,
                           rng.permutation(d.sg_pref_ori.values))
    bins = np.arange(0, 91, 7.5)
    ax.hist(obs, bins=bins, density=True, alpha=0.7, color="tab:blue", label="observed")
    ax.hist(shuf, bins=bins, density=True, histtype="step", lw=2, color="k",
            label="shuffled pairing")
    ax.axhline(1 / 90, ls=":", color="0.4", lw=1)
    ax.set_xlabel("|Δ preferred orientation| (deg)")
    ax.set_ylabel("density")
    ax.set_title(f"median |Δ| = {np.median(obs):.1f}° vs {np.median(shuf):.1f}° shuffled")
    ax.legend(fontsize=8)

    ax = axes[2]
    v = u[u.area.isin(O.VISUAL_CORTEX)]
    ax.plot(v.dg_osi, v.sg_osi, "o", ms=3, alpha=0.3, color="0.35")
    r = np.corrcoef(v.dg_osi.dropna(), v.sg_osi.dropna())[0, 1] if len(v) else np.nan
    ax.set_xlabel("OSI, drifting gratings")
    ax.set_ylabel("OSI, static gratings")
    ax.set_title(f"OSI consistency across stimulus classes\nPearson r = {r:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("Orientation preference replicates across two independent stimulus classes", y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 6


def fig_decoding(dec, fname="fig06_decoding.png"):
    """Population decoding of grating direction / orientation, by brain structure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    palette = {"VISp": "tab:blue", "VISl": "tab:cyan", "VISal": "tab:purple",
               "VISrl": "tab:green", "VISam": "tab:olive", "LP": "tab:orange",
               "CA1": "0.45"}

    for ax, task, chance in ((axes[0], "direction", 1 / 8), (axes[1], "orientation", 1 / 4)):
        d = dec[(dec.task == task) & (dec.kind == "observed")]
        for area, g in d.groupby("area"):
            m = g.groupby("n_units").acc.agg(["mean", "sem", "size"])
            ax.errorbar(m.index, m["mean"], yerr=m["sem"], marker="o", ms=4, lw=1.5,
                        capsize=2, color=palette.get(area, "k"),
                        label=f"{area} ({int(m['size'].max())} sess.)")
        sh = dec[(dec.task == task) & (dec.kind == "shuffled")].groupby("n_units").acc.mean()
        ax.axhline(chance, color="0.6", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(sorted(d.n_units.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("number of simultaneously recorded units")
        ax.set_ylabel("cross-validated accuracy")
        ax.plot(sh.index, sh.values, "k:", lw=1.5, label="label-shuffled")
        ax.set_title(f"{task.capitalize()} decoding "
                     f"({'8-way' if task=='direction' else '4-way'}; chance {chance:.2f})")
        ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    # use the largest population size that most structures actually reach
    counts = dec[dec.task == "orientation"].groupby("n_units").area.nunique()
    nsel = int(counts[counts >= counts.max()].index.max())
    d = dec[(dec.task == "orientation") & (dec.n_units == nsel)]
    obs = d[d.kind == "observed"].groupby("area").acc.agg(["mean", "sem"])
    shf = d[d.kind == "shuffled"].groupby("area").acc.mean()
    areas = obs.index.tolist()
    x = np.arange(len(areas))
    ax.bar(x - 0.2, obs["mean"], 0.4, yerr=obs["sem"], capsize=3,
           color=[palette.get(a, "k") for a in areas], label="observed")
    ax.bar(x + 0.2, shf.reindex(areas), 0.4, color="0.8", label="label-shuffled")
    ax.axhline(0.25, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=45, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Orientation decoding at {nsel} units")
    ax.legend(fontsize=8)

    fig.suptitle("Grating orientation is linearly decodable from visual-cortex populations", y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 7


def fig_glm(glm_df, examples, fname="fig07_glm.png"):
    """NeMoS Poisson-GLM encoding results."""
    import orilib as O

    fig = plt.figure(figsize=(14, 7.6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.35)

    for k, ex in enumerate(examples[:4]):
        ax = fig.add_subplot(gs[0, k])
        ax.errorbar(DIRECTIONS, ex["emp"], yerr=ex["sem"], fmt="o", ms=4, color="0.3",
                    capsize=2, label="observed")
        ax.plot(ex["grid"], ex["fit"], "-", color="tab:red", lw=1.8, label="GLM fit")
        ax.set_xticks(np.arange(0, 361, 90))
        ax.set_xlabel("drift direction (deg)")
        if k == 0:
            ax.set_ylabel("firing rate (Hz)")
            ax.legend(fontsize=7)
        ax.set_title(f"unit {ex['unit_id']} ({ex['area']})\n"
                     f"$\\Delta$LL$_{{dir|speed}}$ = {ex['d']:.2f} nats/trial", fontsize=9)

    areas = [a for a in O.VISUAL_CORTEX + O.THALAMUS + O.CONTROL
             if (glm_df.area == a).sum() >= 20]
    colors = ["tab:blue" if a in O.VISUAL_CORTEX else
              "tab:orange" if a in O.THALAMUS else "0.5" for a in areas]

    ax = fig.add_subplot(gs[1, :2])
    data = [glm_df.loc[glm_df.area == a, "d_dir_given_speed"].values for a in areas]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    ax.boxplot(data, widths=0.15, showfliers=False, medianprops=dict(color="k", lw=1.3))
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xticks(np.arange(1, len(areas) + 1))
    ax.set_xticklabels([f"{a}\n(n={len(d)})" for a, d in zip(areas, data)], fontsize=7.5)
    ax.set_ylabel("held-out $\\Delta$LL (nats/trial)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("Cross-validated gain from adding drift direction to a running-speed-only model")

    ax = fig.add_subplot(gs[1, 2:])
    frac_dir = np.array([(glm_df.loc[glm_df.area == a, "d_dir_given_speed"] > 0).mean()
                         for a in areas])
    frac_spd = np.array([(glm_df.loc[glm_df.area == a, "d_speed"] > 0).mean() for a in areas])
    x = np.arange(len(areas))
    ax.bar(x - 0.2, frac_dir, 0.4, color=colors, alpha=0.9, label="direction | speed")
    ax.bar(x + 0.2, frac_spd, 0.4, color="0.75", label="running speed alone")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of units with $\\Delta$LL > 0")
    ax.set_title("Units whose held-out likelihood improves")
    ax.legend(fontsize=8)

    fig.suptitle("Poisson GLM (NeMoS): grating direction predicts spiking beyond locomotor state",
                 y=0.99)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 8


def fig_summary(pool_, fname="fig08_summary.png"):
    """Preferred-orientation distribution, OSI vs DSI, and cell-type breakdown."""
    import orilib as O

    u = pool_["units"]
    v = u[u.area.isin(O.VISUAL_CORTEX) & (u.dg_p_perm < 0.05)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    bins = np.arange(0, 181, 15)
    ax.hist(v.dg_pref_ori, bins=bins, color="tab:blue", alpha=0.8, edgecolor="w")
    ax.axhline(len(v) / (len(bins) - 1), color="k", ls="--", lw=1, label="uniform")
    for c in (0, 90, 180):
        ax.axvline(c, color="tab:red", ls=":", lw=1.2)
    ax.plot([], [], color="tab:red", ls=":", lw=1.2, label="cardinal (0/90 deg)")
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_xlabel("preferred orientation (deg)")
    ax.set_ylabel("number of units")
    ax.set_title(f"Preferred orientations, visual cortex\n(n = {len(v)} tuned units)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(v.dg_osi, v.dg_dsi, "o", ms=3, alpha=0.35, color="tab:blue")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("OSI (orientation selectivity)")
    ax.set_ylabel("DSI (direction selectivity)")
    ax.set_title("Most tuned units are orientation-\nselective but not direction-selective")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[2]
    narrow = v.waveform_duration < 0.4
    for m, lab, c in ((narrow, "narrow-waveform", "tab:red"),
                      (~narrow, "broad-waveform", "tab:blue")):
        d = v.loc[m, "dg_osi"].dropna().values
        ax.plot(np.sort(d), np.linspace(0, 1, len(d)), color=c, lw=2,
                label=f"{lab} (n={len(d)})")
    ax.set_xlabel("OSI")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Waveform class")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 0.9)

    fig.suptitle("Properties of the orientation-tuned population", y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname
