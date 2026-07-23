"""Figure generation for the orientation-selectivity demonstration."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import analysis as A

mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "legend.frameon": False,
    }
)

REGION_COLORS = {
    "LGd": "#2f2f2f",
    "VISp": "#c1272d",
    "VISl": "#e08214",
    "VISal": "#b8860b",
    "VISrl": "#4a9b8f",
    "VISpm": "#3b6fb6",
    "VISam": "#7b52a1",
}


def fig_raw_traces(sess, tsg, trials, ori, unit, path):
    """Spike raster of a slice of the drifting-gratings block, with stimulus bands."""
    n_show = 16
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    t0 = trials.start[0]
    t1 = trials.end[n_show - 1]
    win = nap.IntervalSet(start=t0 - 2, end=t1 + 2)

    order = np.argsort(sess["depth"])
    loc = sess["location"]
    ax = axes[0]
    for row, u in enumerate(order):
        st = tsg[u].restrict(win).t
        ax.plot(
            st,
            np.full_like(st, row),
            "|",
            color=REGION_COLORS.get(loc[u], "k"),
            markersize=1.6,
            markeredgewidth=0.6,
        )
    for k in range(n_show):
        ax.axvspan(trials.start[k], trials.end[k], color="0.85", alpha=0.35, zorder=0)
    ax.set_ylabel("unit (sorted by depth)")
    ax.set_ylim(-5, len(order) + 5)
    ax.set_title(
        f"{sess['session']}: spiking during the first {n_show} drifting-grating trials"
        "   (grey bands = 2 s grating presentation)",
        pad=26,
    )
    handles = [
        plt.Line2D([], [], color=c, lw=3, label=r)
        for r, c in REGION_COLORS.items()
        if (loc == r).any()
    ]
    ax.legend(
        handles=handles, ncol=6, loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=8
    )

    ax = axes[1]
    st = tsg[unit].restrict(win).t
    ax.plot(st, np.zeros_like(st) + 0.6, "|", color="k", markersize=8)
    rate = tsg[unit].count(0.05, win).smooth(0.1) / 0.05
    ax.plot(rate.t, rate.d / rate.d.max() * 0.45, color=REGION_COLORS["VISp"], lw=1)
    for k in range(n_show):
        ax.axvspan(trials.start[k], trials.end[k], color="0.85", alpha=0.35, zorder=0)
        ax.text(
            0.5 * (trials.start[k] + trials.end[k]),
            0.95,
            f"{int(ori[k])}°",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"example VISp\nunit {sess['unit_id'][unit]}")
    ax.set_title("Numbers above each band give the grating drift direction (deg)", pad=4)
    ax.set_xlim(win.start[0], win.end[0])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_direction_rasters(sess, tsg, trials, ori, unit, path):
    """Trial rasters and PSTHs sorted by drift direction for one unit."""
    fig = plt.figure(figsize=(11, 6.0))
    gs = fig.add_gridspec(2, 8, height_ratios=[2.4, 1], hspace=0.35, wspace=0.25)
    pre, post = 0.5, 2.5
    tc = np.zeros(8)
    for j, dd in enumerate(A.DIRECTIONS):
        sel = ori == dd
        ep = nap.IntervalSet(start=trials.start[sel] - pre, end=trials.start[sel] + post)
        tensor = nap.build_tensor(tsg[[unit]], ep, bin_size=0.05)[0]
        edges = np.arange(-pre, post, 0.05)[: tensor.shape[1]]
        ax = fig.add_subplot(gs[0, j])
        for r in range(tensor.shape[0]):
            idx = np.where(tensor[r] > 0)[0]
            ax.plot(edges[idx], np.full(idx.shape, r), "|", color="k", markersize=2.2)
        ax.axvspan(0, 2, color="#c1272d", alpha=0.12)
        ax.set_xlim(-pre, post)
        ax.set_ylim(-1, tensor.shape[0])
        ax.set_title(f"{int(dd)}°", pad=3)
        if j:
            ax.set_yticks([])
        else:
            ax.set_ylabel("trial")

        ax2 = fig.add_subplot(gs[1, j])
        psth = np.nanmean(tensor, axis=0) / 0.05
        ax2.fill_between(edges, psth, color="#c1272d", alpha=0.75, lw=0)
        ax2.axvspan(0, 2, color="#c1272d", alpha=0.12)
        ax2.set_xlim(-pre, post)
        ax2.set_xticks([0, 2])
        if j:
            ax2.set_yticks([])
        else:
            ax2.set_ylabel("rate (Hz)")
            ax2.set_xlabel("time from onset (s)")
        tc[j] = np.nanmean(psth[(edges >= 0) & (edges <= 2)])
    ymax = max(ax2.get_ylim()[1] for ax2 in fig.axes[1::2])
    for ax2 in fig.axes[1::2]:
        ax2.set_ylim(0, ymax)
    fig.suptitle(
        f"Unit {sess['unit_id'][unit]} (VISp): responses to the 8 drift directions"
        f"   |   gOSI = {A.global_osi(tc[None, :])[0]:.2f}",
        y=0.99,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_example_tuning(sess, tc, sem, units, path, region="VISp"):
    """Cartesian and polar tuning curves with double von Mises fits."""
    n = len(units)
    fig = plt.figure(figsize=(3.2 * n, 6.2))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.15], hspace=0.62, wspace=0.38)
    theta_fine = np.linspace(0, 360, 361)
    for k, u in enumerate(units):
        y = tc[u]
        popt, r2 = A.fit_double_von_mises(y)
        ax = fig.add_subplot(gs[0, k])
        ax.errorbar(
            A.DIRECTIONS, y, yerr=sem[u], fmt="o", color="k", ms=4, lw=1, capsize=2
        )
        ax.plot(
            theta_fine, A.double_von_mises(theta_fine, *popt), color="#c1272d", lw=1.5
        )
        ax.set_xticks(A.DIRECTIONS[::2])
        ax.set_xlim(-15, 375)
        ax.set_xlabel("drift direction (deg)")
        if k == 0:
            ax.set_ylabel("firing rate (Hz)")
        hwhm = A.circular_tuning_width(popt)
        ax.set_title(
            f"unit {sess['unit_id'][u]} ({region})\n"
            f"gOSI={A.global_osi(y[None])[0]:.2f}   gDSI={A.global_dsi(y[None])[0]:.2f}\n"
            f"von Mises $R^2$={r2:.2f}   HWHM={hwhm:.0f}°",
            pad=6,
            fontsize=9,
        )

        axp = fig.add_subplot(gs[1, k], projection="polar")
        th = np.deg2rad(np.r_[A.DIRECTIONS, 360])
        axp.plot(th, np.r_[y, y[0]], "o-", color="k", ms=3.5, lw=1.2)
        axp.plot(
            np.deg2rad(theta_fine),
            A.double_von_mises(theta_fine, *popt),
            color="#c1272d",
            lw=1.4,
        )
        pref = A.preferred_orientation(y[None])[0]
        for a in (pref, pref + 180):
            axp.plot(
                [np.deg2rad(a)] * 2, [0, y.max()], color="#3b6fb6", lw=1.2, ls="--"
            )
        axp.set_theta_zero_location("E")
        axp.set_rlabel_position(135)
        axp.tick_params(labelsize=7, pad=1)
        axp.set_title(f"preferred orientation axis {pref:.0f}°", fontsize=8, pad=14)
    fig.suptitle(
        "Orientation-selective units in mouse primary visual cortex (drifting gratings)",
        y=0.99,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _running_mean_rows(y, w):
    """Smooth along the unit axis; neighbouring rows have similar preferences."""
    if w <= 1:
        return y
    pad = np.vstack([y[: w // 2][::-1], y, y[-(w // 2) :][::-1]])
    ker = np.ones(w) / w
    return np.stack(
        [np.convolve(pad[:, j], ker, mode="same")[w // 2 : w // 2 + len(y)]
         for j in range(y.shape[1])], axis=1
    )


def fig_population_heatmap(tc, tc_a, tc_b, loc, path):
    """Peak-normalised direction tuning curves per structure, plus aligned averages.

    Units are sorted by the preferred orientation estimated from one half of the
    trials, and the other half is displayed, so a diagonal band can only appear
    if the tuning is real. Because each unit peaks at both ends of its preferred
    axis, an orientation-selective population produces two parallel diagonal
    bands offset by 180 deg. The right-hand panel applies the same
    cross-validated alignment and averages across units.
    """
    regions = [r for r in REGION_COLORS if (loc == r).sum() >= 30]
    fig = plt.figure(figsize=(1.85 * len(regions) + 5.0, 4.5))
    gs = fig.add_gridspec(
        1, len(regions) + 3,
        width_ratios=[1] * len(regions) + [0.07, 0.35, 2.1], wspace=0.28,
    )
    for j, r in enumerate(regions):
        ax = fig.add_subplot(gs[0, j])
        sel = loc == r
        # normalised by the peak of the *other* half, so noise in the displayed
        # half cannot manufacture a bright pixel
        y = np.clip(tc_b[sel], 0, None) / np.maximum(
            np.clip(tc_a[sel], 0, None).max(axis=1, keepdims=True), 1e-9
        )
        order = np.argsort(A.preferred_orientation(tc_a[sel]))
        y = _running_mean_rows(y[order], max(1, len(order) // 25))
        im = ax.imshow(
            y, aspect="auto", cmap="magma", vmin=0, vmax=1.1,
            extent=[-22.5, 337.5, len(order), 0], interpolation="nearest",
        )
        ax.set_xticks([0, 180])
        ax.set_title(f"{r}\nn={sel.sum()}", pad=5)
        ax.set_xlabel("direction (deg)")
        if j == 0:
            ax.set_ylabel("unit (sorted by preferred orientation\nfrom the other half of the trials)")
    cax = fig.add_subplot(gs[0, len(regions)])
    fig.colorbar(im, cax=cax).set_label("normalised rate")

    ax = fig.add_subplot(gs[0, -1])
    rel = np.arange(-4, 5) * 45.0
    for r in regions:
        sel = loc == r
        k = np.argmax(np.clip(tc_a[sel], 0, None), axis=1)
        y = np.clip(tc_b[sel], 0, None)
        y = y / np.maximum(y.max(axis=1, keepdims=True), 1e-9)
        rolled = np.stack([np.roll(y[i], -k[i] + 4) for i in range(len(k))])
        rolled = np.column_stack([rolled, rolled[:, 0]])  # wrap -180 to +180
        m, e = rolled.mean(0), rolled.std(0) / np.sqrt(len(rolled))
        ax.plot(rel, m, "o-", color=REGION_COLORS[r], ms=3, lw=1.4, label=r)
        ax.fill_between(rel, m - e, m + e, color=REGION_COLORS[r], alpha=0.2, lw=0)
    ax.axvline(-180, color="0.6", ls=":", lw=1)
    ax.axvline(180, color="0.6", ls=":", lw=1)
    ax.set_xticks(rel[::2])
    ax.set_xlabel("direction relative to preferred (deg)")
    ax.set_ylabel("normalised rate (held-out trials)")
    ax.set_title(
        "Cross-validated aligned average:\nthe rise at $\\pm$180° is the second lobe",
        pad=5,
    )
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_selectivity_distributions(df, path):
    """gOSI / gDSI distributions and significant fractions per structure."""
    regions = [r for r in REGION_COLORS if (df["location"] == r).sum() >= 15]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(
        df["gOSI_null"].values, bins=bins, density=True, color="0.75",
        label="shuffled null", zorder=0,
    )
    for r in regions:
        v = df.loc[df["location"] == r, "gOSI"].values
        ax.hist(
            v, bins=bins, histtype="step", density=True, lw=1.6,
            color=REGION_COLORS[r], label=f"{r} (n={len(v)})",
        )
    ax.set_xlabel("global OSI")
    ax.set_ylabel("probability density")
    ax.set_title("Orientation selectivity by structure", pad=6)
    ax.legend(fontsize=7)

    ax = axes[1]
    pos = np.arange(len(regions))
    for i, r in enumerate(regions):
        v = df.loc[df["location"] == r, "gOSI"].values
        parts = ax.violinplot([v], positions=[i], widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(REGION_COLORS[r])
            b.set_alpha(0.55)
        ax.plot([i], [np.median(v)], "o", color="k", ms=4)
        ax.plot([i - 0.2, i + 0.2], [np.median(v)] * 2, color="k", lw=1.4)
    ax.set_xticks(pos)
    ax.set_xticklabels(regions)
    ax.set_ylabel("global OSI")
    ax.set_title("Thalamus vs. cortex", pad=6)

    ax = axes[2]
    sel = (df["p_osi"] < 0.05) & (df["gOSI"] >= 0.25)
    frac = [(sel & (df["location"] == r)).sum() / (df["location"] == r).sum() * 100
            for r in regions]
    ax.bar(pos, frac, color=[REGION_COLORS[r] for r in regions], alpha=0.85)
    for i, (f, r) in enumerate(zip(frac, regions)):
        n = (df["location"] == r).sum()
        ax.text(i, f + 1.5, f"{f:.0f}%\nn={n}", ha="center", fontsize=7.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(regions)
    ax.set_ylabel("% of units")
    ax.set_ylim(0, max(frac) * 1.28)
    ax.set_title("Orientation selective\n(p < 0.05 and gOSI $\\geq$ 0.25)", pad=6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(pref_a, pref_b, pref_all, gosi, sig, path):
    """Split-half reproducibility and the distribution of preferred orientations."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    ax = axes[0]
    sc = ax.scatter(
        pref_a, pref_b, c=gosi, cmap="viridis", s=18, vmin=0, vmax=np.percentile(gosi, 98)
    )
    ax.plot([0, 180], [0, 180], "k--", lw=1)
    ax.set_xlabel("preferred orientation, odd trials (deg)")
    ax.set_ylabel("preferred orientation, even trials (deg)")
    r, p = A.circ_corr_axial(pref_a, pref_b)
    p_txt = "p < 1e-16" if p < 1e-16 else f"p = {p:.1e}"
    ax.set_title(f"Split-half reproducibility\ncircular r = {r:.2f}, {p_txt}", pad=6)
    plt.colorbar(sc, ax=ax, label="gOSI")

    ax = axes[1]
    diff = (pref_a - pref_b + 90) % 180 - 90
    ax.hist(diff, bins=np.arange(-90, 91, 10), color="#3b6fb6", alpha=0.85)
    ax.set_xlabel("odd - even preferred orientation (deg)")
    ax.set_ylabel("units")
    ax.set_title(
        f"Median |difference| = {np.median(np.abs(diff)):.1f}°", pad=6
    )

    ax = axes[2]
    bins = np.arange(0, 181, 15)
    ax.hist(
        pref_all[sig], bins=bins, color="#c1272d", alpha=0.85,
        label=f"tuned (n={sig.sum()})",
    )
    ax.hist(
        pref_all[~sig], bins=bins, histtype="step", color="0.4", lw=1.5,
        label=f"untuned (n={(~sig).sum()})",
    )
    ax.set_xlabel("preferred orientation (deg)")
    ax.set_ylabel("units")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title("Preferred orientations across the population", pad=6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_static_vs_drifting(pref_dg, pref_sg, gosi_dg_all, gosi_sg_all, gosi_dg, gosi_sg, offset, path):
    """Cross-stimulus validation: static-grating vs drifting-grating preferences."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    ax = axes[0]
    ax.scatter(gosi_dg_all, gosi_sg_all, s=10, color="0.7", alpha=0.6, lw=0,
               label=f"all cortical units (n={len(gosi_dg_all)})")
    ax.scatter(gosi_dg, gosi_sg, s=14, color="#3b6fb6", alpha=0.8, lw=0,
               label=f"tuned to both (n={len(gosi_dg)})")
    lim = max(gosi_dg_all.max(), gosi_sg_all.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    r = np.corrcoef(gosi_dg_all, gosi_sg_all)[0, 1]
    ax.set_xlabel("gOSI, drifting gratings")
    ax.set_ylabel("gOSI, static gratings")
    ax.set_title(f"Selectivity is stimulus-general\nPearson r = {r:.2f}", pad=6)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ax.scatter(pref_dg, pref_sg, s=16, color="#c1272d", alpha=0.75)
    ax.set_xlabel("preferred orientation, drifting (deg)")
    ax.set_ylabel("preferred orientation, static (deg)")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    r, p = A.circ_corr_axial(pref_dg, pref_sg)
    p_txt = "p < 1e-16" if p < 1e-16 else f"p = {p:.1e}"
    ax.set_title(f"Preferred orientation agrees across\nstimulus families: circular r = {r:.2f}, {p_txt}", pad=6)

    ax = axes[2]
    ax.hist(offset, bins=np.arange(-90, 91, 10), color="#7b52a1", alpha=0.85)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("static - drifting preferred orientation (deg)")
    ax.set_ylabel("units")
    ax.set_title(
        "Offset between the two stimulus families\n"
        f"(circular mean {np.rad2deg(np.angle(np.exp(2j*np.deg2rad(offset)).mean()))/2:.0f}°)",
        pad=6,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_decoding(conf, acc, curve_sizes, curve_acc, curve_sem, n_sessions, path):
    """Population decoding of drift direction and the structure of its errors."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    ax = axes[0]
    im = ax.imshow(conf * 100, cmap="magma", vmin=0)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([int(d) for d in A.DIRECTIONS], fontsize=7.5)
    ax.set_yticklabels([int(d) for d in A.DIRECTIONS], fontsize=7.5)
    ax.set_xlabel("decoded direction (deg)")
    ax.set_ylabel("true direction (deg)")
    ax.set_title(
        f"VISp population decoder, mean of {n_sessions} sessions\n"
        f"{acc*100:.1f}% correct (chance 12.5%)", pad=6,
    )
    plt.colorbar(im, ax=ax, label="% of trials")

    ax = axes[1]
    err = (np.arange(8)[None, :] - np.arange(8)[:, None]) * 45
    err = (err + 180) % 360 - 180  # signed error, -180 .. +135
    offsets = np.arange(-180, 180, 45)
    profile = np.array([conf[err == e].mean() for e in offsets])
    ax.bar(offsets, profile * 100, width=32, color="#4a9b8f")
    ax.set_xticks(offsets[::2])
    ax.set_xlabel("decoding error (deg)")
    ax.set_ylabel("% of trials")
    ax.set_title(
        "Errors concentrate at 180°:\nthe axis is decoded better than the direction",
        pad=6,
    )

    ax = axes[2]
    ax.errorbar(curve_sizes, np.array(curve_acc) * 100, yerr=np.array(curve_sem) * 100,
                fmt="o-", color="#c1272d", ms=4, capsize=3, label="direction (8-way)")
    ax.axhline(12.5, color="k", ls="--", lw=1, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("number of VISp units")
    ax.set_ylabel("% correct")
    ax.set_title("Decoding improves with population size", pad=6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_multisession(df, path):
    """Pooled results across sessions."""
    sessions = sorted(df["session"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))

    ax = axes[0]
    regions = [r for r in REGION_COLORS if (df["location"] == r).sum() >= 30]
    width = 0.8 / len(regions)
    for i, r in enumerate(regions):
        med, lo, hi, xs = [], [], [], []
        for j, s in enumerate(sessions):
            v = df.loc[(df["session"] == s) & (df["location"] == r), "gOSI"].values
            if len(v) < 5:
                continue
            med.append(np.median(v))
            boot = [
                np.median(np.random.choice(v, len(v))) for _ in range(500)
            ]
            lo.append(np.median(v) - np.percentile(boot, 2.5))
            hi.append(np.percentile(boot, 97.5) - np.median(v))
            xs.append(j + i * width - 0.4)
        ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", ms=4, color=REGION_COLORS[r],
                    label=r, capsize=2, lw=1)
    ax.legend(fontsize=7, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([s.split("_")[-1].replace("ses-", "") for s in sessions],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("median gOSI (95% CI)")
    ax.set_xlabel("session")
    ax.set_title(f"Per-session medians ({len(sessions)} sessions)", pad=30)

    ax = axes[1]
    for r in regions:
        v = np.sort(df.loc[df["location"] == r, "gOSI"].values)
        ax.plot(v, np.linspace(0, 1, len(v)), color=REGION_COLORS[r], lw=1.8,
                label=f"{r} (n={len(v)})")
    ax.set_xlabel("global OSI")
    ax.set_ylabel("cumulative fraction of units")
    ax.set_title("Pooled across all sessions", pad=6)
    ax.legend(fontsize=7)

    ax = axes[2]
    sig = (df["p_osi"] < 0.05) & (df["gOSI"] >= 0.25)
    resp = df["p_resp"] < 0.05
    frac = [
        (sig & resp & (df["location"] == r)).sum() / max((resp & (df["location"] == r)).sum(), 1) * 100
        for r in regions
    ]
    ax.bar(range(len(regions)), frac, color=[REGION_COLORS[r] for r in regions], alpha=0.85)
    for i, f in enumerate(frac):
        n = (resp & (df["location"] == regions[i])).sum()
        ax.text(i, f + 1.2, f"{f:.0f}%\nn={n}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_ylabel("% of visually responsive units")
    ax.set_ylim(0, max(frac) * 1.3)
    ax.set_title("Orientation-tuned fraction, pooled", pad=6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_glm_and_summary(df, grid, curve_dir, curve_ori, tc_example, sess, unit, path):
    """NeMoS encoding-model results: the 180 deg model versus the 360 deg model."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))

    ax = axes[0]
    dur = 2.0
    ax.plot(A.DIRECTIONS, tc_example, "o", color="k", ms=5, label="measured", zorder=5)
    ax.plot(grid, curve_dir / dur, color="#c1272d", lw=1.8, label="360° direction model")
    ax.plot(grid, curve_ori / dur, color="#3b6fb6", lw=1.8, ls="--",
            label="180° orientation model")
    ax.set_xticks(A.DIRECTIONS[::2])
    ax.set_xlim(-15, 375)
    ax.set_xlabel("drift direction (deg)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"Both GLMs, unit {sess['unit_id'][unit]} (VISp)", pad=6)
    ax.legend(fontsize=7)

    ax = axes[1]
    regions = [r for r in REGION_COLORS if (df["location"] == r).sum() >= 30]
    for r in regions:
        m = df["location"] == r
        ax.scatter(df.loc[m, "glm_r2_dir"], df.loc[m, "glm_r2_ori"], s=9,
                   color=REGION_COLORS[r], alpha=0.55, label=r, lw=0)
    lim = np.nanpercentile(df["glm_r2_dir"], 99.5)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel("cross-validated pseudo-$R^2$, 360° model")
    ax.set_ylabel("cross-validated pseudo-$R^2$, 180° model")
    ax.set_title("The 180° model loses almost nothing", pad=6)
    ax.legend(fontsize=7, markerscale=1.6)

    ax = axes[2]
    ok = df["glm_r2_dir"] > 0.02
    pos = np.arange(len(regions))
    for i, r in enumerate(regions):
        v = (df.loc[ok & (df["location"] == r), "glm_r2_ori"]
             / df.loc[ok & (df["location"] == r), "glm_r2_dir"]).values
        v = v[np.isfinite(v)]
        parts = ax.violinplot([v], positions=[i], widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(REGION_COLORS[r])
            b.set_alpha(0.55)
        ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color="k", lw=1.6)
        ax.text(i, 1.32, f"n={len(v)}", ha="center", fontsize=7)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(regions)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("pseudo-$R^2$ ratio (180° / 360°)")
    ax.set_title("Direction adds little in cortex,\nmore in thalamus", pad=6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
