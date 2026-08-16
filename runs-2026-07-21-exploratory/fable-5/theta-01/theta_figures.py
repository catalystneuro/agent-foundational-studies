"""Figure generation for the theta entrainment / phase precession analysis."""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pynapple as nap

import theta_analysis as A

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

EXC_C, INH_C = "#2c6fbb", "#c0392b"


def theta_guide(ax, n_cycles=2, color="0.55"):
    """Draw a schematic cosine along the top of a phase axis (x in cycles*2pi)."""
    x = np.linspace(0, n_cycles * 2 * np.pi, 400)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.18)
    a = 0.05 * (y1 - y0)
    ax.plot(x, y1 * 1.12 - a + a * np.cos(x), color=color, lw=1.0)


def phase_axis(ax, n_cycles=2):
    ax.set_xlim(0, n_cycles * 2 * np.pi)
    ticks = np.arange(0, n_cycles * 2 * np.pi + 0.1, np.pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(np.degrees(t))}" for t in ticks])


# --------------------------------------------------------------- figure 1
def fig_raw_streams(session, position, lfp, filt, phase, units, runs, direction,
                    psd_f, psd_p, lap_idx, fname):
    lap_start, lap_end = runs[lap_idx].start[0], runs[lap_idx].end[0]
    win = nap.IntervalSet(start=lap_start - 0.3, end=lap_end + 0.3)
    t0 = win.start[0]

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(4, 3, height_ratios=[1, 1.2, 1, 2.3], width_ratios=[3, 3, 1.6],
                  hspace=0.3, wspace=0.4)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(position.restrict(win).t - t0, position.restrict(win).values,
            "k.-", ms=3, lw=0.8)
    ax.set_ylabel("position (m)")
    ax.set_title(f"{session} — one traversal of the linear track "
                 f"({'left to right' if direction[lap_idx] > 0 else 'right to left'})")

    ax2 = fig.add_subplot(gs[1, :2], sharex=ax)
    ax2.plot(lfp.restrict(win).t - t0, lfp.restrict(win).values, color="0.7",
             lw=0.6, label="raw LFP (1250 Hz)")
    ax2.plot(filt.restrict(win).t - t0, filt.restrict(win).values, color="C0",
             lw=1.5, label="6–10 Hz")
    ax2.set_ylabel("LFP (µV)")
    ax2.legend(loc="upper right", ncol=2, fontsize=8)

    ax3 = fig.add_subplot(gs[2, :2], sharex=ax)
    ax3.plot(phase.restrict(win).t - t0, np.degrees(phase.restrict(win).values),
             color="C1", lw=0.8)
    ax3.set_ylabel("theta phase (deg)")
    ax3.set_yticks([0, 180, 360])

    ax4 = fig.add_subplot(gs[3, :2], sharex=ax)
    exc = units[np.array(units.cell_type) == "excitatory"]
    for k, uid in enumerate(exc.index):
        st = exc[uid].restrict(win)
        ax4.plot(st.t - t0, np.full(len(st), k), "|", color="k", ms=3.5, mew=0.7)
    ax4.set_ylabel("excitatory unit")
    ax4.set_xlabel("time from window onset (s)")
    ax4.set_ylim(-2, len(exc.index) + 1)
    ax4.set_xlim(0, win.end[0] - t0)
    for a in (ax, ax2, ax3):
        a.tick_params(labelbottom=False)

    rgs = gs[:, 2].subgridspec(2, 1, hspace=0.45, height_ratios=[1, 1.2])
    axp = fig.add_subplot(rgs[0])
    vis = psd_f <= 25
    axp.semilogy(psd_f[vis], psd_p[vis], "k", lw=1.2)
    axp.axvspan(6, 10, color="C0", alpha=0.2)
    axp.set_xlim(0, 25)
    axp.set_xlabel("frequency (Hz)"); axp.set_ylabel("PSD (µV²/Hz)")
    axp.set_title("LFP spectrum\nduring running")

    axl = fig.add_subplot(rgs[1])
    for i in range(len(runs)):
        p = position.restrict(runs[i])
        axl.plot(p.t - p.t[0], p.values, lw=0.6,
                 color=("C0" if direction[i] > 0 else "C3"), alpha=0.5)
    axl.set_xlabel("time from lap start (s)"); axl.set_ylabel("position (m)")
    axl.set_title(f"All {len(runs)} laps\n(blue L→R, red R→L)", pad=12)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 2
def fig_place_fields(res, fname):
    cells = res["cells"]
    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(2, 4, width_ratios=[1.3, 1.3, 2, 2], hspace=0.45, wspace=0.42)

    for j, lbl in enumerate(["R", "L"]):
        tc = res["rate_maps"][lbl]
        sel = cells[(cells.direction == lbl) & cells.is_place_cell]
        ids = sel.sort_values("peak_pos").unit.values
        M = tc[ids].values.T
        M = M / np.maximum(M.max(axis=1, keepdims=True), 1e-9)
        ax = fig.add_subplot(gs[:, j])
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       extent=[tc.index[0], tc.index[-1], 0, len(ids)],
                       interpolation="nearest")
        ax.set_xlabel("position (m)")
        ax.set_ylabel("place cell (sorted by field peak)" if j == 0 else "")
        ax.set_title(f"{'left→right' if lbl == 'R' else 'right→left'} laps\n"
                     f"n = {len(ids)} place fields")
        plt.colorbar(im, ax=ax, label="normalized rate", pad=0.02)

    # four example fields, both directions
    pcs = cells[cells.is_place_cell].sort_values("peak_rate", ascending=False)
    picks = pcs.drop_duplicates("unit").head(4)
    for k, (_, row) in enumerate(picks.iterrows()):
        ax = fig.add_subplot(gs[k // 2, 2 + k % 2])
        for lbl, c in [("R", "C0"), ("L", "C3")]:
            tc = res["rate_maps"][lbl]
            if row.unit in tc.columns:
                ax.plot(tc.index, tc[row.unit].values, color=c, lw=1.5,
                        label="left→right" if lbl == "R" else "right→left")
        ax.axvspan(row.field_start, row.field_end, color="0.85", zorder=0)
        ax.set_title(f"unit {int(row.unit)} — {row.spatial_info:.2f} bits/spike "
                 f"({'L→R' if row.direction == 'R' else 'R→L'} field shaded)",
                 fontsize=9)
        ax.set_xlabel("position (m)"); ax.set_ylabel("rate (Hz)")
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{res['session']}: directional place fields on the linear track", y=0.98)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3
def fig_entrainment(entrain, hists, edges, mean_wave_phase, mean_wave, fname):
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(3, 3, hspace=0.9, wspace=0.4, height_ratios=[1, 1, 1.15])

    # A: mean theta waveform, defining the phase convention
    ax = fig.add_subplot(gs[0, 0])
    x2 = np.r_[mean_wave_phase, mean_wave_phase + 2 * np.pi]
    ax.plot(x2, np.r_[mean_wave, mean_wave], "k", lw=1.6)
    phase_axis(ax); ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("theta phase (deg)"); ax.set_ylabel("mean LFP (µV)")
    ax.set_title("Phase convention:\n0° = peak of band-passed LFP")

    ctr = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    # B,C: two example units (best-entrained excitatory and inhibitory)
    for k, ctype in enumerate(["excitatory", "inhibitory"]):
        sub = entrain[(entrain.cell_type == ctype) & (entrain.n_spikes > 500)]
        row = sub.sort_values("mrl", ascending=False).iloc[0]
        h = hists[(row.session, row.unit)]
        ax = fig.add_subplot(gs[0, 1 + k])
        ax.bar(np.r_[ctr, ctr + 2 * np.pi], np.r_[h, h], width=width,
               color=EXC_C if ctype == "excitatory" else INH_C)
        phase_axis(ax)
        ax.set_xlabel("theta phase (deg)"); ax.set_ylabel("spike count")
        ax.set_title(f"{ctype} unit {int(row.unit)} ({row.session})\n"
                     f"MRL = {row.mrl:.2f}, φ = {np.degrees(row.pref_phase):.0f}°, "
                     f"p = {row.rayleigh_p:.0e}")
        theta_guide(ax)

    # D: pooled spike-phase histograms by cell type
    ax = fig.add_subplot(gs[1, 0])
    for ctype, c in [("excitatory", EXC_C), ("inhibitory", INH_C)]:
        pooled = np.sum([hists[(r.session, r.unit)]
                         for _, r in entrain[entrain.cell_type == ctype].iterrows()],
                        axis=0).astype(float)
        pooled /= pooled.sum()
        ax.plot(np.r_[ctr, ctr + 2 * np.pi], np.r_[pooled, pooled],
                color=c, lw=1.6, label=ctype)
    phase_axis(ax); ax.legend(fontsize=8)
    ax.set_xlabel("theta phase (deg)"); ax.set_ylabel("fraction of spikes")
    ax.set_title("Pooled spike-phase distribution\n(all sessions, running only)")

    # E: MRL distributions
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, entrain.mrl.max() * 1.05, 30)
    for ctype, c in [("excitatory", EXC_C), ("inhibitory", INH_C)]:
        ax.hist(entrain[entrain.cell_type == ctype].mrl, bins=bins, alpha=0.65,
                color=c, label=ctype)
    ax.set_xlabel("mean resultant length"); ax.set_ylabel("units")
    ax.set_title("Strength of theta locking"); ax.legend(fontsize=8)

    # F: Rayleigh p vs n spikes
    ax = fig.add_subplot(gs[1, 2])
    for ctype, c in [("excitatory", EXC_C), ("inhibitory", INH_C)]:
        s = entrain[entrain.cell_type == ctype]
        ax.scatter(s.n_spikes, np.maximum(s.rayleigh_p, 1e-60), s=12, color=c,
                   alpha=0.7, label=ctype)
    ax.axhline(0.01, color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(3e-61, 30)
    ax.text(0.99, 0.02, "dashed line: p = 0.01\nvalues clipped at $10^{-60}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5)
    ax.set_xlabel("spikes during running")
    ax.set_ylabel("Rayleigh p")
    ax.set_title("Significance of theta locking")
    ax.legend(fontsize=8, loc="lower left")

    # G: polar preferred-phase distribution
    ax = fig.add_subplot(gs[2, 0], projection="polar")
    for ctype, c in [("excitatory", EXC_C), ("inhibitory", INH_C)]:
        s = entrain[(entrain.cell_type == ctype) & (entrain.rayleigh_p < 0.01)]
        ax.scatter(s.pref_phase, s.mrl, s=14, color=c, alpha=0.75, label=ctype)
    ax.set_theta_zero_location("E")
    ax.set_rlabel_position(112)
    ax.set_title("Preferred phase vs locking strength\n(theta-locked units, p < 0.01)",
                 pad=28)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    # H: preferred phase histogram
    ax = fig.add_subplot(gs[2, 1])
    for ctype, c in [("excitatory", EXC_C), ("inhibitory", INH_C)]:
        s = entrain[(entrain.cell_type == ctype) & (entrain.rayleigh_p < 0.01)]
        ax.hist(np.r_[s.pref_phase, s.pref_phase + 2 * np.pi], bins=36,
                range=(0, 4 * np.pi), alpha=0.65, color=c, label=ctype)
    phase_axis(ax); ax.legend(fontsize=8)
    ax.set_xlabel("preferred theta phase (deg)"); ax.set_ylabel("units")
    ax.set_title("Preferred phase across the population")

    # I: fraction entrained per session
    ax = fig.add_subplot(gs[2, 2])
    g = (entrain.assign(sig=entrain.rayleigh_p < 0.01)
         .groupby(["session", "cell_type"]).sig.mean().unstack())
    g = g.reindex(columns=["excitatory", "inhibitory"])
    xs = np.arange(len(g))
    ax.bar(xs - 0.19, g["excitatory"] * 100, 0.36, color=EXC_C, label="excitatory")
    ax.bar(xs + 0.19, g["inhibitory"] * 100, 0.36, color=INH_C, label="inhibitory")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("_", "\n") for s in g.index], fontsize=6.5)
    ax.set_ylabel("% units theta-locked (p < 0.01)")
    ax.set_ylim(0, 105); ax.legend(fontsize=8)
    ax.set_title("Theta locking is reproducible\nacross sessions")

    fig.suptitle("Theta phase entrainment of CA1 units during track running", y=0.995)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4
def fig_precession_examples(res, spikes, fname, n=8, n_perm=500):
    cells = res["cells"]
    track_len = float(res["track_len"])
    ok = (cells.is_place_cell & (cells.perm_p < 0.05) & (cells.slope < 0)
          & (cells.n_spikes_field >= 150)
          & (cells.field_start > 0.05) & (cells.field_end < track_len - 0.05))
    pcs = cells[ok].sort_values("rho").head(n)
    ncol = 4
    nrow = int(np.ceil(len(pcs) / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(3.4 * ncol, 4.1 * nrow),
                             gridspec_kw={"height_ratios": [0.55, 1] * nrow,
                                          "hspace": 0.85, "wspace": 0.38})
    axes = np.atleast_2d(axes)
    for k, (_, row) in enumerate(pcs.iterrows()):
        r, c = (k // ncol) * 2, k % ncol
        tc = res["rate_maps"][row.direction]
        axr = axes[r, c]
        axr.plot(tc.index, tc[row.unit].values, color="k", lw=1.3)
        axr.axvspan(row.field_start, row.field_end, color="C1", alpha=0.2)
        axr.set_xlabel("position (m)", labelpad=1)
        axr.set_ylabel("rate (Hz)")
        axr.set_title(f"unit {int(row.unit)}, {'L→R' if row.direction == 'R' else 'R→L'}",
                      fontsize=9)

        s = spikes[(spikes.session == row.session) & (spikes.unit == row.unit)
                   & (spikes.direction == row.direction)]
        ax = axes[r + 1, c]
        ax.scatter(np.r_[s.x_norm, s.x_norm], np.r_[np.degrees(s.phase),
                                                    np.degrees(s.phase) + 360],
                   s=6, color="0.25", alpha=0.55, edgecolors="none")
        xx = np.linspace(0, 1, 100)
        yy = np.degrees(row.phase0 + 2 * np.pi * row.slope * xx)
        for off in (-360, 0, 360, 720):
            ax.plot(xx, yy + off, color="C1", lw=1.6)
        ax.set_ylim(0, 720); ax.set_xlim(0, 1)
        ax.set_yticks([0, 180, 360, 540, 720])
        ax.set_xlabel("normalized position in field")
        ax.set_ylabel("theta phase (deg)")
        ptxt = ("p < 0.002" if row.perm_p <= 2.0 / (n_perm + 1)
                else f"p = {row.perm_p:.3f}")
        ax.set_title(f"slope {row.slope:.2f} cyc/field, ρ = {row.rho:.2f}, {ptxt}",
                     fontsize=8.5)
    for k in range(len(pcs), nrow * ncol):
        r, c = (k // ncol) * 2, k % ncol
        axes[r, c].axis("off"); axes[r + 1, c].axis("off")
    fig.suptitle(f"{res['session']}: theta phase precession in single place fields",
                 y=0.995)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 5
def fig_precession_population(all_cells, all_spikes, null_rhos, fname):
    pcs = all_cells[all_cells.is_place_cell]
    sig = pcs[pcs.perm_p < 0.05]
    sig_keys = set(zip(sig.session, sig.unit, sig.direction))
    sp = all_spikes[[k in sig_keys for k in
                     zip(all_spikes.session, all_spikes.unit, all_spikes.direction)]]

    fig = plt.figure(figsize=(13, 7.5))
    gs = GridSpec(2, 3, hspace=0.5, wspace=0.32,
                  width_ratios=[1.45, 1, 1])

    # A: pooled density
    ax = fig.add_subplot(gs[:, 0])
    x = np.r_[sp.x_norm, sp.x_norm]
    y = np.r_[np.degrees(sp.phase), np.degrees(sp.phase) + 360]
    H, xe, ye = np.histogram2d(x, y, bins=[25, 48], range=[[0, 1], [0, 720]])
    H = H / H.sum(axis=1, keepdims=True)
    ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
              extent=[0, 1, 0, 720], interpolation="bilinear")
    ax.set_yticks([0, 180, 360, 540, 720])
    ax.set_xlabel("normalized position in place field")
    ax.set_ylabel("theta phase (deg)")
    ax.set_title(f"Pooled spike density\n{len(sig)} fields, {len(sp):,} spikes")

    # B: circular mean phase per position bin
    ax = fig.add_subplot(gs[0, 1])
    edges = np.linspace(0, 1, 13)
    idx = np.digitize(sp.x_norm.values, edges) - 1
    mus, ns = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        mus.append(A.circ_mean(sp.phase.values[m]) if m.sum() > 20 else np.nan)
        ns.append(int(m.sum()))
    mus = np.degrees(np.array(mus))
    ctr = (edges[:-1] + edges[1:]) / 2
    mus_u = np.copy(mus)
    for i in range(1, len(mus_u)):                     # unwrap downward
        while mus_u[i] > mus_u[i - 1] + 180:
            mus_u[i] -= 360
    ax.plot(ctr, mus_u, "o-", color="C1", lw=1.6)
    ax.plot(ctr, mus_u + 360, "o-", color="C1", lw=1.6, alpha=0.4)
    ax.set_xlabel("normalized position in place field")
    ax.set_ylabel("circular mean phase (deg)")
    ax.set_title("Mean spike phase advances\nacross the field")

    # C: slope distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(pcs.slope, bins=np.linspace(-2, 2, 33), color="0.6", label="all fields")
    ax.hist(sig.slope, bins=np.linspace(-2, 2, 33), color="C1",
            label="significant (p < 0.05)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("regression slope (theta cycles per field)")
    ax.set_ylabel("place fields"); ax.legend(fontsize=8)
    ax.set_title(f"Slopes are predominantly negative\n"
                 f"{100 * (pcs.slope < 0).mean():.0f}% of all fields, "
                 f"{100 * (sig.slope < 0).mean():.0f}% of significant")

    # D: rho vs shuffle
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, max(np.abs(pcs.rho).max(), null_rhos.max()) * 1.05, 30)
    ax.hist(null_rhos, bins=bins, color="0.75", density=True, label="shuffled")
    ax.hist(np.abs(pcs.rho), bins=bins, color="C1", alpha=0.75, density=True,
            label="observed")
    ax.set_xlabel("|circular–linear correlation|"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.set_title("Observed vs shuffled correlation")

    # E: per-session summary
    ax = fig.add_subplot(gs[1, 2])
    g = pcs.groupby("session").apply(
        lambda d: pd.Series({"n": len(d),
                             "pct_sig": 100 * (d.perm_p < 0.05).mean(),
                             "pct_neg": 100 * (d.slope < 0).mean()}),
        include_groups=False)
    xs = np.arange(len(g))
    ax.bar(xs - 0.19, g.pct_sig, 0.36, color="C1", label="significant precession")
    ax.bar(xs + 0.19, g.pct_neg, 0.36, color="0.5", label="negative slope")
    for i, n in enumerate(g.n):
        ax.text(i, 103, f"n={int(n)}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("_", "\n") for s in g.index], fontsize=6.5)
    ax.set_ylim(0, 115); ax.set_ylabel("% of place fields")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2)
    ax.set_title("Precession across sessions")

    fig.suptitle("Theta phase precession of CA1 place fields (5 sessions, DANDI:000044)",
                 y=0.995)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
