"""Generate all figures for the Achilles place-cell analysis."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CACHE = "achilles_maze_cache.npz"
RES = "achilles_placefields.npz"


def load_all():
    c = np.load(CACHE, allow_pickle=False)
    r = np.load(RES, allow_pickle=False)
    D = {k: c[k] for k in c.files}
    D.update({k: r[k] for k in r.files})
    D["spikes"] = {u: c[f"spk_{u}"] for u in c["unit_ids"]}
    return D


def pick_window(D, dur=60.0):
    """Find the dur-second window containing the most run-bout starts."""
    starts = D["run_starts"]
    best_t0, best_n = starts[0], 0
    for s in starts:
        n = ((starts >= s) & (starts < s + dur)).sum()
        if n > best_n:
            best_t0, best_n = s, n
    return best_t0, int(best_n)


def fig1_raw_data(D):
    """Session overview: trajectory, position trace with spikes, population raster."""
    t, xy, lin, speed = D["t"], D["xy"], D["lin"], D["speed"]
    unit_ids = D["unit_ids"]
    is_place = D["is_place"]
    ratemaps_sm = D["ratemaps_sm"]
    centers = D["centers"]

    WIN = 60.0
    t0, n_runs = pick_window(D, WIN)
    win = (t >= t0) & (t < t0 + WIN)

    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.38, wspace=0.22,
                  left=0.07, right=0.97, top=0.94, bottom=0.07)

    # --- A: 2D trajectory ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xy[:, 0], xy[:, 1], color="0.8", lw=0.3, zorder=1)
    ax.plot(xy[win, 0], xy[win, 1], color="tab:blue", lw=1.0, zorder=2,
            label=f"{int(WIN)} s window shown below")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("A  Trajectory on the 1.6 m linear maze\n(MazeEpoch, 34.5 min)", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_aspect("equal")

    # --- B: speed trace over the window with run bouts shaded ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t[win] - t0, np.abs(speed[win]) * 100, color="0.4", lw=0.7)
    for s, e in zip(D["run_starts"], D["run_ends"]):
        if e < t0 or s > t0 + WIN:
            continue
        ax.axvspan(max(s, t0) - t0, min(e, t0 + WIN) - t0, color="tab:green",
                   alpha=0.15, lw=0)
    ax.axhline(5, color="k", ls="--", lw=0.8, label="run threshold (5 cm/s)")
    ax.set_xlim(0, WIN)
    ax.set_xlabel("time in window (s)")
    ax.set_ylabel("|speed| (cm/s)")
    ax.set_title(f"B  Speed during the window ({n_runs} runs, shaded)", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # --- C: linearized position with spikes of example place cells ---
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t[win] - t0, lin[win], color="0.5", lw=0.8, zorder=1)
    place_idx = np.where(is_place)[0]
    peaks = np.array([centers[np.nanargmax(ratemaps_sm[i])] if np.isfinite(ratemaps_sm[i]).any() else np.nan
                      for i in place_idx])
    order = np.argsort(peaks)
    picks = place_idx[order[np.linspace(0, len(order) - 1, 6).astype(int)]]
    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for k, ui in enumerate(picks):
        spk = D["spikes"][unit_ids[ui]]
        spk_win = spk[(spk >= t0) & (spk < t0 + WIN)]
        pos_spk = np.interp(spk_win, t, lin)
        ax.plot(spk_win - t0, pos_spk, "o", ms=3, color=colors[k],
                label=f"unit {unit_ids[ui]}", zorder=3)
    ax.set_xlim(0, WIN)
    ax.set_ylim(-0.05, 1.8)
    ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
    ax.set_xlabel("time in window (s)")
    ax.set_ylabel("track position (m)")
    ax.set_title("C  Linearized position with spikes of six example place cells", loc="left")
    ax.legend(loc="upper right", ncol=6, frameon=False, fontsize=8,
              columnspacing=0.8, handletextpad=0.1)

    # --- D: raster of all place cells sorted by field location ---
    ax = fig.add_subplot(gs[2, :])
    sort_idx = place_idx[np.argsort(peaks)]
    for row, ui in enumerate(sort_idx):
        spk = D["spikes"][unit_ids[ui]]
        spk_win = spk[(spk >= t0) & (spk < t0 + WIN)]
        ax.plot(spk_win - t0, np.full_like(spk_win, row), "|", ms=2.5,
                color="k", alpha=0.6)
    ax.set_xlim(0, WIN)
    ax.set_ylim(-1, len(sort_idx))
    ax.set_xlabel("time in window (s)")
    ax.set_ylabel("place cells (sorted)")
    ax.set_title("D  Raster of all significant place cells, sorted by place-field location", loc="left")

    fig.savefig("fig1_raw_data.png")
    plt.close(fig)
    print("saved fig1_raw_data.png")


def fig2_occupancy(D):
    """Behavioral coverage: occupancy per bin, per direction; speed distribution."""
    occ_s, occ_p, occ_n = D["occ_s"], D["occ_s_posdir"], D["occ_s_negdir"]
    centers = D["centers"]
    speed, t = D["speed"], D["t"]
    lin = D["lin"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16, wspace=0.3)

    ax = axes[0]
    w = centers[1] - centers[0]
    ax.bar(centers, occ_s, width=w, color="0.75", label="all runs")
    ax.step(centers, occ_p, where="mid", color="tab:red", lw=1.4, label="positive dir.")
    ax.step(centers, occ_n, where="mid", color="tab:blue", lw=1.4, label="negative dir.")
    ax.set_xlabel("track position (m)")
    ax.set_ylabel("occupancy (s)")
    ax.set_title("A  Position occupancy during runs", loc="left")
    ax.legend(frameon=False)

    ax = axes[1]
    sp = np.abs(speed[~np.isnan(speed)]) * 100
    ax.hist(sp, bins=np.linspace(0, 120, 61), color="0.5")
    ax.axvline(5, color="k", ls="--", lw=1, label="run threshold (5 cm/s)")
    ax.set_xlabel("|speed| (cm/s)")
    ax.set_ylabel("samples")
    ax.set_title("B  Speed distribution (MazeEpoch)", loc="left")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    # position vs time over the whole epoch (downsampled)
    step = 20
    ax.plot(t[::step] - t[0], lin[::step], ".", ms=1.2, color="0.4", alpha=0.6)
    ax.set_xlabel("time in MazeEpoch (s)")
    ax.set_ylabel("track position (m)")
    ax.set_title("C  Position over the full maze epoch", loc="left")

    fig.savefig("fig2_behavior.png")
    plt.close(fig)
    print("saved fig2_behavior.png")


def fig3_examples(D):
    """Example place cells: 2D spike scatter + direction-split rate maps."""
    unit_ids = D["unit_ids"]
    is_place = D["is_place"]
    si = D["si"]
    ratemaps_sm = D["ratemaps_sm"]
    rm_pos, rm_neg = D["ratemaps_pos"], D["ratemaps_neg"]
    centers = D["centers"]
    xy, t, lin = D["xy"], D["t"], D["lin"]

    place_idx = np.where(is_place)[0]
    peaks = np.array([centers[np.nanargmax(ratemaps_sm[i])] for i in place_idx])
    # one high-SI cell per eighth of the track, so fields tile the track
    picks = []
    for b in range(8):
        lo, hi = b * 0.2, (b + 1) * 0.2
        cand = place_idx[(peaks >= lo) & (peaks < hi)]
        if len(cand) == 0:
            continue
        picks.append(cand[np.argmax(si[cand])])

    fig = plt.figure(figsize=(12, 5.2))
    gs = GridSpec(2, len(picks), hspace=0.28, wspace=0.45,
                  left=0.05, right=0.98, top=0.88, bottom=0.12)
    for k, ui in enumerate(picks):
        u = unit_ids[ui]
        spk = D["spikes"][u]
        # 2D scatter
        ax = fig.add_subplot(gs[0, k])
        ax.plot(xy[:, 0], xy[:, 1], color="0.75", lw=0.3, zorder=1)
        spk_xy = np.column_stack([np.interp(spk, t, xy[:, 0]),
                                  np.interp(spk, t, xy[:, 1])])
        ax.plot(spk_xy[:, 0], spk_xy[:, 1], "o", ms=1.1, color="tab:red",
                alpha=0.45, zorder=2, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"unit {u}\nSI = {si[ui]:.2f} bits/spk", fontsize=8)
        # rate map
        ax = fig.add_subplot(gs[1, k])
        ax.fill_between(centers, rm_pos[ui], color="tab:red", alpha=0.35, lw=0)
        ax.plot(centers, rm_pos[ui], color="tab:red", lw=1.2, label="pos. dir.")
        ax.fill_between(centers, rm_neg[ui], color="tab:blue", alpha=0.35, lw=0)
        ax.plot(centers, rm_neg[ui], color="tab:blue", lw=1.2, label="neg. dir.")
        ax.set_xlim(0, 1.6)
        ax.set_xticks([0, 0.8, 1.6])
        if k == 0:
            ax.set_ylabel("firing rate (Hz)")
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.set_xlabel("position (m)", labelpad=1)
        ax.tick_params(labelsize=7)

    fig.suptitle("Example place cells: spike locations on the maze (top) and direction-split rate maps (bottom)",
                 fontsize=11, y=0.97)
    fig.savefig("fig3_example_place_cells.png")
    plt.close(fig)
    print("saved fig3_example_place_cells.png")


def fig4_population(D):
    """Population rate maps sorted by peak location, per direction."""
    is_place = D["is_place"]
    rm_pos, rm_neg = D["ratemaps_pos"], D["ratemaps_neg"]
    centers = D["centers"]

    place_idx = np.where(is_place)[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.13, wspace=0.12)
    for ax, (rm, name) in zip(axes, [(rm_pos, "positive direction"),
                                     (rm_neg, "negative direction")]):
        R = rm[place_idx]
        peak_pos = np.array([centers[np.nanargmax(r)] if np.isfinite(r).any() else np.inf
                             for r in R])
        order = np.argsort(peak_pos)
        R = R[order]
        with np.errstate(invalid="ignore", divide="ignore"):
            Rn = R / np.nanmax(R, axis=1, keepdims=True)
        im = ax.imshow(Rn, aspect="auto", cmap="viridis", origin="upper",
                       extent=[0, 1.6, len(Rn), 0], vmin=0, vmax=1)
        ax.set_xlabel("track position (m)")
        ax.set_title(f"{name} (n={len(Rn)} place cells)", loc="left")
    axes[0].set_ylabel("place cells (sorted by peak)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("normalized firing rate")
    fig.suptitle("Population rate maps tile the linear track", fontsize=11, y=0.97)
    fig.savefig("fig4_population_ratemaps.png")
    plt.close(fig)
    print("saved fig4_population_ratemaps.png")


def fig5_spatial_info(D):
    """Spatial information distributions and place-cell classification."""
    si = D["si"]
    si_sh = D["si_shuffle"]
    thresh = D["si_thresh_95"]
    cell_type = D["cell_type"]
    peak = D["peak_rate"]
    is_place = D["is_place"]

    exc = cell_type == "excitatory"
    inh = cell_type == "inhibitory"

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16, wspace=0.32)

    ax = axes[0]
    bins = np.linspace(0, np.nanmax(si[exc]) * 1.05, 50)
    ax.hist(si[exc & np.isfinite(si)], bins=bins, color="tab:red", alpha=0.75,
            label=f"excitatory (n={exc.sum()})")
    ax.hist(si[inh & np.isfinite(si)], bins=bins, color="tab:blue", alpha=0.75,
            label=f"inhibitory (n={inh.sum()})")
    ax.set_xlabel("spatial information (bits/spike)")
    ax.set_ylabel("units")
    ax.set_title("A  Spatial information by cell type", loc="left")
    ax.legend(frameon=False)

    ax = axes[1]
    # observed vs shuffle 95th percentile (excitatory only)
    m = exc & np.isfinite(si) & np.isfinite(thresh)
    ax.scatter(thresh[m], si[m], s=10, alpha=0.6,
               c=np.where(is_place[m], "tab:red", "0.6"))
    lim = [0, max(np.nanmax(si[m]), np.nanmax(thresh[m])) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("shuffle 95th percentile (bits/spike)")
    ax.set_ylabel("observed SI (bits/spike)")
    ax.set_title("B  Observed vs shuffle threshold", loc="left")

    ax = axes[2]
    m = exc & np.isfinite(si)
    ax.scatter(peak[m & ~is_place], si[m & ~is_place], s=10, color="0.6", alpha=0.6,
               label="not significant")
    ax.scatter(peak[m & is_place], si[m & is_place], s=10, color="tab:red", alpha=0.7,
               label=f"place cells (n={int(is_place.sum())})")
    ax.set_xscale("log")
    ax.set_xlabel("peak firing rate (Hz)")
    ax.set_ylabel("spatial information (bits/spike)")
    ax.set_title("C  Place-cell classification", loc="left")
    ax.legend(frameon=False, loc="upper left")

    fig.savefig("fig5_spatial_info.png")
    plt.close(fig)
    print("saved fig5_spatial_info.png")


def fig6_directionality(D):
    """Direction selectivity of place fields."""
    is_place = D["is_place"]
    rm_pos, rm_neg = D["ratemaps_pos"], D["ratemaps_neg"]
    centers = D["centers"]

    place_idx = np.where(is_place)[0]
    # peak rates per direction and correlation between direction maps
    peak_pos = np.nanmax(rm_pos[place_idx], axis=1)
    peak_neg = np.nanmax(rm_neg[place_idx], axis=1)

    def dir_corr(i):
        a, b = rm_pos[i], rm_neg[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
            return np.nan  # cell silent (or flat) in one direction
        return np.corrcoef(a[m], b[m])[0, 1]

    corr = np.array([dir_corr(i) for i in place_idx])
    # peak locations per direction
    loc_pos = centers[np.nanargmax(rm_pos[place_idx], axis=1)]
    loc_neg = centers[np.nanargmax(rm_neg[place_idx], axis=1)]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16, wspace=0.32)

    ax = axes[0]
    ax.scatter(peak_pos, peak_neg, s=12, color="0.4", alpha=0.7)
    lim = [0, max(peak_pos.max(), peak_neg.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("peak rate, positive dir. (Hz)")
    ax.set_ylabel("peak rate, negative dir. (Hz)")
    ax.set_title("A  Peak rates by direction", loc="left")

    ax = axes[1]
    corr_f = corr[np.isfinite(corr)]
    ax.hist(corr_f, bins=np.linspace(-0.4, 1, 37), color="0.5")
    ax.axvline(np.median(corr_f), color="tab:red", ls="--", lw=1,
               label=f"median r = {np.median(corr_f):.2f}")
    ax.set_xlabel("correlation between direction maps")
    ax.set_ylabel("place cells")
    ax.set_title("B  Directional stability of rate maps", loc="left")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.scatter(loc_pos, loc_neg, s=12, color="0.4", alpha=0.7)
    ax.plot([0, 1.6], [0, 1.6], "k--", lw=1)
    ax.set_xlim(0, 1.6); ax.set_ylim(0, 1.6)
    ax.set_xlabel("peak location, positive dir. (m)")
    ax.set_ylabel("peak location, negative dir. (m)")
    ax.set_title("C  Field location by direction", loc="left")

    fig.savefig("fig6_directionality.png")
    plt.close(fig)
    print("saved fig6_directionality.png")
    return corr


def main():
    D = load_all()
    fig1_raw_data(D)
    fig2_occupancy(D)
    fig3_examples(D)
    fig4_population(D)
    fig5_spatial_info(D)
    corr = fig6_directionality(D)
    corr_f = corr[np.isfinite(corr)]
    print(f"direction-map correlation: median {np.median(corr_f):.3f} "
          f"(n={len(corr_f)} with both-direction fields), "
          f"fraction r>0.5: {(corr_f > 0.5).mean():.2f}")


if __name__ == "__main__":
    main()
