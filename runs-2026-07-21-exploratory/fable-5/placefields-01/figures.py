"""Figure generation for the DANDI:000044 place-cell demonstration."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import place_cell_lib as pcl

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

DIR_COLOR = {"rightward": "#1b6ca8", "leftward": "#d1495b"}


def _peak_order(per_dir, direction, idx):
    """Indices `idx` reordered by place-field peak position in `direction`."""
    peaks = per_dir[direction]["peak_pos"][idx]
    return idx[np.argsort(peaks)]


# ---------------------------------------------------------------------------
# Figure 1 - session and raw-data overview
# ---------------------------------------------------------------------------
def fig_session_overview(S, stats, per_dir, lfp_channel, fname, pad=1.5):
    units = S["units"]
    pc_mask = stats["is_place_cell"].values
    pc_idx = np.flatnonzero(pc_mask)
    ref_dir = "rightward" if "rightward" in per_dir else list(per_dir)[0]
    order = _peak_order(per_dir, ref_dir, pc_idx)

    # Zoom on a single brisk traversal: use the speed-masked running epochs so
    # the window is not dominated by the pause at the reward port.
    runs = per_dir[ref_dir]["epochs"]
    dur = runs.end - runs.start
    long_enough = np.flatnonzero(dur > np.percentile(dur, 75))
    j = int(long_enough[np.argmin(np.abs(dur[long_enough]
                                         - np.median(dur[long_enough])))])
    t0, t1 = float(runs.start[j]) - pad, float(runs.end[j]) + pad
    zoom = pcl.nap.IntervalSet(start=t0, end=t1)

    fig = plt.figure(figsize=(14, 13.5))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.7, 1.7, 0.5],
                  hspace=0.62, wspace=0.24)

    # (a) 2D trajectory ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    p2 = S["position_2d"].restrict(S["maze_ep"])
    ax.plot(p2["x"].values, p2["y"].values, color="0.8", lw=0.4, alpha=0.8)
    for name in per_dir:
        ep_d = per_dir[name]["epochs"]
        for i in range(len(ep_d)):
            seg = S["position_2d"].restrict(ep_d[i])
            ax.plot(seg["x"].values, seg["y"].values, color=DIR_COLOR[name],
                    lw=0.5, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"(a) Trajectory, {S['maze_name']}\n"
                 f"{S['subject_id']}, session {S['session_id']}")

    # (b) session timeline ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ep = S["epochs"]
    colors = {"PREEpoch": "#8ecae6", "MazeEpoch": "#ffb703", "POSTEpoch": "#adb5bd"}
    for i in range(len(ep)):
        lab = ep.metadata["label"].iloc[i]
        ax.barh(0, (ep.end[i] - ep.start[i]) / 60, left=ep.start[i] / 60,
                height=0.35, color=colors.get(lab, "0.6"),
                label=lab.replace("Epoch", ""))
    ax.set_yticks([])
    ax.set_ylim(-1.7, 1.1)
    ax.set_xlabel("time from session start (min)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.set_title("(b) Session structure: sleep - track - sleep")
    ax.text(0.0, 0.30,
            f"{len(units)} sorted units "
            f"({(stats.cell_type == 'excitatory').sum()} excitatory, "
            f"{(stats.cell_type == 'inhibitory').sum()} inhibitory)\n"
            f"{len(S['run_ep'])} track traversals, {pc_mask.sum()} place cells",
            transform=ax.transAxes, va="top", fontsize=8.5)

    # (c) linearised position over the whole maze epoch ----------------------
    ax = fig.add_subplot(gs[1, :])
    pos = S["position"]
    ax.plot(pos.t, pos.values, color="0.8", lw=0.5)
    for name in per_dir:
        ep_d = per_dir[name]["epochs"]
        for i in range(len(ep_d)):
            seg = pos.restrict(ep_d[i])
            ax.plot(seg.t, seg.values, color=DIR_COLOR[name], lw=1.0)
    ax.axvspan(t0, t1, color="gold", alpha=0.6, zorder=0)
    ax.set_xlim(float(S["maze_ep"].start[0]), float(S["maze_ep"].end[0]))
    ax.set_ylim(-10, S["track_cm"] + 60)
    ax.set_ylabel("position (cm)")
    ax.set_xlabel("time (s)")
    ax.set_title("(c) Linearised position over the maze epoch; coloured segments "
                 "are the running laps kept for the rate maps")
    ax.legend(handles=[Line2D([], [], color=DIR_COLOR[n], lw=2, label=n)
                       for n in per_dir]
              + [Line2D([], [], color="gold", lw=6, label="window in (d)")],
              loc="upper right", ncol=3, fontsize=8)

    # (d) raster of place cells ordered by field position --------------------
    ax = fig.add_subplot(gs[2, :])
    for row, k in enumerate(order):
        uid = units.index[k]
        st = units[uid].restrict(zoom).t - t0
        ax.plot(st, np.full(st.size, row), "|", color="k", ms=5, mew=0.9)
    ax.set_ylim(-2, len(order) + 1)
    ax.set_ylabel(f"place cell, sorted by field position (n={len(order)})")
    ax.set_xlim(0, t1 - t0)
    ax.set_xlabel("time from start of window (s)")
    ax.set_title("(d) Raw spiking during one traversal: activity sweeps through "
                 "the ensemble in field order as the animal advances")
    axp = ax.twinx()
    seg = pos.restrict(zoom)
    axp.plot(seg.t - t0, seg.values, color="#ffb703", lw=2.5)
    axp.set_ylim(-5, S["track_cm"] + 5)
    axp.set_ylabel("position (cm)", color="#c98900")
    axp.tick_params(axis="y", colors="#c98900")
    axp.spines["right"].set_visible(True)
    axp.spines["right"].set_color("#c98900")

    # (e) LFP ----------------------------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    lfp = pcl.lfp_snippet(S["nwbfile"], t0, t1, lfp_channel)
    ax.plot(lfp.t - t0, lfp.values * 1e3, color="#2a9d8f", lw=0.7)
    ax.set_xlim(0, t1 - t0)
    ax.set_xlabel("time from start of window (s)")
    ax.set_ylabel("LFP (mV)")
    ax.set_title(f"(e) Simultaneous CA1 local field potential (channel "
                 f"{lfp_channel}); the running-speed theta rhythm is visible")

    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Figure 2 - example place cells
# ---------------------------------------------------------------------------
def select_examples(S, fields, n_examples=8, max_sparsity=0.4,
                    min_stability=0.8, min_peak_rate=3.0):
    """Compact, stable, well-driven fields spread along the length of the track.

    Selection deliberately does not rank on spatial information: raw bits/spike
    is inflated by low spike counts and the z-scored version rewards high firing
    rates, so neither picks out the textbook single-peaked field. Sparsity,
    split-half stability and peak rate do.
    """
    good = fields[(fields["sparsity"] < max_sparsity)
                  & (fields["stability"] > min_stability)
                  & (fields["peak_rate"] > min_peak_rate)]
    if len(good) < n_examples:            # fall back to a looser cut
        good = fields.nsmallest(max(n_examples * 3, 12), "sparsity")
    quality = (1 - good["sparsity"].values) * good["stability"].values
    peaks = good["peak_pos"].values
    chosen = []
    for target in np.linspace(0.08, 0.92, n_examples) * S["track_cm"]:
        score = np.abs(peaks - target) - 25.0 * quality
        for c in np.argsort(score):
            if int(good.iloc[c]["k"]) not in chosen:
                chosen.append(int(good.iloc[c]["k"]))
                break
    chosen = np.array(chosen)
    peak_of_chosen = np.array([peaks[list(good["k"].values).index(k)]
                               for k in chosen])
    return chosen[np.argsort(peak_of_chosen)]


def fig_example_cells(S, stats, per_dir, fields, fname, n_examples=8):
    units = S["units"]
    chosen = select_examples(S, fields, n_examples)

    ncol = 4
    nrow = int(np.ceil(len(chosen) / ncol))
    fig = plt.figure(figsize=(4.1 * ncol, 5.0 * nrow))
    gs = GridSpec(2 * nrow, ncol, figure=fig, height_ratios=[2.0, 1.0] * nrow,
                  hspace=0.75, wspace=0.38)

    for m, k in enumerate(chosen):
        r, c = divmod(m, ncol)
        uid = units.index[k]
        ax_r = fig.add_subplot(gs[2 * r, c])
        ax_t = fig.add_subplot(gs[2 * r + 1, c], sharex=ax_r)

        lap_offset = 0
        for j, name in enumerate(per_dir):
            laps = per_dir[name]["laps"]
            block_start = lap_offset
            for i in range(len(laps)):
                st = units[uid].restrict(laps[i])
                if len(st):
                    xp = st.value_from(S["position"]).values
                    ax_r.plot(xp, np.full(xp.size, lap_offset), "|",
                              color=DIR_COLOR[name], ms=3.5, mew=0.8)
                lap_offset += 1
            ax_r.text(0.99, (block_start + lap_offset) / 2.0, name[0].upper() + " ",
                      color=DIR_COLOR[name], fontsize=8, ha="right", va="center",
                      transform=ax_r.get_yaxis_transform(), clip_on=False)
            if j < len(per_dir) - 1:
                ax_r.axhline(lap_offset - 0.5, color="0.6", lw=0.8)
        ax_r.set_ylim(-1, lap_offset)
        ax_r.set_ylabel("lap")
        ax_r.set_title(f"unit {uid}\n"
                       + "   ".join(f"{n[0].upper()} {per_dir[n]['si'][k]:.2f}"
                                    for n in per_dir)
                       + " bits/spike", fontsize=9)
        plt.setp(ax_r.get_xticklabels(), visible=False)

        for name in per_dir:
            d = per_dir[name]
            ax_t.plot(d["centres"], d["rates"][k], color=DIR_COLOR[name], lw=1.6,
                      label=name)
        ax_t.set_xlabel("position (cm)")
        ax_t.set_ylabel("rate (Hz)")
        ax_t.set_xlim(0, S["track_cm"])
        if m == 0 and len(per_dir) > 1:
            ax_t.legend(fontsize=7, loc="upper center", ncol=2)

    fig.suptitle("Example CA1 place cells, ordered by field position along the "
                 "track\nTop of each pair: position of every spike, lap by lap "
                 "(blue = rightward laps, red = leftward laps).  "
                 "Bottom: occupancy-normalised rate map.", y=0.995, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Figure 3 - population rate maps
# ---------------------------------------------------------------------------
def _norm_map(rates):
    return rates / np.maximum(rates.max(axis=1, keepdims=True), 1e-9)


def fig_population_maps(S, stats, per_dir, fname):
    pc_idx = np.flatnonzero(stats["is_place_cell"].values)
    dirs = list(per_dir)
    ref = dirs[0]
    order_ref = _peak_order(per_dir, ref, pc_idx)

    panels = [(d, _peak_order(per_dir, d, pc_idx), f"sorted by its own peak")
              for d in dirs]
    if len(dirs) > 1:
        other = dirs[1]
        panels.append((other, order_ref, f"sorted by the {ref} peak"))

    ncol = len(panels) + (1 if len(dirs) > 1 else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.7 * ncol, 5.4))
    axes = np.atleast_1d(axes)
    letters = "abcdefg"
    for i, (d, order, sub) in enumerate(panels):
        ax = axes[i]
        im = ax.imshow(_norm_map(per_dir[d]["rates"][order]), aspect="auto",
                       origin="lower", cmap="viridis", vmin=0, vmax=1,
                       extent=[0, S["track_cm"], 0, len(order)])
        ax.set_xlabel("position (cm)")
        ax.set_ylabel("place cell #")
        ax.set_title(f"({letters[i]}) {d} runs,\n{sub}", fontsize=9.5)
    fig.colorbar(im, ax=axes[len(panels) - 1], label="rate / peak rate",
                 fraction=0.06, pad=0.04)

    if len(dirs) > 1:
        ax = axes[-1]
        a, b = dirs[0], dirs[1]
        pa, pb = per_dir[a]["peak_pos"][pc_idx], per_dir[b]["peak_pos"][pc_idx]
        ax.scatter(pa, pb, s=20, color="0.25", alpha=0.8)
        lim = [0, S["track_cm"]]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"peak position, {a} (cm)")
        ax.set_ylabel(f"peak position, {b} (cm)")
        ax.set_title(f"({letters[len(panels)]}) the two directional maps\n"
                     f"are largely independent (r = {np.corrcoef(pa, pb)[0,1]:.2f})",
                     fontsize=9.5)

    fig.suptitle("Population rate maps: place fields tile the whole track, and "
                 "the ordering is specific to the direction of travel", y=1.02,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Figure 4 - spatial information and the shuffle test
# ---------------------------------------------------------------------------
def fig_spatial_information(S, stats, per_dir, fname):
    exc = (stats["cell_type"] == "excitatory").values
    inh = (stats["cell_type"] == "inhibitory").values
    pc = stats["is_place_cell"].values
    ref = list(per_dir)[0]
    d = per_dir[ref]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    fig.subplots_adjust(hspace=0.42, wspace=0.28)

    ax = axes[0, 0]
    bins = np.linspace(0, max(3.0, np.nanpercentile(d["si"][exc], 99.5)), 40)
    ax.hist(d["null"][:, exc].ravel(), bins=bins, density=True, color="0.7",
            label="circularly shifted spikes (null)")
    ax.hist(d["si"][exc], bins=bins, density=True, histtype="step", lw=2,
            color="#1b6ca8", label="observed")
    ax.set_xlabel("spatial information (bits/spike)")
    ax.set_ylabel("density")
    ax.set_title(f"(a) Observed spatial information far exceeds the\n"
                 f"shift-control null ({ref} runs, excitatory cells)")
    ax.legend()

    ax = axes[0, 1]
    thr = np.percentile(d["null"], 100 * (1 - pcl.SHUFFLE_ALPHA), axis=0)
    active = (d["n_spikes"] >= pcl.MIN_SPIKES_ON_TRACK) & \
             (d["peak_rate"] >= pcl.MIN_PEAK_RATE_HZ)
    sig = d["pval"] < pcl.SHUFFLE_ALPHA
    ax.scatter(thr[active & exc & ~sig], d["si"][active & exc & ~sig], s=16,
               color="0.6", label="not significant")
    ax.scatter(thr[active & exc & sig], d["si"][active & exc & sig], s=18,
               color="#1b6ca8", label="significant (p < 0.01)")
    ax.scatter(thr[active & inh], d["si"][active & inh], s=26, color="#d1495b",
               marker="^", label="interneuron")
    lim = [0, max(thr[active].max(), d["si"][active].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"per-cell {100*(1-pcl.SHUFFLE_ALPHA):.0f}th percentile of null "
                  "(bits/spike)")
    ax.set_ylabel("observed spatial information (bits/spike)")
    ax.set_title(f"(b) Every cell is tested against its own null;\n"
                 f"points above the diagonal are significant\n"
                 f"({(~active).sum()} sparsely firing cells not shown)",
                 fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    eligible = exc & sig & active
    best = np.argmax(np.where(eligible, d["si"], -np.inf))
    ax.hist(d["null"][:, best], bins=40, color="0.7", label="null")
    ax.axvline(d["si"][best], color="#1b6ca8", lw=2)
    ax.text(d["si"][best], ax.get_ylim()[1] * 0.92,
            f"  observed\n  p < {max(d['pval'][best], 1/d['null'].shape[0]):.3f}",
            color="#1b6ca8", va="top")
    ax.set_xlim(0, max(d["si"][best] * 1.15, d["null"][:, best].max()))
    ax.set_xlabel("spatial information (bits/spike)")
    ax.set_ylabel(f"count (of {d['null'].shape[0]} shuffles)")
    ax.set_title(f"(c) Null distribution for the most informative\n"
                 f"place cell (unit {S['units'].index[best]})")

    ax = axes[1, 1]
    # Observed spatial information is only interpretable next to each cell's own
    # null: a sparsely firing cell has a noisy rate map and therefore a high
    # value under both. Pairing the two makes the comparison honest.
    null_med = np.median(d["null"], axis=0)
    pos, ticks, labels = [], [], []
    for i, (m, lab, col) in enumerate(((pc, "place cells", "#1b6ca8"),
                                       (exc & ~pc, "other\nexcitatory", "0.6"),
                                       (inh, "interneurons", "#d1495b"))):
        if not m.sum():
            continue
        x = 2 * i
        for dx, vals, fc, ec in ((-0.32, d["si"][m], col, col),
                                 (0.32, null_med[m], "white", col)):
            bp = ax.boxplot([vals], positions=[x + dx], widths=0.55,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(fc)
            bp["boxes"][0].set_edgecolor(ec)
            bp["boxes"][0].set_alpha(0.8)
            ax.scatter(np.random.default_rng(1).normal(x + dx, 0.06, vals.size),
                       vals, s=8, color=col, alpha=0.6, zorder=3)
        ticks.append(x)
        labels.append(f"{lab}\n(n={m.sum()})")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-1.2, 2 * 2 + 1.2)
    ax.set_ylabel("spatial information (bits/spike)")
    ax.legend(handles=[Patch(facecolor="0.35", edgecolor="0.35", label="observed"),
                       Patch(facecolor="white", edgecolor="0.35",
                             label="median of that cell's own null")],
              fontsize=8, loc="upper right")
    ax.set_title("(d) Observed information beside each cell's own null.\n"
                 "Place cells exceed theirs several-fold; the high raw values\n"
                 "among other excitatory cells are matched by a high null.",
                 fontsize=9.5)

    fig.suptitle("Place fields are statistically reliable, not a by-product of "
                 "uneven sampling of the track", y=0.98, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Figure 5 - field properties
# ---------------------------------------------------------------------------
def fig_field_properties(S, stats, per_dir, fields, fname):
    """`fields` is the one-row-per-(cell, direction) table from pcl.field_table."""
    pc = stats["is_place_cell"].values
    exc = (stats["cell_type"] == "excitatory").values
    dirs = list(per_dir)
    best_name, best_z = pcl.best_direction(per_dir)

    def per_cell(key, mask):
        """Value of `key` in each cell's best-tuned direction."""
        v = np.array([per_dir[n][key][k] for k, n in enumerate(best_name)])
        return v[mask]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    fig.subplots_adjust(hspace=0.5, wspace=0.33)

    ax = axes[0, 0]
    v = fields["peak_rate"].values
    ax.hist(v, bins=np.linspace(0, np.percentile(v, 99), 30), color="#1b6ca8",
            alpha=0.85)
    ax.axvline(np.median(v), color="k", ls="--", lw=1)
    ax.set_xlabel("in-field peak rate (Hz)")
    ax.set_ylabel("number of fields")
    ax.set_title(f"(a) Peak firing rate\nmedian {np.median(v):.1f} Hz")

    ax = axes[0, 1]
    v = fields["width"].values
    v = v[v > 0]
    ax.hist(v, bins=25, color="#1b6ca8", alpha=0.85)
    ax.axvline(np.median(v), color="k", ls="--", lw=1)
    ax.set_xlabel("field width at half maximum (cm)")
    ax.set_ylabel("number of fields")
    ax.set_title(f"(b) Field width\nmedian {np.median(v):.0f} cm "
                 f"({100*np.median(v)/S['track_cm']:.0f}% of the track)")

    ax = axes[0, 2]
    other = per_cell("participation", exc & ~pc)
    other = other[np.isfinite(other)]
    mine = fields["participation"].values
    mine = mine[np.isfinite(mine)]
    bins = np.linspace(0, 1, 21)
    ax.hist(other, bins=bins, color="0.65", alpha=0.9,
            label=f"other excitatory (median {np.median(other):.0%})")
    ax.hist(mine, bins=bins, color="#1b6ca8", alpha=0.8,
            label=f"place fields (median {np.median(mine):.0%})")
    ax.set_xlabel("fraction of laps with a spike inside the field")
    ax.set_ylabel("count")
    ax.set_title("(c) A place field fires on nearly every pass;\n"
                 "an incidentally peaked map does not")
    ax.legend(fontsize=7.5, loc="upper center")

    ax = axes[1, 0]
    for m, lab, col in ((pc, "place cells", "#1b6ca8"),
                        (exc & ~pc, "other excitatory", "0.65")):
        if m.sum():
            v = per_cell("stability", m)
            v = v[np.isfinite(v)]
            ax.hist(v, bins=np.linspace(-1, 1, 30), alpha=0.75, color=col,
                    label=f"{lab} (median {np.median(v):.2f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("odd-lap vs even-lap map correlation")
    ax.set_ylabel("number of cells")
    ax.set_title("(d) Fields are stable across independent laps")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    for d in dirs:
        sub = fields[fields["direction"] == d]
        ax.hist(sub["peak_pos"], bins=np.linspace(0, S["track_cm"], 21),
                histtype="step", lw=2, color=DIR_COLOR[d],
                label=f"{d} (n={len(sub)})")
    ax.set_xlabel("field peak position (cm)")
    ax.set_ylabel("number of fields")
    ax.set_title("(e) Fields cover the whole track, with the\n"
                 "usual over-representation of the reward ends")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    mid = fields[(fields["peak_pos"] > 0.15 * S["track_cm"])
                 & (fields["peak_pos"] < 0.85 * S["track_cm"])]
    row = mid.loc[mid["si_z"].idxmax()]
    d0, k = per_dir[row["direction"]], int(row["k"])
    ax.plot(d0["centres"], d0["rates_odd"][k], color="#1b6ca8", lw=1.8,
            label="odd laps")
    ax.plot(d0["centres"], d0["rates_even"][k], color="#f4a261", lw=1.8,
            label="even laps")
    ax.set_xlabel("position (cm)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(f"(f) Independent halves of the data give the\n"
                 f"same field (unit {int(row['unit'])}, {row['direction']}, "
                 f"r = {row['stability']:.2f})")
    ax.legend()

    fig.suptitle("Properties of the identified place fields "
                 f"({len(fields)} fields from {fields['unit'].nunique()} cells)",
                 y=0.98, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Figure 6 - Bayesian decoding
# ---------------------------------------------------------------------------
def decoding_results(S, stats, per_dir, bin_size=0.25, rng=None):
    """Cross-validated decoding for each direction, plus an ensemble-size sweep."""
    rng = np.random.default_rng(0) if rng is None else rng
    pc_ids = stats.index[stats["is_place_cell"]].values
    ensemble = S["units"][list(pc_ids)]
    out = {}
    for name in per_dir:
        laps = per_dir[name]["laps"]
        t, dec, true = pcl.crossvalidated_decoding(
            ensemble, S["position"], laps, S["speed"], S["dt"],
            bin_size=bin_size, n_bins=S["n_bins"], track_cm=S["track_cm"],
            circular=S["maze_type"] == "circular")
        out[name] = dict(t=t, decoded=dec, true=true, error=np.abs(dec - true))
    # chance: pair each true position with a decoded value from another time bin
    allerr = np.concatenate([v["error"] for v in out.values()])
    alltrue = np.concatenate([v["true"] for v in out.values()])
    alldec = np.concatenate([v["decoded"] for v in out.values()])
    chance = np.concatenate([np.abs(rng.permutation(alldec) - alltrue)
                             for _ in range(20)])

    sizes = [n for n in (2, 5, 10, 20, 40, 80) if n <= 0.7 * len(pc_ids)]
    sizes.append(len(pc_ids))
    sweep = {}
    ref = list(per_dir)[0]
    laps = per_dir[ref]["laps"]
    for n in sizes:
        errs = []
        for _ in range(8 if n < len(pc_ids) else 1):
            sub = rng.choice(pc_ids, size=n, replace=False)
            _, dcd, tru = pcl.crossvalidated_decoding(
                S["units"][list(sub)], S["position"], laps, S["speed"], S["dt"],
                bin_size=bin_size, n_bins=S["n_bins"], track_cm=S["track_cm"],
                circular=S["maze_type"] == "circular")
            errs.append(np.median(np.abs(dcd - tru)))
        sweep[n] = np.array(errs)

    # A held-out block of consecutive laps, kept with its posterior for display.
    n_block = min(6, max(2, len(laps) // 4))
    start = max(0, len(laps) // 2 - n_block // 2)
    block = pcl.decode_lap_block(
        ensemble, S["position"], laps, S["speed"], S["dt"],
        np.arange(start, start + n_block), bin_size=bin_size,
        n_bins=S["n_bins"], track_cm=S["track_cm"],
        circular=S["maze_type"] == "circular")

    return dict(per_dir=out, error=allerr, chance=chance, sweep=sweep,
                block=block, ref=ref, n_cells=len(pc_ids), bin_size=bin_size)


def fig_decoding(S, stats, per_dir, dec, fname):
    fig = plt.figure(figsize=(15, 9.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.42,
                  height_ratios=[1.0, 0.95])
    ref = dec["ref"]
    blk = dec["block"]

    # (a) posterior over a held-out block of laps, laps drawn side by side so
    # that the pauses between traversals are not spanned by a line.
    ax = fig.add_subplot(gs[0, :])
    laps = blk["test_laps"]
    edges = np.linspace(0, S["track_cm"], S["n_bins"] + 1)
    x_off = 0.0
    ticks, labels = [], []
    for i in range(len(laps)):
        dsub = blk["decoded"].restrict(laps[i])
        if len(dsub) == 0:
            continue
        psub = blk["proba"].restrict(laps[i])
        tsub = blk["truth"].restrict(laps[i])
        rel = dsub.t - float(laps[i].start[0])
        dt_bin = dec["bin_size"]
        tedges = np.concatenate([rel - dt_bin / 2, [rel[-1] + dt_bin / 2]])
        pm = ax.pcolormesh(x_off + tedges, edges, np.asarray(psub).T,
                           cmap="magma", vmin=0, vmax=0.35, shading="flat")
        ax.plot(x_off + rel, tsub.values, color="#8ecae6", lw=2.2)
        ticks.append(x_off + rel.mean())
        labels.append(f"lap {i+1}")
        x_off += rel[-1] + 2 * dt_bin
        ax.axvline(x_off - dt_bin, color="0.5", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, x_off - dt_bin)
    ax.set_ylim(0, S["track_cm"])
    ax.set_ylabel("position (cm)")
    fig.colorbar(pm, ax=ax, fraction=0.02, pad=0.01,
                 label="posterior probability")
    ax.set_title(f"(a) Posterior probability over position, decoded from "
                 f"{dec['n_cells']} place cells in {int(dec['bin_size']*1000)} ms "
                 f"bins, on {len(laps)} consecutive {ref} laps held out of the "
                 f"encoding model\n"
                 f"(blue line: the animal's true position)", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    tr = np.concatenate([v["true"] for v in dec["per_dir"].values()])
    dc = np.concatenate([v["decoded"] for v in dec["per_dir"].values()])
    edges = np.linspace(0, S["track_cm"], S["n_bins"] + 1)
    H, _, _ = np.histogram2d(tr, dc, bins=[edges, edges])
    H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(H.T, origin="lower", aspect="equal", cmap="magma",
                   extent=[0, S["track_cm"], 0, S["track_cm"]])
    ax.plot([0, S["track_cm"]], [0, S["track_cm"]], "w--", lw=0.8)
    ax.set_xlabel("true position (cm)")
    ax.set_ylabel("decoded position (cm)")
    ax.set_title("(b) Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(decoded | true)")

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, S["track_cm"], 40)
    ax.hist(dec["chance"], bins=bins, density=True, color="0.75",
            label=f"chance (median {np.median(dec['chance']):.0f} cm)")
    ax.hist(dec["error"], bins=bins, density=True, histtype="step", lw=2,
            color="#e76f51",
            label=f"observed (median {np.median(dec['error']):.1f} cm)")
    ax.set_xlabel("absolute decoding error (cm)")
    ax.set_ylabel("density")
    ax.set_title("(c) Decoding error vs chance")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ns = sorted(dec["sweep"])
    med = [np.median(dec["sweep"][n]) for n in ns]
    lo = [np.min(dec["sweep"][n]) for n in ns]
    hi = [np.max(dec["sweep"][n]) for n in ns]
    ax.fill_between(ns, lo, hi, color="#e76f51", alpha=0.25)
    ax.plot(ns, med, "-o", color="#e76f51")
    ax.axhline(np.median(dec["chance"]), color="0.5", ls="--", label="chance")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("number of place cells in the ensemble")
    ax.set_ylabel("median decoding error (cm)")
    ax.set_title("(d) Accuracy improves with ensemble size\n"
                 "(shaded: range over 8 random subsets)")
    ax.legend()

    fig.suptitle("The population code is read-out-able: position is recoverable "
                 "from held-out laps to within a few centimetres", y=0.97,
                 fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Figure 7 - GLM: position versus running speed
# ---------------------------------------------------------------------------
def fig_glm(S, stats, per_dir, fields, glm, direction, fname, n_examples=3):
    pc = stats["is_place_cell"].values
    d = per_dir[direction]
    ok = np.isfinite(glm["gains"]["position"])

    fig = plt.figure(figsize=(15, 8.2))
    gs = GridSpec(2, 6, figure=fig, hspace=0.5, wspace=1.1)

    # top row: model-based tuning next to the empirical rate map
    sub = fields[(fields["direction"] == direction)
                 & fields["k"].isin(np.flatnonzero(ok))]
    sub = sub.nlargest(n_examples * 4, "participation").nlargest(n_examples, "peak_rate")
    for m, (_, row) in enumerate(sub.iterrows()):
        k = int(row["k"])
        ax = fig.add_subplot(gs[0, 2 * m:2 * m + 2])
        ax.plot(d["centres"], d["rates"][k], color="0.4", lw=1.6,
                label="binned rate map")
        ax.plot(glm["grid"], glm["tuning"][:, k], color="#6a4c93", lw=2.0,
                label="Poisson GLM")
        ax.set_xlabel("position (cm)")
        ax.set_ylabel("rate (Hz)")
        ax.set_title(f"({'abc'[m]}) unit {int(row['unit'])}", fontsize=9.5)
        if m == 0:
            ax.legend(fontsize=8)

    gp = glm["gains"]["position"]
    gsp = glm["gains"]["speed"]
    gb = glm["gains"]["position+speed"]

    ax = fig.add_subplot(gs[1, 0:2])
    ax.scatter(gsp[ok & ~pc], gp[ok & ~pc], s=16, color="0.6", label="other units")
    ax.scatter(gsp[ok & pc], gp[ok & pc], s=20, color="#1b6ca8", label="place cells")
    lim = [min(0, np.nanmin(gsp[ok])) - 0.05, np.nanmax(gp[ok]) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("speed-only model (bits/spike)")
    ax.set_ylabel("position-only model (bits/spike)")
    frac = float(np.mean(gp[ok & pc] > gsp[ok & pc]))
    ax.set_title(f"(d) Held-out likelihood gain: position beats\n"
                 f"speed for {frac:.0%} of place cells", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_subplot(gs[1, 2:4])
    names = ["speed", "position", "position+speed"]
    for i, name in enumerate(names):
        for j, (m, col, off) in enumerate(((pc & ok, "#1b6ca8", -0.18),
                                           (~pc & ok, "0.6", 0.18))):
            v = glm["gains"][name][m]
            bp = ax.boxplot([v], positions=[i + off], widths=0.3,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(col)
            bp["boxes"][0].set_alpha(0.75)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("cross-validated gain (bits/spike)")
    ax.legend(handles=[Patch(facecolor="#1b6ca8", alpha=0.75, label="place cells"),
                       Patch(facecolor="0.6", alpha=0.75, label="other units")],
              fontsize=8, loc="upper left")
    ax.set_title("(e) Speed alone explains little of the\nspiking; position explains most of it",
                 fontsize=9.5)

    ax = fig.add_subplot(gs[1, 4:6])
    extra = gb - gp
    ax.hist(extra[ok & pc], bins=25, color="#1b6ca8", alpha=0.85)
    ax.axvline(np.nanmedian(extra[ok & pc]), color="k", ls="--", lw=1)
    ax.set_xlabel("gain from adding speed to the position model\n(bits/spike)")
    ax.set_ylabel("number of place cells")
    ax.set_title(f"(f) Once position is in the model, speed adds\n"
                 f"almost nothing (median "
                 f"{np.nanmedian(extra[ok & pc]):.3f} bits/spike)", fontsize=9.5)

    fig.suptitle("A Poisson GLM confirms the fields are spatial, not a by-product "
                 f"of the speed profile along the track ({direction} runs)",
                 y=0.99, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Figure 8 - across sessions
# ---------------------------------------------------------------------------
MAZE_COLOR = {"linear": "#1b6ca8", "circular": "#8ac926"}


def fig_multisession(summary, pooled_fields, pooled_maps, pooled_glm, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
    fig.subplots_adjust(hspace=0.75, wspace=0.32)
    order = summary.sort_values(["maze_type", "session"]).index
    sm = summary.loc[order]
    x = np.arange(len(sm))
    labels = [f"{s}\n{int(t)} cm {mt}" for s, t, mt in
              zip(sm["session"], sm["track_cm"], sm["maze_type"])]
    cols = [MAZE_COLOR[m] for m in sm["maze_type"]]

    ax = axes[0, 0]
    ax.bar(x, 100 * sm["place_cell_fraction"], color=cols)
    for xi, (frac, n, ne) in enumerate(zip(sm["place_cell_fraction"],
                                           sm["n_place_cells"],
                                           sm["n_excitatory"])):
        ax.text(xi, 100 * frac + 1.5, f"{int(n)}/{int(ne)}", ha="center",
                fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("place cells (% of excitatory units)")
    ax.set_ylim(0, 108)
    lo, hi = 100 * sm["place_cell_fraction"].min(), 100 * sm["place_cell_fraction"].max()
    ax.set_title(f"(a) A large fraction of CA1 pyramidal cells has\n"
                 f"a place field in every session ({lo:.0f}-{hi:.0f}%)")
    ax.legend(handles=[Patch(facecolor=c, label=m) for m, c in MAZE_COLOR.items()],
              fontsize=8, loc="upper center", ncol=2)

    ax = axes[0, 1]
    groups = [pooled_fields.loc[pooled_fields["session"] == s, "si"].values
              for s in sm["session"]]
    bp = ax.boxplot(groups, positions=x, widths=0.6, patch_artist=True,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for g, xi, c in zip(groups, x, cols):
        ax.scatter(np.random.default_rng(2).normal(xi, 0.08, g.size), g, s=6,
                   color=c, alpha=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("spatial information (bits/spike)")
    ax.set_title("(b) Spatial information per field")

    ax = axes[0, 2]
    ax.bar(x, sm["decode_error_cm"], color=cols)
    ax.plot(x, sm["chance_error_cm"], "k_", ms=18, mew=2, label="chance")
    for xi, (e, c) in enumerate(zip(sm["decode_error_cm"], sm["chance_error_cm"])):
        ax.text(xi, e + 1, f"{e:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("median decoding error (cm)")
    ax.set_title("(c) Held-out position is decodable\nin every session")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    maps = pooled_maps / np.maximum(pooled_maps.max(axis=1, keepdims=True), 1e-9)
    maps = maps[np.argsort(np.argmax(maps, axis=1))]
    im = ax.imshow(maps, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=1, extent=[0, 100, 0, maps.shape[0]])
    ax.set_xlabel("position (% of track)")
    ax.set_ylabel("place field #")
    ax.set_title(f"(d) All {maps.shape[0]} fields from all "
                 f"{len(sm)} sessions, sorted by peak")
    fig.colorbar(im, ax=ax, fraction=0.05, label="rate / peak rate")

    ax = axes[1, 1]
    for key, col, lab in (("width", "#1b6ca8", "field width (cm)"),):
        ax.hist(pooled_fields[key], bins=30, color=col, alpha=0.85)
        ax.axvline(pooled_fields[key].median(), color="k", ls="--", lw=1)
        ax.set_xlabel(lab)
    ax.set_ylabel("number of fields")
    rel = (pooled_fields["width"] / pooled_fields["track_cm"]).median()
    ax.set_title(f"(e) Pooled field width\n"
                 f"median {pooled_fields['width'].median():.0f} cm "
                 f"({rel:.0%} of the animal's own track)")

    ax = axes[1, 2]
    g = pooled_glm
    ax.scatter(g["speed"], g["position"], s=14, c=[MAZE_COLOR[m] for m in g["maze_type"]],
               alpha=0.75)
    lim = [min(0, g["speed"].min()) - 0.05, g["position"].max() * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("speed-only GLM (bits/spike)")
    ax.set_ylabel("position-only GLM (bits/spike)")
    ax.legend(handles=[Patch(facecolor=c, label=m) for m, c in MAZE_COLOR.items()],
              fontsize=8, loc="lower right")
    ax.set_title(f"(f) Position beats speed for {(g['position'] > g['speed']).mean():.0%}\n"
                 f"of place cells, pooled over sessions")

    fig.suptitle("The same result holds across all 8 sessions of DANDI:000044 "
                 "(4 rats, three maze geometries)", y=0.98, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname
