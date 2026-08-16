"""
Visualization suite for theta phase entrainment: an example LFP/phase/raster
snippet, per-unit polar phase histograms, and population-level summaries of
phase-locking strength split by cell type.
"""

import sys
from importlib import import_module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap

sys.path.insert(0, ".")
mod = import_module("01_load_and_inspect")

THETA_BAND = (6.0, 10.0)


if __name__ == "__main__":
    nwb, nwbfile = mod.load_nwb()
    epochs = nwb["epochs"]
    maze = epochs[epochs.label == "MazeEpoch"]
    units = nwb["units"]

    results = pd.read_csv("cache/phase_locking_results.csv")
    running_df = pd.read_csv("cache/running_epochs.csv")
    running = nap.IntervalSet(start=running_df["start"].values, end=running_df["end"].values)
    sig = np.load("cache/theta_signal.npz")
    t_theta = sig["t"]
    filt = sig["filtered"]
    phase = sig["phase"]
    amp = sig["amplitude"]
    raw = sig["raw"]

    phase_tsd = nap.Tsd(t=t_theta, d=phase)
    filt_tsd = nap.Tsd(t=t_theta, d=filt)
    raw_tsd = nap.Tsd(t=t_theta, d=raw)

    # =====================================================================
    # Figure 1: example snippet - raw LFP, theta-filtered LFP with theta
    # trough markers, and a spike raster of well-sampled strongly- and
    # weakly-locked units, during a running bout.
    # =====================================================================
    well_sampled = results[results["n_spikes"] >= 500]
    strongest = well_sampled.sort_values("mrl", ascending=False).iloc[0]
    weakest = well_sampled.sort_values("mrl", ascending=True).iloc[0]
    example_units = [int(strongest.unit_id), int(weakest.unit_id)]

    # pick a window centered on the longest running bout
    window = 6.0
    durations = np.asarray(running.end) - np.asarray(running.start)
    i_longest = int(np.argmax(durations))
    mid = (running.start[i_longest] + running.end[i_longest]) / 2
    t0 = mid - window / 2
    t1 = mid + window / 2

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.3, 1.5]},
    )

    raw_win = raw_tsd.restrict(nap.IntervalSet(t0, t1))
    filt_win = filt_tsd.restrict(nap.IntervalSet(t0, t1))
    phase_win = phase_tsd.restrict(nap.IntervalSet(t0, t1))

    # theta trough times (phase wraps through pi) mark theta cycle boundaries
    phase_unwrapped = np.unwrap(phase_win.d)
    cycle_crossings = phase_win.t[:-1][
        (np.mod(phase_unwrapped[:-1], 2 * np.pi) < np.pi)
        & (np.mod(phase_unwrapped[1:], 2 * np.pi) >= np.pi)
    ] - t0

    ax = axes[0]
    ax.plot(raw_win.t - t0, raw_win.d, color="steelblue", lw=0.8)
    for c in cycle_crossings:
        ax.axvline(c, color="gray", lw=0.5, alpha=0.5, ls="--")
    ax.set_ylabel("Raw LFP (a.u.)")
    ax.set_title(f"Theta channel — example {window:.0f} s running snippet, "
                 "dashed lines = theta troughs")

    ax = axes[1]
    ax.plot(filt_win.t - t0, filt_win.d, color="darkorange", lw=1.2)
    for c in cycle_crossings:
        ax.axvline(c, color="gray", lw=0.5, alpha=0.5, ls="--")
    ax.set_ylabel(f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz\nfiltered LFP")

    ax = axes[2]
    for i, uid in enumerate(example_units):
        spk = units[uid].restrict(nap.IntervalSet(t0, t1))
        ax.vlines(spk.t - t0, i, i + 0.8, color="k", lw=1.2)
    for c in cycle_crossings:
        ax.axvline(c, color="gray", lw=0.5, alpha=0.5, ls="--")
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(
        [f"unit {example_units[0]} ({strongest.cell_type})\nMRL={strongest.mrl:.2f}",
         f"unit {example_units[1]} ({weakest.cell_type})\nMRL={weakest.mrl:.2f}"],
        fontsize=8,
    )
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, window)

    fig.tight_layout()
    fig.savefig("figures/03_theta_lfp_phase_raster.png", dpi=150)
    plt.close(fig)
    print("Saved figures/03_theta_lfp_phase_raster.png")

    # =====================================================================
    # Figure 2: polar phase histograms for the strongest-locked pyramidal
    # cell, strongest-locked interneuron, and a weakly-locked cell.
    # =====================================================================
    exc = results[results.cell_type == "excitatory"].sort_values("mrl", ascending=False)
    inh = results[results.cell_type == "inhibitory"].sort_values("mrl", ascending=False)
    examples = [
        ("Strongest pyramidal cell", exc.iloc[0]),
        ("Strongest interneuron", inh.iloc[0]),
        ("Weakest (non-significant) unit", results.sort_values("mrl").iloc[0]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), subplot_kw={"projection": "polar"})
    for ax, (title, row) in zip(axes, examples):
        uid = int(row.unit_id)
        spk = units[uid].restrict(running)
        spk_phase = spk.value_from(phase_tsd).d
        n_bins = 24
        counts, bin_edges = np.histogram(spk_phase, bins=n_bins, range=(0, 2 * np.pi))
        width = 2 * np.pi / n_bins
        ax.bar(bin_edges[:-1], counts, width=width, align="edge",
               color="steelblue", edgecolor="white", alpha=0.85)
        ax.plot([row.mean_phase_rad, row.mean_phase_rad], [0, counts.max()],
                color="crimson", lw=2)
        ax.set_title(
            f"{title}\nunit {uid} ({row.cell_type})\n"
            f"MRL={row.mrl:.2f}, p={row.rayleigh_p:.1e}, n={int(row.n_spikes)}",
            fontsize=9,
        )
        ax.set_theta_zero_location("N")

    fig.tight_layout()
    fig.savefig("figures/04_example_polar_phase_histograms.png", dpi=150)
    plt.close(fig)
    print("Saved figures/04_example_polar_phase_histograms.png")

    # =====================================================================
    # Figure 3: population summary - MRL distributions by cell type,
    # fraction significant, and preferred-phase distribution.
    # =====================================================================
    fig = plt.figure(figsize=(13, 4.5))

    ax1 = fig.add_subplot(1, 3, 1)
    for ct, color in [("excitatory", "steelblue"), ("inhibitory", "darkorange")]:
        sub = results[results.cell_type == ct]
        ax1.hist(sub.mrl, bins=20, range=(0, 0.6), alpha=0.6, color=color, label=ct)
    ax1.set_xlabel("Mean resultant length (MRL)")
    ax1.set_ylabel("Number of units")
    ax1.set_title("Theta phase-locking strength by cell type")
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    order = ["excitatory", "inhibitory"]
    data = [results[results.cell_type == ct].mrl.values for ct in order]
    bp = ax2.boxplot(data, tick_labels=order, showmeans=True)
    for i, d in enumerate(data):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(d))
        ax2.scatter(np.full(len(d), i + 1) + jitter, d, s=10, alpha=0.5, color="gray")
    ax2.set_ylabel("MRL")
    ax2.set_title("Excitatory vs inhibitory phase locking")

    ax3 = fig.add_subplot(1, 3, 3, projection="polar")
    for ct, color in [("excitatory", "steelblue"), ("inhibitory", "darkorange")]:
        sub = results[(results.cell_type == ct) & results.significant]
        ax3.scatter(sub.mean_phase_rad, sub.mrl, s=25, alpha=0.7, color=color, label=ct)
    ax3.set_theta_zero_location("N")
    ax3.set_title("Preferred theta phase\n(significant units)", fontsize=9, pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/05_population_summary.png", dpi=150)
    plt.close(fig)
    print("Saved figures/05_population_summary.png")

    print("\nSummary statistics:")
    print(results.groupby("cell_type")[["mrl", "significant"]].agg(
        ["mean", "count"]
    ))
