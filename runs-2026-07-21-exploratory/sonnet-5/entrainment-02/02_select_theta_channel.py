"""
Select the LFP channel with the strongest theta rhythm during running.

We compute the power spectral density (1-40 Hz) for a sparse sample of
channels (one every 8th of the 128) over a chunk of the MazeEpoch, and pick
the channel with the highest ratio of theta-band (6-10 Hz) power to
broadband (1-40 Hz) power. Chunks along time are per-channel in this NWB
file (HDF5 chunk shape (170221, 1)) so subsetting to one channel does not
require reading the other 127.
"""

import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from pynwb import NWBHDF5IO

from importlib import import_module
import sys

sys.path.insert(0, ".")
mod = import_module("01_load_and_inspect")

THETA_BAND = (6.0, 10.0)
BROAD_BAND = (1.0, 40.0)
CANDIDATE_CHANNELS = list(range(0, 128, 8))


if __name__ == "__main__":
    nwb, nwbfile = mod.load_nwb()
    epochs = nwb["epochs"]
    maze = epochs[epochs.label == "MazeEpoch"]
    lfp = nwb["LFP"]

    # Use a 200 s window well inside the maze run for channel selection.
    t0 = maze.start[0] + 100.0
    snippet = lfp.get(t0, t0 + 200.0)[:, CANDIDATE_CHANNELS]

    ratios = {}
    psds = {}
    for i, ch in enumerate(CANDIDATE_CHANNELS):
        sig = nap.Tsd(t=snippet.t, d=snippet[:, i].d)
        psd = nap.compute_power_spectral_density(sig, fs=lfp.rate)
        f = psd.index.values
        p = psd.iloc[:, 0].values.real
        p = np.abs(p)
        theta_mask = (f >= THETA_BAND[0]) & (f <= THETA_BAND[1])
        broad_mask = (f >= BROAD_BAND[0]) & (f <= BROAD_BAND[1])
        ratio = p[theta_mask].sum() / p[broad_mask].sum()
        ratios[ch] = ratio
        psds[ch] = (f, p)

    best_channel = max(ratios, key=ratios.get)
    print("Theta/broadband power ratio per candidate channel:")
    for ch in CANDIDATE_CHANNELS:
        marker = "  <-- best" if ch == best_channel else ""
        print(f"  channel {ch:3d}: {ratios[ch]:.4f}{marker}")
    print(f"\nSelected theta channel: {best_channel}")

    with open("cache/theta_channel.txt", "w") as fh:
        fh.write(str(best_channel))

    # --- Plot PSDs for all candidate channels, highlighting the winner
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for ch in CANDIDATE_CHANNELS:
        f, p = psds[ch]
        mask = (f >= 0.5) & (f <= BROAD_BAND[1])
        color = "crimson" if ch == best_channel else "gray"
        lw = 2.0 if ch == best_channel else 0.8
        alpha = 1.0 if ch == best_channel else 0.5
        ax.semilogy(f[mask], p[mask], color=color, lw=lw, alpha=alpha,
                    label=f"ch {ch} (best)" if ch == best_channel else None)
    ax.axvspan(*THETA_BAND, color="gold", alpha=0.2, label="theta band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("PSD across candidate channels (running epoch)")
    ax.legend()

    ax = axes[1]
    chs = list(ratios.keys())
    vals = [ratios[c] for c in chs]
    colors = ["crimson" if c == best_channel else "steelblue" for c in chs]
    ax.bar([str(c) for c in chs], vals, color=colors)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Theta / broadband power ratio")
    ax.set_title("Channel ranking")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig("figures/02_theta_channel_selection.png", dpi=150)
    plt.close(fig)
    print("Saved figures/02_theta_channel_selection.png")
