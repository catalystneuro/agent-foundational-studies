"""
Core analysis: extract the instantaneous theta phase from the selected LFP
channel during running, assign each spike its theta phase, and compute
circular statistics (mean phase, mean resultant length, Rayleigh test) for
every CA1 unit to test for theta-phase entrainment.
"""

import sys
from importlib import import_module

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import hilbert

sys.path.insert(0, ".")
mod = import_module("01_load_and_inspect")

THETA_BAND = (6.0, 10.0)
MIN_SPEED = 0.05  # m/s, minimum running speed to include a period
MIN_SPIKES = 50  # minimum spikes during running for a unit to be analyzed


def rayleigh_test(phases):
    """Rayleigh test for non-uniformity of a circular distribution.

    Returns (R, p) where R is the mean resultant vector length and p is the
    Rayleigh test p-value (Zar, Biostatistical Analysis, approximation).
    """
    n = len(phases)
    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R = np.sqrt(C ** 2 + S ** 2) / n
    z = n * R ** 2
    p = np.exp(-z) * (
        1
        + (2 * z - z ** 2) / (4 * n)
        - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2)
    )
    return R, p


def compute_speed(position_2d):
    """Speed (m/s) from a 2-column position TsdFrame, lightly smoothed."""
    d = np.diff(position_2d.d, axis=0)
    dt = np.diff(position_2d.t)
    speed = np.sqrt((d ** 2).sum(axis=1)) / dt
    speed = np.concatenate([[speed[0]], speed])
    speed_tsd = nap.Tsd(t=position_2d.t, d=speed)
    return speed_tsd.smooth(std=0.25, windowsize=1.0)


if __name__ == "__main__":
    nwb, nwbfile = mod.load_nwb()
    epochs = nwb["epochs"]
    maze = epochs[epochs.label == "MazeEpoch"]
    lfp = nwb["LFP"]
    units = nwb["units"]

    with open("cache/theta_channel.txt") as fh:
        theta_channel = int(fh.read().strip())
    print(f"Using theta channel: {theta_channel}")

    # --- Running epochs: threshold speed computed from raw 2D position
    pos2d_ts = nwbfile.processing["behavior"]["1.6mLinearMazePosition"][
        "1.6mLinearMazeSpatialSeries"
    ]
    true_rate = 1.0 / pos2d_ts.rate
    t_pos = pos2d_ts.starting_time + np.arange(pos2d_ts.data.shape[0]) / true_rate
    pos2d = nap.TsdFrame(t=t_pos, d=pos2d_ts.data[:])
    pos2d = pos2d.restrict(maze)

    speed = compute_speed(pos2d)
    running = speed.threshold(MIN_SPEED, method="above").time_support
    running = running.intersect(maze)
    print(f"Running epochs: {len(running)} intervals, "
          f"{running.tot_length():.1f} s total out of "
          f"{float(maze.tot_length()):.1f} s maze epoch")

    # --- Load the theta channel LFP for the full maze epoch and filter
    print("Loading full theta channel over the maze epoch (streamed)...")
    lfp_channel = lfp[:, theta_channel].restrict(maze)
    print(f"Loaded {len(lfp_channel)} samples")

    filtered = nap.apply_bandpass_filter(
        lfp_channel, THETA_BAND, fs=lfp.rate, mode="butter", order=4
    )
    analytic = hilbert(filtered.d)
    phase = np.mod(np.angle(analytic), 2 * np.pi)
    phase_tsd = nap.Tsd(t=filtered.t, d=phase, time_support=filtered.time_support)
    amplitude = nap.Tsd(t=filtered.t, d=np.abs(analytic), time_support=filtered.time_support)

    # --- Restrict everything to running periods only
    phase_run = phase_tsd.restrict(running)
    units_run = units.restrict(running)

    # --- Per-unit circular statistics
    rows = []
    for uid in units.index:
        spk = units_run[uid]
        n_spk = len(spk)
        if n_spk < MIN_SPIKES:
            continue
        spk_phase = spk.value_from(phase_run)
        R, p = rayleigh_test(spk_phase.d)
        mean_phase = np.angle(np.mean(np.exp(1j * spk_phase.d))) % (2 * np.pi)
        rows.append(
            dict(
                unit_id=uid,
                cell_type=units.get_info("cell_type")[uid],
                location=units.get_info("location")[uid],
                n_spikes=n_spk,
                mean_phase_rad=mean_phase,
                mrl=R,
                rayleigh_p=p,
                significant=p < 0.05,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv("cache/phase_locking_results.csv", index=False)
    print(f"\nAnalyzed {len(results)} units (of {len(units)} total, "
          f"{MIN_SPIKES}+ spikes during running)")
    print(f"Significantly phase-locked (Rayleigh p<0.05): "
          f"{results['significant'].sum()} / {len(results)} "
          f"({100 * results['significant'].mean():.1f}%)")
    print(results.groupby("cell_type")["mrl"].describe())

    # Save phase/amplitude arrays (downsampled) and spike phases for plotting
    np.savez(
        "cache/theta_signal.npz",
        t=filtered.t,
        filtered=filtered.d,
        phase=phase,
        amplitude=amplitude.d,
        raw_t=lfp_channel.t,
        raw=lfp_channel.d,
    )
    pd.DataFrame({"start": running.start, "end": running.end}).to_csv(
        "cache/running_epochs.csv", index=False
    )
    print("Saved cache/phase_locking_results.csv, cache/theta_signal.npz, "
          "cache/running_epochs.csv")
