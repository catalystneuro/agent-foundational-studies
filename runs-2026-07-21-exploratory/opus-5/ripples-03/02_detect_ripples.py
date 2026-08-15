"""Stage 2: detect sharp-wave ripples on the full 9.7 h recording.

Standard envelope-threshold detection on the 130-250 Hz band of the best CA1
pyramidal-layer channel, with movement periods excluded.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm

import swr_utils as su

SESSION = "Achilles-10252013"
LOW, HIGH = 130.0, 250.0
PEAK_Z, EDGE_Z = 4.0, 2.0
MIN_DUR, MAX_DUR, MERGE_GAP = 0.020, 0.200, 0.020
SPEED_THRESH = 4.0  # cm/s; ripples are an immobility/sleep phenomenon


def bandpass(x, lo, hi, fs, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def envelope(x, fs, chunk=2 ** 22):
    """Hilbert envelope, computed in overlapping chunks to bound memory."""
    pad = int(2 * fs)
    out = np.empty(len(x), dtype=np.float32)
    for i in tqdm(range(0, len(x), chunk), desc="hilbert"):
        a, b = max(0, i - pad), min(len(x), i + chunk + pad)
        e = np.abs(hilbert(x[a:b].astype(np.float64)))
        out[i:min(i + chunk, len(x))] = e[i - a: i - a + min(chunk, len(x) - i)]
    return out


def detect(env_z, peak_z=PEAK_Z, edge_z=EDGE_Z):
    """Threshold-crossing detection: edges at edge_z, kept if the peak exceeds peak_z."""
    cand = env_z.threshold(edge_z, "above").time_support
    # merge events separated by less than MERGE_GAP
    starts, ends = list(cand.start), list(cand.end)
    ms, me = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - me[-1] < MERGE_GAP:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
    cand = nap.IntervalSet(start=np.array(ms), end=np.array(me))
    dur = cand.end - cand.start
    cand = cand[(dur >= MIN_DUR) & (dur <= MAX_DUR)]

    keep, peak_t, peak_v = [], [], []
    for i in range(len(cand)):
        seg = env_z.get(cand.start[i], cand.end[i])
        if len(seg) == 0:
            continue
        j = int(np.argmax(seg.values))
        if seg.values[j] >= peak_z:
            keep.append(i)
            peak_t.append(float(seg.index[j]))
            peak_v.append(float(seg.values[j]))
    return cand[np.array(keep)], np.array(peak_t), np.array(peak_v)


if __name__ == "__main__":
    h5 = su.open_session(SESSION)
    epochs = su.load_epochs(h5)
    states = su.load_states(h5)
    pos = su.load_position(h5)
    best_ch, alt_ch = np.load("channels.npy")
    _, fs, _ = su.lfp_meta(h5)

    print("reading full LFP channel", best_ch, "...")
    lfp = su.load_lfp_channel(h5, int(best_ch))
    print("  %.2f M samples" % (len(lfp) / 1e6))

    filt = bandpass(lfp.values, LOW, HIGH, fs)
    env = envelope(filt, fs)
    env_z = nap.Tsd(t=lfp.index, d=(env - env.mean()) / env.std())

    ripples, peak_t, peak_z = detect(env_z)
    print("candidate ripples:", len(ripples))

    # Exclude periods of locomotion on the maze (speed from linearized position).
    speed = su.compute_speed(pos)
    np.savez("speed.npz", t=speed.index, d=speed.values)
    run_ep = speed.threshold(SPEED_THRESH, "above").time_support.drop_short_intervals(0.5)
    print("running time on maze: %.0f s of %.0f s" % (run_ep.tot_length(), epochs["MAZE"].tot_length()))

    in_run = np.zeros(len(ripples), dtype=bool)
    for s, e in zip(run_ep.start, run_ep.end):
        in_run |= (peak_t >= s) & (peak_t <= e)
    ripples = ripples[~in_run]
    peak_t, peak_z = peak_t[~in_run], peak_z[~in_run]
    print("ripples after removing locomotion: %d (%.2f Hz overall)"
          % (len(ripples), len(ripples) / (lfp.index[-1] - lfp.index[0])))

    np.savez("ripples.npz", start=ripples.start, end=ripples.end,
             peak_t=peak_t, peak_z=peak_z, channel=best_ch, fs=fs)
    np.save("lfp_filt.npy", filt.astype(np.float32))
    np.save("lfp_raw.npy", lfp.values.astype(np.float32))
    np.save("env_z.npy", env_z.values.astype(np.float32))

    # ---- rates by epoch and by brain state ----------------------------------
    rows = []
    for name, ep in epochs.items():
        n = int(((peak_t[:, None] >= ep.start) & (peak_t[:, None] <= ep.end)).any(1).sum())
        rows.append((name, n, ep.tot_length(), n / ep.tot_length()))
    for name, ep in states.items():
        n = int(((peak_t[:, None] >= ep.start) & (peak_t[:, None] <= ep.end)).any(1).sum())
        rows.append((name, n, ep.tot_length(), n / ep.tot_length()))
    for r in rows:
        print("  %-10s n=%5d  dur=%8.0f s  rate=%.3f Hz" % r)
    np.save("ripple_rates.npy", np.array(rows, dtype=object), allow_pickle=True)
