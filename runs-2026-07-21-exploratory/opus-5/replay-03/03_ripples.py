"""Stage 3: detect sharp-wave ripples in the PRE and POST sleep epochs."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.signal import welch
from tqdm import tqdm

import replaylib as R

SESSION = os.environ.get("SESSION", "Achilles-10252013")
FIG, CACHE = "figures", "cache"
os.makedirs(FIG, exist_ok=True)

LOW_THR, HIGH_THR = 3.0, 6.0     # robust SD (median/MAD) of the ripple envelope
MIN_DUR, MAX_DUR = 0.03, 0.30    # s
BLOCK = 1800.0                   # s per streaming block
PAD = 5.0                        # s of overlap discarded on each side

meta = np.load(f"{CACHE}/session_meta.npz", allow_pickle=True)
best_ch = int(meta["best_ch"])

h5, nwbfile, nwb = R.open_session(SESSION)
epochs = {row.label: nap.IntervalSet(start=row.start_time, end=row.stop_time)
          for row in nwbfile.epochs.to_dataframe().itertuples()}
states_df = nwbfile.processing["behavior"]["states"].to_dataframe()
rate, t0, n_t, n_ch = R.lfp_meta(h5)
print(f"ripple channel {best_ch}, LFP {rate} Hz")

rem = nap.IntervalSet(start=states_df.loc[states_df.label == "REM", "start_time"].values,
                      end=states_df.loc[states_df.label == "REM", "stop_time"].values)
nrem = nap.IntervalSet(start=states_df.loc[states_df.label == "Non-REM", "start_time"].values,
                       end=states_df.loc[states_df.label == "Non-REM", "stop_time"].values)


def envelope_for_epoch(ep):
    """Ripple-band envelope over an epoch, streamed in overlapping blocks."""
    start, stop = float(ep.start[0]), float(ep.end[0])
    envs, times, filts = [], [], []
    blocks = np.arange(start, stop, BLOCK)
    for b in tqdm(blocks, desc=f"LFP {start:.0f}-{stop:.0f}s"):
        lo = max(start, b - PAD)
        hi = min(stop, b + BLOCK + PAD)
        seg = R.read_lfp(h5, [best_ch], lo, hi)
        filt, env = R.ripple_envelope(seg.values[:, 0], rate)
        keep = (seg.t >= b) & (seg.t < min(b + BLOCK, stop))
        envs.append(env[keep]); times.append(seg.t[keep]); filts.append(filt[keep])
    return np.concatenate(times), np.concatenate(envs), np.concatenate(filts)


results = {}
for name in ("PREEpoch", "POSTEpoch"):
    ep = epochs[name]
    t, env, filt = envelope_for_epoch(ep)
    # Scale against the epoch's own robust baseline, then drop REM (theta-rich,
    # and ripples do not occur there) before calling events.
    iv, pk_t, pk_z = R.detect_ripples(env, t, rate, LOW_THR, HIGH_THR,
                                      MIN_DUR, MAX_DUR)
    in_rem = np.zeros(len(pk_t), dtype=bool)
    for s, e in zip(rem.start, rem.end):
        in_rem |= (pk_t >= s) & (pk_t <= e)
    iv = nap.IntervalSet(start=iv.start[~in_rem], end=iv.end[~in_rem])
    pk_t, pk_z = pk_t[~in_rem], pk_z[~in_rem]
    dur = iv.end - iv.start
    print(f"{name}: {len(iv)} ripples ({len(iv) / ep.tot_length():.3f} Hz), "
          f"median duration {np.median(dur) * 1000:.0f} ms, "
          f"median peak {np.median(pk_z):.1f} SD")
    results[name] = dict(start=iv.start, end=iv.end, peak_t=pk_t, peak_z=pk_z)
    if name == "POSTEpoch":
        post_t, post_env, post_filt = t, env, filt

np.savez(f"{CACHE}/ripples.npz", best_ch=best_ch, lfp_rate=rate,
         **{f"{k}_{f}": v[f] for k, v in results.items()
            for f in ("start", "end", "peak_t", "peak_z")})

# ---------------------------------------------------------------- figure 3
zpost = R.robust_z(post_env)
pk = results["POSTEpoch"]["peak_t"]
order = np.argsort(results["POSTEpoch"]["peak_z"])[::-1]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 3, hspace=0.65, wspace=0.3)

# Three example ripples: raw LFP, filtered, envelope.
for k, j in enumerate(order[[3, 20, 60]]):
    tc = pk[j]
    seg = R.read_lfp(h5, [best_ch], tc - 0.25, tc + 0.25)
    raw = seg.values[:, 0]
    f_, e_ = R.ripple_envelope(raw, rate)
    ax = fig.add_subplot(gs[0, k])
    ax.plot((seg.t - tc) * 1000, raw, "k", lw=0.7)
    ax.axvspan((results["POSTEpoch"]["start"][j] - tc) * 1000,
               (results["POSTEpoch"]["end"][j] - tc) * 1000, color="tab:orange", alpha=0.25)
    ax.set_title(f"raw LFP, ripple #{j} (peak {results['POSTEpoch']['peak_z'][j]:.1f} SD)",
                 fontsize=9)
    ax.set_ylabel("a.u." if k == 0 else "")
    ax = fig.add_subplot(gs[1, k])
    ax.plot((seg.t - tc) * 1000, f_, "k", lw=0.7)
    ax.plot((seg.t - tc) * 1000, e_, "tab:red", lw=1.2)
    ax.set_xlabel("time from ripple peak (ms)")
    ax.set_ylabel("140-250 Hz" if k == 0 else "")
    ax.set_title("ripple-band + envelope", fontsize=9)

# Ripple-triggered average of the filtered signal and its spectrum.
win = int(0.15 * rate)
idx = np.searchsorted(post_t, pk)
idx = idx[(idx > win) & (idx < len(post_t) - win)]
snips = np.stack([post_filt[i - win:i + win] for i in idx[:4000]])
ax = fig.add_subplot(gs[2, 0])
lag = (np.arange(-win, win) / rate) * 1000
ax.plot(lag, snips.mean(axis=0), "k", lw=1)
ax.set_xlabel("time from peak (ms)")
ax.set_ylabel("mean filtered LFP")
ax.set_title("Ripple-triggered average", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
raw_snips_f, raw_snips_p = welch(snips, fs=rate, nperseg=min(256, snips.shape[1]))
ax.semilogy(raw_snips_f, raw_snips_p.mean(axis=0), "k")
ax.set_xlim(0, 400)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("power")
ax.set_title("Spectrum of detected events", fontsize=10)

ax = fig.add_subplot(gs[2, 2])
for name, col in (("PREEpoch", "tab:blue"), ("POSTEpoch", "tab:red")):
    d = (results[name]["end"] - results[name]["start"]) * 1000
    ax.hist(d, bins=np.arange(MIN_DUR * 1000, MAX_DUR * 1000 + 5, 5), alpha=0.55,
            density=True, color=col, label=name.replace("Epoch", ""))
ax.set_xlabel("ripple duration (ms)")
ax.set_ylabel("density")
ax.set_title("Event durations", fontsize=10)
ax.legend(fontsize=8)

# Ripple rate over the session, and the NREM overlay.
ax = fig.add_subplot(gs[3, :])
allpk = np.concatenate([results[n]["peak_t"] for n in results])
bins = np.arange(0, n_t / rate + 60, 60)
cnt, _ = np.histogram(allpk, bins=bins)
ax.plot(bins[:-1] / 3600, cnt / 60, "k", lw=0.8)
for s, e in zip(nrem.start, nrem.end):
    ax.axvspan(s / 3600, e / 3600, color="tab:green", alpha=0.12, lw=0)
for name, col in (("PREEpoch", "tab:blue"), ("MazeEpoch", "0.5"), ("POSTEpoch", "tab:red")):
    ax.axvline(epochs[name].start[0] / 3600, color=col, ls="--")
    ax.text(epochs[name].start[0] / 3600 + 0.05, ax.get_ylim()[1] * 0.9,
            name.replace("Epoch", ""), color=col, fontsize=9)
ax.set_xlabel("time (h)")
ax.set_ylabel("ripple rate (Hz)")
ax.set_title("Ripple rate across the session (green = scored non-REM)", fontsize=10)

fig.savefig(f"{FIG}/03_ripples.png", dpi=130, bbox_inches="tight")
print("wrote", f"{FIG}/03_ripples.png")
