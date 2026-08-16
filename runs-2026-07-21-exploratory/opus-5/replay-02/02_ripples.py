"""Stage 2: pick the best ripple channel, detect sharp-wave ripples in NREM sleep."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pynapple as nap
import common

SESSION = "Achilles_10252013"
nwbfile, h = common.open_session(SESSION)
lfp_ds = h["processing/ecephys/LFP/LFP/data"]
FS = nwbfile.processing["ecephys"]["LFP"]["LFP"].rate
CONV = nwbfile.processing["ecephys"]["LFP"]["LFP"].conversion

ep_df = nwbfile.epochs.to_dataframe()
epochs = {r.label: nap.IntervalSet(start=r.start_time, end=r.stop_time) for r in ep_df.itertuples()}
st = nwbfile.processing["behavior"]["states"].to_dataframe()
nrem = nap.IntervalSet(start=st.start_time[st.label == "Non-REM"].values,
                       end=st.stop_time[st.label == "Non-REM"].values)
post_nrem = nrem.intersect(epochs["POSTEpoch"])
pre_nrem = nrem.intersect(epochs["PREEpoch"])
print("POST NREM", len(post_nrem), post_nrem.tot_length(), "s | PRE NREM", pre_nrem.tot_length(), "s")


def bp(x, lo, hi, fs=FS, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


# ------------------------------------------------------------------ channel scan
# use the longest POST NREM bout as the probe window
dur = post_nrem.end - post_nrem.start
k = int(np.argmax(dur))
w0, w1 = post_nrem.start[k], min(post_nrem.end[k], post_nrem.start[k] + 300)
i0, i1 = int(w0 * FS), int(w1 * FS)
print(f"channel scan window {w0:.0f}-{w1:.0f} s (NREM bout {k}, {dur[k]:.0f} s long)")

ripple_sd, hf_sd = np.zeros(128), np.zeros(128)
for ch in tqdm(range(128), desc="scanning channels"):
    x = lfp_ds[i0:i1, ch].astype(np.float64) * CONV * 1e6      # uV
    ripple_sd[ch] = np.std(bp(x, 140, 230))
    hf_sd[ch] = np.std(bp(x, 300, 500))
score = ripple_sd / hf_sd
best = int(np.argmax(ripple_sd * score))
print(f"best ripple channel = {best} (ripple SD {ripple_sd[best]:.1f} uV, ripple/HF {score[best]:.2f})")

# ------------------------------------------------------------------ full-channel detection
print("reading full LFP for the selected channel ...")
raw = lfp_ds[:, best].astype(np.float64) * CONV * 1e6
t_lfp = np.arange(len(raw)) / FS
lfp = nap.Tsd(t=t_lfp, d=raw)
filt = nap.Tsd(t=t_lfp, d=bp(raw, 140, 230))
env_raw = np.abs(hilbert(filt.d))
env_raw = gaussian_filter1d(env_raw, sigma=0.0075 * FS)          # ~7.5 ms smoothing

# z-score against NREM statistics only
nrem_mask = nap.Tsd(t=t_lfp, d=env_raw).restrict(nrem).d
z = (env_raw - nrem_mask.mean()) / nrem_mask.std()
zenv = nap.Tsd(t=t_lfp, d=z)

PEAK_Z, EDGE_Z = 4.0, 1.0
MIN_DUR, MAX_DUR, MERGE = 0.020, 0.250, 0.020


def detect(ep, label):
    zz = zenv.restrict(ep)
    cand = zz.threshold(EDGE_Z, "above").time_support
    cand = cand.merge_close_intervals(MERGE)
    cand = cand[(cand.end - cand.start) >= MIN_DUR]
    keep, pk, pt = [], [], []
    for s, e in zip(cand.start, cand.end):
        seg = zz.get(s, e)
        if len(seg) == 0:
            continue
        m = int(np.argmax(seg.d))
        if seg.d[m] >= PEAK_Z and (e - s) <= MAX_DUR:
            keep.append((s, e)); pk.append(seg.d[m]); pt.append(seg.t[m])
    keep = np.array(keep)
    ips = nap.IntervalSet(start=keep[:, 0], end=keep[:, 1])
    print(f"{label}: {len(ips)} ripples in {ep.tot_length():.0f} s NREM "
          f"({len(ips)/ep.tot_length():.3f} Hz), median dur {np.median(keep[:,1]-keep[:,0])*1e3:.0f} ms")
    return ips, np.array(pk), np.array(pt)


post_rip, post_pk, post_pt = detect(post_nrem, "POST")
pre_rip, pre_pk, pre_pt = detect(pre_nrem, "PRE")

# ------------------------------------------------------------------ validation figure
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, hspace=.55, wspace=.3)

ax = fig.add_subplot(gs[0, 0])
ax.plot(ripple_sd, ".-", lw=.6, ms=3)
ax.axvline(best, color="r", ls="--"); ax.set(xlabel="channel", ylabel="140-230 Hz SD (uV)",
                                             title=f"Ripple-band power per channel\n(selected ch {best})")

ax = fig.add_subplot(gs[0, 1:])
ex = int(np.argmax(post_pk))
c = post_pt[ex]
seg = lfp.get(c - .3, c + .3); segf = filt.get(c - .3, c + .3); segz = zenv.get(c - .3, c + .3)
ax.plot(seg.t - c, seg.d, color="k", lw=.8, label="raw LFP")
ax.plot(segf.t - c, segf.d - 350, color="#e63946", lw=.8, label="140-230 Hz")
ax.plot(segz.t - c, segz.d * 25 - 700, color="#457b9d", lw=.8, label="z-scored envelope")
s0 = post_rip.start[ex]; e0 = post_rip.end[ex]
ax.axvspan(s0 - c, e0 - c, color="#ffb703", alpha=.3)
ax.set(xlabel="time from ripple peak (s)", title="Example detected sharp-wave ripple", yticks=[])
ax.legend(fontsize=8, loc="upper right")

# ripple-triggered average of the filtered signal + raw
ax = fig.add_subplot(gs[1, 0])
win = int(.1 * FS)
idx = (post_pt * FS).astype(int)
idx = idx[(idx > win) & (idx < len(raw) - win)]
sel = idx[np.linspace(0, len(idx) - 1, min(2000, len(idx))).astype(int)]
snips = np.stack([raw[i - win:i + win] for i in sel])
snf = np.stack([filt.d[i - win:i + win] for i in sel])
tt = (np.arange(-win, win)) / FS * 1e3
ax.plot(tt, snips.mean(0), "k"); ax.set(xlabel="ms from peak", ylabel="uV", title="Ripple-triggered LFP average")
ax = fig.add_subplot(gs[1, 1])
ax.plot(tt, snf.mean(0), color="#e63946"); ax.set(xlabel="ms from peak", title="Ripple-triggered 140-230 Hz average")
ax = fig.add_subplot(gs[1, 2])
ax.imshow(np.abs(hilbert(snf, axis=1))[:200], aspect="auto", extent=[tt[0], tt[-1], 200, 0], cmap="magma")
ax.set(xlabel="ms from peak", ylabel="event #", title="Ripple envelopes (200 events)")

ax = fig.add_subplot(gs[2, 0])
ax.hist((post_rip.end - post_rip.start) * 1e3, bins=40, color="#457b9d")
ax.set(xlabel="duration (ms)", ylabel="count", title=f"POST ripple durations (n={len(post_rip)})")
ax = fig.add_subplot(gs[2, 1])
ax.hist(post_pk, bins=40, color="#457b9d"); ax.set(xlabel="peak envelope (z)", title="Ripple peak amplitude")
ax = fig.add_subplot(gs[2, 2])
rate_pre = len(pre_rip) / pre_nrem.tot_length(); rate_post = len(post_rip) / post_nrem.tot_length()
ax.bar(["PRE", "POST"], [rate_pre, rate_post], color=["#8ecae6", "#90be6d"])
ax.set(ylabel="ripples / s", title="Ripple incidence in NREM")

plt.savefig("figures/02_ripple_detection.png", dpi=140, bbox_inches="tight")
np.savez("cache/02_ripples.npz",
         best_channel=best, ripple_sd=ripple_sd,
         post_start=post_rip.start, post_end=post_rip.end, post_peak_t=post_pt, post_peak_z=post_pk,
         pre_start=pre_rip.start, pre_end=pre_rip.end, pre_peak_t=pre_pt, pre_peak_z=pre_pk,
         nrem_start=nrem.start, nrem_end=nrem.end)
print("saved cache/02_ripples.npz and figures/02_ripple_detection.png")
