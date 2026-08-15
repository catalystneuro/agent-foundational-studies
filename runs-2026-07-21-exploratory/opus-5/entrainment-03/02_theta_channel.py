"""Stage 2: pick the LFP channel with the strongest theta and validate phase extraction."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.signal import welch
from tqdm import tqdm

import theta_utils as tu

SESSION = "Achilles-10252013"
h = tu.open_session(tu.SESSIONS[SESSION])
meta = tu.load_metadata(h)
maze = meta["epochs"]["MazeEpoch"]

# ------------------------------------------------- scan channels for theta power
# A 100 s window in the middle of the maze epoch is enough to rank channels.
t0 = maze.start[0] + 900
t1 = t0 + 100
ratios, powers = [], []
for ch in tqdm(range(meta["n_lfp_channels"]), desc="scanning channels"):
    x = tu.load_lfp_channel(h, ch, t0, t1).d
    ratios.append(tu.theta_delta_ratio(x))
    powers.append(tu.band_power(x, tu.LFP_FS, tu.THETA_BAND))
ratios = np.array(ratios)
powers = np.array(powers)
best_ch = int(np.argmax(ratios))
print(f"best theta channel: {best_ch}  theta/delta = {ratios[best_ch]:.2f}")
np.save("theta_channel_ratios.npy", ratios)

# ---------------------------------------------------- run epochs and state PSDs
pos = meta["position"].restrict(maze)
speed = tu.compute_speed(pos, rate=meta["position_rate"])
run = speed.threshold(10.0).time_support.drop_short_intervals(1.0)
print(f"run epochs: {len(run)} intervals, {run.tot_length():.0f} s "
      f"of {maze.tot_length():.0f} s maze")

lfp_maze = tu.load_lfp_channel(h, best_ch, maze.start[0], maze.end[0])
rem = meta["states"]["REM"].intersect(meta["epochs"]["POSTEpoch"])
nrem = meta["states"]["Non-REM"].intersect(meta["epochs"]["POSTEpoch"])
# take a bounded amount of sleep LFP to keep the download modest
rem = rem[: min(len(rem), 12)]
nrem = nrem[: min(len(nrem), 12)]
lfp_rem = tu.load_lfp_channel(h, best_ch, rem.start[0], rem.end[-1])
lfp_nrem = tu.load_lfp_channel(h, best_ch, nrem.start[0], nrem.end[-1])

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].plot(ratios, ".-", ms=4, lw=0.6, color="0.4")
axs[0].plot(best_ch, ratios[best_ch], "o", color="C3", ms=9,
            label=f"selected ch {best_ch}")
axs[0].set_xlabel("LFP channel"); axs[0].set_ylabel("theta / delta power")
axs[0].set_title("channel selection (100 s of maze)")
axs[0].legend(fontsize=8)

for lbl, sig, c in [("run (maze)", lfp_maze.restrict(run).d, "C0"),
                    ("REM sleep", lfp_rem.restrict(rem).d, "C2"),
                    ("non-REM sleep", lfp_nrem.restrict(nrem).d, "C3")]:
    f, p = welch(sig, fs=tu.LFP_FS, nperseg=int(4 * tu.LFP_FS))
    m = (f > 0.5) & (f < 30)
    axs[1].semilogy(f[m], p[m], color=c, label=lbl)
axs[1].axvspan(*tu.THETA_BAND, color="C1", alpha=0.15)
axs[1].set_xlabel("frequency (Hz)"); axs[1].set_ylabel("PSD (µV²/Hz)")
axs[1].set_title(f"power spectrum by brain state (ch {best_ch})")
axs[1].legend(fontsize=8)

# phase extraction sanity check: theta-trough-triggered average of the raw LFP
filt, phase, amp = tu.theta_phase_amplitude(lfp_maze)
ph_run = phase.restrict(run).d
troughs = np.where((np.diff(np.sign(np.mod(ph_run + np.pi, 2 * np.pi) - np.pi)) < 0))[0]
raw_run = lfp_maze.restrict(run).d
w = int(0.15 * tu.LFP_FS)
seg = np.array([raw_run[i - w:i + w] for i in troughs[5:2000]
                if i - w >= 0 and i + w < len(raw_run)])
tt = np.arange(-w, w) / tu.LFP_FS
axs[2].plot(tt, seg.mean(0), color="C0")
axs[2].fill_between(tt, seg.mean(0) - seg.std(0) / np.sqrt(len(seg)),
                    seg.mean(0) + seg.std(0) / np.sqrt(len(seg)),
                    color="C0", alpha=0.3)
axs[2].axvline(0, color="k", ls="--", lw=0.8)
axs[2].set_xlabel("time from detected theta trough (s)")
axs[2].set_ylabel("raw LFP (µV)")
axs[2].set_title(f"trough-triggered average, n={len(seg)} cycles")
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig03_theta_channel_and_spectra.png", dpi=150)
print("saved fig03_theta_channel_and_spectra.png")

print("theta/delta ratio: run=%.2f REM=%.2f nonREM=%.2f" % (
    tu.theta_delta_ratio(lfp_maze.restrict(run).d),
    tu.theta_delta_ratio(lfp_rem.restrict(rem).d),
    tu.theta_delta_ratio(lfp_nrem.restrict(nrem).d)))
