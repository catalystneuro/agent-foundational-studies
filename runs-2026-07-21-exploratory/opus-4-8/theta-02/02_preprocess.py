"""Select a theta-rich LFP channel and build position / speed / run epochs."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from tqdm import tqdm
import pynapple as nap

import hc11

SESSION = "Achilles_10252013"

nwbfile, io = hc11.open_session(SESSION)
maze = hc11.maze_epoch(nwbfile)
print("maze epoch:", maze)

pos_all = hc11.load_position(nwbfile)
tracked = maze.intersect(pos_all.time_support)
pos = pos_all.restrict(tracked)
print("position:", pos.shape, "range", pos.min(), pos.max())
print(f"tracked: {tracked.tot_length():.0f} s of {maze.tot_length():.0f} s "
      f"in {len(tracked)} intervals")

units = hc11.load_units(nwbfile)
print(units)

# ---- one candidate LFP channel per shank -----------------------------------
el = nwbfile.electrodes.to_dataframe()
candidates = []
for g, sub in el.groupby("group_name", sort=False):
    idx = sub.index.to_numpy()
    candidates.append(int(idx[len(idx) // 2]))
print("candidate channels:", candidates)

scores, psds = [], {}
for ch in tqdm(candidates, desc="scanning LFP channels"):
    lfp = hc11.load_lfp_channel(nwbfile, ch, maze)
    fr, pxx = welch(lfp.values, fs=hc11.LFP_FS, nperseg=int(4 * hc11.LFP_FS))
    th = pxx[(fr >= 6) & (fr <= 10)].mean()
    delta = pxx[(fr >= 1) & (fr <= 4)].mean()
    scores.append(th / delta)
    psds[ch] = (fr, pxx)
scores = np.array(scores)
best_ch = candidates[int(np.argmax(scores))]
print("theta/delta per channel:", dict(zip(candidates, np.round(scores, 2))))
print("best channel:", best_ch)

lfp = hc11.load_lfp_channel(nwbfile, best_ch, maze)
filt, phase = hc11.theta_phase(lfp)

# ---- speed and running epochs ---------------------------------------------
vel = hc11.compute_velocity(pos)
speed, run, run_right, run_left = hc11.run_epochs(vel)
print(f"run: {run.tot_length():.0f} s  right: {run_right.tot_length():.0f} s "
      f"left: {run_left.tot_length():.0f} s")

np.savez("preproc_%s.npz" % SESSION, best_ch=best_ch,
         cand=candidates, scores=scores)

# ---- validation figure -----------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 11))

axes[0].plot(pos.t - maze.start[0], pos.values, lw=0.7, color="k")
axes[0].set_ylabel("linearized\nposition (m)")
axes[0].set_title(f"{SESSION}: maze epoch behaviour and LFP")
axes[0].set_xlim(0, maze.tot_length())

axes[1].plot(speed.t - maze.start[0], speed.values, lw=0.5, color="tab:gray")
axes[1].axhline(hc11.SPEED_THRESH, color="tab:red", ls="--", label="run threshold")
axes[1].set_ylabel("speed (m/s)")
axes[1].set_ylim(0, 1.2)
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_xlim(0, maze.tot_length())

for ch, (fr, pxx) in psds.items():
    axes[2].semilogy(fr, pxx, lw=0.8, alpha=0.5,
                     color="tab:red" if ch == best_ch else "tab:gray")
axes[2].axvspan(*hc11.THETA_BAND, color="tab:blue", alpha=0.15)
axes[2].set_xlim(0, 30)
axes[2].set_xlabel("frequency (Hz)")
axes[2].set_ylabel("PSD (V$^2$/Hz)")
axes[2].set_title(f"LFP power spectra, one channel per shank "
                  f"(red = selected ch {best_ch})", fontsize=10)

t0 = run_right.start[10]
win = nap.IntervalSet(t0, t0 + 2)
axes[3].plot(lfp.restrict(win).t - t0, lfp.restrict(win).values * 1e3,
             color="tab:gray", lw=0.8, label="raw LFP")
axes[3].plot(filt.restrict(win).t - t0, filt.restrict(win).values * 1e3,
             color="tab:blue", lw=1.5, label="6-10 Hz")
ax3b = axes[3].twinx()
ax3b.plot(phase.restrict(win).t - t0, phase.restrict(win).values,
          color="tab:orange", lw=0.8, alpha=0.7)
ax3b.set_ylabel("theta phase (rad)", color="tab:orange")
axes[3].set_xlabel("time from start of a running bout (s)")
axes[3].set_ylabel("LFP (mV)")
axes[3].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("fig01_preprocessing.png", dpi=150)
print("saved fig01_preprocessing.png")
