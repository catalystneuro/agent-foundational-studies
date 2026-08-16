"""02: Extract maze-epoch LFP on the reference channel, compute theta phase, build running mask."""
import remfile, h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal

with open("s3_url.txt") as f:
    BARE_URL = f.read().strip().split("?")[0]
ref = np.load("results/reference_channel.npz")
ref_ch, fs = int(ref["ref_ch"]), float(ref["fs"])

disk_cache = remfile.DiskCache('/tmp/remfile_cache_theta')
h5py_file = h5py.File(remfile.File(BARE_URL, disk_cache=disk_cache), "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

lfp_es = h5py_file["processing"]["ecephys"]["LFP"]["LFP"]
data = lfp_es["data"]
conversion = float(lfp_es["data"].attrs["conversion"])

maze_start, maze_end = 18079.5, 20147.0
i0, i1 = int(maze_start * fs), int(maze_end * fs)
# pad by 2 s each side so filter edge effects can be cropped
pad = int(2 * fs)
lfp = data[i0 - pad:i1 + pad, ref_ch] * conversion  # V
t_lfp = (np.arange(i0 - pad, i1 + pad) / fs).astype(np.float64)
print("lfp loaded:", lfp.shape, f"{t_lfp[0]:.1f}-{t_lfp[-1]:.1f} s")

# --- theta bandpass (5-11 Hz), zero-phase FIR ---
nyq = fs / 2
ntaps = int(np.ceil(3 * fs / 5))  # 3 cycles of 5 Hz
ntaps += 1 - ntaps % 2  # make odd
b = signal.firwin(ntaps, [5, 11], pass_zero=False, fs=fs)
theta = signal.filtfilt(b, [1.0], lfp)
phase = np.angle(signal.hilbert(theta))

# crop padding
sl = slice(pad, len(lfp) - pad)
lfp_c, theta_c, phase_c, t_c = lfp[sl], theta[sl], phase[sl], t_lfp[sl]

# --- speed from 2D position ---
pos = nwb["1.6mLinearMazeSpatialSeries"]
print(pos)
pos_maze = pos.get(maze_start, maze_end)  # TsdFrame restricted by time
xy = pos_maze.values
t_pos = pos_maze.index.values
print("position:", xy.shape, "columns:", pos_maze.columns)
# check units scale
print("x range:", np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0]))

dt_pos = np.median(np.diff(t_pos))
print(f"position dt: {dt_pos*1000:.1f} ms ({1/dt_pos:.1f} Hz)")

vx = np.gradient(xy[:, 0], t_pos)
vy = np.gradient(xy[:, 1], t_pos)
speed = np.sqrt(vx**2 + vy**2)
speed[np.isnan(xy[:, 0])] = np.nan
# smooth with 0.5 s Gaussian
sig = 0.5 / dt_pos
k = signal.windows.gaussian(int(6 * sig) | 1, sig)
k /= k.sum()
speed_s = np.convolve(np.nan_to_num(speed), k, mode="same")
speed_s[np.isnan(speed)] = np.nan

np.savez("results/theta_phase_maze.npz",
         t_lfp=t_c, lfp=lfp_c, theta=theta_c, phase=phase_c,
         t_pos=t_pos, xy=xy, speed=speed_s, fs=fs, ref_ch=ref_ch)

# --- epoch overview figure: speed + theta amplitude over the whole maze epoch ---
theta_amp = np.abs(signal.hilbert(theta_c))
# downsample envelope to ~10 Hz for plotting
ds = int(fs / 10)
t_ds = t_c[::ds]
amp_ds = theta_amp[::ds] * 1e6

fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax = axes[0]
ax.plot(t_pos, speed_s * 100, color="tab:green", lw=0.6)
ax.axhline(10, color="k", ls="--", lw=0.8, label="10 cm/s run threshold")
ax.set_ylabel("Speed (cm/s)")
ax.set_ylim(0, 120)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("Maze epoch overview: running speed and theta amplitude")
ax = axes[1]
ax.plot(t_ds, amp_ds, color="tab:blue", lw=0.5)
ax.set_ylabel("Theta amplitude (µV)")
ax.set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("figures/fig2_epoch_overview.png", dpi=150)
print("saved figures/fig2_epoch_overview.png")

# --- validation figure: 3 s window ---
units = nwb["units"]
t0 = t_c[0] + 500.0  # arbitrary window inside maze epoch
win = (t_c >= t0) & (t_c < t0 + 3.0)

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 2, 1.2]})
ax = axes[0]
ax.plot(t_c[win], lfp_c[win] * 1e6, color="0.5", lw=0.8, label="raw LFP")
ax.plot(t_c[win], theta_c[win] * 1e6, color="tab:blue", lw=1.4, label="theta (5-11 Hz)")
ax.set_ylabel("LFP (µV)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title(f"Theta phase extraction, reference channel {ref_ch}")

ax = axes[1]
ax.plot(t_c[win], phase_c[win], color="tab:purple", lw=1.0)
ax.set_ylabel("Theta phase (rad)")
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels(["-π", "0", "π"])
ax.set_ylim(-np.pi - 0.3, np.pi + 0.3)

# overlay spikes of 6 example units on the phase panel
keys = list(units.keys())[:6]
for j, k in enumerate(keys):
    spk = units[k].get(t0, t0 + 3.0)
    spk_t = spk.index.values
    ax.plot(spk_t, np.full_like(spk_t, np.pi + 0.15 + 0.05 * j), "|",
            color="tab:red", ms=6)
ax = axes[2]
ax.plot(t_pos, speed_s * 100, color="tab:green", lw=1.0)
ax.axhline(10, color="k", ls="--", lw=0.8, label="10 cm/s run threshold")
ax.set_ylabel("Speed (cm/s)")
ax.set_xlabel("Time (s)")
ax.set_xlim(t0, t0 + 3.0)
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig3_theta_phase_validation.png", dpi=150)
print("saved figures/fig3_theta_phase_validation.png")
print(f"speed: median {np.nanmedian(speed_s)*100:.1f} cm/s, "
      f"fraction >10 cm/s: {np.nanmean(speed_s > 0.10):.2f}")
