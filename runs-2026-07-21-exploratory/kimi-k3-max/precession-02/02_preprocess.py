# 02_preprocess.py — maze-epoch position, run bouts from linearized position, theta phase from LFP ch 117
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
import pynapple as nap
from achilles_loader import open_nwb

THETA_CH = 117
LFP_FS = 1250.0
THETA_BAND = (6, 12)

nwbfile, nwb, h5 = open_nwb()

# --- epochs ---
epochs = nwb["epochs"]
labels = np.asarray(epochs.label)
maze_mask = labels == "MazeEpoch"
maze_start = float(np.asarray(epochs.start)[maze_mask][0])
maze_end = float(np.asarray(epochs.end)[maze_mask][0])
print(f"MazeEpoch: {maze_start:.1f} - {maze_end:.1f} s ({maze_end - maze_start:.1f} s)")

# --- position ---
pos = nwb["1.6mLinearMazeSpatialSeries"]   # TsdFrame x,y
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]  # TsdFrame 1 col
pos_t = np.asarray(pos.t)
pos_xy = np.asarray(pos.values)
lin_v = np.asarray(lin.values).ravel()
dt = np.median(np.diff(pos_t))
print("position: n =", len(pos_t), "dt =", dt, "span", pos_t[0], "->", pos_t[-1])
print("lin valid fraction:", np.mean(np.isfinite(lin_v)),
      "range:", np.nanmin(lin_v), np.nanmax(lin_v))

# --- 2D speed (for overview only) ---
valid2d = np.isfinite(pos_xy[:, 0]) & np.isfinite(pos_xy[:, 1])
xy = pos_xy.copy()
for c in range(2):
    xy[:, c] = np.interp(pos_t, pos_t[valid2d], pos_xy[valid2d, c])
vx = np.gradient(xy[:, 0], dt)
vy = np.gradient(xy[:, 1], dt)
speed2d = np.sqrt(vx**2 + vy**2)
win = int(round(0.25 / dt))
gw = signal.windows.gaussian(win, std=win / 6)
gw /= gw.sum()
speed2d = np.convolve(speed2d, gw, mode="same")

# --- run bouts: contiguous valid stretches of the linearized coordinate ---
# (the linearized series is NaN except during on-track runs)
lin_ok = np.isfinite(lin_v)
# merge stretches separated by < 0.3 s (brief tracking dropouts)
d = np.diff(lin_ok.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if lin_ok[0]:
    starts = [0] + starts
if lin_ok[-1]:
    ends = ends + [len(lin_ok)]
segs = list(zip(starts, ends))
merged = []
for s, e in segs:
    if merged and (pos_t[s] - pos_t[merged[-1][1] - 1]) < 0.3:
        merged[-1] = (merged[-1][0], e)
    else:
        merged.append((s, e))
# keep segments >= 1.0 s with clear direction and decent speed
bouts = []
for s, e in merged:
    dur = (e - s) * dt
    if dur < 1.0:
        continue
    seg_lin = lin_v[s:e]
    seg_t = pos_t[s:e]
    ok = np.isfinite(seg_lin)
    if ok.mean() < 0.8:
        continue
    v = np.abs(np.diff(seg_lin[ok])) / dt
    med_speed = np.median(v)
    dpos = np.diff(seg_lin[ok])
    direction = 1 if np.median(dpos) > 0 else -1
    span = seg_lin[ok].max() - seg_lin[ok].min()
    if med_speed < 0.15 or span < 0.3:
        continue
    bouts.append((seg_t[0], seg_t[-1], direction, med_speed, span))
bout_dir = np.array([d for _, _, d, _, _ in bouts], dtype=int)
bout_speed = np.array([v for _, _, _, v, _ in bouts])
bout_span = np.array([sp for _, _, _, _, sp in bouts])
bouts = np.array([(s, e) for s, e, _, _, _ in bouts])
print(f"run bouts: {len(bouts)}, total {(bouts[:, 1] - bouts[:, 0]).sum():.0f} s")
print(f"  +dir: {(bout_dir == 1).sum}, -dir: {(bout_dir == -1).sum}" if False else
      f"  +dir: {(bout_dir == 1).sum()}, -dir: {(bout_dir == -1).sum()}")
print(f"  median bout speed {np.median(bout_speed):.2f} m/s, median span {np.median(bout_span):.2f} m")
run_ep = nap.IntervalSet(start=bouts[:, 0], end=bouts[:, 1])

# --- LFP theta channel: read maze epoch, channel 117, in chunks ---
lfp_es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
i0 = int(maze_start * LFP_FS)
i1 = int(maze_end * LFP_FS)
n = i1 - i0
print(f"LFP maze slice: samples {i0}:{i1} = {n / LFP_FS:.1f} s")
chunk = int(100 * LFP_FS)  # 100 s chunks
lfp_raw = np.empty(n, dtype=np.float32)
for s in tqdm(range(0, n, chunk), desc="reading LFP ch117"):
    e = min(s + chunk, n)
    lfp_raw[s:e] = lfp_es.data[i0 + s:i0 + e, THETA_CH]
lfp_raw = lfp_raw.astype(np.float64) * lfp_es.conversion * 1e6  # uV
lfp_t = maze_start + np.arange(n) / LFP_FS

# --- theta bandpass + Hilbert phase ---
b, a = signal.butter(4, [THETA_BAND[0] / (LFP_FS / 2), THETA_BAND[1] / (LFP_FS / 2)], btype="band")
lfp_filt = signal.filtfilt(b, a, lfp_raw)
analytic = signal.hilbert(lfp_filt)
theta_phase = np.mod(np.angle(analytic), 2 * np.pi)  # [0, 2pi), 0 = LFP peak

# --- PSD on a run-heavy 200 s window (18500-18700 s) ---
m = (lfp_t >= 18500) & (lfp_t < 18700)
f, psd = signal.welch(lfp_raw[m], fs=LFP_FS, nperseg=int(4 * LFP_FS))
theta_pow = psd[(f >= 6) & (f <= 12)].mean()
delta_pow = psd[(f >= 1) & (f <= 4)].mean()
peak_f = f[(f >= 4) & (f <= 12)][np.argmax(psd[(f >= 4) & (f <= 12)])]
print(f"PSD (18500-18700s): theta/delta = {theta_pow / delta_pow:.2f}, peak {peak_f:.2f} Hz")

np.savez_compressed(
    "cache_preproc.npz",
    maze_start=maze_start, maze_end=maze_end,
    pos_t=pos_t, pos_xy=pos_xy, lin=lin_v, speed2d=speed2d,
    bout_starts=bouts[:, 0], bout_ends=bouts[:, 1],
    bout_dir=bout_dir, bout_speed=bout_speed, bout_span=bout_span,
    lfp_t=lfp_t, lfp_raw=lfp_raw, lfp_filt=lfp_filt, theta_phase=theta_phase,
    psd_f=f, psd=psd,
)

# --- validation figure ---
fig, axes = plt.subplots(4, 1, figsize=(12, 11))
ax = axes[0]
ax.plot(pos_t, lin_v, lw=0.5, color="k")
ax.set_ylabel("linearized pos (m)")
ax.set_xlim(maze_start, maze_end)
ax.set_title("Linearized position over maze epoch (NaN except on-track runs)")

ax = axes[1]
ax.plot(pos_t, speed2d, lw=0.5, color="tab:blue")
for (s, e), ddir in zip(bouts, bout_dir):
    ax.axvspan(s, e, color="tab:green" if ddir > 0 else "tab:orange", alpha=0.25, lw=0)
ax.set_ylabel("2D speed (m/s)")
ax.set_xlim(maze_start, maze_end)
ax.set_ylim(0, 1.5)
ax.set_title(f"Run bouts from linearized position (n={len(bouts)}; green=+dir, orange=-dir)")

ax = axes[2]
t0 = 18500.0
m2 = (lfp_t >= t0) & (lfp_t < t0 + 1.0)
ax.plot(lfp_t[m2], lfp_raw[m2], lw=0.4, color="gray", label="raw")
ax.plot(lfp_t[m2], lfp_filt[m2], lw=1.2, color="tab:purple", label="6-12 Hz")
# mark phase 0 (peaks)
pk = (np.diff(np.sign(np.diff(lfp_filt[m2]))) < 0)
ax.set_ylabel("LFP (uV)")
ax.set_title(f"Theta reference channel {THETA_CH}, 1 s snippet at {t0:.0f} s")
ax.legend(loc="upper right")

ax = axes[3]
ax.semilogy(f, psd, color="k")
ax.axvspan(6, 12, color="tab:purple", alpha=0.2, label="theta band")
ax.axvline(peak_f, color="tab:purple", ls="--", lw=0.8)
ax.set_xlim(0, 30)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (uV$^2$/Hz)")
ax.set_title(f"LFP PSD during running (18500-18700 s): theta/delta = {theta_pow / delta_pow:.1f}, peak {peak_f:.1f} Hz")
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig("fig01_data_overview.png", dpi=150)
print("saved fig01_data_overview.png")
