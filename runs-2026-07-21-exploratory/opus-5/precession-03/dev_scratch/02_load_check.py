"""Load position + LFP for the maze epoch and validate visually."""
import h5py, remfile, numpy as np, matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

ASSET = "5349c68b-c0a7-46c0-9900-cda050722fa4"
URL = f"https://api.dandiarchive.org/api/dandisets/000044/versions/0.250624.0426/assets/{ASSET}/download/"
h5 = h5py.File(remfile.File(URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache")), "r")

pos_g = h5["processing/behavior/1.6mLinearMazeLinearizedPosition/1.6mLinearMazeLinearizedTimeSeries"]
t0 = pos_g["starting_time"][()]
dt = pos_g["starting_time"].attrs["rate"]  # actually the sampling PERIOD in this conversion
lin = pos_g["data"][:, 0]
tpos = t0 + np.arange(len(lin)) * dt
print("pos t0", t0, "dt", dt, "n", len(lin), "t range", tpos[0], tpos[-1])
print("nan frac", np.isnan(lin).mean(), "min", np.nanmin(lin), "max", np.nanmax(lin))

xy = h5["processing/behavior/1.6mLinearMazePosition/1.6mLinearMazeSpatialSeries"]["data"][:]
print("xy nan frac", np.isnan(xy).mean(), "x range", np.nanmin(xy[:,0]), np.nanmax(xy[:,0]),
      "y range", np.nanmin(xy[:,1]), np.nanmax(xy[:,1]))

# LFP over the maze epoch, sampling a subset of channels for theta selection
lfp_d = h5["processing/ecephys/LFP/LFP/data"]
fs = 1250.0
conv = lfp_d.attrs["conversion"]
i0, i1 = int(round(t0 * fs)), int(round(tpos[-1] * fs))
print("lfp maze samples", i1 - i0)

sel = np.arange(0, 128, 8)
seg = slice(i0, i0 + int(300 * fs))
theta_ratio = []
for ch in sel:
    x = lfp_d[seg, ch].astype(float) * conv
    fr, P = welch(x, fs=fs, nperseg=2048)
    th = P[(fr >= 6) & (fr <= 10)].mean()
    de = P[((fr >= 2) & (fr <= 4)) | ((fr >= 12) & (fr <= 20))].mean()
    theta_ratio.append(th / de)
    print(f"ch {ch:3d} theta/ref = {th/de:.2f}")
best = sel[int(np.argmax(theta_ratio))]
print("best channel", best)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(tpos, lin, lw=0.5)
axes[0].set(xlabel="time (s)", ylabel="linear pos (m)", title="Linearized position, maze epoch")
m = (tpos > t0 + 200) & (tpos < t0 + 260)
axes[1].plot(tpos[m], lin[m], lw=1)
axes[1].set(xlabel="time (s)", ylabel="pos (m)", title="60 s zoom")
tl = np.arange(int(60 * fs)) / fs + t0 + 200
x = lfp_d[i0 + int(200 * fs): i0 + int(260 * fs), best].astype(float) * conv * 1e3
axes[2].plot(tl, x, lw=0.4)
axes[2].set(xlabel="time (s)", ylabel="LFP (mV)", title=f"LFP ch {best}")
plt.tight_layout(); plt.savefig("check_raw.png", dpi=110)
print("saved check_raw.png")
