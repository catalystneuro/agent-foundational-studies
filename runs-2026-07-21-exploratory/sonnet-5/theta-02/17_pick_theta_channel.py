import h5py
import remfile
import numpy as np
import pynapple as nap
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import time

url = open('s3_url.txt').read().strip()
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")

lin_grp2 = h5f['processing']['behavior']['1.6mLinearMazeLinearizedPosition']
lss2 = lin_grp2['1.6mLinearMazeLinearizedTimeSeries']
lin_grp = h5f['processing']['behavior']['1.6mLinearMazePosition']
lss = lin_grp['1.6mLinearMazeSpatialSeries']
starting_time = lss['starting_time'][()]
rate = lss['starting_time'].attrs['rate'] * 1000
data = lss2['data'][:, 0]
t = starting_time + np.arange(len(data)) / rate
good = ~np.isnan(data)
lin_pos = nap.Tsd(t=t[good], d=data[good])
maze_ep = nap.IntervalSet(start=18079.5, end=20147.0)
lin_pos_maze = lin_pos.restrict(maze_ep)

tt = lin_pos_maze.t; xx = lin_pos_maze.values
dtv = np.diff(tt); dxv = np.diff(xx)
cont_mask = dtv < 0.15
vel = dxv[cont_mask] / dtv[cont_mask]
tmid = (tt[:-1] + dtv/2)[cont_mask]
vel_tsd = nap.Tsd(t=tmid, d=vel)
outbound_ep = vel_tsd.threshold(0.1, method="above").time_support.drop_short_intervals(0.3)
inbound_ep = vel_tsd.threshold(-0.1, method="below").time_support.drop_short_intervals(0.3)
run_ep = outbound_ep.union(inbound_ep)
print("run_ep total dur", run_ep.tot_length(), "n", len(run_ep))

io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
lfp = nwb["LFP"]
print("lfp rate", lfp.rate)

t0 = time.time()
candidates = list(range(0, 128, 8))
psds = {}
for ch in candidates:
    sig = lfp[:, ch].restrict(maze_ep)
    psd = nap.compute_power_spectral_density(sig, fs=1250.0)
    psds[ch] = psd
print("time", time.time()-t0)

fig, ax = plt.subplots(figsize=(10,6))
theta_ratios = {}
for ch, psd in psds.items():
    freqs = psd.index.values
    p = np.abs(psd.values)
    theta_mask = (freqs>=6)&(freqs<=10)
    broad_mask = (freqs>=2)&(freqs<=40)
    ratio = p[theta_mask].sum()/p[broad_mask].sum()
    theta_ratios[ch] = ratio
    ax.plot(freqs, p, label=f"ch{ch} ratio={ratio:.2f}")
ax.set_xlim(0,30)
ax.set_yscale('log')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("check_theta_channels.png", dpi=120)

best_ch = max(theta_ratios, key=theta_ratios.get)
print("theta ratios:", theta_ratios)
print("Best channel:", best_ch)
io.close()
