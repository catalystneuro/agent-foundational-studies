"""Overview sanity figure: position, spikes, LFP across epochs."""
import numpy as np
import h5py
import remfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal

with open('/tmp/achilles_url.txt') as f:
    bare = f.read().strip().split('?')[0]

disk_cache = remfile.DiskCache('/tmp/remfile_cache_ripples03')
h5py_file = h5py.File(remfile.File(bare, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())

units = nwb['units']
pos = nwb['1.6mLinearMazeSpatialSeries']
lin = nwb['1.6mLinearMazeLinearizedTimeSeries']
epochs = nwb['epochs']
states = nwb['states']

lfp_data = h5py_file['processing/ecephys/LFP/LFP/data']
conv = lfp_data.attrs['conversion']
fs = 1250.0

# ---------- figure ----------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

# (a) linearized position over maze epoch
ax = fig.add_subplot(gs[0, 0])
lt = lin.t
lv = np.asarray(lin.values).ravel()
ax.plot(lt[::10], lv[::10], lw=0.4, color='k')
ax.set_title('Linearized position (maze epoch)')
ax.set_xlabel('time (s)')
ax.set_ylabel('track position (m)')
ax.set_xlim(18079.5, 20147)

# (b) 2D trajectory
ax = fig.add_subplot(gs[0, 1])
pv = np.asarray(pos.values)
ax.plot(pv[::5, 0], pv[::5, 1], lw=0.2, color='darkblue')
ax.set_title('2D position (linear track)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# (c) LFP snippet during maze (theta) and during POST Non-REM
ax = fig.add_subplot(gs[1, 0])
ch = 117
i0 = int(19000 * fs)
seg = np.asarray(lfp_data[i0:i0 + int(2 * fs), ch]) * conv * 1e6  # uV
tseg = np.arange(seg.size) / fs
ax.plot(tseg, seg, lw=0.4, color='k')
ax.set_title('LFP ch117 during run (19000-19002 s)')
ax.set_xlabel('time (s)')
ax.set_ylabel('uV')

ax = fig.add_subplot(gs[1, 1])
# find a POST Non-REM interval
nrem = states[states.label == 'Non-REM']
post_nrem = nrem[(nrem.start > 20147)]
print('first POST Non-REM:', post_nrem.start[0], post_nrem.end[0])
t0 = post_nrem.start[0] + 5
i0 = int(t0 * fs)
seg = np.asarray(lfp_data[i0:i0 + int(2 * fs), ch]) * conv * 1e6
ax.plot(tseg, seg, lw=0.4, color='k')
ax.set_title(f'LFP ch117 during POST Non-REM ({t0:.0f}-{t0+2:.0f} s)')
ax.set_xlabel('time (s)')
ax.set_ylabel('uV')

# (d) PSD awake-run vs Non-REM on ch117
ax = fig.add_subplot(gs[2, 0])
for (label, tstart) in [('run (maze)', 19000), ('POST Non-REM', t0)]:
    i0 = int(tstart * fs)
    seg = np.asarray(lfp_data[i0:i0 + int(60 * fs), ch]) * conv
    f, pxx = signal.welch(seg, fs=fs, nperseg=int(4 * fs))
    ax.semilogy(f, pxx * 1e12, label=label, lw=1)
ax.set_xlim(0, 300)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('PSD (uV^2/Hz)')
ax.set_title('LFP power spectrum (ch117)')
ax.legend()
ax.axvspan(100, 250, color='red', alpha=0.08)

# (e) spike raster snippet during maze
ax = fig.add_subplot(gs[2, 1])
t0r, t1r = 19000, 19010
for i, u in enumerate(list(units.keys())[:60]):
    sp = units[u].t
    sp = sp[(sp >= t0r) & (sp < t1r)]
    ax.plot(sp - t0r, np.full_like(sp, i), '|', color='k', ms=2)
ax.set_title('Raster, first 60 units (19000-19010 s)')
ax.set_xlabel('time (s)')
ax.set_ylabel('unit index')

fig.savefig('fig01_session_overview.png', dpi=150)
print('saved fig01_session_overview.png')
print('position valid fraction (maze):', np.mean(np.isfinite(lv)))
