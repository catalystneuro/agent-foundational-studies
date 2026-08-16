"""04_detect_ripples.py — detect sharp-wave ripples on the best channel, full session.

Pipeline: bandpass 100-250 Hz (SOS butter4, filtfilt) -> Hilbert envelope ->
gaussian smooth (sigma 4 ms) -> threshold stats on Non-REM samples ->
events: peak > mean+4SD, edges at mean+1SD, merge gaps <30 ms,
keep durations 30-500 ms fully inside Non-REM.
"""
import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import signal
from scipy.ndimage import gaussian_filter1d

FS = 1250.0
s3_url = open("scripts/s3_url.txt").read().strip()
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_ripples02")), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())

ch = int(np.load("scripts/ripple_channel.npy")[0])
print(f"Ripple channel: {ch}")

states = nwb["states"]
nonrem = states[states.label == "Non-REM"]
epochs = nwb["epochs"]

print("Reading full LFP channel (43.6M samples)...")
dset = h5py_file["processing/ecephys/LFP/LFP/data"]
conv = dset.attrs["conversion"]
lfp = dset[:, ch].astype(np.float64) * conv * 1e6  # uV
t_lfp = np.arange(len(lfp)) / FS
print("done. duration:", t_lfp[-1], "s")

# --- ripple-band envelope ---
sos = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
filt = signal.sosfiltfilt(sos, lfp)
env = np.abs(signal.hilbert(filt))
sigma_samp = 0.004 * FS  # 4 ms
env_s = gaussian_filter1d(env, sigma_samp)
print("envelope computed")

# --- threshold stats on Non-REM samples only ---
nr_mask = np.zeros(len(lfp), dtype=bool)
for s, e in zip(nonrem.start, nonrem.end):
    nr_mask[int(s * FS):int(e * FS)] = True
mu, sd = env_s[nr_mask].mean(), env_s[nr_mask].std()
print(f"Non-REM envelope mean {mu:.2f} uV, SD {sd:.2f} uV")
peak_thr = mu + 4 * sd
edge_thr = mu + 1 * sd

# --- event detection ---
above = env_s > edge_thr
d = np.diff(above.astype(int))
starts = np.where(d == 1)[0] + 1
ends = np.where(d == -1)[0] + 1
if above[0]:
    starts = np.r_[0, starts]
if above[-1]:
    ends = np.r_[ends, len(above)]
events = np.column_stack([starts, ends])  # sample indices, [start, end)

# merge events separated by < 30 ms
merged = []
for s, e in events:
    if merged and (s - merged[-1][1]) < 0.030 * FS:
        merged[-1][1] = e
    else:
        merged.append([s, e])
merged = np.array(merged)

# keep events with a peak above peak_thr and duration 30-500 ms
keep = []
for s, e in merged:
    dur = (e - s) / FS
    if dur < 0.030 or dur > 0.500:
        continue
    if env_s[s:e].max() < peak_thr:
        continue
    keep.append((s, e))
keep = np.array(keep)
print(f"Detected {len(keep)} candidate ripples (pre Non-REM restriction)")

# --- restrict to events fully inside Non-REM ---
rip_t = keep / FS
in_nr = np.zeros(len(rip_t), dtype=bool)
for s, e in zip(nonrem.start, nonrem.end):
    in_nr |= (rip_t[:, 0] >= s) & (rip_t[:, 1] <= e)
rip_t = rip_t[in_nr]
print(f"{in_nr.sum()} ripples fully inside Non-REM")

ripples = nap.IntervalSet(start=rip_t[:, 0], end=rip_t[:, 1])
pre = epochs[epochs.label == "PREEpoch"]
post = epochs[epochs.label == "POSTEpoch"]
rip_pre = ripples.intersect(pre)
rip_post = ripples.intersect(post)
# intersect clips to epoch bounds; events are already fully inside Non-REM which respects epochs
print(f"PRE ripples: {len(rip_pre)}  ({len(rip_pre) / ((pre.end[0] - pre.start[0]) / 60):.2f}/min over full PRE)")
print(f"POST ripples: {len(rip_post)} ({len(rip_post) / ((post.end[0] - post.start[0]) / 60):.2f}/min over full POST)")

# rate per minute of Non-REM (more meaningful)
pre_nr = nonrem.intersect(pre)
post_nr = nonrem.intersect(post)
pre_nr_dur = float((pre_nr.end - pre_nr.start).sum())
post_nr_dur = float((post_nr.end - post_nr.start).sum())
print(f"PRE Non-REM duration {pre_nr_dur:.0f} s -> {len(rip_pre) / (pre_nr_dur / 60):.2f} ripples/min")
print(f"POST Non-REM duration {post_nr_dur:.0f} s -> {len(rip_post) / (post_nr_dur / 60):.2f} ripples/min")

durs = ripples.end - ripples.start
print(f"Ripple durations: median {np.median(durs) * 1e3:.0f} ms, "
      f"IQR {np.percentile(durs, 25) * 1e3:.0f}-{np.percentile(durs, 75) * 1e3:.0f} ms")

np.savez("scripts/ripples.npz",
         start=ripples.start, end=ripples.end,
         pre_start=rip_pre.start, pre_end=rip_pre.end,
         post_start=rip_post.start, post_end=rip_post.end,
         channel=ch, env_mean=mu, env_sd=sd)

# save the smoothed envelope + filtered trace for plotting (downsampled env at 1250 Hz is too big; keep full, it's 350 MB float64... instead keep every 4th sample)
np.savez("scripts/lfp_ripple_channel.npz",
         t=t_lfp[::4], env=env_s[::4], filt=filt[::4], raw=lfp[::4])
print("saved scripts/ripples.npz and scripts/lfp_ripple_channel.npz")
