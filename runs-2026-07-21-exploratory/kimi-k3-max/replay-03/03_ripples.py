"""Step 3: ripple-channel selection and SWR detection on the full session.

Recipe (validated on this session in prior runs):
- Channel pick: on a 200 s POST Non-REM block, per channel compute the ripple-band
  (100-250 Hz) envelope; score = (p99/median peakiness) x (ripple/delta PSD ratio).
  Expect ch 117 (CA1 pyramidal layer).
- Detection: SOS butter4 100-250 Hz -> Hilbert envelope -> gaussian smooth sigma 4 ms.
  Thresholds from Non-REM envelope samples: peak > mean+4SD, edges > mean+1SD.
  Merge events <30 ms apart; keep 30-500 ms events fully inside Non-REM.
- Expect ~5400 PRE + ~2900 POST SWRs (~31-33/min of Non-REM, median ~60 ms).
"""
import pickle
import numpy as np
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from tqdm import tqdm

# ---------------- load ----------------
dl = "https://api.dandiarchive.org/api/assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/"
s3_url = requests.get(dl, allow_redirects=False).headers["Location"]
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache")), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())

FS = 1250.0
lfp_ds = h5py_file["processing/ecephys/LFP/LFP/data"]
CONV = 3.815e-7  # V per count
N_SAMP = lfp_ds.shape[0]
print(f"LFP: {N_SAMP} samples x {lfp_ds.shape[1]} ch @ {FS} Hz, {N_SAMP/FS/3600:.2f} h")

states = nwb["states"]
nonrem = states[states.label == "Non-REM"]
nr_start = np.asarray(nonrem.start)
nr_end = np.asarray(nonrem.end)
maze_end = 20147.0
post_mask = nr_start > maze_end
print(f"Non-REM: {len(nr_start)} blocks ({np.sum(post_mask)} POST), "
      f"PRE {np.sum(nr_end[~post_mask]-nr_start[~post_mask]):.0f} s, "
      f"POST {np.sum(nr_end[post_mask]-nr_start[post_mask]):.0f} s")

sos = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")

def ripple_env(trace):
    filt = signal.sosfiltfilt(sos, trace)
    env = np.abs(signal.hilbert(filt))
    return env

# ---------------- channel selection on a POST Non-REM block ----------------
blk_s = nr_start[post_mask][0]
blk = (int(blk_s * FS), int((blk_s + 200) * FS))
print(f"channel scan on POST Non-REM block {blk_s:.0f}-{blk_s+200:.0f} s")
scores = np.zeros(128)
for ch in tqdm(range(128), desc="channel scan"):
    tr = lfp_ds[blk[0]:blk[1], ch].astype(float) * CONV
    env = ripple_env(tr)
    win = signal.windows.gaussian(int(FS * 0.008), int(FS * 0.004))
    env_s = np.convolve(env, win / win.sum(), mode="same")
    peakiness = np.percentile(env_s, 99) / np.median(env_s)
    f, psd = signal.welch(tr, FS, nperseg=int(FS * 4))
    rd = psd[(f >= 100) & (f <= 250)].mean() / psd[(f >= 1) & (f <= 4)].mean()
    scores[ch] = peakiness * rd
best_ch = int(np.argmax(scores))
print(f"best ripple channel: {best_ch} (score {scores[best_ch]:.3g}); "
      f"ch117 rank {int(np.sum(scores > scores[117])) + 1}")

# ---------------- full-session envelope on the best channel ----------------
print("streaming best channel for full session...")
t0 = 0
full = np.empty(N_SAMP, dtype=np.float32)
CHUNK = int(FS * 600)  # 10 min chunks
for i0 in tqdm(range(0, N_SAMP, CHUNK), desc="LFP stream"):
    i1 = min(i0 + CHUNK, N_SAMP)
    full[i0:i1] = lfp_ds[i0:i1, best_ch].astype(np.float32) * CONV

print("filtering + envelope...")
env = ripple_env(full.astype(float))
del full
win = signal.windows.gaussian(int(FS * 0.008), int(FS * 0.004))
env_s = np.convolve(env, win / win.sum(), mode="same").astype(np.float32)
del env

# ---------------- thresholds from Non-REM samples ----------------
nr_mask = np.zeros(N_SAMP, dtype=bool)
for s, e in zip(nr_start, nr_end):
    nr_mask[int(s * FS):int(e * FS)] = True
mu, sd = env_s[nr_mask].mean(), env_s[nr_mask].std()
thr_peak, thr_edge = mu + 4 * sd, mu + 1 * sd
print(f"envelope mean {mu*1e6:.1f} uV, sd {sd*1e6:.1f} uV; "
      f"peak thr {thr_peak*1e6:.1f} uV, edge thr {thr_edge*1e6:.1f} uV")

# ---------------- event detection ----------------
above = env_s > thr_edge
d = np.diff(above.astype(int))
ev_s = list(np.where(d == 1)[0] + 1)
ev_e = list(np.where(d == -1)[0] + 1)
if above[0]:
    ev_s = [0] + ev_s
if above[-1]:
    ev_e = ev_e + [N_SAMP]
events = []
for s, e in zip(ev_s, ev_e):
    if env_s[s:e].max() > thr_peak:
        events.append([s, e])
# merge < 30 ms
merged = [events[0]]
for ev in events[1:]:
    if ev[0] - merged[-1][1] < int(0.030 * FS):
        merged[-1][1] = ev[1]
    else:
        merged.append(list(ev))
# duration 30-500 ms, fully inside Non-REM
kept = []
for s, e in merged:
    dur = (e - s) / FS
    if 0.030 <= dur <= 0.500 and nr_mask[s] and nr_mask[e - 1]:
        # fully inside one Non-REM block
        bi = np.searchsorted(nr_end, s / FS)
        if bi < len(nr_end) and e / FS <= nr_end[bi] and s / FS >= nr_start[bi]:
            kept.append((s / FS, e / FS))
kept = np.array(kept)
pre = kept[kept[:, 0] < maze_end]
post = kept[kept[:, 0] > maze_end]
pre_dur = np.sum(nr_end[~post_mask] - nr_start[~post_mask]) / 60
post_dur = np.sum(nr_end[post_mask] - nr_start[post_mask]) / 60
print(f"\nSWRs: {len(pre)} PRE ({len(pre)/pre_dur:.1f}/min), "
      f"{len(post)} POST ({len(post)/post_dur:.1f}/min), "
      f"median duration {np.median((kept[:,1]-kept[:,0]))*1000:.0f} ms")

with open("ripples.pkl", "wb") as f:
    pickle.dump(dict(events=kept, best_ch=best_ch, scores=scores,
                     thr_peak=thr_peak, thr_edge=thr_edge, mu=mu, sd=sd,
                     nr_start=nr_start, nr_end=nr_end), f)
# save a few example envelope snippets for figures
snips = {}
for i, ev in enumerate(post[:200]):
    c = int(((ev[0] + ev[1]) / 2) * FS)
    w = int(0.4 * FS)
    snips[i] = env_s[c - w:c + w]
with open("ripple_snips.pkl", "wb") as f:
    pickle.dump(snips, f)
print("saved ripples.pkl")
