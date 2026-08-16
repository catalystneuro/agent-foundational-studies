"""Replication of the velocity-tuning result on MC_RTT (DANDI:000129, monkey Indy).

Self-paced random-target reaching. Unlike the center-out maze task, successive targets appear at
random locations, so hand position and hand velocity are far less correlated. That makes this the
right dataset for the position-vs-velocity model comparison that MC_Maze cannot settle.
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from tqdm.auto import tqdm

import lindi
import nemos as nmo
import pynapple as nap
from pynwb import NWBHDF5IO
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CACHE = "./cache"
BIN = 0.02

RTT_LINDI = ("https://lindi.neurosift.org/dandi/dandisets/000129/assets/"
             "2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/nwb.lindi.json")


def load_rtt():
    f = lindi.LindiH5pyFile.from_lindi_file(
        RTT_LINDI, local_cache=lindi.LocalCache(cache_dir=f"{CACHE}/lindi"))
    nwbfile = NWBHDF5IO(file=f, mode="r").read()
    beh = nwbfile.processing["behavior"].data_interfaces
    cp = beh["cursor_pos"]
    t = cp.starting_time + np.arange(cp.data.shape[0]) / cp.rate

    npz = f"{CACHE}/rtt_behavior.npz"
    if os.path.exists(npz):
        z = np.load(npz)
        pos, vel, tgt = z["pos"], z["vel"], z["tgt"]
    else:
        pos = np.asarray(cp.data[:], dtype=np.float32)
        vel = np.asarray(beh["finger_vel"].data[:], dtype=np.float32)
        tgt = np.asarray(beh["target_pos"].data[:], dtype=np.float32)
        np.savez_compressed(npz, t=t, pos=pos, vel=vel, tgt=tgt)

    spz = f"{CACHE}/rtt_spikes.npz"
    if os.path.exists(spz):
        z = np.load(spz)
        st = [z[str(i)] for i in range(len(z.files))]
    else:
        st = [np.asarray(s) for s in nwbfile.units["spike_times"][:]]
        np.savez_compressed(spz, **{str(i): s for i, s in enumerate(st)})
    obs = np.asarray(nwbfile.units["obs_intervals"][0], dtype=float)
    return nwbfile, t, pos, vel, tgt, st, obs


nwbfile, t_beh, pos_raw, vel_raw, tgt_raw, spike_times, obs = load_rtt()
print(nwbfile.session_description[:110])
print("subject:", nwbfile.subject.subject_id, "| duration %.0f s" % (t_beh[-1] - t_beh[0]))
print("observed intervals:", obs)

# behaviour has 600 NaN samples; drop them before building pynapple objects
finite = np.isfinite(pos_raw).all(1) & np.isfinite(vel_raw).all(1) & np.isfinite(tgt_raw).all(1)
print("dropping %d non-finite behaviour samples of %d" % ((~finite).sum(), len(finite)))
t_beh, pos_raw, vel_raw, tgt_raw = t_beh[finite], pos_raw[finite], vel_raw[finite], tgt_raw[finite]

vel = nap.TsdFrame(t=t_beh, d=vel_raw.astype(float), columns=["vx", "vy"])
pos = nap.TsdFrame(t=t_beh, d=pos_raw.astype(float), columns=["x", "y"])
spikes_all = nap.TsGroup({i: s for i, s in enumerate(spike_times)})

obs_ep = nap.IntervalSet(start=obs[:, 0], end=obs[:, 1])
keep = np.array([u for u in spikes_all.index if spikes_all[u].restrict(obs_ep).rate > 1.0])
spikes = spikes_all[list(keep)]
print("kept %d of %d units above 1 Hz" % (len(keep), len(spikes_all)))

# reach onsets: the target jumps to a new random location
jump = np.flatnonzero(np.abs(np.diff(tgt_raw, axis=0)).sum(1) > 1e-6)
reach_onset = t_beh[jump + 1]
print("%d target jumps, median inter-target interval %.2f s"
      % (len(reach_onset), np.median(np.diff(reach_onset))))

# ---------------------------------------------------------------- binned matrix
counts_tsd = spikes.count(BIN, ep=obs_ep)
vel_b = vel.bin_average(BIN, ep=obs_ep)
pos_b = pos.bin_average(BIN, ep=obs_ep)
seg = np.asarray(obs_ep.in_interval(nap.Ts(counts_tsd.index.values))).astype(float)

ok = np.isfinite(vel_b.values).all(1) & np.isfinite(pos_b.values).all(1) & np.isfinite(seg)
M = dict(t=counts_tsd.index.values[ok],
         counts=np.asarray(counts_tsd.values)[ok],
         vel=np.asarray(vel_b.values)[ok],
         pos=np.asarray(pos_b.values)[ok],
         seg=seg[ok].astype(int))
M["speed"] = np.hypot(M["vel"][:, 0], M["vel"][:, 1])
M["vel_angle"] = np.arctan2(M["vel"][:, 1], M["vel"][:, 0])
# blocks of 5 s used as CV groups (the session is continuous, so trials are not available)
M["block"] = (M["t"] // 5.0).astype(int)
print("binned matrix:", M["counts"].shape, "| moving bins (>100 mm/s):", (M["speed"] > 100).sum())

with open(f"{CACHE}/rtt_binned.pkl", "wb") as fh:
    pickle.dump(dict(M=M, keep=keep, reach_onset=reach_onset), fh)
print("wrote rtt_binned.pkl")
