"""Inspect behavior streams, units observation intervals, and trial geometry."""
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

beh = nwbfile.processing["behavior"]
for name in ["hand_pos", "hand_vel", "cursor_pos", "eye_pos"]:
    ts = beh[name]
    print(name, type(ts).__name__, "shape", ts.data.shape, "unit", ts.unit,
          "rate", ts.rate, "starting_time", ts.starting_time,
          "timestamps", None if ts.timestamps is None else ts.timestamps.shape,
          "conversion", ts.conversion)

units = nwbfile.units
print("\nheldout:", np.unique(units["heldout"][:], return_counts=True))
oi = units["obs_intervals"][0]
print("obs_intervals[0] shape", np.asarray(oi).shape)
print(np.asarray(oi)[:5])
print("obs total coverage (unit 0):", np.sum(np.diff(np.asarray(oi), axis=1)))

st = units["spike_times"][0]
print("unit0 nspikes", len(st), "range", st[0], st[-1])

# pynapple
nwb = nap.NWBFile(nwbfile)
print("\n", nwb)

hand_vel = nwb["hand_vel"]
hand_pos = nwb["hand_pos"]
print("\nhand_vel", type(hand_vel), hand_vel.shape, hand_vel.time_support)
print("hand_pos", hand_pos.shape)
print("dt", np.median(np.diff(hand_vel.t)))
v = hand_vel.values
print("vel min/max per col", v.min(0), v.max(0))
print("nan count", np.isnan(v).sum())
speed = np.sqrt((v**2).sum(1))
print("speed percentiles", np.percentile(speed, [5, 50, 90, 99, 100]))
p = hand_pos.values
print("pos min/max", p.min(0), p.max(0))

# trials
tdf = nwbfile.trials.to_dataframe()
tgt = np.array([np.asarray(r)[a] for r, a in zip(tdf.target_pos, tdf.active_target)])
ang = np.degrees(np.arctan2(tgt[:, 1], tgt[:, 0])) % 360
print("\nactive target positions: unique count", len(np.unique(tgt, axis=0)))
print("angles unique (rounded):", np.unique(np.round(ang)))
print("n barrier==0 trials:", (tdf.num_barriers == 0).sum())
sub = tdf[tdf.num_barriers == 0]
tgt0 = np.array([np.asarray(r)[a] for r, a in zip(sub.target_pos, sub.active_target)])
ang0 = np.degrees(np.arctan2(tgt0[:, 1], tgt0[:, 0])) % 360
u, c = np.unique(np.round(ang0, 1), return_counts=True)
print("barrier-free trial angles:", list(zip(u, c)))
print("radius:", np.unique(np.round(np.hypot(tgt0[:, 0], tgt0[:, 1]))))

# Quick raw plot: hand position and velocity for a few seconds
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
t0, t1 = 100.0, 130.0
ep = nap.IntervalSet(t0, t1)
hp = hand_pos.restrict(ep)
hv = hand_vel.restrict(ep)
axes[0].plot(hp.t, hp.values[:, 0], label="x")
axes[0].plot(hp.t, hp.values[:, 1], label="y")
axes[0].set_ylabel("hand pos (mm)")
axes[0].legend()
axes[1].plot(hv.t, hv.values[:, 0], label="vx")
axes[1].plot(hv.t, hv.values[:, 1], label="vy")
axes[1].set_ylabel("hand vel (mm/s)")
axes[1].legend()
spikes = nwb["units"]
for i, u_ in enumerate(list(spikes.keys())[:40]):
    s = spikes[u_].restrict(ep)
    axes[2].plot(s.t, np.full(len(s), i), "|", color="k", ms=3)
axes[2].set_ylabel("unit #")
axes[2].set_xlabel("time (s)")
for tr in tdf.itertuples():
    if t0 < tr.move_onset_time < t1:
        for a in axes:
            a.axvline(tr.move_onset_time, color="r", lw=0.7, alpha=0.6)
fig.suptitle("MC_Maze raw data: hand kinematics + spike raster (red = movement onset)")
fig.tight_layout()
fig.savefig("fig_raw_check.png", dpi=130)
print("saved fig_raw_check.png")
