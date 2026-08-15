"""Validate the raw data streams of one session before analysing them."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import hd_analysis as ha
import hd_io

SESSION = "A3707"

assets = hd_io.list_assets("000939")
path, aid, size = [a for a in assets if SESSION in a[0]][0]
print(path, f"{size/1e9:.1f} GB")
nwb, nwbfile, io = hd_io.open_session(aid, dandiset="000939")

eps = ha.get_epochs(nwbfile)
print({k: float(v.tot_length()) for k, v in eps.items()})
ep = eps["wake_square"]

hd = ha.clean_head_direction(nwb["head-direction"], ep)
pos = nwb["position"].restrict(ep)
units = nwb["units"]
md = units.metadata
print(f"{len(units)} units, {md.is_head_direction.sum()} flagged as HD by the authors")
print(f"HD: {len(hd)} samples, {hd.rate:.1f} Hz")

tc = ha.tuning_curves(units, hd, ep)
mvl, pref = ha._mvl_and_pref(tc)
rate = np.array([len(units[u].restrict(ep)) / ep.tot_length() for u in units.index])

# three well-tuned cells with clearly different preferred directions, plus one
# untuned cell for contrast
ok = np.where((mvl > 0.5) & (rate > 2))[0]
examples = []
for target in np.radians([30, 150, 270]):
    cand = ok[np.argsort(np.abs(ha.circ_diff(pref[ok], target)))]
    cand = [c for c in cand if c not in examples]
    examples.append(cand[0])
untuned = np.where((mvl < 0.05) & (rate > 2))[0]
examples.append(untuned[np.argmax(rate[untuned])])
ids = [units.index[i] for i in examples]

fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.45, height_ratios=[1, 1.15, 1.1])

# --- head direction and position over the 60 s window that samples the widest
#     range of directions, so all three example cells get a chance to fire
cands = np.arange(ep.start[0], ep.end[-1] - 60, 20.0)
spread = []
for c in cands:
    v = hd.restrict(nap.IntervalSet(start=c, end=c + 60)).values
    p = np.histogram(v, bins=np.linspace(0, 2 * np.pi, 25))[0].astype(float)
    p /= max(p.sum(), 1)
    spread.append(-np.nansum(np.where(p > 0, p * np.log(p), 0)))
t0 = cands[int(np.argmax(spread))]
win = nap.IntervalSet(start=t0, end=t0 + 60)
h = hd.restrict(win)

ax = fig.add_subplot(gs[0, :3])
ax.plot(h.index.values - t0, np.degrees(h.values), ".", ms=1.5, color="k")
ax.set(xlabel="time (s)", ylabel="head direction (deg)", ylim=(0, 360),
       title="Head direction during open-field foraging (60 s excerpt)")

ax = fig.add_subplot(gs[0, 3])
p = pos.restrict(win)
ax.plot(pos["x"].values, pos["y"].values, lw=0.3, color="0.8")
ax.plot(p["x"].values, p["y"].values, lw=1.2, color="crimson")
ax.set(xlabel="x (cm)", ylabel="y (cm)", title="Trajectory\n(red = same 60 s)")
ax.set_aspect("equal")

# --- spikes of the three tuned cells plotted at the animal's head direction
ax = fig.add_subplot(gs[1, :3])
trace = np.degrees(h.values).copy()
trace[np.abs(np.diff(trace, prepend=trace[0])) > 180] = np.nan  # hide 0/360 wraps
ax.plot(h.index.values - t0, trace, "-", lw=0.8, color="0.75", zorder=1)
colors = ["#1b6ca8", "#c0392b", "#2e8b57"]
unwrapped = np.unwrap(h.values)
for uid, c in zip(ids[:3], colors):
    s = units[uid].restrict(win).index.values
    sd = np.degrees(np.mod(np.interp(s, h.index.values, unwrapped), 2 * np.pi))
    ax.plot(s - t0, sd, ".", ms=4, color=c, zorder=2, label=f"unit {uid}")
ax.set(xlabel="time (s)", ylabel="head direction (deg)", ylim=(0, 360),
       title="Each tuned cell fires only when the head points in its own direction")
ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.9)

# --- occupancy
occ, edges = ha.occupancy_hist(hd)
centres = (edges[:-1] + edges[1:]) / 2
ax = fig.add_subplot(gs[1, 3], projection="polar")
ax.bar(centres, occ, width=np.diff(edges), color="0.5")
ax.set_title("Occupancy\n(s per direction)", pad=22, fontsize=10)
ax.set_yticklabels([])
ax.tick_params(labelsize=7)

# --- polar tuning curves
labels = ["tuned", "tuned", "tuned", "untuned"]
for k, (i, uid) in enumerate(zip(examples, ids)):
    ax = fig.add_subplot(gs[2, k], projection="polar")
    col = colors[k] if k < 3 else "0.4"
    th = np.append(tc.index.values, tc.index.values[0])
    r = np.append(tc[uid].values, tc[uid].values[0])
    ax.plot(th, r, color=col)
    ax.fill(th, r, alpha=0.25, color=col)
    ax.set_title(f"unit {uid} ({labels[k]})\nMVL={mvl[i]:.2f}, peak {r.max():.1f} Hz",
                 pad=22, fontsize=9)
    ax.set_rticks(np.round(np.linspace(0, r.max(), 3)[1:], 0))
    ax.tick_params(labelsize=7)

fig.suptitle(f"DANDI:000939  {path.split('/')[-1]}  postsubiculum", y=0.97)
fig.savefig("fig01_raw_data_validation.png", dpi=150, bbox_inches="tight")
print("wrote fig01_raw_data_validation.png")
