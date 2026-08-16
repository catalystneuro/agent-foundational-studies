"""Single-session figures: behavior overview, example place cells, population maps."""
import pickle
import warnings

warnings.simplefilter("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import pf_lib as pf

SESSION = "Achilles_10252013"
COL = {"right": "#1f77b4", "left": "#d62728"}
DIRS = ("right", "left")

urls = pf.session_urls()
nwb, nwbfile, io = pf.load_session(urls[SESSION])
maze = pf.maze_epoch(nwbfile)
pos, L = pf.get_position(nwb, nwbfile)
pos2d = pf.get_position_2d(nwb, nwbfile)
vel, eps = pf.running_epochs(pos)
units = nwb["units"]
pyr = pf.select_pyramidal(units, maze)
print(f"{SESSION}: {len(units)} sorted units -> {len(pyr)} putative CA1 pyramidal cells")
print(f"traversals: {len(eps['right'])} rightward, {len(eps['left'])} leftward, "
      f"{eps['run'].tot_length():.0f} s of running")

rng = np.random.default_rng(0)
res = {d: pf.direction_metrics(pyr, pos, eps[d], L, rng=rng) for d in DIRS}
tcs = {d: res[d]["tc_smooth"] for d in DIRS}
occs = {d: res[d]["occ"] for d in DIRS}
mets = {d: res[d]["metrics"] for d in DIRS}
bins = res["right"]["bins"]
centers = 0.5 * (bins[:-1] + bins[1:])
uidx = np.asarray(pyr.index)
for d in DIRS:
    m = mets[d]
    print(f"  {d}: {int(m.is_place_cell.sum())} place cells; "
          f"median SI {np.median(m.si[m.is_place_cell]):.2f} bits/spike, "
          f"median width {np.median(m.width[m.is_place_cell]) * 100:.0f} cm")

pc_either = mets["right"].is_place_cell.values | mets["left"].is_place_cell.values
print(f"  place cell in at least one direction: {pc_either.sum()}/{len(pyr)} "
      f"({100 * pc_either.mean():.0f}%)")

with open("single_session.pkl", "wb") as fh:
    pickle.dump(
        {
            "session": SESSION,
            "centers": centers,
            "bins": bins,
            "tc": {d: tcs[d] for d in DIRS},
            "occ": occs,
            "metrics": mets,
            "unit_index": uidx,
            "n_traversals": {d: len(eps[d]) for d in DIRS},
        },
        fh,
    )


def spike_positions(ts, ep):
    """Interpolated track position of each spike, plus its traversal number."""
    xs, trav = [], []
    for k in range(len(ep)):
        s = np.asarray(ts.restrict(ep[k : k + 1]).t)
        if len(s) == 0:
            continue
        xs.append(np.interp(s, pos.t, pos.values))
        trav.append(np.full(len(s), k))
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(trav)


# ----------------------------------------------------------------- figure 1
order = np.argsort(mets["right"].peak_pos.values)
sorted_units = uidx[order]

t0 = float(eps["right"].start[6])
win = nap.IntervalSet(start=t0 - 10, end=t0 + 290)

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(
    3, 3, height_ratios=[1, 1, 2.6], width_ratios=[1.15, 1.15, 1.0], hspace=0.4, wspace=0.55
)

ax = fig.add_subplot(gs[0, 0:2])
p = pos.restrict(win)
ax.plot(p.t, p.values, "k.", ms=2)
for d in DIRS:
    ei = eps[d].intersect(win)
    for k in range(len(ei)):
        ax.axvspan(float(ei.start[k]), float(ei.end[k]), color=COL[d], alpha=0.25, lw=0)
ax.set_ylabel("track position (m)")
ax.set_title(f"{SESSION}: shuttling on the {L:.1f} m linear track "
             "(blue = rightward, red = leftward)", fontsize=11)
ax.set_xlim(float(win.start[0]), float(win.end[0]))
ax.set_xticklabels([])

ax = fig.add_subplot(gs[1, 0:2])
v = vel.restrict(win)
ax.plot(v.t, v.values, "k-", lw=0.7)
ax.axhline(0, color="0.6", lw=0.5)
for s in (pf.SPEED_THRESH, -pf.SPEED_THRESH):
    ax.axhline(s, color="g", ls="--", lw=0.7)
ax.set_ylabel("velocity (m/s)")
ax.set_xlabel("time (s)")
ax.set_xlim(float(win.start[0]), float(win.end[0]))

for col, d in enumerate(DIRS):
    e = eps[d][6:7]
    pad = 0.5
    zw = nap.IntervalSet(start=float(e.start[0]) - pad, end=float(e.end[0]) + pad)
    ax = fig.add_subplot(gs[2, col])
    for row, uid in enumerate(sorted_units):
        s = np.asarray(pyr[uid].restrict(zw).t)
        ax.plot(s - float(e.start[0]), np.full(len(s), row), "|", color="k", ms=4, mew=0.9)
    ax.axvspan(0, float(e.end[0]) - float(e.start[0]), color=COL[d], alpha=0.15, lw=0)
    ax2 = ax.twinx()
    pz = pos.restrict(zw)
    ax2.plot(np.asarray(pz.t) - float(e.start[0]), pz.values, color=COL[d], lw=2)
    ax2.set_ylim(0, L)
    ax2.set_ylabel("position (m)", color=COL[d])
    ax2.tick_params(axis="y", colors=COL[d])
    ax.set_xlim(-pad, float(e.end[0]) - float(e.start[0]) + pad)
    ax.set_ylim(-1, len(sorted_units))
    ax.set_xlabel("time from run onset (s)")
    if col == 0:
        ax.set_ylabel("CA1 pyramidal cell\n(sorted by rightward field position)")
    ax.set_title(f"single {d}ward traversal", fontsize=11)

ax = fig.add_subplot(gs[0:2, 2])
pp = pos2d.restrict(maze)
lin_at = np.interp(np.asarray(pp.t), pos.t, pos.values, left=np.nan, right=np.nan)
sc = ax.scatter(pp.values[:, 0], pp.values[:, 1], c=lin_at, s=1, cmap="viridis")
plt.colorbar(sc, ax=ax, label="linearized (m)", fraction=0.08)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("2D tracking, whole maze epoch", fontsize=11)

ax = fig.add_subplot(gs[2, 2])
for d in DIRS:
    ax.plot(centers, occs[d], color=COL[d], label=f"{d} ({len(eps[d])} runs)")
ax.set_xlabel("track position (m)")
ax.set_ylabel("occupancy (s / 2 cm bin)")
ax.set_title("Occupancy", fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(0, None)
fig.savefig("fig01_behavior_and_raster.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig01 done")

# ----------------------------------------------------------------- figure 2
# Examples: place cells in both directions, reliable, spread across the track.
score = np.where(
    mets["right"].is_place_cell.values & mets["left"].is_place_cell.values,
    mets["right"].reliability.values + mets["left"].reliability.values,
    -1.0,
)
cand = np.argsort(-score)
chosen, taken = [], []
for i in cand:
    if score[i] < 0:
        break
    pk = mets["right"].peak_pos.values[i]
    if any(abs(pk - q) < 0.15 for q in taken):
        continue
    chosen.append(i)
    taken.append(pk)
    if len(chosen) == 6:
        break
order_by_pos = np.argsort([mets["right"].peak_pos.values[i] for i in chosen])
chosen = [chosen[k] for k in order_by_pos]

fig, axs = plt.subplots(
    3, 6, figsize=(17, 8.5),
    gridspec_kw={"height_ratios": [2, 2, 1.5], "hspace": 0.32, "wspace": 0.34},
)
for j, i in enumerate(chosen):
    uid = uidx[i]
    for r, d in enumerate(DIRS):
        ax = axs[r, j]
        x, tr = spike_positions(pyr[uid], eps[d])
        ax.plot(x, tr, ".", color=COL[d], ms=3)
        ax.set_xlim(0, L)
        ax.set_ylim(-1, len(eps[d]))
        if j == 0:
            ax.set_ylabel(f"{d}ward\ntraversal #")
        if r == 0:
            ax.set_title(
                f"unit {uid} ({pyr.get_info('location')[uid]})\n"
                f"SI {mets['right'].si[uid]:.1f} / {mets['left'].si[uid]:.1f} bits per spike",
                fontsize=10,
            )
        ax.set_xticklabels([])
    ax = axs[2, j]
    for d in DIRS:
        ax.plot(centers, tcs[d][uid].values, color=COL[d], lw=1.8, label=d)
    ax.set_xlim(0, L)
    ax.set_ylim(0, None)
    ax.set_xlabel("position (m)")
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
    if j == 5:
        ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle(
    "Example CA1 place cells: spikes recur at the same track location on traversal after traversal",
    fontsize=13, y=0.97,
)
fig.savefig("fig02_example_place_cells.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig02 done")

# ----------------------------------------------------------------- figure 3
fig, axs = plt.subplots(1, 3, figsize=(16, 6))


def norm_maps(d, sel):
    m = tcs[d].values[:, sel].T
    mx = m.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return m / mx


sel = {d: np.where(mets[d].is_place_cell.values)[0] for d in DIRS}
ordr = {d: sel[d][np.argsort(mets[d].peak_pos.values[sel[d]])] for d in DIRS}

for ax, d in zip(axs[:2], DIRS):
    o = ordr[d]
    im = ax.imshow(norm_maps(d, o), aspect="auto", origin="lower",
                   extent=[0, L, 0, len(o)], cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("track position (m)")
    ax.set_ylabel("place cell (sorted by field peak)")
    ax.set_title(f"{d}ward runs: {len(o)} place cells", fontsize=11)
    plt.colorbar(im, ax=ax, label="rate / peak rate", fraction=0.046)

ax = axs[2]
o = ordr["right"]
im = ax.imshow(norm_maps("left", o), aspect="auto", origin="lower",
               extent=[0, L, 0, len(o)], cmap="magma", vmin=0, vmax=1)
ax.set_xlabel("track position (m)")
ax.set_ylabel("same cells, same order as left panel")
ax.set_title("Rightward-sorted cells, leftward maps:\nthe diagonal breaks up (directionality)",
             fontsize=11)
plt.colorbar(im, ax=ax, label="rate / peak rate", fraction=0.046)
fig.suptitle(f"{SESSION}: the CA1 place-cell ensemble tiles the whole track", fontsize=13)
fig.tight_layout()
fig.savefig("fig03_population_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig03 done")
