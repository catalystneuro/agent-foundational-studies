import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from load_data import load_cache
from replay import compute_speed, place_fields

d = load_cache()
maze = nap.IntervalSet(*d["epochs"]["MazeEpoch"])

# Position Tsd (linearized), restricted to Maze
pos = nap.Tsd(t=d["pos_t"], d=d["pos_data"])
pos = pos.restrict(maze)
# drop NaNs
good = ~np.isnan(pos.values)
pos = nap.Tsd(t=pos.index.values[good], d=pos.values[good])
print("pos range:", np.nanmin(pos.values), np.nanmax(pos.values), "n", len(pos))

speed = compute_speed(pos)
run_ep = speed.threshold(0.04).time_support  # 4 cm/s on ~1.6m maze (units ~ m)
run_ep = run_ep.drop_short_intervals(0.5)
print("run time (s):", float(run_ep.tot_length()), "of maze", float(maze.tot_length()))

# pyramidal units only
ct = d["cell_type"]
pyr_idx = np.flatnonzero(ct == "excitatory")
spikes = {int(i): nap.Ts(d["spikes"][i]) for i in pyr_idx}
grp = nap.TsGroup(spikes)

tc, centers = place_fields(grp, pos, run_ep, nbins=50, sigma=1.5)
peak_rate = tc.max(axis=0)
peak_loc = centers[np.argmax(tc.values, axis=0)]

# place cells: peak rate > 1 Hz and decent spatial info
place_mask = peak_rate.values > 1.0
place_units = np.array(list(tc.columns))[place_mask]
print("pyramidal:", len(pyr_idx), " place cells (>1Hz peak):", place_mask.sum())

# order by peak location
order = np.argsort(peak_loc[place_mask])
ordered_units = place_units[order]

# normalized place-field matrix, sorted
M = tc[place_units].values.T[order]  # (cells, bins)
Mn = M / M.max(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im = ax[0].imshow(Mn, aspect="auto", origin="lower",
                  extent=[centers[0], centers[-1], 0, Mn.shape[0]], cmap="viridis")
ax[0].set_xlabel("linearized position (m)"); ax[0].set_ylabel("place cell (sorted by peak)")
ax[0].set_title("Place fields on the linear maze (%d cells)" % Mn.shape[0])
plt.colorbar(im, ax=ax[0], label="norm. rate")

# example place fields
for u in ordered_units[::max(1, len(ordered_units)//6)][:6]:
    ax[1].plot(centers, tc[u].values, label="unit %d" % u)
ax[1].set_xlabel("position (m)"); ax[1].set_ylabel("firing rate (Hz)")
ax[1].set_title("Example place fields"); ax[1].legend(fontsize=7)

# position trace + speed
ax[2].plot(pos.index.values - maze.start[0], pos.values, "k", lw=0.5)
ax[2].set_xlabel("time in maze (s)"); ax[2].set_ylabel("position (m)")
ax[2].set_title("Linearized trajectory")
plt.tight_layout()
plt.savefig("fig_place_fields.png", dpi=130)
print("saved fig_place_fields.png")

np.savez("placefields.npz", tc=tc.values, columns=np.array(tc.columns),
         centers=centers, place_units=place_units, peak_loc=peak_loc,
         peak_rate=peak_rate.values, ordered_units=ordered_units)
