"""Stage 3: place fields of CA1 pyramidal cells during running on the track."""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import swr_common as C

nwb = C.open_nwb()
epochs = C.get_epochs(nwb)
units = C.get_units(nwb)
maze = nap.IntervalSet(start=epochs["MAZE"][0], end=epochs["MAZE"][1])

# --- Position (metres) restricted to the MAZE epoch ---
tpos, xpos = C.get_position(nwb)
good = np.isfinite(xpos)
pos = nap.Tsd(t=tpos[good], d=xpos[good]).restrict(maze)

# --- Speed and running epochs (>5 cm/s) ---
dt = np.median(np.diff(pos.t))
speed_v = np.abs(np.gradient(pos.d, pos.t))
speed = nap.Tsd(t=pos.t, d=speed_v)
# smooth speed a bit
from scipy.ndimage import uniform_filter1d
speed = nap.Tsd(t=pos.t, d=uniform_filter1d(speed_v, size=int(0.25 / dt)))
run_ep = speed.threshold(0.08).time_support   # sustained locomotion (>8 cm/s)
run_ep = run_ep.drop_short_intervals(0.5)
print(f"Running: {run_ep.tot_length():.0f}s of {maze.tot_length():.0f}s maze")

# --- Excitatory (pyramidal) units as a TsGroup ---
exc = np.where(units["cell_type"] == "excitatory")[0]
spk = {int(i): nap.Ts(t=units["spike_times"][i]) for i in exc}
tsg = nap.TsGroup(spk).restrict(maze)

# --- Directional split for cleaner fields ---
xd = pos.d
vsign = np.gradient(xd)
dir_ts = nap.Tsd(t=pos.t, d=vsign)
rightward = dir_ts.threshold(0).time_support.intersect(run_ep)
leftward = dir_ts.threshold(0, "below").time_support.intersect(run_ep)

NB = 50
minmax = (0.0, 1.6)
from scipy.ndimage import gaussian_filter1d
def smooth_tc(tc):
    v = gaussian_filter1d(tc.values, sigma=1.2, axis=0, mode="nearest")
    return type(tc)(v, index=tc.index, columns=tc.columns)
tc_run = smooth_tc(nap.compute_1d_tuning_curves(tsg, pos, NB, ep=run_ep, minmax=minmax))
tc_R = smooth_tc(nap.compute_1d_tuning_curves(tsg, pos, NB, ep=rightward, minmax=minmax))
tc_L = smooth_tc(nap.compute_1d_tuning_curves(tsg, pos, NB, ep=leftward, minmax=minmax))
bins = tc_run.index.values * 100  # cm

# --- Spatial information (bits/spike) to define place cells ---
occ, _ = np.histogram(pos.restrict(run_ep).d, bins=np.linspace(0, 1.6, NB + 1))
p_occ = occ / occ.sum()
def spatial_info(tc):
    r = tc.values
    rbar = np.sum(p_occ[:, None] * r, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.nansum(p_occ[:, None] * (r / rbar) * np.log2(r / rbar), axis=0)
    return si, rbar
si, meanrate = spatial_info(tc_run)
peakrate = tc_run.values.max(0)
is_place = (peakrate > 1.0) & (si > 0.3)
place_ids = tc_run.columns[is_place]
print(f"Pyramidal cells: {len(exc)} | place cells (peak>1Hz & SI>0.3): {is_place.sum()}")

# --- Figure ---
fig = plt.figure(figsize=(14, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

# (a) ordered place-field heatmap (running, both directions)
ax = fig.add_subplot(gs[:, 0])
pf = tc_run[place_ids].values.T
pf_norm = pf / pf.max(1, keepdims=True)
order = np.argsort(np.argmax(pf, axis=1))
im = ax.imshow(pf_norm[order], aspect="auto", cmap="viridis",
               extent=[0, 160, len(order), 0], interpolation="nearest")
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Place cell (sorted by peak)")
ax.set_title(f"(a) CA1 place fields (n={len(order)})")
fig.colorbar(im, ax=ax, label="norm. rate", shrink=0.6)

# (b) example place cells
ax = fig.add_subplot(gs[0, 1])
ex_order = place_ids[order][np.linspace(0, len(order) - 1, 6).astype(int)]
for uid in ex_order:
    ax.plot(bins, tc_run[uid].values, lw=1.5, label=f"u{uid}")
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Firing rate (Hz)")
ax.set_title("(b) Example place fields"); ax.legend(fontsize=7, ncol=2)

# (c) directional fields for one example cell
ax = fig.add_subplot(gs[0, 2])
uid = ex_order[len(ex_order) // 2]
ax.plot(bins, tc_R[uid].values, color="#3182bd", lw=2, label="rightward")
ax.plot(bins, tc_L[uid].values, color="#e6550d", lw=2, label="leftward")
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Firing rate (Hz)")
ax.set_title(f"(c) Directional field (unit {uid})"); ax.legend(fontsize=8)

# (d) spatial information distribution
ax = fig.add_subplot(gs[1, 1])
ax.hist(si, bins=20, color="#756bb1", edgecolor="w")
ax.axvline(0.3, color="k", ls="--", label="place-cell cut")
ax.set_xlabel("Spatial information (bits/spike)"); ax.set_ylabel("Cell count")
ax.legend(fontsize=8); ax.set_title("(d) Spatial information")

# (e) occupancy
ax = fig.add_subplot(gs[1, 2])
ax.bar(bins, occ * dt, width=160 / NB, color="#31a354", edgecolor="w")
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Occupancy (s)")
ax.set_title("(e) Track occupancy (running)")

fig.savefig("fig3_placefields.png", dpi=130)
print("saved fig3_placefields.png")

# Save decoding template (place cells, running fields) for stage 4
np.savez("cache_placefields.npz",
         tc_values=tc_run[place_ids].values, tc_bins=tc_run.index.values,
         place_ids=np.array(place_ids), order=order,
         run_start=run_ep.start, run_end=run_ep.end)
print("saved cache_placefields.npz")
