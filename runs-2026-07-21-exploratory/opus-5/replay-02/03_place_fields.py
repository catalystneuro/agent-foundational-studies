"""Stage 3: build direction-specific place fields on the linear track and validate the decoder."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pynapple as nap
import common

SESSION = "Achilles_10252013"
NBINS = 50
nwbfile, h = common.open_session(SESSION)
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
udf = nwbfile.units.to_dataframe()
ep_df = nwbfile.epochs.to_dataframe()
epochs = {r.label: nap.IntervalSet(start=r.start_time, end=r.stop_time) for r in ep_df.itertuples()}

# ---------------------------------------------------------------- position -> traversals
ls = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t_all, d_all = ls.t, np.asarray(ls.d).squeeze()
good = ~np.isnan(d_all)
tg, dg = t_all[good], d_all[good]
dt = np.median(np.diff(t_all))
brk = np.where(np.diff(tg) > 5 * dt)[0]
starts = np.r_[0, brk + 1]
ends = np.r_[brk, len(tg) - 1]
nsamp = ends - starts + 1
starts, ends = starts[nsamp >= 5], ends[nsamp >= 5]
lap_ep = nap.IntervalSet(start=tg[starts] - dt / 2, end=tg[ends] + dt / 2)
pos = nap.Tsd(t=tg, d=dg, time_support=lap_ep)
print(f"{len(lap_ep)} track traversals, {lap_ep.tot_length():.0f} s of tracked running")

# direction and mean speed per traversal
lap_dir, lap_speed = [], []
for s, e in zip(lap_ep.start, lap_ep.end):
    p = pos.get(s, e)
    lap_dir.append(np.sign(p.d[-1] - p.d[0]))
    lap_speed.append(abs(p.d[-1] - p.d[0]) / (p.t[-1] - p.t[0]))
lap_dir, lap_speed = np.array(lap_dir), np.array(lap_speed)
# keep well-formed traversals: covers >60% of the track at >0.1 m/s
span = np.array([np.ptp(pos.get(s, e).d) for s, e in zip(lap_ep.start, lap_ep.end)])
ok = (span > 0.6 * 1.6) & (lap_speed > 0.1)
print(f"{ok.sum()}/{len(ok)} traversals pass the span/speed criterion; "
      f"{(lap_dir[ok]>0).sum()} rightward, {(lap_dir[ok]<0).sum()} leftward, "
      f"mean speed {lap_speed[ok].mean():.2f} m/s")

ep_dir = {}
for name, sgn in [("rightward", 1), ("leftward", -1)]:
    m = ok & (lap_dir == sgn)
    ep_dir[name] = nap.IntervalSet(start=lap_ep.start[m], end=lap_ep.end[m])
lap_id = {name: np.where(ok & (lap_dir == sgn))[0] for name, sgn in [("rightward", 1), ("leftward", -1)]}

# ---------------------------------------------------------------- tuning curves
exc = udf.index[udf.cell_type == "excitatory"].values
pyr = units[list(exc)]
bins = np.linspace(0, 1.6, NBINS + 1)
centers = (bins[:-1] + bins[1:]) / 2


def tuning(group, ep, sigma=1.0):
    tc = nap.compute_1d_tuning_curves(group=group, feature=pos.restrict(ep), nb_bins=NBINS,
                                      minmax=(0, 1.6), ep=ep)
    arr = tc.values.copy()
    arr = np.apply_along_axis(lambda v: gaussian_filter1d(v, sigma, mode="nearest"), 0, arr)
    return arr                                    # (NBINS, n_units)


tc_all = {k: tuning(pyr, ep_dir[k]) for k in ep_dir}


def spatial_info(rate, occ):
    p = occ / occ.sum()
    mr = (p * rate).sum()
    if mr <= 0:
        return 0.0
    r = np.where(rate > 0, rate, np.nan)
    return np.nansum(p * r / mr * np.log2(r / mr))


occ = {k: np.histogram(pos.restrict(ep_dir[k]).d, bins)[0] * dt for k in ep_dir}
si = {k: np.array([spatial_info(tc_all[k][:, i], occ[k]) for i in range(len(exc))]) for k in ep_dir}
peak = {k: tc_all[k].max(0) for k in ep_dir}
mean_rate = np.array([len(pyr[u].restrict(lap_ep)) / lap_ep.tot_length() for u in pyr.index])

# place cells: appreciable peak rate and spatial information in at least one direction
is_pc = ((peak["rightward"] > 1.0) & (si["rightward"] > 0.5)) | \
        ((peak["leftward"] > 1.0) & (si["leftward"] > 0.5))
pc_idx = np.where(is_pc)[0]
print(f"{len(pc_idx)}/{len(exc)} pyramidal cells classified as place cells "
      f"(peak > 1 Hz and SI > 0.5 bits/spike in >=1 direction)")

# ---------------------------------------------------------------- decoder validation on RUN
# cross-validated: fields from even-numbered traversals, decode odd-numbered ones
pc_units = pyr[list(exc[pc_idx])]
tc_train, ep_test = {}, {}
for k in ep_dir:
    ids = lap_id[k]
    tr = nap.IntervalSet(start=lap_ep.start[ids[0::2]], end=lap_ep.end[ids[0::2]])
    te = nap.IntervalSet(start=lap_ep.start[ids[1::2]], end=lap_ep.end[ids[1::2]])
    tc_train[k] = tuning(pc_units, tr)
    ep_test[k] = te

import pandas as pd
errs, dec_examples = [], {}
for k in ep_dir:
    tcdf = pd.DataFrame(index=centers, data=tc_train[k], columns=pc_units.index)
    dec, prob = nap.decode_1d(tuning_curves=tcdf, group=pc_units, ep=ep_test[k],
                              bin_size=0.25, feature=pos.restrict(ep_test[k]))
    true = np.interp(dec.t, pos.t, pos.d)
    e = dec.d - true
    errs.append(e)
    dec_examples[k] = (dec, true)
err = np.concatenate(errs)
median_err = np.median(np.abs(err))
print(f"cross-validated decoding error on running laps: median |error| = {median_err*100:.1f} cm "
      f"(chance ~ {np.median(np.abs(np.random.uniform(0,1.6,10000)-np.random.uniform(0,1.6,10000)))*100:.0f} cm)")

# ---------------------------------------------------------------- figure
order = np.argsort(np.argmax(tc_all["rightward"][:, pc_idx], axis=0))
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, hspace=.5, wspace=.32)

for j, k in enumerate(["rightward", "leftward"]):
    ax = fig.add_subplot(gs[0, j])
    M = tc_all[k][:, pc_idx][:, order].T
    M = M / np.maximum(M.max(1, keepdims=True), 1e-9)
    ax.imshow(M, aspect="auto", extent=[0, 1.6, len(pc_idx), 0], cmap="viridis")
    ax.set(xlabel="position (m)", ylabel="place cell (sorted)", title=f"Place fields, {k} runs")

ax = fig.add_subplot(gs[0, 2])
for i in pc_idx[order[::4]]:
    ax.plot(centers, tc_all["rightward"][:, i], lw=1)
ax.set(xlabel="position (m)", ylabel="firing rate (Hz)", title="Example place fields (rightward)")

ax = fig.add_subplot(gs[1, 0])
ax.hist(si["rightward"], bins=30, alpha=.6, label="rightward")
ax.hist(si["leftward"], bins=30, alpha=.6, label="leftward")
ax.axvline(0.5, color="k", ls="--"); ax.legend(fontsize=8)
ax.set(xlabel="spatial information (bits/spike)", ylabel="count", title="Spatial information")

ax = fig.add_subplot(gs[1, 1])
ax.scatter(np.argmax(tc_all["rightward"][:, pc_idx], 0) * 1.6 / NBINS,
           np.argmax(tc_all["leftward"][:, pc_idx], 0) * 1.6 / NBINS, s=12, color="k")
ax.plot([0, 1.6], [0, 1.6], "r--", lw=.8)
ax.set(xlabel="peak position, rightward (m)", ylabel="peak position, leftward (m)",
       title="Directionality of place fields")

ax = fig.add_subplot(gs[1, 2])
ax.hist(np.abs(err) * 100, bins=40, color="#457b9d")
ax.axvline(median_err * 100, color="r")
ax.set(xlabel="|decoding error| (cm)", ylabel="time bins",
       title=f"Cross-validated decoding error\nmedian {median_err*100:.1f} cm")

ax = fig.add_subplot(gs[2, :])
dec, true = dec_examples["rightward"]
n = min(400, len(dec))
ax.plot(dec.t[:n], true[:n], "k-", lw=1.5, label="true position")
ax.plot(dec.t[:n], dec.d[:n], ".", color="#e63946", ms=4, label="decoded")
ax.set(xlabel="time (s)", ylabel="position (m)", title="Decoded vs. true position, held-out running laps (250 ms bins)")
ax.legend(fontsize=8)

plt.savefig("figures/03_place_fields.png", dpi=140, bbox_inches="tight")
np.savez("cache/03_place_fields.npz",
         centers=centers, unit_ids=exc, pc_idx=pc_idx,
         tc_right=tc_all["rightward"], tc_left=tc_all["leftward"],
         si_right=si["rightward"], si_left=si["leftward"],
         mean_rate=mean_rate, median_decode_err=median_err,
         lap_start=lap_ep.start, lap_end=lap_ep.end, lap_dir=lap_dir, lap_ok=ok)
print("saved cache/03_place_fields.npz, figures/03_place_fields.png")
