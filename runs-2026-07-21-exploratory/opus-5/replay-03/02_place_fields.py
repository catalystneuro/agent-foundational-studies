"""Stage 2: direction-specific place fields on the linear track + decoder validation."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

import replaylib as R

SESSION = os.environ.get("SESSION", "Achilles-10252013")
FIG, CACHE = "figures", "cache"
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

BIN_CM = 4.0          # spatial bin width
SMOOTH_BINS = 1.5     # Gaussian smoothing sigma, in bins
MIN_SPEED = 0.05      # m/s, running threshold
MIN_PEAK_RATE = 1.0   # Hz
MIN_RUN_SPIKES = 50   # spikes emitted while running

h5, nwbfile, nwb = R.open_session(SESSION)
epochs = {row.label: nap.IntervalSet(start=row.start_time, end=row.stop_time)
          for row in nwbfile.epochs.to_dataframe().itertuples()}
units = nwb["units"]
cell_type = np.array(nwbfile.units["cell_type"][:])

position, _ = R.load_position(h5)
speed = R.compute_speed(position)
run_dirs = R.direction_intervals(position, speed, min_speed=MIN_SPEED)
run_all = run_dirs["right"].union(run_dirs["left"])

# ---------------------------------------------------------------- spatial grid
lo, hi = np.percentile(position.restrict(run_all).d, [0.2, 99.8])
n_pos = int(np.ceil((hi - lo) / (BIN_CM / 100)))
edges = np.linspace(lo, lo + n_pos * BIN_CM / 100, n_pos + 1)
centers = (edges[:-1] + edges[1:]) / 2
print(f"spatial grid: {n_pos} bins of {BIN_CM} cm spanning {lo:.2f}-{edges[-1]:.2f} m")

# Pyramidal cells only; interneurons carry little spatial information and
# dominate ripple spiking.
pyr = np.array(units.index)[cell_type == "excitatory"]
pyr_units = units[list(pyr)]
print(f"{len(pyr)} putative pyramidal cells")


def tuning(group, ep):
    tc = nap.compute_1d_tuning_curves(group, position, nb_bins=n_pos, ep=ep,
                                      minmax=(edges[0], edges[-1]))
    arr = tc.values.T  # (n_units, n_pos)
    arr = np.nan_to_num(arr, nan=0.0)
    return gaussian_filter1d(arr, SMOOTH_BINS, axis=1, mode="nearest")


tc = {d: tuning(pyr_units, run_dirs[d]) for d in ("right", "left")}
tc_all = tuning(pyr_units, run_all)


def spatial_info(rates, occ):
    """Skaggs spatial information (bits/spike)."""
    p = occ / occ.sum()
    mean_r = (p * rates).sum()
    if mean_r <= 0:
        return 0.0
    r = np.clip(rates, 1e-6, None)
    return float((p * (r / mean_r) * np.log2(r / mean_r)).sum())


occ_all, _ = np.histogram(position.restrict(run_all).d, bins=edges)
si = np.array([spatial_info(tc_all[i], occ_all) for i in range(len(pyr))])
peak = tc_all.max(axis=1)
# Selection is on firing, not on spatial information: a fixed information
# threshold left as few as 11 cells in some sessions of this dandiset, which more
# than doubled the cross-validated decoding error.  The Bayesian decoder tolerates
# weakly tuned cells because their near-flat tuning curves contribute an almost
# position-independent term to the likelihood.
n_run_spikes = np.array([len(pyr_units[i].restrict(run_all)) for i in pyr_units.index])
is_place = (peak >= MIN_PEAK_RATE) & (n_run_spikes >= MIN_RUN_SPIKES)
print(f"decoding ensemble: {is_place.sum()} / {len(pyr)} pyramidal cells "
      f"(peak >= {MIN_PEAK_RATE} Hz and >= {MIN_RUN_SPIKES} spikes while running)")

place_ids = pyr[is_place]
templates = {d: tc[d][is_place] for d in ("right", "left")}   # (n_cells, n_pos)

np.savez(f"{CACHE}/place_fields.npz",
         place_ids=place_ids, centers=centers, edges=edges,
         tc_right=templates["right"], tc_left=templates["left"],
         tc_all=tc_all[is_place], si=si, peak=peak, is_place=is_place, pyr=pyr)

# ---------------------------------------------------------------- decoder check
# Decode the animal's true position during running with the same Bayesian
# decoder used for ripples.  If the templates are sound, decoded position must
# track the real trajectory.
VAL_BIN = 0.25
counts_val = pyr_units[list(place_ids)].count(VAL_BIN, run_all)
true_pos = position.bin_average(VAL_BIN, run_all)
# Direction label for every validation bin.
in_right = np.zeros(len(counts_val), dtype=bool)
for s, e in zip(run_dirs["right"].start, run_dirs["right"].end):
    in_right |= (counts_val.t >= s) & (counts_val.t <= e)

occ_r, _ = np.histogram(position.restrict(run_dirs["right"]).d, bins=edges)
occ_l, _ = np.histogram(position.restrict(run_dirs["left"]).d, bins=edges)
prior = {"right": occ_r / max(occ_r.sum(), 1), "left": occ_l / max(occ_l.sum(), 1)}

post_val = np.zeros((len(counts_val), n_pos))
for d, mask in (("right", in_right), ("left", ~in_right)):
    if mask.sum():
        post_val[mask] = R.bayesian_decode(counts_val.values[mask], templates[d],
                                           VAL_BIN, prior=prior[d])
decoded = centers[np.argmax(post_val, axis=1)]
ok = np.isfinite(true_pos.values) & (counts_val.values.sum(axis=1) > 0)
err = np.abs(decoded[ok] - true_pos.values[ok])
print(f"decoder validation ({VAL_BIN * 1000:.0f} ms bins, n={ok.sum()}): "
      f"median error {np.median(err) * 100:.1f} cm, "
      f"r = {np.corrcoef(decoded[ok], true_pos.values[ok])[0, 1]:.3f}")

# Held-out check: fit the templates on odd traversals only and decode the even
# ones, so the error above is not inflated by fitting and testing on the same laps.
cv_err = []
for d in ("right", "left"):
    iv = run_dirs[d]
    odd = nap.IntervalSet(start=iv.start[::2], end=iv.end[::2])
    even = nap.IntervalSet(start=iv.start[1::2], end=iv.end[1::2])
    tmpl_odd = tuning(pyr_units[list(place_ids)], odd)
    c = pyr_units[list(place_ids)].count(VAL_BIN, even)
    p_even = R.bayesian_decode(c.values, tmpl_odd, VAL_BIN)
    truth = position.bin_average(VAL_BIN, even)
    good = np.isfinite(truth.values) & (c.values.sum(axis=1) > 0)
    cv_err.append(np.abs(centers[np.argmax(p_even, axis=1)][good] - truth.values[good]))
cv_err = np.concatenate(cv_err)
print(f"cross-validated (odd laps -> even laps) median error "
      f"{np.median(cv_err) * 100:.1f} cm, n={len(cv_err)}")
np.savez(f"{CACHE}/decoder_cv.npz", cv_err=cv_err)
np.savez(f"{CACHE}/decoder_validation.npz", decoded=decoded[ok],
         true=true_pos.values[ok], err=err, t=counts_val.t[ok])

# ---------------------------------------------------------------- figure 2
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.38, wspace=0.42)

order = np.argsort(np.argmax(templates["right"], axis=1))
for k, d in enumerate(("right", "left")):
    ax = fig.add_subplot(gs[0, k])
    norm = templates[d][order] / np.clip(templates[d][order].max(axis=1, keepdims=True), 1e-9, None)
    im = ax.imshow(norm, aspect="auto", origin="lower", cmap="viridis",
                   extent=[centers[0], centers[-1], 0, len(place_ids)])
    ax.set_xlabel("position (m)")
    ax.set_ylabel("place cell (sorted by rightward peak)" if k == 0 else "")
    ax.set_title(f"{d}ward-run place fields (n={len(place_ids)})", fontsize=11)
    plt.colorbar(im, ax=ax, label="normalised rate", fraction=0.046)

ax = fig.add_subplot(gs[0, 2])
for i in np.linspace(0, len(place_ids) - 1, 8).astype(int):
    j = order[i]
    ax.plot(centers, templates["right"][j], lw=1.5)
ax.set_xlabel("position (m)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("Example place fields (rightward runs)", fontsize=11)

ax = fig.add_subplot(gs[1, 0])
ax.hist(si, bins=30, color="0.4", label="all pyramidal")
ax.hist(si[is_place], bins=30, color="tab:red", alpha=0.7, label="decoding ensemble")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("cells")
ax.set_title("Spatial information", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
seg = slice(0, 400)
ax.plot(np.arange(seg.stop - seg.start) * VAL_BIN, true_pos.values[ok][seg],
        "k-", lw=2, label="actual")
ax.plot(np.arange(seg.stop - seg.start) * VAL_BIN, decoded[seg], ".",
        color="tab:red", ms=4, label="decoded")
ax.set_xlabel("time within concatenated run bins (s)")
ax.set_ylabel("position (m)")
ax.set_title(f"Decoder validation ({VAL_BIN * 1000:.0f} ms bins)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[1, 2])
bins_e = np.linspace(0, 60, 40)
ax.hist(err * 100, bins=bins_e, color="0.4", alpha=0.7, density=True,
        label=f"in-sample (median {np.median(err) * 100:.1f} cm)")
ax.hist(cv_err * 100, bins=bins_e, color="tab:red", alpha=0.55, density=True,
        label=f"held-out laps (median {np.median(cv_err) * 100:.1f} cm)")
ax.set_xlabel("decoding error (cm)")
ax.set_ylabel("density")
ax.set_title("Decoding error during running", fontsize=11)
ax.legend(fontsize=7)

fig.savefig(f"{FIG}/02_place_fields.png", dpi=130, bbox_inches="tight")
print("wrote", f"{FIG}/02_place_fields.png")
