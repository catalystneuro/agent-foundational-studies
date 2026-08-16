"""Refined decoding: mask zero-spike bins; auto-pick wide-coverage display window."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

from hd_utils import (
    SESSIONS, load_session, compute_head_direction, get_wake_epochs, tuning_curves_hd,
)

name = "Mouse17-130128"
nwb, io = load_session(SESSIONS[name], cache_dir="cache2")
units = nwb["units"]
hd = compute_head_direction(nwb)
wake = get_wake_epochs(nwb, "Awake")

res = np.load("cache/prototype_results.npz", allow_pickle=True)
keys = list(res["keys"])
hd_keys = [k for k, h in zip(keys, res["is_hd"]) if h]

rates_emp, centers, occupancy = tuning_curves_hd(units, hd, wake, bins=60)
hd_group = units[hd_keys]
bin_size = 0.1
count = hd_group.count(bin_size, ep=wake)
tc_filled = rates_emp.sel(unit=hd_keys).fillna(0.0)
decoded, proba = nap.decode_bayes(tc_filled, count, wake, bin_size)

valid = ~np.isnan(hd.values)
t_v, a_v = hd.t[valid], hd.values[valid]
cos_i = np.interp(decoded.t, t_v, np.cos(a_v))
sin_i = np.interp(decoded.t, t_v, np.sin(a_v))
hd_actual = np.arctan2(sin_i, cos_i) % (2 * np.pi)

n_spikes_bin = count.values.sum(axis=1)
informative = n_spikes_bin >= 2
err = np.angle(np.exp(1j * (decoded.values - hd_actual)))
print(f"bins: {len(err)}, informative (>=2 spikes): {informative.sum()} ({100*informative.mean():.0f}%)")
print("median |err| all bins:", np.rad2deg(np.nanmedian(np.abs(err))))
print("median |err| informative:", np.rad2deg(np.nanmedian(np.abs(err[informative]))))
print("frac within 30 deg (informative):", np.mean(np.abs(err[informative]) < np.deg2rad(30)))

# chance level: decode with shuffled... simpler: circular-uniform error median = 90 deg
# auto-pick 60 s window with broadest HD coverage
hd_wake = hd.restrict(wake)
win = 60.0
best_t0, best_cov = None, -1
for t0 in np.arange(wake.start[0], wake.end[-1] - win, 30.0):
    ep = nap.IntervalSet(t0, t0 + win)
    if len(ep.intersect(wake)) == 0 or ep.intersect(wake).tot_length() < win - 1:
        continue
    v = hd.restrict(ep).values
    v = v[~np.isnan(v)]
    if len(v) < 500:
        continue
    occ, _ = np.histogram(v, bins=36, range=(0, 2 * np.pi))
    cov = np.mean(occ > 0)
    if cov > best_cov:
        best_cov, best_t0 = cov, t0
print(f"best window t0={best_t0:.0f} coverage={best_cov:.2f}")

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
t0, t1 = best_t0, best_t0 + win
sl = (decoded.t >= t0) & (decoded.t <= t1) & informative
axes[0].plot(decoded.t[sl], hd_actual[sl], "k.", ms=3, label="actual HD")
axes[0].plot(decoded.t[sl], decoded.values[sl] % (2 * np.pi), "r.", ms=3, alpha=0.55, label="decoded HD")
axes[0].set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
axes[0].set_ylabel("HD (rad)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title(f"{name}: Bayesian HD decoding from {len(hd_keys)} HD cells (100 ms bins, >=2 spikes)")
axes[1].hist(np.rad2deg(err[informative]), bins=72, range=(-180, 180), color="steelblue")
axes[1].axvline(0, color="k", lw=0.8)
med = np.rad2deg(np.nanmedian(np.abs(err[informative])))
axes[1].set_xlabel("Decoding error (deg)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Error distribution (median |err| = {med:.1f} deg; chance ~ 90 deg)")
fig.savefig("figures/06_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/06_decoding.png")
io.close()
