"""Wake: the HD population forms a bump on a ring that tracks the actual head direction.

- Sort HD cells by preferred angle -> population activity is a localized bump.
- Bayesian-decode HD from 100 ms population vectors -> decoded angle tracks true HD.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from common import SESSIONS, load_session, get_hd_angle, get_states, sorted_units, flat_tuning

name = "Mouse28-140310"
nwb = load_session(SESSIONS[name])
units = sorted_units(nwb)
hd = get_hd_angle(nwb)
states = get_states(nwb)
wake = states["Awake"]

d = np.load("data/mouse28_tuning.npz")
rates, bin_centers = d["rates"], d["bin_centers"]
s = np.load("data/mouse28_hd_stats.npz")
is_hd, pref = s["is_hd"], s["pref"]

hd_idx = np.where(is_hd)[0]
order = np.argsort(pref[hd_idx])
hd_sorted = hd_idx[order]  # unit indices sorted by preferred direction
hd_units = nap.TsGroup({i: units[int(u)] for i, u in enumerate(hd_sorted)})

# tuning DataArray for decoding, rows sorted by pref direction; tuning curves
# are scaled to flatten the population rate landscape (removes the static
# decode bias toward under-represented directions)
tc_rates, scales = flat_tuning(rates[hd_sorted])
print(f"landscape flattening: {(scales > 0.01).sum()}/{len(hd_sorted)} cells used")
tuning_da = xr.DataArray(
    tc_rates,
    dims=("unit", "hd"),
    coords={"unit": np.arange(len(hd_sorted)), "hd": bin_centers},
)

# ---- Bayesian decoding on wake, 100 ms bins ----
decoded, posterior = nap.decode_bayes(tuning_da, hd_units, epochs=wake, bin_size=0.1)
print(f"decoded {len(decoded)} wake bins")

# actual HD at the same bins
actual = decoded.value_from(hd)
valid = ~np.isnan(actual.values) & ~np.isnan(decoded.values)
err = np.angle(np.exp(1j * (decoded.values[valid] - actual.values[valid])))
med_err = np.degrees(np.median(np.abs(err)))
print(f"median |circular error| on wake: {med_err:.1f} deg ({valid.sum()} bins)")
np.savez("data/mouse28_wake_decoding.npz", decoded_t=decoded.t, decoded=decoded.values,
         actual=actual.values, err_deg=np.degrees(err))

# ---- figure ----
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3,
                      height_ratios=[1.2, 1.2, 1])

# pick a 60 s window inside the longest wake epoch
dur = wake.end - wake.start
i_long = int(np.argmax(dur))
t0 = wake.start[i_long] + (dur[i_long] - 60) / 2
ep = nap.IntervalSet(start=t0, end=t0 + 60)

# (a) population bump: binned, smoothed rates sorted by pref direction
counts = hd_units.count(0.05, ep)  # 50 ms bins
sm = gaussian_filter1d(counts.values.astype(float), sigma=2, axis=0)  # 100 ms temporal smooth
norm = sm / (sm.max(axis=1, keepdims=True) + 1e-9)
ax = fig.add_subplot(gs[0, :])
ax.imshow(norm.T, aspect="auto", cmap="viridis", origin="lower",
          extent=[counts.t[0], counts.t[-1], -0.5, len(hd_sorted) - 0.5])
hd_ep = hd.restrict(ep)
ax.plot(hd_ep.t, np.interp(hd_ep.values % (2 * np.pi), pref[hd_sorted] % (2 * np.pi),
                           np.arange(len(hd_sorted)), period=len(hd_sorted)),
        color="red", lw=1.2, label="true HD")
ax.set_ylabel("HD cell (sorted by pref. dir.)")
ax.set_xlabel("time (s)")
ax.set_title("Population activity: a single bump moves along the ring")
ax.legend(loc="upper right", fontsize=8)

# (b) posterior heatmap + true HD
dec_ep = decoded.restrict(ep)
post_ep = posterior.restrict(ep)
ax = fig.add_subplot(gs[1, :])
P = post_ep.values.T
ax.imshow(P / (P.sum(axis=0, keepdims=True) + 1e-12), aspect="auto", cmap="magma",
          origin="lower", extent=[dec_ep.t[0], dec_ep.t[-1], 0, 360])
ax.plot(hd_ep.t, np.degrees(hd_ep.values), color="cyan", lw=1.0, label="true HD")
ax.plot(dec_ep.t, np.degrees(dec_ep.values), ".", color="white", ms=1.5, label="decoded HD")
ax.set_ylabel("head direction (deg)")
ax.set_xlabel("time (s)")
ax.set_title("Bayesian posterior over HD (100 ms bins, wake tuning curves)")
ax.legend(loc="upper right", fontsize=8)

# (c) decoded vs actual scatter (circular)
ax = fig.add_subplot(gs[2, 0])
ax.plot(np.degrees(actual.values[valid]), np.degrees(decoded.values[valid]), ".",
        ms=1, alpha=0.3, color="k")
ax.plot([0, 360], [0, 360], "r--", lw=1)
ax.set_xlabel("true HD (deg)")
ax.set_ylabel("decoded HD (deg)")
ax.set_title(f"Wake decoding (median err {med_err:.0f} deg)")

# (d) error distribution
ax = fig.add_subplot(gs[2, 1])
ax.hist(np.degrees(err), bins=72, range=(-180, 180), color="k")
ax.axvline(0, color="r", ls="--", lw=1)
ax.set_xlabel("circular error (deg)")
ax.set_ylabel("count")
ax.set_title("Decoding error distribution")

fig.suptitle(f"{name}: ring-attractor readout during wake", fontsize=13)
fig.savefig("figures/03_wake_ring_decoding.png", dpi=150, bbox_inches="tight")
print("saved figures/03_wake_ring_decoding.png")
