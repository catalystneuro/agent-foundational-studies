"""Prototype: Bayesian HD decoding + wake-sleep correlation preservation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

from hd_utils import (
    SESSIONS, load_session, compute_head_direction, get_wake_epochs,
    tuning_curves_hd,
)

name = "Mouse17-130128"
nwb, io = load_session(SESSIONS[name])
units = nwb["units"]
hd = compute_head_direction(nwb)
wake = get_wake_epochs(nwb, "Awake")

res = np.load("cache/prototype_results.npz", allow_pickle=True)
keys = list(res["keys"])
hd_keys = [k for k, h in zip(keys, res["is_hd"]) if h]
print("HD cells:", hd_keys)

rates_emp, centers, occupancy = tuning_curves_hd(units, hd, wake, bins=60)

# ------------------------------------------------------------------ decoding
hd_group = units[hd_keys]
bin_size = 0.1
count = hd_group.count(bin_size, ep=wake)
# decode_bayes wants tuning curves as xarray with proper dims; rates_emp is (unit, bin)
tc_hd = rates_emp.sel(unit=hd_keys)
# fill NaN bins (zero occupancy) with 0 rate for decoding
tc_filled = tc_hd.fillna(0.0)
decoded, proba = nap.decode_bayes(tc_filled, count, wake, bin_size)
print("decoded shape:", decoded.shape)

# actual HD at decode times (circular-safe interpolation)
valid = ~np.isnan(hd.values)
t_v, a_v = hd.t[valid], hd.values[valid]
cos_i = np.interp(decoded.t, t_v, np.cos(a_v))
sin_i = np.interp(decoded.t, t_v, np.sin(a_v))
hd_actual = np.arctan2(sin_i, cos_i) % (2 * np.pi)
err = np.angle(np.exp(1j * (decoded.values - hd_actual)))  # circular error
print("median |error| (deg):", np.rad2deg(np.nanmedian(np.abs(err))))
print("frac within 30 deg:", np.mean(np.abs(err) < np.deg2rad(30)))

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
t0, t1 = 17900.0, 17960.0
sl = (decoded.t >= t0) & (decoded.t <= t1)
axes[0].plot(decoded.t[sl], hd_actual[sl], "k.", ms=2, label="actual HD")
axes[0].plot(decoded.t[sl], decoded.values[sl] % (2 * np.pi), "r.", ms=2, alpha=0.6, label="decoded HD")
axes[0].set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
axes[0].set_ylabel("HD (rad)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title(f"{name}: Bayesian decoding of HD from {len(hd_keys)} HD cells (100 ms bins)")
axes[1].hist(np.rad2deg(err), bins=72, range=(-180, 180), color="steelblue")
axes[1].set_xlabel("Decoding error (deg)")
axes[1].set_ylabel("Count")
axes[1].axvline(0, color="k", lw=0.8)
med = np.rad2deg(np.nanmedian(np.abs(err)))
axes[1].set_title(f"Error distribution (median |err| = {med:.1f}°)")
fig.savefig("figures/06_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/06_decoding.png")

# ------------------------------------------------------- wake-sleep correlations
states = nwb["states"]
def state_epochs(label):
    return states[states["label"] == label]

def pairwise_corr_matrix(group, epochs, bin_size=0.1):
    """Pearson correlation matrix of binned spike counts across units."""
    cnt = group.count(bin_size, ep=epochs)
    X = cnt.values  # (time, units)
    if X.shape[0] < 50:
        return None
    C = np.corrcoef(X.T)
    return C

C_wake = pairwise_corr_matrix(hd_group, wake)
C_rem = pairwise_corr_matrix(hd_group, state_epochs("REM"))
C_nrem = pairwise_corr_matrix(hd_group, state_epochs("Non-REM"))

iu = np.triu_indices(len(hd_keys), k=1)
def upper(C):
    return C[iu] if C is not None else None

pairs = {
    "wake": upper(C_wake),
    "REM": upper(C_rem),
    "NREM": upper(C_nrem),
}
for s in ["REM", "NREM"]:
    if pairs[s] is not None:
        r = np.corrcoef(pairs["wake"], pairs[s])[0, 1]
        print(f"wake vs {s} corr-of-corr: r = {r:.3f} (n pairs = {len(pairs['wake'])})")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
vmin, vmax = -0.6, 0.6
for ax, (lab, C) in zip(axes, [("Wake", C_wake), ("REM", C_rem), ("Non-REM", C_nrem)]):
    if C is None:
        ax.axis("off"); continue
    im = ax.imshow(C, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"{lab} pairwise correlations")
    ax.set_xticks(range(len(hd_keys)), hd_keys, fontsize=7)
    ax.set_yticks(range(len(hd_keys)), hd_keys, fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
fig.savefig("figures/07_state_correlations.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/07_state_correlations.png")

# scatter wake vs sleep pairwise correlations
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(pairs["wake"], pairs["REM"], s=30, c="darkorange", label=f"REM (r={np.corrcoef(pairs['wake'], pairs['REM'])[0,1]:.2f})")
ax.scatter(pairs["wake"], pairs["NREM"], s=30, c="steelblue", alpha=0.7, label=f"NREM (r={np.corrcoef(pairs['wake'], pairs['NREM'])[0,1]:.2f})")
ax.plot([-0.6, 0.8], [-0.6, 0.8], "k--", lw=0.8)
ax.set_xlabel("Wake pairwise correlation")
ax.set_ylabel("Sleep pairwise correlation")
ax.set_title(f"{name}: HD-cell pairwise correlations preserved in sleep")
ax.legend(fontsize=9)
fig.savefig("figures/08_wake_sleep_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/08_wake_sleep_scatter.png")
io.close()
