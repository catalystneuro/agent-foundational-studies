"""Prototype: continuous velocity tuning — lagged regression of rate on hand velocity."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

S3_MCMAZE = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

disk_cache = remfile.DiskCache("cache/remfile_cache")
rem_file = remfile.File(S3_MCMAZE, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

hand_vel = nwb["hand_vel"]
units = nwb["units"]
df = nwbfile.intervals["trials"].to_dataframe()
move_on = df["move_onset_time"].values

BIN = 0.025  # 25 ms bins
t0, t1 = hand_vel.t[0], hand_vel.t[-1]
bin_edges = np.arange(t0, t1 + BIN, BIN)
bin_centers = bin_edges[:-1] + BIN / 2
nb = len(bin_centers)
print("bins:", nb)

# binned spike counts -> smoothed rate (Hz)
counts = np.zeros((len(units), nb), dtype=np.float32)
for j, u in enumerate(tqdm(units.index, desc="binning spikes")):
    counts[j] = np.histogram(units[u].t, bins=bin_edges)[0]
rate = counts / BIN
rate_s = gaussian_filter1d(rate, sigma=2.0, axis=1)  # 50 ms sigma

# binned velocity (mean within bin)
vx, _ = np.histogram(hand_vel.t, bins=bin_edges, weights=hand_vel.d[:, 0])
vy, _ = np.histogram(hand_vel.t, bins=bin_edges, weights=hand_vel.d[:, 1])
n_in_bin, _ = np.histogram(hand_vel.t, bins=bin_edges)
vx = vx / np.maximum(n_in_bin, 1)
vy = vy / np.maximum(n_in_bin, 1)
speed = np.hypot(vx, vy)

# perimovement mask: [move_onset-0.15, move_onset+0.65] for all trials
mask = np.zeros(nb, dtype=bool)
for m in move_on:
    i0 = int((m - 0.15 - t0) / BIN)
    i1 = int((m + 0.65 - t0) / BIN)
    mask[max(i0, 0):min(i1, nb)] = True
print("perimovement bins:", mask.sum(), "=", mask.sum() * BIN / 60, "min")

# lagged regression: rate(t) ~ [1, vx(t+lag), vy(t+lag)]
lags = np.arange(-0.25, 0.251, BIN)  # seconds; positive = velocity leads spikes
lag_bins = (lags / BIN).astype(int)
X = np.stack([vx, vy], axis=1)  # (nb, 2)

def fit_r2(y, x1, x2, m):
    Xm = np.stack([np.ones(m.sum()), x1[m], x2[m]], axis=1)
    ym = y[m]
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    pred = Xm @ beta
    ss_res = np.sum((ym - pred) ** 2)
    ss_tot = np.sum((ym - ym.mean()) ** 2)
    return 1 - ss_res / ss_tot, beta

r2_vs_lag = np.zeros((len(units), len(lags)), dtype=np.float32)
for li, lb in enumerate(tqdm(lag_bins, desc="lag sweep")):
    Xs = np.roll(X, -lb, axis=0)  # Xs[t] = X[t+lb]
    for j in range(len(units)):
        r2, _ = fit_r2(rate_s[j], Xs[:, 0], Xs[:, 1], mask)
        r2_vs_lag[j, li] = r2

best_li = np.argmax(r2_vs_lag, axis=1)
best_lag = lags[best_li]
best_r2 = r2_vs_lag[np.arange(len(units)), best_li]
print("best lag median (ms):", np.median(best_lag) * 1000)
print("best R2 median:", np.median(best_r2))

# population mean R2 vs lag
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(lags * 1000, r2_vs_lag.mean(axis=0), "k")
ax.set_xlabel("lag (ms)  [+ = velocity leads spikes]")
ax.set_ylabel("population mean $R^2$")
ax.axvline(0, color="gray", ls=":")
ax.set_title("Velocity encoding vs lag")

ax = axes[1]
ax.hist(best_lag * 1000, bins=lags * 1000)
ax.set_xlabel("best lag per unit (ms)")
ax.set_ylabel("units")
ax.set_title("Best-lag distribution")

ax = axes[2]
ax.hist(best_r2, bins=30)
ax.set_xlabel("$R^2$ at best lag")
ax.set_ylabel("units")
ax.set_title("Velocity-encoding strength")
fig.tight_layout()
fig.savefig("figures/04_velocity_lag_population.png", dpi=150)
print("saved figures/04_velocity_lag_population.png")

# example unit detail: pick the unit with best R2
j = int(np.argmax(best_r2))
u = units.index[j]
lb = lag_bins[best_li[j]]
Xs = np.roll(X, -lb, axis=0)
r2, beta = fit_r2(rate_s[j], Xs[:, 0], Xs[:, 1], mask)
pref_vel_dir = np.arctan2(beta[2], beta[1])
print(f"example unit {u}: R2={r2:.3f}, lag={best_lag[j]*1000:.0f} ms, pref vel dir={np.degrees(pref_vel_dir):.0f} deg")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
ax = axes[0, 0]
ax.plot(lags * 1000, r2_vs_lag[j], "k")
ax.axvline(best_lag[j] * 1000, color="r", ls="--")
ax.set_xlabel("lag (ms)"); ax.set_ylabel("$R^2$")
ax.set_title(f"unit {u}: lag sweep")

ax = axes[0, 1]
# velocity direction tuning at best lag: bin by direction of velocity
vdir = np.arctan2(Xs[:, 1], Xs[:, 0])
m2 = mask & (np.hypot(Xs[:, 0], Xs[:, 1]) > 100)  # moving bins only
edges = np.linspace(-np.pi, np.pi, 17)
centers = (edges[:-1] + edges[1:]) / 2
means = [rate_s[j][m2 & (vdir >= edges[k]) & (vdir < edges[k+1])].mean() for k in range(16)]
ax.plot(np.degrees(centers), means, "o-")
ax.set_xlabel("velocity direction (deg)"); ax.set_ylabel("rate (Hz)")
ax.set_title(f"unit {u}: velocity-direction tuning (speed>100 mm/s)")

ax = axes[1, 0]
spd = np.hypot(Xs[:, 0], Xs[:, 1])
sbins = np.arange(0, 1000, 75)
sc = (sbins[:-1] + sbins[1:]) / 2
smeans = [rate_s[j][mask & (spd >= sbins[k]) & (spd < sbins[k+1])].mean() for k in range(len(sbins)-1)]
ax.plot(sc, smeans, "o-")
r_spd = np.corrcoef(rate_s[j][mask], spd[mask])[0, 1]
ax.set_xlabel("speed (mm/s)"); ax.set_ylabel("rate (Hz)")
ax.set_title(f"unit {u}: rate vs speed (r={r_spd:.2f})")

ax = axes[1, 1]
tt = bin_centers[mask]
sel = (tt > 500) & (tt < 512)
ax.plot(tt[sel], rate_s[j][mask][sel] / rate_s[j][mask][sel].max(), "k", label="rate (norm)")
ax.plot(tt[sel], spd[mask][sel] / spd[mask][sel].max(), "r", alpha=0.7, label="speed (norm)")
ax.set_xlabel("time (s)"); ax.set_ylabel("normalized")
ax.set_title(f"unit {u}: rate and speed (lag-corrected)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/05_velocity_example_unit.png", dpi=150)
print("saved figures/05_velocity_example_unit.png")
