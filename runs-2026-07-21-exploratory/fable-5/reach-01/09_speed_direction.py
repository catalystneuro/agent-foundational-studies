"""Does speed act as a gain on the directional signal, or is it coded separately?

For each unit, firing rate is measured as a function of hand speed separately for
movements toward its preferred direction, orthogonal to it, and away from it.
A pure speed code predicts three parallel rising curves; a velocity (speed x
direction) code predicts curves that fan out.
"""
import numpy as np
import pynapple as nap
import mcmaze_io as mio

nap.nap_config.suppress_conversion_warnings = True

BIN = 0.02
nwbfile, nwb = mio.open_nwb()
spikes = nwb["units"]
trials_ep = nwb["trials"]
hand_pos, hand_vel = mio.load_kinematics(nwbfile)
fits = np.load("tuning_fits.npz")
pdir, sig = fits["pdir"], fits["pval"] < 0.01
LAG = float(np.load("trial_data.npz")["best_lag"])

vel = nap.TsdFrame(t=hand_vel.t - LAG, d=hand_vel.values, columns=["vx", "vy"]).restrict(trials_ep)
counts = spikes.count(BIN, ep=trials_ep)
V = vel.interpolate(counts, ep=counts.time_support)
sp = np.hypot(V["vx"].values, V["vy"].values)
th = np.arctan2(V["vy"].values, V["vx"].values)
C = counts.values.astype(float)
n_units = C.shape[1]

SP_EDGES = np.array([100, 250, 400, 550, 700, 850, 1400.0])
sp_c = (SP_EDGES[:-1] + SP_EDGES[1:]) / 2
sp_c[-1] = 1000
sbin = np.digitize(sp, SP_EDGES) - 1
valid_sp = (sbin >= 0) & (sbin < len(SP_EDGES) - 1)

CATS = [("toward PD", 0, 45), ("orthogonal", 45, 135), ("away from PD", 135, 180)]
prof = np.full((len(CATS), len(sp_c), n_units), np.nan)
nobs = np.zeros((len(CATS), len(sp_c)))
for j in range(n_units):
    dth = np.degrees(np.abs((th - pdir[j] + np.pi) % (2 * np.pi) - np.pi))
    for ci, (_, lo, hi) in enumerate(CATS):
        m0 = valid_sp & (dth >= lo) & (dth < hi)
        for si in range(len(sp_c)):
            m = m0 & (sbin == si)
            if m.sum() > 20:
                prof[ci, si, j] = C[m, j].mean() / BIN
            if j == 0:
                nobs[ci, si] = m.sum()
print("bins per (category, speed) cell:\n", nobs.astype(int))

# normalise each unit by its overall mean rate so units can be pooled
mean_rate = C[valid_sp].mean(0) / BIN
prof_n = prof / mean_rate[None, None, :]

# per-unit slope of rate vs speed within each category
slopes = np.full((len(CATS), n_units), np.nan)
for ci in range(len(CATS)):
    for j in range(n_units):
        y = prof[ci, :, j]
        ok = np.isfinite(y)
        if ok.sum() >= 4:
            slopes[ci, j] = np.polyfit(sp_c[ok], y[ok], 1)[0] * 100  # Hz per 100 mm/s

for ci, (name, _, _) in enumerate(CATS):
    print(f"{name:14s} median slope = {np.nanmedian(slopes[ci][sig]):+.3f} Hz per 100 mm/s")

from scipy import stats

w = stats.wilcoxon(slopes[0][sig], slopes[2][sig], nan_policy="omit")
print(f"toward vs away slopes: Wilcoxon p = {w.pvalue:.3g}")

# ---- population map in PD-relative polar velocity coordinates (speed x delta-theta)
NDT = 16
dt_edges = np.linspace(-np.pi, np.pi, NDT + 1)
dt_c = (dt_edges[:-1] + dt_edges[1:]) / 2
pop_map = np.full((len(sp_c), NDT, n_units), np.nan)
for j in range(n_units):
    dth = (th - pdir[j] + np.pi) % (2 * np.pi) - np.pi
    dbin = np.digitize(dth, dt_edges) - 1
    for si in range(len(sp_c)):
        for di in range(NDT):
            m = valid_sp & (sbin == si) & (dbin == di)
            if m.sum() > 15:
                pop_map[si, di, j] = C[m, j].mean() / BIN
pop_map_n = pop_map / mean_rate[None, None, :]
print("population map coverage:", np.isfinite(pop_map_n[:, :, sig]).mean())

np.savez("speed_direction.npz", prof=prof, prof_n=prof_n, sp_c=sp_c, slopes=slopes,
         cat_names=np.array([c[0] for c in CATS]), sig=sig, mean_rate=mean_rate,
         sp_edges=SP_EDGES, pop_map_n=pop_map_n, dt_edges=dt_edges, dt_c=dt_c,
         wilcoxon_p=w.pvalue)
print("saved speed_direction.npz")
