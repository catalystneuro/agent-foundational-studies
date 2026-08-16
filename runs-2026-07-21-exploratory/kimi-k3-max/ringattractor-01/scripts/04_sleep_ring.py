"""Sleep: the ring structure is internally maintained without sensory input.

Analysis A (correlation preservation, Peyrache et al. 2015):
  Pairwise correlations of HD-cell activity (0.5 s bins, per-epoch Fisher-z
  averaged). Wake correlations are computed during active head-movement
  periods (top-quartile angular speed), where the bump rotates and rates
  co-modulate. Correlate wake vs sleep matrices across pairs;
  label-permutation null.

Analysis B (sleep decoding):
  Bayesian-decode a virtual HD during REM using wake tuning curves
  (100 ms bins, 5-bin sliding window). If the ring attractor is internally
  maintained, the decoded angle forms a smooth, slowly drifting trajectory.
  Control: circular time-shift of each unit's spikes within each epoch
  (preserves rates and the static rate landscape, destroys cross-cell
  coordination).
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
from tqdm import tqdm
from common import SESSIONS, load_session, get_hd_angle, get_states, sorted_units, flat_tuning

rng = np.random.default_rng(0)
CORR_BIN = 0.5
DEC_BIN = 0.1
MIN_EPOCH_S = 5.0

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
hd_sorted = hd_idx[order]
pref_sorted = pref[hd_sorted]
n_hd = len(hd_sorted)
hd_units = nap.TsGroup({i: units[int(u)] for i, u in enumerate(hd_sorted)})
iu = np.triu_indices(n_hd, 1)

# ---- active-wake epochs (top-quartile HD angular speed) ----
hd_w = hd.restrict(wake)
v = np.abs(np.angle(np.exp(1j * np.diff(hd_w.values)))) / np.diff(hd_w.t)
v = np.concatenate([[0], v])
v[np.isnan(v)] = 0
v_sm = gaussian_filter1d(v, sigma=39)  # ~1 s smoothing at 39 Hz
active_ep = nap.Tsd(t=hd_w.t, d=v_sm).threshold(
    np.percentile(v_sm, 75), "above").time_support.intersect(wake)
print(f"active wake: {active_ep.tot_length('s'):.0f} s of {wake.tot_length('s'):.0f} s")


def state_corr(epochs, label):
    """Mean pairwise Pearson corr across epochs (Fisher-z averaged, NaN-aware)."""
    zs, ws = [], []
    for st, en in zip(epochs.start, epochs.end):
        if en - st < MIN_EPOCH_S:
            continue
        counts = hd_units.count(CORR_BIN, nap.IntervalSet(st, en)).values.astype(float)
        if counts.shape[0] < 20:
            continue
        with np.errstate(invalid="ignore"):
            C = np.corrcoef(counts.T)
        z = np.arctanh(np.clip(C[iu], -0.999, 0.999))
        zs.append(z)
        ws.append(counts.shape[0])
    Z = np.stack(zs)
    W = np.broadcast_to(np.array(ws, dtype=float)[:, None], Z.shape).copy()
    W[np.isnan(Z)] = 0.0
    Z = np.nan_to_num(Z)
    denom = W.sum(axis=0)
    z_mean = np.where(denom > 0, Z.sum(axis=0) / np.maximum(denom, 1e-12), np.nan)
    Cmean = np.full((n_hd, n_hd), np.nan)
    Cmean[iu] = np.tanh(z_mean)
    Cmean[(iu[1], iu[0])] = Cmean[iu]
    np.fill_diagonal(Cmean, 1.0)
    print(f"{label}: {len(zs)} epochs, {np.isnan(z_mean).sum()} pairs without data")
    return Cmean


C_wake = state_corr(active_ep, "active wake")
C_rem = state_corr(states["REM"], "REM")
C_nrem = state_corr(states["Non-REM"], "NREM")

pw, pr, pn = C_wake[iu], C_rem[iu], C_nrem[iu]
finite = np.isfinite(pw) & np.isfinite(pr) & np.isfinite(pn)
pw, pr, pn = pw[finite], pr[finite], pn[finite]
r_rem = np.corrcoef(pw, pr)[0, 1]
r_nrem = np.corrcoef(pw, pn)[0, 1]
print(f"corr-of-corr wake-REM: {r_rem:.3f}, wake-NREM: {r_nrem:.3f} ({finite.sum()} pairs)")

N_PERM = 1000
null_rem, null_nrem = np.zeros(N_PERM), np.zeros(N_PERM)
for i in tqdm(range(N_PERM), desc="permutation null"):
    p = rng.permutation(n_hd)
    null_rem[i] = np.corrcoef(pw, C_rem[np.ix_(p, p)][iu][finite])[0, 1]
    null_nrem[i] = np.corrcoef(pw, C_nrem[np.ix_(p, p)][iu][finite])[0, 1]
p_rem = (np.sum(null_rem >= r_rem) + 1) / (N_PERM + 1)
p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (N_PERM + 1)
print(f"permutation p: REM {p_rem:.4f}, NREM {p_nrem:.4f}")

# correlation vs angular distance between preferred directions
d_ang = np.abs(np.angle(np.exp(1j * (pref_sorted[:, None] - pref_sorted[None, :]))))[iu][finite]
bins = np.linspace(0, np.pi, 7)
bin_id = np.digitize(d_ang, bins) - 1
centers = 0.5 * (bins[:-1] + bins[1:])
def binned(c):
    return np.array([np.nanmean(c[bin_id == b]) for b in range(len(centers))])
bw, br, bn = binned(pw), binned(pr), binned(pn)

np.savez("data/mouse28_sleep_corr.npz", C_wake=C_wake, C_rem=C_rem, C_nrem=C_nrem,
         r_rem=r_rem, r_nrem=r_nrem, p_rem=p_rem, p_nrem=p_nrem,
         null_rem=null_rem, null_nrem=null_nrem,
         ang_centers=centers, b_wake=bw, b_rem=br, b_nrem=bn,
         pref_sorted=pref_sorted)

# ================= Analysis B: REM decoding with time-shift null =================
tc_rates, scales = flat_tuning(rates[hd_sorted])
print(f"landscape flattening: {(scales > 0.01).sum()}/{n_hd} cells used, "
      f"landscape CV {(tc_rates.sum(0).std()/tc_rates.sum(0).mean()):.3f}")
tuning_da = xr.DataArray(tc_rates, dims=("unit", "hd"),
                         coords={"unit": np.arange(n_hd), "hd": bin_centers})
rem = states["REM"]
dec_rem_all, post_rem_all = nap.decode_bayes(tuning_da, hd_units, epochs=rem,
                                             bin_size=DEC_BIN)
counts_rem = hd_units.count(DEC_BIN, rem)
enough = counts_rem.values.sum(axis=1) >= 2
print(f"REM bins with >=2 spikes: {enough.mean()*100:.0f}% ({enough.sum()} bins)")


def angular_speed(dec, keep):
    t, vv = dec.t, dec.values
    dt = np.diff(t)
    good = (dt < 0.2) & ~np.isnan(vv[:-1]) & ~np.isnan(vv[1:]) & keep[:-1] & keep[1:]
    dv = np.abs(np.angle(np.exp(1j * np.diff(vv))))
    return dv[good] / dt[good]


def posterior_autocorr(post, keep, max_lag=50):
    P = post.values
    Pn = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    t = post.t
    out = np.full(max_lag + 1, np.nan)
    for k in range(1, max_lag + 1):
        dt = t[k:] - t[:-k]
        ok = (np.abs(dt - k * DEC_BIN) < 0.02) & keep[k:] & keep[:-k]
        if ok.sum() < 10:
            continue
        a, b = Pn[:-k][ok], Pn[k:][ok]
        num = np.sum(a * b, axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        out[k] = np.mean(num / den)
    return out


def time_shift_group(group, epochs, rng):
    data = {}
    for k in group.keys():
        pieces = []
        for st, en in zip(epochs.start, epochs.end):
            tt = group[k].t[(group[k].t >= st) & (group[k].t < en)]
            L = en - st
            if len(tt) and L > 0:
                tt = st + (tt - st + rng.uniform(0, L)) % L
            pieces.append(tt)
        data[k] = nap.Ts(np.sort(np.concatenate(pieces)))
    return nap.TsGroup(data)


sp_real = angular_speed(dec_rem_all, enough)
ac_real = posterior_autocorr(post_rem_all, enough)
p90_real = np.percentile(sp_real, 90)
print(f"real REM: median angular speed {np.median(sp_real):.0f} deg/s, "
      f"p90 {p90_real:.0f} deg/s, "
      f"autocorr lag-1 {ac_real[1]:.3f}, lag-10 {ac_real[10]:.3f}, lag-30 {ac_real[30]:.3f}")

N_SHIFT = 100
shift_p90_speed = np.zeros(N_SHIFT)
shift_ac = np.full((N_SHIFT, 51), np.nan)
for i in tqdm(range(N_SHIFT), desc="time-shift null"):
    shifted = time_shift_group(hd_units, rem, rng)
    dec_s, post_s = nap.decode_bayes(tuning_da, shifted, epochs=rem, bin_size=DEC_BIN)
    keep_s = shifted.count(DEC_BIN, rem).values.sum(axis=1) >= 2
    shift_p90_speed[i] = np.percentile(angular_speed(dec_s, keep_s), 90)
    shift_ac[i] = posterior_autocorr(post_s, keep_s)
p_speed = (np.sum(shift_p90_speed <= p90_real) + 1) / (N_SHIFT + 1)
ac_null_mean = np.nanmean(shift_ac, axis=0)
p_ac10 = (np.sum(shift_ac[:, 10] >= ac_real[10]) + 1) / (N_SHIFT + 1)
print(f"time-shift null p90 speed: {np.median(shift_p90_speed):.0f} deg/s; "
      f"real slower with p={p_speed:.4f}")
print(f"autocorr lag-10: real {ac_real[10]:.3f} vs null {ac_null_mean[10]:.3f} (p={p_ac10:.4f})")

# posterior concentration
def posterior_R(post, keep):
    P = post.values[keep]
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    th = bin_centers[None, :]
    return np.sqrt(np.sum(P * np.sin(th), axis=1) ** 2 + np.sum(P * np.cos(th), axis=1) ** 2)

R_rem = posterior_R(post_rem_all, enough)
R_rem = R_rem[~np.isnan(R_rem)]
dec_wake, post_wake = nap.decode_bayes(tuning_da, hd_units, epochs=wake,
                                       bin_size=DEC_BIN)
enough_wake = hd_units.count(DEC_BIN, wake).values.sum(axis=1) >= 2
R_wake = posterior_R(post_wake, enough_wake)
R_wake = R_wake[~np.isnan(R_wake)]
print(f"posterior concentration R: wake {np.median(R_wake):.3f}, REM {np.median(R_rem):.3f}")

# example time-shifted decode + best 90 s window of the longest REM episode
shifted_ex = time_shift_group(hd_units, rem, rng)
dec_shift_ex, _ = nap.decode_bayes(tuning_da, shifted_ex, epochs=rem, bin_size=DEC_BIN)
i_rem = int(np.argmax(rem.end - rem.start))
rem_start, rem_end = rem.start[i_rem], rem.end[i_rem]
# choose the 90 s sub-window with the most HD-cell spikes (best-informed decode)
cnt_ep = counts_rem.restrict(nap.IntervalSet(rem_start, rem_end)).values.sum(axis=1)
t_ep = counts_rem.restrict(nap.IntervalSet(rem_start, rem_end)).t
win = int(90 / DEC_BIN)
if len(cnt_ep) > win:
    csum = np.convolve(cnt_ep, np.ones(win), mode="valid")
    best = int(np.argmax(csum))
    w_start = t_ep[best]
else:
    w_start = rem_start
ep_rem = nap.IntervalSet(start=w_start, end=min(w_start + 90, rem_end))
dec_rem = dec_rem_all.restrict(ep_rem)
post_rem = post_rem_all.restrict(ep_rem)
dec_shuf_rem = dec_shift_ex.restrict(ep_rem)
enough_rem = counts_rem.restrict(ep_rem).values.sum(axis=1) >= 2

np.savez("data/mouse28_sleep_decoding.npz",
         dec_rem_t=dec_rem_all.t, dec_rem=dec_rem_all.values, enough=enough,
         R_rem=R_rem, R_wake=R_wake, sp_real=sp_real,
         shift_p90_speed=shift_p90_speed, p90_real=p90_real,
         ac_real=ac_real, shift_ac=shift_ac)

# ================= figures =================
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.4)

vmax = np.nanpercentile(np.abs(np.concatenate([C_wake[iu], C_rem[iu], C_nrem[iu]]))
                        [np.isfinite(np.concatenate([C_wake[iu], C_rem[iu], C_nrem[iu]]))], 99)
for j, (C, ttl) in enumerate([(C_wake, "wake (active)"), (C_rem, "REM"), (C_nrem, "NREM")]):
    ax = fig.add_subplot(gs[0, j])
    Cplot = C.copy()
    np.fill_diagonal(Cplot, np.nan)
    im = ax.imshow(Cplot, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"pairwise corr, {ttl}")
    ax.set_xlabel("HD cell (sorted)")
    if j == 0:
        ax.set_ylabel("HD cell (sorted)")
fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")

ax = fig.add_subplot(gs[1, 0])
ax.plot(pw, pr, ".", ms=4, alpha=0.5, color="darkorange")
lim = max(abs(pw).max(), abs(pr).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("wake pairwise corr")
ax.set_ylabel("REM pairwise corr")
ax.set_title(f"wake vs REM: r={r_rem:.2f}, p={p_rem:.4f}")

ax = fig.add_subplot(gs[1, 1])
ax.plot(pw, pn, ".", ms=4, alpha=0.5, color="seagreen")
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("wake pairwise corr")
ax.set_ylabel("NREM pairwise corr")
ax.set_title(f"wake vs NREM: r={r_nrem:.2f}, p={p_nrem:.4f}")

ax = fig.add_subplot(gs[1, 2])
ax.hist(null_rem, bins=40, alpha=0.6, color="darkorange", label="REM null")
ax.hist(null_nrem, bins=40, alpha=0.6, color="seagreen", label="NREM null")
ax.axvline(r_rem, color="darkorange", lw=2)
ax.axvline(r_nrem, color="seagreen", lw=2)
ax.set_xlabel("corr-of-corr under label permutation")
ax.set_ylabel("count")
ax.legend(fontsize=8)
ax.set_title("Permutation null")

ax = fig.add_subplot(gs[2, 0])
ax.plot(np.degrees(centers), bw, "o-", color="k", label="wake (active)")
ax.plot(np.degrees(centers), br, "o-", color="darkorange", label="REM")
ax.plot(np.degrees(centers), bn, "o-", color="seagreen", label="NREM")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("angular distance between pref. dirs (deg)")
ax.set_ylabel("mean pairwise corr")
ax.legend(fontsize=8)
ax.set_title("Ring metric structure in all states")

ax = fig.add_subplot(gs[2, 1:])
P = post_rem.values.T
ax.imshow(P / (P.sum(axis=0, keepdims=True) + 1e-12), aspect="auto", cmap="magma",
          origin="lower", extent=[dec_rem.t[0], dec_rem.t[-1], 0, 360])
ax.plot(dec_rem.t[enough_rem], np.degrees(dec_rem.values[enough_rem]), ".",
        color="cyan", ms=1.5, alpha=0.35, label="decoded (real, 100 ms bins)")
# 1-s smoothed decoded trajectory (circular moving mean) to guide the eye
dv = dec_rem.values.copy()
dv[~enough_rem] = np.nan
kern = np.ones(10) / 10
sm_s = np.convolve(np.nan_to_num(np.sin(dv)), kern, mode="same")
sm_c = np.convolve(np.nan_to_num(np.cos(dv)), kern, mode="same")
cnt = np.convolve((~np.isnan(dv)).astype(float), kern, mode="same")
sm_ang = np.where(cnt > 0.5, np.arctan2(sm_s, sm_c) % (2 * np.pi), np.nan)
# break the line where the angle wraps around the circle
sm_deg = np.degrees(sm_ang)
sm_deg[np.abs(np.diff(sm_deg, prepend=sm_deg[0])) > 180] = np.nan
ax.plot(dec_rem.t, sm_deg, color="cyan", lw=1.8, label="decoded (1 s smoothed)")
ax.plot(dec_shuf_rem.t, np.degrees(dec_shuf_rem.values), ".",
        color="white", ms=1, alpha=0.25, label="decoded (time-shifted)")
ax.set_ylabel("virtual HD (deg)")
ax.set_xlabel("time (s)")
ax.set_title(f"REM episode, best 90 s window: internally maintained bump")
ax.legend(loc="upper right", fontsize=8)

fig.suptitle(f"{name}: ring structure is maintained during sleep", fontsize=13)
fig.savefig("figures/04_sleep_ring.png", dpi=150, bbox_inches="tight")

fig2, axes = plt.subplots(1, 4, figsize=(17, 4))
axes[0].hist(shift_p90_speed, bins=20, color="0.4", alpha=0.8,
             label="time-shift null")
axes[0].axvline(p90_real, color="darkorange", lw=2,
                label=f"real {p90_real:.0f} deg/s")
axes[0].set_xlabel("90th percentile |d(decoded HD)/dt| (deg/s)")
axes[0].set_ylabel("count")
axes[0].legend(fontsize=8)
axes[0].set_title(f"Bump drifts slowly (p={p_speed:.3f})")

lags = np.arange(51) * DEC_BIN
axes[1].plot(lags, ac_real, color="darkorange", lw=2, label="real REM")
m = np.nanmean(shift_ac, axis=0)
sd = np.nanstd(shift_ac, axis=0)
axes[1].plot(lags, m, color="0.4", lw=1.5, label="time-shift null")
axes[1].fill_between(lags, m - 2 * sd, m + 2 * sd, color="0.4", alpha=0.3)
axes[1].set_xlabel("lag (s)")
axes[1].set_ylabel("posterior autocorrelation")
axes[1].legend(fontsize=8)
axes[1].set_title(f"Bump persists over seconds (lag-1s p={p_ac10:.3f})")

axes[2].hist(R_wake, bins=50, alpha=0.7, density=True, color="k", label="wake")
axes[2].hist(R_rem, bins=50, alpha=0.7, density=True, color="darkorange", label="REM")
axes[2].set_xlabel("posterior concentration (resultant length)")
axes[2].set_ylabel("density")
axes[2].legend(fontsize=8)
axes[2].set_title("Bump remains localized in REM")

occ_sleep, edges = np.histogram(dec_rem_all.values[enough & ~np.isnan(dec_rem_all.values)],
                                bins=36, range=(0, 2 * np.pi))
axes[3].bar(np.degrees(0.5 * (edges[:-1] + edges[1:])), occ_sleep / occ_sleep.sum(),
            width=8, color="darkorange")
axes[3].set_xlabel("decoded virtual HD (deg)")
axes[3].set_ylabel("occupancy")
axes[3].set_title("REM bump visits the whole ring")

fig2.tight_layout()
fig2.savefig("figures/05_sleep_continuity.png", dpi=150, bbox_inches="tight")
print("saved figures/04_sleep_ring.png and figures/05_sleep_continuity.png")
