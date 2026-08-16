"""Scale the ring-attractor analysis to 4 more sessions (one per mouse).

Per session: HD cell selection (MVL + random-time null), wake decoding error,
correlation preservation (active-wake vs REM/NREM, label-permutation null),
REM bump autocorrelation vs a small time-shift null.
Saves per-session npz + a pooled summary figure.
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
from common import (SESSIONS, load_session, get_hd_angle, get_states, sorted_units,
                    mean_vector_length, flat_tuning)

rng = np.random.default_rng(7)
CORR_BIN = 0.5
DEC_BIN = 0.1
N_SHUFFLE_HD = 1000
N_PERM = 1000
N_SHIFT = 20
MVL_FLOOR = 0.3
SPIKE_CAP = 100_000

PANEL = ["Mouse25-140123", "Mouse17-130128", "Mouse20-130514", "Mouse24-131213"]


def analyze_session(name):
    print(f"\n===== {name} =====")
    nwb = load_session(SESSIONS[name])
    units = sorted_units(nwb)
    hd = get_hd_angle(nwb)
    states = get_states(nwb)
    wake = states["Awake"]

    # ---- HD cell selection ----
    hd_wake = hd.restrict(wake)
    hd_valid_v = hd_wake.values[~np.isnan(hd_wake.values)]
    counts_da = nap.compute_tuning_curves(units, hd, bins=60, range=[(0, 2 * np.pi)],
                                          epochs=wake, return_counts=True,
                                          feature_names=["hd"])
    counts = counts_da.values
    occupancy = counts_da.attrs["occupancy"] / counts_da.attrs["fs"]
    rates = np.where(occupancy[None, :] > 0, counts / occupancy[None, :], np.nan)
    bin_centers = counts_da.coords["hd"].values

    n_units = len(units)
    mvl_obs = np.zeros(n_units)
    p_val = np.ones(n_units)
    pref = np.zeros(n_units)
    for i, k in enumerate(units.keys()):
        spk = units[k].restrict(wake)
        if len(spk) == 0:
            continue
        ang = spk.value_from(hd)
        ang = ang[~np.isnan(ang)]
        if len(ang) > SPIKE_CAP:
            ang = ang[rng.choice(len(ang), SPIKE_CAP, replace=False)]
        if len(ang) < 50:
            continue
        mvl_obs[i] = mean_vector_length(ang)
        pref[i] = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2 * np.pi)
        n = len(ang)
        null = np.zeros(N_SHUFFLE_HD)
        for cs in range(0, N_SHUFFLE_HD, 100):
            m = min(100, N_SHUFFLE_HD - cs)
            samp = hd_valid_v[rng.integers(0, len(hd_valid_v), size=(m, n))]
            null[cs:cs + m] = np.sqrt(np.sin(samp).mean(1) ** 2 + np.cos(samp).mean(1) ** 2)
        p_val[i] = (np.sum(null >= mvl_obs[i]) + 1) / (N_SHUFFLE_HD + 1)
    is_hd = (p_val < 0.05) & (mvl_obs > MVL_FLOOR)
    n_hd = int(is_hd.sum())
    print(f"HD cells: {n_hd}/{n_units}")
    if n_hd < 5:
        print("too few HD cells, skipping")
        return None

    hd_idx = np.where(is_hd)[0]
    order = np.argsort(pref[hd_idx])
    hd_sorted = hd_idx[order]
    pref_sorted = pref[hd_sorted]
    hd_units = nap.TsGroup({i: units[int(u)] for i, u in enumerate(hd_sorted)})
    iu = np.triu_indices(n_hd, 1)

    # ---- wake decoding (flattened landscape) ----
    tc_rates, scales = flat_tuning(rates[hd_sorted])
    tuning_da = xr.DataArray(tc_rates, dims=("unit", "hd"),
                             coords={"unit": np.arange(n_hd), "hd": bin_centers})
    dec_wake, _ = nap.decode_bayes(tuning_da, hd_units, epochs=wake, bin_size=DEC_BIN)
    actual = dec_wake.value_from(hd)
    valid = ~np.isnan(actual.values) & ~np.isnan(dec_wake.values)
    err = np.angle(np.exp(1j * (dec_wake.values[valid] - actual.values[valid])))
    med_err = np.degrees(np.median(np.abs(err)))
    print(f"wake decoding median error: {med_err:.1f} deg")

    # ---- correlation preservation ----
    hd_w = hd.restrict(wake)
    v = np.abs(np.angle(np.exp(1j * np.diff(hd_w.values)))) / np.diff(hd_w.t)
    v = np.concatenate([[0], v])
    v[np.isnan(v)] = 0
    v_sm = gaussian_filter1d(v, sigma=39)
    active_ep = nap.Tsd(t=hd_w.t, d=v_sm).threshold(
        np.percentile(v_sm, 75), "above").time_support.intersect(wake)

    def state_corr(epochs):
        zs, ws = [], []
        for st, en in zip(epochs.start, epochs.end):
            if en - st < 5.0:
                continue
            c = hd_units.count(CORR_BIN, nap.IntervalSet(st, en)).values.astype(float)
            if c.shape[0] < 20:
                continue
            with np.errstate(invalid="ignore"):
                C = np.corrcoef(c.T)
            zs.append(np.arctanh(np.clip(C[iu], -0.999, 0.999)))
            ws.append(c.shape[0])
        Z = np.stack(zs)
        W = np.broadcast_to(np.array(ws, float)[:, None], Z.shape).copy()
        W[np.isnan(Z)] = 0
        Z = np.nan_to_num(Z)
        den = W.sum(0)
        return np.where(den > 0, Z.sum(0) / np.maximum(den, 1e-12), np.nan)

    zw = np.tanh(state_corr(active_ep))
    zr = np.tanh(state_corr(states["REM"]))
    zn = np.tanh(state_corr(states["Non-REM"]))
    fin = np.isfinite(zw) & np.isfinite(zr) & np.isfinite(zn)
    r_rem = np.corrcoef(zw[fin], zr[fin])[0, 1]
    r_nrem = np.corrcoef(zw[fin], zn[fin])[0, 1]
    # permutation null
    C_rem = np.full((n_hd, n_hd), np.nan)
    C_rem[iu] = zr
    C_rem[(iu[1], iu[0])] = zr
    C_nrem = np.full((n_hd, n_hd), np.nan)
    C_nrem[iu] = zn
    C_nrem[(iu[1], iu[0])] = zn
    null_rem = np.zeros(N_PERM)
    null_nrem = np.zeros(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(n_hd)
        null_rem[i] = np.corrcoef(zw[fin], C_rem[np.ix_(p, p)][iu][fin])[0, 1]
        null_nrem[i] = np.corrcoef(zw[fin], C_nrem[np.ix_(p, p)][iu][fin])[0, 1]
    p_rem = (np.sum(null_rem >= r_rem) + 1) / (N_PERM + 1)
    p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (N_PERM + 1)
    print(f"corr-of-corr: wake-REM {r_rem:.3f} (p={p_rem:.4f}), "
          f"wake-NREM {r_nrem:.3f} (p={p_nrem:.4f}), pairs {fin.sum()}")

    # ---- REM bump autocorrelation vs time-shift null ----
    rem = states["REM"]
    dec_rem, post_rem = nap.decode_bayes(tuning_da, hd_units, epochs=rem, bin_size=DEC_BIN)
    keep = hd_units.count(DEC_BIN, rem).values.sum(axis=1) >= 2

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
            out[k] = np.mean(np.sum(a * b, 1) /
                             (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12))
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

    ac_real = posterior_autocorr(post_rem, keep)
    ac_null = np.full((N_SHIFT, 51), np.nan)
    for i in range(N_SHIFT):
        sh = time_shift_group(hd_units, rem, rng)
        _, ps = nap.decode_bayes(tuning_da, sh, epochs=rem, bin_size=DEC_BIN)
        ks = sh.count(DEC_BIN, rem).values.sum(axis=1) >= 2
        ac_null[i] = posterior_autocorr(ps, ks)
    p_ac = (np.nansum(ac_null[:, 10] >= ac_real[10]) + 1) / (N_SHIFT + 1)
    print(f"REM autocorr lag-1s: real {ac_real[10]:.3f} vs null "
          f"{np.nanmean(ac_null[:,10]):.3f} (p={p_ac:.3f})")

    return dict(name=name, n_hd=n_hd, n_units=n_units, med_err=med_err,
                r_rem=r_rem, r_nrem=r_nrem, p_rem=p_rem, p_nrem=p_nrem,
                zw=zw[fin], zr=zr[fin], zn=zn[fin],
                d_ang=np.abs(np.angle(np.exp(1j * (pref_sorted[:, None] -
                                                   pref_sorted[None, :]))))[iu][fin],
                ac_real=ac_real, ac_null_mean=np.nanmean(ac_null, axis=0), p_ac=p_ac)


results = []
for name in PANEL:
    res = analyze_session(name)
    if res is not None:
        np.savez(f"data/{res['name']}_summary.npz", **{k: v for k, v in res.items()
                 if isinstance(v, np.ndarray) or np.isscalar(v)})
        results.append(res)

# add Mouse28 (from the single-session analysis)
d28 = np.load("data/mouse28_sleep_corr.npz")
s28 = np.load("data/mouse28_sleep_decoding.npz")
pref28 = d28["pref_sorted"]
iu28 = np.triu_indices(len(pref28), 1)
fin28 = np.isfinite(d28["C_wake"][iu28]) & np.isfinite(d28["C_rem"][iu28]) & np.isfinite(d28["C_nrem"][iu28])
results.append(dict(
    name="Mouse28-140310", n_hd=20, n_units=47, med_err=19.9,
    r_rem=float(d28["r_rem"]), r_nrem=float(d28["r_nrem"]),
    p_rem=float(d28["p_rem"]), p_nrem=float(d28["p_nrem"]),
    zw=d28["C_wake"][iu28][fin28], zr=d28["C_rem"][iu28][fin28], zn=d28["C_nrem"][iu28][fin28],
    d_ang=np.abs(np.angle(np.exp(1j * (pref28[:, None] - pref28[None, :]))))[iu28][fin28],
    ac_real=s28["ac_real"], ac_null_mean=np.nanmean(s28["shift_ac"], axis=0), p_ac=0.0099,
))

# ---- pooled summary figure ----
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

names = [r["name"] for r in results]
r_rems = [r["r_rem"] for r in results]
r_nrems = [r["r_nrem"] for r in results]
x = np.arange(len(names))
axes[0].bar(x - 0.2, r_rems, width=0.4, color="darkorange", label="wake-REM")
axes[0].bar(x + 0.2, r_nrems, width=0.4, color="seagreen", label="wake-NREM")
axes[0].set_xticks(x)
axes[0].set_xticklabels([n.replace("-", "\n") for n in names], fontsize=7)
axes[0].set_ylabel("corr-of-corr")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=8)
axes[0].set_title("Structure preserved per session")
for xi, r in zip(x, results):
    star = "***" if max(r["p_rem"], r["p_nrem"]) <= 0.001 else ""
    axes[0].text(xi, max(r["r_rem"], r["r_nrem"]) + 0.03, star, ha="center", fontsize=10)

# pooled wake vs sleep scatter
zw_all = np.concatenate([r["zw"] for r in results])
zr_all = np.concatenate([r["zr"] for r in results])
zn_all = np.concatenate([r["zn"] for r in results])
axes[1].plot(zw_all, zr_all, ".", ms=3, alpha=0.4, color="darkorange", label="REM")
axes[1].plot(zw_all, zn_all, ".", ms=3, alpha=0.4, color="seagreen", label="NREM")
lim = np.percentile(np.abs(np.concatenate([zw_all, zr_all, zn_all])), 99) * 1.2
axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
axes[1].set_xlim(-lim, lim)
axes[1].set_ylim(-lim, lim)
r_pool_rem = np.corrcoef(zw_all, zr_all)[0, 1]
r_pool_nrem = np.corrcoef(zw_all, zn_all)[0, 1]
axes[1].set_xlabel("wake pairwise corr")
axes[1].set_ylabel("sleep pairwise corr")
axes[1].legend(fontsize=8)
axes[1].set_title(f"pooled: REM r={r_pool_rem:.2f}, NREM r={r_pool_nrem:.2f}")

# pooled corr vs angular distance
d_all = np.concatenate([r["d_ang"] for r in results])
bins = np.linspace(0, np.pi, 7)
bid = np.digitize(d_all, bins) - 1
centers = np.degrees(0.5 * (bins[:-1] + bins[1:]))
for vals, c, lbl in [(zw_all, "k", "wake (active)"), (zr_all, "darkorange", "REM"),
                     (zn_all, "seagreen", "NREM")]:
    m = [np.mean(vals[bid == b]) for b in range(6)]
    axes[2].plot(centers, m, "o-", color=c, label=lbl)
axes[2].axhline(0, color="0.7", lw=0.8)
axes[2].set_xlabel("angular distance between pref. dirs (deg)")
axes[2].set_ylabel("mean pairwise corr")
axes[2].legend(fontsize=8)
axes[2].set_title("Ring metric structure (pooled)")

# REM bump autocorrelation across sessions
lags = np.arange(51) * DEC_BIN
for r in results:
    axes[3].plot(lags, r["ac_real"] - r["ac_null_mean"], lw=1.5,
                 label=f"{r['name']} (p={r['p_ac']:.3f})")
axes[3].axhline(0, color="0.5", ls="--", lw=0.8)
axes[3].set_xlabel("lag (s)")
axes[3].set_ylabel("excess posterior autocorrelation\n(real - time-shift null)")
axes[3].legend(fontsize=6.5)
axes[3].set_title("REM bump persistence above null")

fig.suptitle("Ring attractor maintained during sleep: 5 sessions, 5 mice", fontsize=13)
fig.tight_layout()
fig.savefig("figures/06_multisession_summary.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/06_multisession_summary.png")
print(f"pooled corr-of-corr: REM {r_pool_rem:.3f}, NREM {r_pool_nrem:.3f} "
      f"({len(zw_all)} pairs, {len(results)} sessions)")
