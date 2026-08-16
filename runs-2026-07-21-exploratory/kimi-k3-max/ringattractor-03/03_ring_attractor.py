"""Ring attractor analyses on Mouse28-140310: bump, decoding, correlations."""
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

ASSET = "656704ea-a4cd-40f4-8158-a6533ebf2eee"  # Mouse28-140310

def load_session(asset_id, cache_dir="/tmp/remfile_cache_hd"):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile)

def compute_hd(nwb):
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    rv = np.asarray(red.values)
    bv = np.asarray(blue.values)
    valid = (rv[:, 0] > 0) & (rv[:, 1] > 0) & (bv[:, 0] > 0) & (bv[:, 1] > 0)
    ang = np.full(rv.shape[0], np.nan)
    ang[valid] = np.arctan2(rv[valid, 1] - bv[valid, 1], rv[valid, 0] - bv[valid, 0]) % (2 * np.pi)
    return nap.Tsd(t=red.t, d=ang, time_support=red.time_support)

def clean_units(units):
    """Sorted-spike TsGroup of nonzero-rate units (handles unsorted-unit gotcha)."""
    d = {}
    for k in units.keys():
        t = np.sort(units[k].t)
        if len(t) > 0:
            d[k] = nap.Ts(t)
    return nap.TsGroup(d)

def identify_hd_cells(units, hd, wake, n_shuffle=1000, mvl_floor=0.3, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    hd_w = hd.restrict(wake)
    ok = ~np.isnan(hd_w.d)
    t_valid, a_valid = hd_w.t[ok], hd_w.d[ok]
    mvl, pval = {}, {}
    for k in units.keys():
        spk = units[k].restrict(wake)
        ang = spk.value_from(hd)
        ang = ang[~np.isnan(ang)]
        n = len(ang)
        if n < 100:
            continue
        if n > 100_000:
            ang = ang[rng.choice(n, 100_000, replace=False)]
            n = 100_000
        obs = np.abs(np.mean(np.exp(1j * ang)))
        null = np.empty(n_shuffle)
        chunk = 100
        for c in range(0, n_shuffle, chunk):
            idx = rng.integers(0, len(t_valid), size=(chunk, n))
            null[c:c + chunk] = np.abs(np.mean(np.exp(1j * a_valid[idx]), axis=1))
        mvl[k] = obs
        pval[k] = (np.sum(null >= obs) + 1) / (n_shuffle + 1)
    hd_cells = [k for k in mvl if pval[k] < alpha and mvl[k] > mvl_floor]
    return mvl, pval, hd_cells

def circ_dist(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))

nwb = load_session(ASSET)
units_all = clean_units(nwb["units"])
states = nwb["states"]
wake = states[states["label"] == "Awake"]
rem = states[states["label"] == "REM"]
nrem = states[states["label"] == "Non-REM"]

hd = compute_hd(nwb)
mvl, pval, hd_cells = identify_hd_cells(units_all, hd, wake)
units = units_all[hd_cells]
print(f"{len(hd_cells)} HD cells")

# tuning curves (rate, Hz) for HD cells
tc = nap.compute_tuning_curves(units, hd, bins=60, range=(0, 2 * np.pi),
                               epochs=wake, return_counts=True)
occ = tc.attrs["occupancy"] / tc.attrs["fs"]
tc_rate = tc / np.where(occ == 0, np.nan, occ)
bin_centers = tc.coords[tc.dims[1]].values
prefs = np.array([bin_centers[np.nanargmax(tc_rate.values[i])] for i in range(len(hd_cells))])
order = np.argsort(prefs)
prefs_sorted = prefs[order]

# ---------- 1. wake bump + decoding validation ----------
bin_size = 0.1  # 100 ms
counts_wake = units.count(bin_size, wake)
decoded_wake, post_wake = nap.decode_bayes(tc_rate, units, wake, bin_size)
n_spk = counts_wake.values.sum(axis=1)
valid_bins = n_spk >= 2
actual = counts_wake.value_from(hd)  # Tsd of HD at bin times? value_from gives Tsd
err = np.angle(np.exp(1j * (decoded_wake.values - actual.values)))
err_valid = err[valid_bins & ~np.isnan(actual.values)]
med_err = np.degrees(np.median(np.abs(err_valid)))
print(f"wake decode: median |err| = {med_err:.1f} deg over {valid_bins.sum()} bins")

# ---------- 2. sleep decoding ----------
decoded_rem, _ = nap.decode_bayes(tc_rate, units, rem, bin_size)
decoded_nrem, _ = nap.decode_bayes(tc_rate, units, nrem, bin_size)
counts_rem = units.count(bin_size, rem)
valid_rem = counts_rem.values.sum(axis=1) >= 2

def angular_speed(decoded, valid, bin_size):
    a = decoded.values[valid]
    t = decoded.t[valid]
    da = np.angle(np.exp(1j * np.diff(a)))
    dt = np.diff(t)
    ok = dt < 2 * bin_size  # consecutive bins only
    return np.degrees(np.abs(da[ok]) / dt[ok])

speed_rem = angular_speed(decoded_rem, valid_rem, bin_size)
# control: shuffle time bins of the count matrix within each REM epoch
rng = np.random.default_rng(1)
counts_shuf = counts_rem.values.copy()
for ep_i in range(len(rem)):
    m = (counts_rem.t >= rem.start[ep_i]) & (counts_rem.t <= rem.end[ep_i])
    idx = np.where(m)[0]
    counts_shuf[idx] = counts_shuf[idx][rng.permutation(len(idx))]
counts_shuf_tsd = nap.TsdFrame(t=counts_rem.t, d=counts_shuf,
                               columns=counts_rem.columns, time_support=rem)
decoded_shuf, _ = nap.decode_bayes(tc_rate, counts_shuf_tsd, rem, bin_size)
valid_shuf = counts_shuf.sum(axis=1) >= 2
speed_shuf = angular_speed(decoded_shuf, valid_shuf, bin_size)
print(f"REM angular speed: real median {np.median(speed_rem):.0f} deg/s, "
      f"bin-shuffle median {np.median(speed_shuf):.0f} deg/s")

# ---------- 3. correlation structure per state ----------
def state_corr(units, ep, bin_size=0.1):
    c = units.count(bin_size, ep)
    X = c.values  # time x units
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(X.T)  # NaN row/col for units silent in this state
    return C

Cc = {}
for name, ep in [("wake", wake), ("REM", rem), ("NREM", nrem)]:
    Cc[name] = state_corr(units, ep)
    print(name, "corr matrix", Cc[name].shape,
          "finite pairs:", np.isfinite(Cc[name][np.triu_indices(len(hd_cells), 1)]).sum())

iu = np.triu_indices(len(hd_cells), k=1)
ang_dist = circ_dist(prefs[iu[0]], prefs[iu[1]])

def corr_of_corr(C1, C2):
    v1, v2 = C1[iu], C2[iu]
    ok = np.isfinite(v1) & np.isfinite(v2)
    return np.corrcoef(v1[ok], v2[ok])[0, 1]

r_rem = corr_of_corr(Cc["wake"], Cc["REM"])
r_nrem = corr_of_corr(Cc["wake"], Cc["NREM"])
print(f"corr-of-corr wake-REM: {r_rem:.3f}, wake-NREM: {r_nrem:.3f}")

# label-permutation null for corr-of-corr
n_perm = 1000
null_rem, null_nrem = np.empty(n_perm), np.empty(n_perm)
for p in range(n_perm):
    perm = rng.permutation(len(hd_cells))
    null_rem[p] = corr_of_corr(Cc["wake"], Cc["REM"][np.ix_(perm, perm)])
    null_nrem[p] = corr_of_corr(Cc["wake"], Cc["NREM"][np.ix_(perm, perm)])
p_rem = (np.sum(null_rem >= r_rem) + 1) / (n_perm + 1)
p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (n_perm + 1)
print(f"permutation p: REM {p_rem:.4f}, NREM {p_nrem:.4f}")

np.savez("/tmp/ring_mouse28.npz", prefs=prefs, order=order, hd_cells=hd_cells,
         Cw=Cc["wake"], Crem=Cc["REM"], Cnrem=Cc["NREM"],
         r_rem=r_rem, r_nrem=r_nrem, null_rem=null_rem, null_nrem=null_nrem,
         speed_rem=speed_rem, speed_shuf=speed_shuf, med_err=med_err)

# ---------- figures ----------
from scipy.ndimage import gaussian_filter1d

def norm_rows(X, sigma):
    Xs = gaussian_filter1d(X.astype(float), sigma, axis=1)
    mx = np.nanmax(Xs, axis=1, keepdims=True)
    return np.nan_to_num(Xs / np.where(mx == 0, 1, mx))

def to_angle_grid(X_sorted, prefs_sorted, n_grid=60):
    """X_sorted: (n_units, n_time) with units sorted by pref angle.
    Returns (n_grid, n_time) interpolated onto a uniform angular grid so the
    y-axis is truly angular and aligns with HD overlays."""
    grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    ang_ext = np.concatenate([prefs_sorted - 2 * np.pi, prefs_sorted, prefs_sorted + 2 * np.pi])
    out = np.empty((n_grid, X_sorted.shape[1]))
    for j in range(X_sorted.shape[1]):
        v_ext = np.concatenate([X_sorted[:, j]] * 3)
        out[:, j] = np.interp(grid, ang_ext, v_ext)
    return grid, out

# pick a 60 s wake window with the best HD coverage for the bump display
hdw = hd.restrict(wake)
best, best_t0 = -1, None
for cand in np.arange(wake.start[0], wake.end[-1] - 60, 20):
    mm = (hdw.t >= cand) & (hdw.t < cand + 60) & ~np.isnan(hdw.d)
    if mm.sum() < 100:
        continue
    cov = len(np.unique(np.floor(hdw.d[mm] / (2 * np.pi) * 36)))
    if cov > best:
        best, best_t0 = cov, cand
t0, t1 = best_t0, best_t0 + 60
print(f"bump window: {t0:.0f}-{t1:.0f} s, HD coverage {best}/36 bins")

# Fig A: wake bump + decode validation
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
m = (counts_wake.t >= t0) & (counts_wake.t <= t1)
Xs = norm_rows(counts_wake.values[m][:, order].T, 2)
grid, Xg = to_angle_grid(Xs, prefs_sorted)
ax.imshow(Xg, aspect="auto", origin="lower", cmap="viridis",
          extent=[counts_wake.t[m][0], counts_wake.t[m][-1], 0, 2 * np.pi])
hdm = (hd.t >= t0) & (hd.t <= t1)
ax.plot(hd.t[hdm], hd.d[hdm], "r.", ms=1.5, label="actual HD")
dm = (decoded_wake.t >= t0) & (decoded_wake.t <= t1)
ax.plot(decoded_wake.t[dm], decoded_wake.values[dm], "w.", ms=1.5, label="decoded HD")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
ax.set_ylabel("angle (rad)")
ax.set_title(f"Population bump (20 HD cells sorted by preferred direction) — wake")
ax.legend(loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.plot(actual.values[valid_bins], decoded_wake.values[valid_bins], ".", ms=1, alpha=0.3)
ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "r-", lw=1)
ax.set_xlabel("actual HD (rad)"); ax.set_ylabel("decoded HD (rad)")
ax.set_title("Wake decoding")
ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])

ax = fig.add_subplot(gs[1, 1])
ax.hist(np.degrees(err_valid), bins=60, range=(-180, 180))
ax.set_xlabel("decode error (deg)")
ax.set_title(f"Wake error, median |err| = {med_err:.0f}°")
fig.savefig("fig_wake_bump_decode.png", dpi=150)
print("saved fig_wake_bump_decode.png")

# Fig B: REM bump + speed
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
# pick longest REM epoch
lens = rem.end - rem.start
iep = int(np.argmax(lens))
t0, t1 = float(rem.start[iep]), float(rem.end[iep])
m = (counts_rem.t >= t0) & (counts_rem.t <= t1)
Xs = norm_rows(counts_rem.values[m][:, order].T, 3)
grid, Xg = to_angle_grid(Xs, prefs_sorted)
ax.imshow(Xg, aspect="auto", origin="lower", cmap="viridis",
          extent=[counts_rem.t[m][0], counts_rem.t[m][-1], 0, 2 * np.pi])
dm = (decoded_rem.t >= t0) & (decoded_rem.t <= t1)
ax.plot(decoded_rem.t[dm], decoded_rem.values[dm], "w.", ms=2, label="decoded internal HD")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
ax.set_ylabel("angle (rad)")
ax.set_title(f"REM sleep episode ({t1 - t0:.0f} s): internally generated bump drifts")
ax.legend(loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.plot(decoded_rem.t[dm], np.unwrap(decoded_rem.values[dm]), lw=1)
ax.set_xlabel("time (s)"); ax.set_ylabel("unwrapped decoded angle (rad)")
ax.set_title("Decoded angle drifts continuously during REM")

ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, 720, 60)
ax.hist(speed_rem, bins=bins, alpha=0.7, density=True, label="REM (real)")
ax.hist(speed_shuf, bins=bins, alpha=0.7, density=True, label="time-bin shuffle")
ax.set_xlabel("|angular speed| (deg/s)")
ax.legend()
ax.set_title(f"median {np.median(speed_rem):.0f} vs {np.median(speed_shuf):.0f} deg/s")
fig.savefig("fig_rem_bump_speed.png", dpi=150)
print("saved fig_rem_bump_speed.png")

# Fig C: correlation matrices + corr vs distance
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 4, wspace=0.4)
vm = 0.6
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("lightgray")
mat_axes = []
for i, name in enumerate(["wake", "REM", "NREM"]):
    ax = fig.add_subplot(gs[0, i])
    Cs = np.ma.masked_invalid(Cc[name][np.ix_(order, order)])
    im = ax.imshow(Cs, cmap=cmap, vmin=-vm, vmax=vm, origin="lower")
    ax.set_title(f"{name}")
    ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("HD cells (sorted by pref. dir.)")
    mat_axes.append(ax)
fig.colorbar(im, ax=mat_axes, fraction=0.03, pad=0.02, label="pairwise correlation")
ax = fig.add_subplot(gs[0, 3])
for name, c in [("wake", "k"), ("REM", "r"), ("NREM", "b")]:
    v = Cc[name][iu]
    # bin by angular distance
    edges = np.linspace(0, np.pi, 13)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    mn = [np.nanmean(v[(ang_dist >= edges[j]) & (ang_dist < edges[j + 1])]) for j in range(12)]
    ax.plot(np.degrees(ctr), mn, ".-", color=c, label=name, ms=4, lw=1)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("Δ preferred direction (deg)")
ax.set_ylabel("pairwise correlation")
ax.legend(fontsize=8)
ax.set_title(f"corr-of-corr: REM {r_rem:.2f}, NREM {r_nrem:.2f}")
fig.savefig("fig_corr_matrices.png", dpi=150)
print("saved fig_corr_matrices.png")
