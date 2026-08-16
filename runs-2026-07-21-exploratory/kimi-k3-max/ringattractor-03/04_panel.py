"""Multi-session panel: corr-of-corr wake vs sleep across 5 sessions (one per mouse)."""
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

ASSETS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}

def load_session(asset_id, cache_dir="/tmp/remfile_cache_hd"):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    return nap.NWBFile(io.read())

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
        ang = units[k].restrict(wake).value_from(hd)
        ang = ang[~np.isnan(ang)]
        n = len(ang)
        if n < 100:
            continue
        if n > 100_000:
            ang = ang[rng.choice(n, 100_000, replace=False)]
            n = 100_000
        obs = np.abs(np.mean(np.exp(1j * ang)))
        null = np.empty(n_shuffle)
        for c in range(0, n_shuffle, 100):
            idx = rng.integers(0, len(t_valid), size=(100, n))
            null[c:c + 100] = np.abs(np.mean(np.exp(1j * a_valid[idx]), axis=1))
        mvl[k] = obs
        pval[k] = (np.sum(null >= obs) + 1) / (n_shuffle + 1)
    return mvl, pval, [k for k in mvl if pval[k] < alpha and mvl[k] > mvl_floor]

def state_corr(units, ep, bin_size=0.1):
    X = units.count(bin_size, ep).values
    with np.errstate(invalid="ignore"):
        return np.corrcoef(X.T)

def corr_of_corr(C1, C2, iu):
    v1, v2 = C1[iu], C2[iu]
    ok = np.isfinite(v1) & np.isfinite(v2)
    return np.corrcoef(v1[ok], v2[ok])[0, 1]

results = {}
for ses, aid in ASSETS.items():
    print(f"\n=== {ses} ===")
    nwb = load_session(aid)
    units_all = clean_units(nwb["units"])
    states = nwb["states"]
    wake = states[states["label"] == "Awake"]
    rem = states[states["label"] == "REM"]
    nrem = states[states["label"] == "Non-REM"]
    hd = compute_hd(nwb)
    mvl, pval, hd_cells = identify_hd_cells(units_all, hd, wake)
    print(f"{len(hd_cells)}/{len(units_all)} HD cells")
    units = units_all[hd_cells]
    tc = nap.compute_tuning_curves(units, hd, bins=60, range=(0, 2 * np.pi),
                                   epochs=wake, return_counts=True)
    occ = tc.attrs["occupancy"] / tc.attrs["fs"]
    tc_rate = tc / np.where(occ == 0, np.nan, occ)
    bin_centers = tc.coords[tc.dims[1]].values
    prefs = np.array([bin_centers[np.nanargmax(tc_rate.values[i])] for i in range(len(hd_cells))])
    Cw = state_corr(units, wake)
    Cr = state_corr(units, rem)
    Cn = state_corr(units, nrem)
    iu = np.triu_indices(len(hd_cells), k=1)
    r_rem = corr_of_corr(Cw, Cr, iu)
    r_nrem = corr_of_corr(Cw, Cn, iu)
    rng = np.random.default_rng(2)
    n_perm = 1000
    null_rem = np.empty(n_perm)
    null_nrem = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(len(hd_cells))
        null_rem[p] = corr_of_corr(Cw, Cr[np.ix_(perm, perm)], iu)
        null_nrem[p] = corr_of_corr(Cw, Cn[np.ix_(perm, perm)], iu)
    p_rem = (np.sum(null_rem >= r_rem) + 1) / (n_perm + 1)
    p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (n_perm + 1)
    print(f"corr-of-corr REM {r_rem:.3f} (p={p_rem:.4f}), NREM {r_nrem:.3f} (p={p_nrem:.4f})")
    results[ses] = dict(hd_cells=hd_cells, n_units=len(units_all), prefs=prefs,
                        Cw=Cw, Cr=Cr, Cn=Cn, r_rem=r_rem, r_nrem=r_nrem,
                        null_rem=null_rem, null_nrem=null_nrem,
                        p_rem=p_rem, p_nrem=p_nrem, iu=iu)

np.savez("/tmp/ring_panel.npz", **{ses: results[ses] for ses in results})

# ---------- panel figure ----------
sessions = list(results.keys())
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
x = np.arange(len(sessions))
ax.bar(x, [results[s]["n_units"] for s in sessions], color="lightgray", label="all units")
ax.bar(x, [len(results[s]["hd_cells"]) for s in sessions], color="steelblue", label="HD cells")
ax.set_xticks(x, [s.replace("Mouse", "M") for s in sessions], fontsize=8)
ax.set_ylabel("unit count")
ax.legend()
ax.set_title("HD cell yield per session")

ax = axes[0, 1]
w = 0.35
for j, (key, nullkey, pkey, c) in enumerate([("r_rem", "null_rem", "p_rem", "r"),
                                             ("r_nrem", "null_nrem", "p_nrem", "b")]):
    vals = [results[s][key] for s in sessions]
    null_hi = [np.percentile(results[s][nullkey], 99) for s in sessions]
    ax.bar(x + (j - 0.5) * w, vals, w, color=c, alpha=0.8,
           label="wake–REM" if j == 0 else "wake–NREM")
    ax.scatter(x + (j - 0.5) * w, null_hi, marker="_", color="k", s=60,
               label="99th pctile of permutation null" if j == 0 else None)
ax.set_xticks(x, [s.replace("Mouse", "M") for s in sessions], fontsize=8)
ax.set_ylabel("corr-of-corr (Pearson r)")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.set_title("Wake vs sleep correlation-structure preservation")

# pooled corr vs angular distance
ax = axes[1, 0]
edges = np.linspace(0, np.pi, 13)
ctr = np.degrees(0.5 * (edges[:-1] + edges[1:]))
for key, c, lab in [("Cw", "k", "wake"), ("Cr", "r", "REM"), ("Cn", "b", "NREM")]:
    all_d, all_v = [], []
    for s in sessions:
        prefs = results[s]["prefs"]
        iu = results[s]["iu"]
        d = np.abs(np.angle(np.exp(1j * (prefs[iu[0]] - prefs[iu[1]]))))
        v = results[s][key][iu]
        ok = np.isfinite(v)
        all_d.append(d[ok]); all_v.append(v[ok])
    all_d = np.concatenate(all_d); all_v = np.concatenate(all_v)
    mn = [np.mean(all_v[(all_d >= edges[j]) & (all_d < edges[j + 1])]) for j in range(12)]
    se = [np.std(all_v[(all_d >= edges[j]) & (all_d < edges[j + 1])]) /
          np.sqrt(np.sum((all_d >= edges[j]) & (all_d < edges[j + 1]))) for j in range(12)]
    ax.errorbar(ctr, mn, yerr=se, fmt=".-", color=c, label=lab, ms=5, lw=1, capsize=2)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("Δ preferred direction (deg)")
ax.set_ylabel("pairwise correlation")
ax.legend()
ax.set_title(f"Ring topology preserved in sleep (pooled pairs)")

# pooled scatter wake vs REM
ax = axes[1, 1]
for key, c, lab in [("Cr", "r", "REM"), ("Cn", "b", "NREM")]:
    vw, vs = [], []
    for s in sessions:
        iu = results[s]["iu"]
        a, b = results[s]["Cw"][iu], results[s][key][iu]
        ok = np.isfinite(a) & np.isfinite(b)
        vw.append(a[ok]); vs.append(b[ok])
    vw = np.concatenate(vw); vs = np.concatenate(vs)
    ax.scatter(vw, vs, s=3, alpha=0.3, color=c,
               label=f"{lab} (r={np.corrcoef(vw, vs)[0,1]:.2f})")
ax.plot([-0.6, 1], [-0.6, 1], "k--", lw=0.8)
ax.set_xlabel("wake pairwise correlation")
ax.set_ylabel("sleep pairwise correlation")
ax.legend(fontsize=8, markerscale=3)
ax.set_title("Pair-level preservation")
fig.tight_layout()
fig.savefig("fig_panel.png", dpi=150)
print("\nsaved fig_panel.png")
