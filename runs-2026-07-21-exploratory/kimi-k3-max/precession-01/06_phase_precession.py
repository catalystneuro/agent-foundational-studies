"""Step 6: theta phase precession — circular-linear regression of spike phase on position.

For each place cell, spikes from runs in the cell's preferred direction are taken,
restricted to the place field. The circular-linear regression slope k (cycles/m) is
found by maximizing the resultant length R(k) = |mean(exp(i(phi - 2*pi*k*x)))|
(Kempter et al. 2012). Significance is assessed against 500 phase-permutation
shuffles per cell. Slopes are expressed relative to field progress (negative =
precession to earlier phases as the animal advances through the field).
"""
import numpy as np
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng(42)

# ---------- load ----------
ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
r = requests.get(f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/",
                 allow_redirects=False)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_precession")
h5py_file = h5py.File(remfile.File(r.headers["Location"], disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
units = nwb["units"]
lin_ts = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t_pos = lin_ts.index.values
lin = np.asarray(lin_ts.values)[:, 0]

th = np.load("theta_phase.npz")
t_lfp, phase = th["t_lfp"], th["phase"]

pf = np.load("place_fields.npz", allow_pickle=True)
place_keys = pf["place_keys"]
place_pref = pf["place_pref"]
place_lo = pf["place_lo"]
place_hi = pf["place_hi"]
place_peak = pf["place_peak"]
dir_pos, dir_neg = pf["dir_pos"], pf["dir_neg"]

def eps_to_intervalset(arr):
    return nap.IntervalSet(arr[:, 0], arr[:, 1])

ep_by_dir = {"pos": eps_to_intervalset(dir_pos), "neg": eps_to_intervalset(dir_neg)}

# ---------- circular-linear regression ----------
K_GRID = np.linspace(-6, 6, 241)  # cycles per meter

def circ_lin_fit(x, phi):
    """Return (best slope k, R at best slope, phase offset phi0)."""
    E = np.exp(1j * (phi[None, :] - 2 * np.pi * K_GRID[:, None] * x[None, :]))
    R = np.abs(E.mean(axis=1))
    i = int(np.argmax(R))
    phi0 = np.angle(E[i].mean())
    return K_GRID[i], R[i], phi0

N_SHUFF = 500
MIN_SPIKES = 30

results = []
spike_cache = {}  # per cell: (x_infield, phi_infield, progress, all spike df for examples)

for k, pref, lo, hi, pk in tqdm(zip(place_keys, place_pref, place_lo, place_hi, place_peak),
                                total=len(place_keys), desc="cells"):
    ep = ep_by_dir[pref]
    spk = units[int(k)].restrict(ep).index.values
    if len(spk) < MIN_SPIKES:
        continue
    # position and theta phase at spike times
    ip = np.searchsorted(t_pos, spk).clip(0, len(t_pos) - 1)
    x = lin[ip]
    il = np.searchsorted(t_lfp, spk).clip(0, len(t_lfp) - 1)
    phi = phase[il]
    ok = ~np.isnan(x)
    x, phi = x[ok], phi[ok]
    # restrict to place field
    infield = (x >= lo) & (x <= hi)
    xf, phif = x[infield], phi[infield]
    if len(xf) < MIN_SPIKES:
        continue

    k_best, R_best, phi0 = circ_lin_fit(xf, phif)
    # shuffle: permute phases among spikes
    R_sh = np.empty(N_SHUFF)
    for s in range(N_SHUFF):
        phi_s = phif[rng.permutation(len(phif))]
        E = np.exp(1j * (phi_s[None, :] - 2 * np.pi * K_GRID[:, None] * xf[None, :]))
        R_sh[s] = np.abs(E.mean(axis=1)).max()
    p_val = (np.sum(R_sh >= R_best) + 1) / (N_SHUFF + 1)

    # slope sign relative to field progress: for 'neg' direction, progress = -x
    sign = 1.0 if pref == "pos" else -1.0
    k_prog = k_best * sign
    # normalized field progress for each spike in [0, 1]
    prog = (xf - lo) / (hi - lo) if pref == "pos" else (hi - xf) / (hi - lo)

    results.append(dict(unit=int(k), pref=pref, lo=lo, hi=hi, peak=pk,
                        n_spikes=len(xf), k=k_best, k_prog=k_prog, R=R_best,
                        phi0=phi0, p=p_val, R_sh_max=np.max(R_sh),
                        R_sh_med=np.median(R_sh)))
    spike_cache[int(k)] = (xf, phif, prog)

print(f"\ncells analyzed: {len(results)}")
n_sig = sum(1 for r_ in results if r_["p"] < 0.05)
n_sig_neg = sum(1 for r_ in results if r_["p"] < 0.05 and r_["k_prog"] < 0)
print(f"significant (p<0.05): {n_sig}/{len(results)}")
print(f"significant with negative (precessing) slope: {n_sig_neg}")
med_k = np.median([r_["k_prog"] for r_ in results])
print(f"median progress-referenced slope: {med_k:.3f} cycles/m")
med_R = np.median([r_["R"] for r_ in results])
med_Rsh = np.median([r_["R_sh_med"] for r_ in results])
print(f"median R: {med_R:.3f}, median shuffle R: {med_Rsh:.3f}")

np.savez("precession_results.npz",
         results=np.array(results, dtype=object), spike_cache=np.array(list(spike_cache.items()), dtype=object))

# ---------- figure: example cells ----------
sig_neg = [r_ for r_ in results if r_["p"] < 0.05 and r_["k_prog"] < 0]
sig_neg.sort(key=lambda r_: r_["R"], reverse=True)
examples = sig_neg[:6]

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, r_ in zip(axes.flat, examples):
    xf, phif, prog = spike_cache[r_["unit"]]
    # plot phase over two cycles
    ax.scatter(xf, phif, s=6, color="C0", alpha=0.6)
    ax.scatter(xf, phif + 2 * np.pi, s=6, color="C0", alpha=0.6)
    xg = np.linspace(r_["lo"], r_["hi"], 50)
    yg = r_["phi0"] + 2 * np.pi * r_["k"] * xg
    yg_wr = (yg + np.pi) % (2 * np.pi) - np.pi
    # plot wrapped line segments in both cycles
    for off in (0, 2 * np.pi):
        yline = yg_wr + off
        jumps = np.where(np.abs(np.diff(yline)) > np.pi)[0]
        xs, ys = xg.copy(), yline.copy()
        xs = np.insert(xs, jumps + 1, np.nan)
        ys = np.insert(ys, jumps + 1, np.nan)
        ax.plot(xs, ys, color="k", lw=1.5)
    ax.set_ylim(-np.pi, 3 * np.pi)
    ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi],
                  ["-π", "0", "π", "2π", "3π"])
    ax.set_xlabel("position on track (m)")
    ax.set_ylabel("theta phase (rad)")
    ax.set_title(f"unit {r_['unit']} ({r_['pref']} dir): slope {r_['k_prog']:.2f} cyc/m, "
                 f"R={r_['R']:.2f}, p={r_['p']:.3f}", fontsize=9)
fig.suptitle("Theta phase precession — example place cells", y=0.995)
fig.tight_layout()
fig.savefig("fig_precession_examples.png", dpi=150)
print("saved fig_precession_examples.png")

# ---------- figure: population ----------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
kv = np.array([r_["k_prog"] for r_ in results])
ax.hist(kv, bins=30, color="0.4")
ax.axvline(0, color="k", lw=1)
ax.axvline(np.median(kv), color="C3", ls="--",
           label=f"median {np.median(kv):.2f}")
n_neg = np.sum(kv < 0)
ax.set_xlabel("slope (cycles/m, progress-referenced)")
ax.set_ylabel("# place cells")
ax.set_title(f"Precession slopes ({n_neg}/{len(kv)} negative)")
ax.legend()

ax = axes[1]
Rv = np.array([r_["R"] for r_ in results])
Rsh = np.array([r_["R_sh_med"] for r_ in results])
bins = np.linspace(0, max(Rv.max(), Rsh.max()), 30)
ax.hist(Rsh, bins=bins, color="0.7", label="shuffle (median)")
ax.hist(Rv, bins=bins, color="C0", alpha=0.7, label="data")
ax.set_xlabel("circular-linear correlation R")
ax.set_ylabel("# place cells")
ax.set_title("Correlation vs phase-permutation shuffle")
ax.legend()

ax = axes[2]
# pooled phase vs normalized field progress for significant precessing cells;
# each cell's constant phase offset (fitted phase at mid-field) is removed so
# the bands align across cells. Slopes in the data are untouched by this shift.
all_prog, all_phi = [], []
for r_ in sig_neg:
    xf, phif, prog = spike_cache[r_["unit"]]
    x_mid = 0.5 * (r_["lo"] + r_["hi"])
    phi_mid = r_["phi0"] + 2 * np.pi * r_["k"] * x_mid
    phi_al = (phif - phi_mid + np.pi) % (2 * np.pi)  # centered on pi at mid-field
    all_prog.append(prog)
    all_phi.append(phi_al)
all_prog = np.concatenate(all_prog)
all_phi = np.concatenate(all_phi)
hb = ax.hexbin(np.concatenate([all_prog, all_prog]),
               np.concatenate([all_phi - np.pi, all_phi + np.pi]),
               gridsize=25, cmap="viridis", mincnt=1, extent=[0, 1, -np.pi, 3 * np.pi])
# overlay circular-mean phase in progress bins
pbins = np.linspace(0, 1, 11)
pmid = 0.5 * (pbins[:-1] + pbins[1:])
cmean = []
for a, b in zip(pbins[:-1], pbins[1:]):
    m = (all_prog >= a) & (all_prog < b)
    cmean.append(np.angle(np.mean(np.exp(1j * all_phi[m]))))
cmean = np.array(cmean) - np.pi  # back to display offset
ax.plot(pmid, cmean, color="white", lw=2)
ax.plot(pmid, cmean + 2 * np.pi, color="white", lw=2)
ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi], ["-π", "0", "π", "2π", "3π"])
ax.set_xlabel("normalized field progress")
ax.set_ylabel("theta phase (rad, offset-removed)")
ax.set_title(f"Pooled precession ({len(sig_neg)} sig. cells)")
fig.colorbar(hb, ax=ax, label="# spikes")
fig.tight_layout()
fig.savefig("fig_precession_population.png", dpi=150)
print("saved fig_precession_population.png")

# ---------- figure: single passes for a strong cell ----------
# pick the highest-R significant cell that has >= 6 passes with >= 5 in-field spikes
def cell_pass_stats(r_):
    ep = ep_by_dir[r_["pref"]]
    spk = units[r_["unit"]].restrict(ep).index.values
    ip = np.searchsorted(t_pos, spk).clip(0, len(t_pos) - 1)
    x = lin[ip]
    ok = ~np.isnan(x)
    spk, x = spk[ok], x[ok]
    inf = (x >= r_["lo"]) & (x <= r_["hi"])
    spk_inf = spk[inf]
    ep_arr = ep.values
    n_qual = 0
    for s, e in ep_arr:
        if np.sum((spk_inf >= s) & (spk_inf <= e)) >= 5:
            n_qual += 1
    return n_qual

cand = sorted(sig_neg, key=lambda r_: r_["R"], reverse=True)
best = None
for r_ in cand[:15]:
    if cell_pass_stats(r_) >= 6:
        best = r_
        break
if best is None:
    best = cand[0]
u = best["unit"]
print(f"single-pass example: unit {u}, R={best['R']:.2f}, slope={best['k_prog']:.2f}")

ep = ep_by_dir[best["pref"]]
spk = units[u].restrict(ep).index.values
ip = np.searchsorted(t_pos, spk).clip(0, len(t_pos) - 1)
x_all = lin[ip]
il = np.searchsorted(t_lfp, spk).clip(0, len(t_lfp) - 1)
phi_all = phase[il]
ok = ~np.isnan(x_all)
spk, x_all, phi_all = spk[ok], x_all[ok], phi_all[ok]
infield = (x_all >= best["lo"]) & (x_all <= best["hi"])

# assign each spike to a run bout
ep_arr = ep.values
bout_idx = np.full(len(spk), -1)
for bi, (s, e) in enumerate(ep_arr):
    bout_idx[(spk >= s) & (spk <= e)] = bi

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
t_rel = spk[infield]
ax.plot(t_pos - t_pos[0], lin, color="0.8", lw=0.5)
ax.plot(t_rel - t_pos[0], x_all[infield], "o", color="C0", ms=3)
ax.set_xlim(0, 2068)
ax.set_xlabel("time in maze epoch (s)")
ax.set_ylabel("linearized position (m)")
ax.set_title(f"unit {u}: spikes on track ({best['pref']} dir runs)")

ax = axes[1]
cmap = plt.get_cmap("tab10")
bouts = np.unique(bout_idx[infield])
bouts = bouts[bouts >= 0]
n_plotted = 0
MAX_PASSES = 8
for bi in bouts:
    if n_plotted >= MAX_PASSES:
        break
    m = (bout_idx == bi) & infield
    if m.sum() >= 5:
        # order by position along travel direction and unwrap phase for continuity
        order = np.argsort(x_all[m]) if best["pref"] == "pos" else np.argsort(-x_all[m])
        xs = x_all[m][order]
        ph = np.unwrap(phi_all[m][order])
        ax.plot(xs, ph, "o-", color=cmap(n_plotted % 10), ms=4, lw=1,
                label=f"pass {n_plotted + 1}")
        n_plotted += 1
ax.axhline(np.pi, color="0.7", lw=0.5)
ax.axhline(0, color="0.7", lw=0.5)
ax.axhline(-np.pi, color="0.7", lw=0.5)
ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi], ["-2π", "-π", "0", "π"])
ax.set_xlabel("position on track (m)")
ax.set_ylabel("theta phase (rad, unwrapped per pass)")
ax.set_title(f"unit {u}: first {n_plotted} passes through the field")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig_single_pass.png", dpi=150)
print("saved fig_single_pass.png")
