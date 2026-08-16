# Compute peri-movement PETHs per direction bin; save rasters/PETHs for example units
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

with open("analysis_results.pkl", "rb") as f:
    R = pickle.load(f)
trials = R["trials"]

s3 = "https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
rf = remfile.File(s3, disk_cache=remfile.DiskCache("/tmp/remfile_cache_mcmaze"))
f5 = h5py.File(rf, "r")
io = NWBHDF5IO(file=f5)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]

onset_t = trials["move_onset_time"].values
PRE, POST, BIN = 0.4, 0.8, 0.01
nbins = int(round((PRE + POST) / BIN))
ep = nap.IntervalSet(start=onset_t - PRE, end=onset_t + POST)
counts = units.count(BIN, ep=ep)  # rows sequential per trial
print("counts shape:", counts.shape, "expected:", (len(onset_t) * nbins, len(units)))
C = counts.values.reshape(len(onset_t), nbins, len(units))  # (trials, time, units)
t_ax = np.linspace(-PRE + BIN / 2, POST - BIN / 2, nbins)

dir_bin = R["dir_bin"]
NBINS = 8
centers = R["bin_centers"]
populated = sorted(np.unique(dir_bin))
# colors per direction bin (hsv by angle)
bin_color = {b: plt.cm.hsv((centers[b] + np.pi) / (2 * np.pi)) for b in populated}

tun = R["tun_move"]; sig = R["sig_move"]
unit_ids = np.load("unit_ids.npy")

# example units from fig3
order = np.argsort(-tun["mod_depth"] * sig)
cands = [u for u in order if sig[u]]
chosen = []
for u in cands:
    if all(abs(np.angle(np.exp(1j * (tun["pd"][u] - tun["pd"][v])))) > 0.6 for v in chosen):
        chosen.append(u)
    if len(chosen) == 2:
        break

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 1.2], hspace=0.08))
for col, u in enumerate(chosen):
    ax_r = axes[0, col]
    ax_p = axes[1, col]
    # raster: trials sorted by direction bin, up to 25 per bin
    row = 0
    yticks_dir = []
    for b in populated:
        tr_idx = np.where(dir_bin == b)[0][:25]
        for tr in tr_idx:
            spk_bins = np.where(C[tr, :, u] > 0)[0]
            ax_r.scatter(t_ax[spk_bins], np.full_like(spk_bins, row, dtype=float),
                         s=1, color=bin_color[b], marker="|")
            row += 1
        yticks_dir.append(row - 1)
        ax_r.axhline(row - 0.5, color="gray", lw=0.3, alpha=0.5)
    ax_r.axvline(0, color="k", ls="--", lw=1)
    ax_r.set_ylabel("trials (by direction)")
    ax_r.set_title(f"unit {unit_ids[u]} (PD={np.degrees(tun['pd'][u]):.0f}°)", fontsize=10)
    ax_r.set_ylim(-0.5, row - 0.5)
    # PETH per direction bin
    for b in populated:
        tr_idx = np.where(dir_bin == b)[0]
        peth = C[tr_idx, :, u].mean(axis=0) / BIN
        peth = gaussian_filter1d(peth, sigma=3)
        ax_p.plot(t_ax, peth, color=bin_color[b], lw=1.2,
                  label=f"{int(round(np.degrees(centers[b])))}°")
    ax_p.axvline(0, color="k", ls="--", lw=1)
    ax_p.set_xlabel("time from move onset (s)")
    if col == 0:
        ax_p.set_ylabel("firing rate (Hz)")
    ax_p.legend(fontsize=7, title="direction", title_fontsize=7, ncol=2, loc="upper right")
fig.suptitle("Peri-movement spiking by reach direction (raster: 25 trials per direction bin)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig4_psth_by_direction.png", dpi=150)
print("saved fig4_psth_by_direction.png")

# population PETH: preferred vs anti-preferred direction (tuned units)
pop_pref, pop_anti = [], []
for u in np.where(sig)[0]:
    pd_u = tun["pd"][u]
    d_ang = np.angle(np.exp(1j * (trials["reach_dir"].values - pd_u)))
    pref_tr = np.abs(d_ang) < np.pi / 4
    anti_tr = np.abs(d_ang) > 3 * np.pi / 4
    if pref_tr.sum() < 5 or anti_tr.sum() < 5:
        continue
    p = gaussian_filter1d(C[pref_tr][:, :, u].mean(axis=0) / BIN, 3)
    a = gaussian_filter1d(C[anti_tr][:, :, u].mean(axis=0) / BIN, 3)
    base = C[:, : int(PRE / BIN) // 2, u].mean() / BIN  # pre-onset baseline
    pop_pref.append(p)
    pop_anti.append(a)
pop_pref = np.array(pop_pref)
pop_anti = np.array(pop_anti)
np.savez("peth_population.npz", t_ax=t_ax, pop_pref=pop_pref, pop_anti=pop_anti)
print("population PETH:", pop_pref.shape, pop_anti.shape)
print("saved peth_population.npz")
