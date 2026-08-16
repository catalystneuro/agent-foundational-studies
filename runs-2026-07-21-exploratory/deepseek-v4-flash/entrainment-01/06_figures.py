# %% [markdown]
# # 06 — Figures
# Full figure suite demonstrating theta phase entrainment: session overview
# with raw LFP + filtered theta + spikes + speed, theta PSD, population polar
# histograms, per-cell statistics (exc vs inh), example cells, and the
# shuffle-null control.

# %%
import numpy as np
import pandas as pd
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import stats as sstats
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"

def resolve_s3_url(asset_id):
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    r = requests.get(dl, allow_redirects=False, timeout=120)
    return r.headers["Location"]

def load_nwb():
    s3_url = resolve_s3_url(ASSET_ID)
    disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment01")
    rem_file = remfile.File(s3_url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    return nap.NWBFile(io.read())

pd_ = np.load("phase_data.npz", allow_pickle=False)
lfp_t, lfp_uV = pd_["lfp_t"], pd_["lfp_uV"]
theta_f = pd_["theta_f"]
phase = pd_["phase"]
run_epoch = nap.IntervalSet(start=pd_["run_start"], end=pd_["run_end"])
df = pd.read_csv("unit_stats.csv")
df_exc = df[df["cell_type"] == "excitatory"]
df_inh = df[df["cell_type"] == "inhibitory"]

d = np.load("theta_channel.npz", allow_pickle=False)
best_ch = int(d["best_ch"])
freqs, psd_best = d["freqs"], d["psd_selected"]

C_EXC, C_INH, C_TH = "#2b7bba", "#d62728", "#08519c"
C_RAW, C_SPD, C_NULL = "#9ecae1", "#666666", "#8c8c8c"

nwb = load_nwb()
units = nwb["units"].restrict(run_epoch)

pos = nwb["1.6mLinearMazeSpatialSeries"]
pt = np.asarray(pos.t)
px = np.asarray(pos["x"], dtype=float)
py = np.asarray(pos["y"], dtype=float)
v = np.zeros(len(pt))
v[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) / (pt[2:] - pt[:-2])
v = np.nan_to_num(v, nan=0.0)

# per-unit spike phases cached for all cells in df
spike_phases = {}
for uid in df["unit"]:
    spike_phases[uid] = np.interp(units[uid].t, lfp_t, phase)

# ====== Figure 1: session overview ======
fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.35, 0.5], hspace=0.32)

ax = fig.add_subplot(gs[0])
t0, t1 = 18550.0, 18552.5
m = (lfp_t >= t0) & (lfp_t <= t1)
ax.plot(lfp_t[m], lfp_uV[m], color=C_RAW, lw=0.7, label="Raw LFP (ch %d)" % best_ch)
ax.plot(lfp_t[m], theta_f[m], color=C_TH, lw=1.2, label="6-11 Hz filter")
ax.set_ylabel("LFP (µV)")
ax.set_title("Raw LFP bandpass-filtered to the theta band during running")
ax.legend(loc="upper right", frameon=False)
ax.set_xticks([])

ax = fig.add_subplot(gs[1])
t0, t1 = 18540.0, 18570.0
m = (lfp_t >= t0) & (lfp_t <= t1)
th_n = theta_f[m] / (4 * np.percentile(np.abs(theta_f[m]), 90))
ax.plot(lfp_t[m], th_n - 0.5, color=C_TH, lw=0.8, label="theta (normalized)")
for s_, e_ in zip(np.asarray(run_epoch.start), np.asarray(run_epoch.end)):
    if e_ >= t0 and s_ <= t1:
        ax.axvspan(max(s_, t0), min(e_, t1), color="#f2f0f5", zorder=0)
ids = df_exc.sort_values("M", ascending=False)["unit"].iloc[:15]
for j, uid in enumerate(ids):
    st = units[uid].t
    st = st[(st >= t0) & (st <= t1)]
    ax.scatter(st, np.full(len(st), j), s=1.0, color=C_EXC, linewidths=0)
ax.set_ylim(-1.4, 15)
ax.set_ylabel("15 most phase-locked\npyramidal units (top MRL)")
ax.set_yticks([])
ax.legend(loc="upper right", frameon=False)

ax = fig.add_subplot(gs[2])
spd = np.interp(lfp_t[m], pt, v)
ax.plot(lfp_t[m], spd, color=C_SPD, lw=0.9)
ax.set_xlabel("Session time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_ylim(0, max(spd) * 1.35)

fig.suptitle("Hippocampal theta rhythm and entrained unit firing, session 'Achilles-10252013'")
plt.savefig("fig01_lfp_theta_overview.png")
plt.close(fig)

# ====== Figure 2: theta PSD ======
fig, ax = plt.subplots(figsize=(7, 4))
m = (freqs >= 0.5) & (freqs <= 40)
ax.semilogy(freqs[m], psd_best[m], color="#333", lw=1.1)
ax.axvspan(6, 11, color=C_TH, alpha=0.15, label="theta band (6-11 Hz)")
pk = freqs[np.argmax(psd_best[(freqs >= 4) & (freqs <= 14)])]
ax.axvline(pk, color=C_INH, ls="--", lw=1, label="peak = %.1f Hz" % pk)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (µV²/Hz)")
ax.set_title("Power spectral density, reference electrode ch %d" % best_ch)
ax.legend(frameon=False)
plt.savefig("fig02_theta_psd.png")
plt.close(fig)

# ====== Figure 3: population phase distributions (polar) ======
def cell_weighted_hist(phases, bins):
    "Average per-cell (density-normalized) phase histograms."
    dens = np.zeros(len(bins) - 1)
    for ph in phases:
        h, _ = np.histogram(ph, bins=bins, density=True)
        dens += h
    return dens / len(phases)

sig_exc = df_exc[df_exc["p_rayleigh"] < 0.01]
sig_inh = df_inh[df_inh["p_rayleigh"] < 0.01]
bins = np.linspace(-np.pi, np.pi, 37)
d_exc = cell_weighted_hist([spike_phases[u] for u in sig_exc["unit"]], bins)
d_inh = cell_weighted_hist([spike_phases[u] for u in sig_inh["unit"]], bins)
centers = (bins[:-1] + bins[1:]) / 2

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), subplot_kw={"projection": "polar"})
for ax, dd, color, n in zip(axes, [d_exc, d_inh], [C_EXC, C_INH], [len(sig_exc), len(sig_inh)]):
    th = np.concatenate([centers, [centers[0]]])
    rr = np.concatenate([dd, [dd[0]]])
    ax.plot(th, rr, color=color, lw=1.3)
    ax.fill(th, rr, color=color, alpha=0.3)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0°", "90°", "180°\ntrough", "270°"], fontsize=8)
    ax.set_title("Significant cells, n=%d" % n, pad=18)
    ax.set_yticks([])

ax = axes[2]
for u in sig_exc["unit"]:
    mu = df[df["unit"] == u]["mu"].iloc[0]
    ax.plot([mu, mu], [0, 1], color=C_EXC, alpha=0.3, lw=0.8)
for u in sig_inh["unit"]:
    mu = df[df["unit"] == u]["mu"].iloc[0]
    ax.plot([mu, mu], [0, 1], color=C_INH, alpha=0.3, lw=0.8)
for color, sub in zip([C_EXC, C_INH], [sig_exc, sig_inh]):
    mu_p = np.arctan2(np.sin(sub["mu"]).mean(), np.cos(sub["mu"]).mean())
    ax.plot([mu_p, mu_p], [0, 0.9], color=color, lw=3)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels(["0°", "90°", "180°\ntrough", "270°"], fontsize=8)
ax.set_title("Single-cell preferred\nphase vectors", fontsize=11)
ax.set_yticks([])

fig.suptitle("Theta phase of spikes during running (0° = theta peak at the LFP)")
plt.tight_layout()
plt.savefig("fig03_population_phase.png")
plt.close(fig)

# ====== Figure 4: per-cell stats exc vs inh ======
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
mv = [df_exc["M"].values, df_inh["M"].values]
bp = ax[0].boxplot(mv, labels=["excitatory", "inhibitory"], widths=0.5,
                   patch_artist=True, medianprops=dict(color="k"))
for patch, col in zip(bp["boxes"], [C_EXC, C_INH]):
    patch.set_facecolor(col)
    patch.set_alpha(0.65)
o, pv = sstats.mannwhitneyu(df_exc["M"], df_inh["M"])
ax[0].text(0.5, 1.0, "MW U p = %.2e" % pv, ha="center",
           transform=ax[0].transAxes, fontweight="bold")
ax[0].set_ylabel("|M| mean resultant length")
ax[0].set_title("Phase-locking strength by cell type")

ax[1].scatter(df_exc["M"], np.degrees(df_exc["mu"]) % 360, s=9, color=C_EXC, alpha=0.6, label="excitatory")
ax[1].scatter(df_inh["M"], np.degrees(df_inh["mu"]) % 360, s=9, color=C_INH, alpha=0.7, label="inhibitory")
ax[1].set_ylabel("Preferred theta phase (°)")
ax[1].set_xlabel("|M|")
ax[1].set_title("Preferred phase vs locking strength")
ax[1].legend(frameon=False, loc="upper left")
fig.suptitle("Cell-level theta entrainment (n exc=%d, n inh=%d)" % (len(df_exc), len(df_inh)))
plt.tight_layout()
plt.savefig("fig04_cell_phase_stats.png")
plt.close(fig)

# ====== Figure 5: example cells ======
ex_exc = df_exc.sort_values("M", ascending=False).iloc[[0, 6, 20]]
ex_inh = df_inh.sort_values("M", ascending=False).iloc[:3]
example = pd.concat([ex_exc, ex_inh])
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
for ax_, (_, row) in zip(axes.ravel(), example.iterrows()):
    col = C_INH if row["cell_type"] == "inhibitory" else C_EXC
    th_d = np.degrees(spike_phases[row["unit"]]) % 360
    ax_.hist(th_d, bins=30, range=(0, 360), color=col, alpha=0.8)
    ax_.axvline(np.degrees(row["mu"]) % 360, color="k", ls="--", lw=1)
    ax_.set_xticks([0, 90, 180, 270, 360])
    ax_.set_xticklabels(["0", "90", "180", "270", "360"])
    ax_.set_xlabel("theta phase (°)")
    ax_.set_title("%s unit %d\nn=%d, R=%.2f, p=%.1e" %
                  (row["cell_type"].title(), row["unit"], row["n_spk"],
                   row["M"], row["p_rayleigh"]), fontsize=9)
fig.suptitle("Example cells: spike-count histograms vs theta phase")
fig.tight_layout()
plt.savefig("fig05_example_cells.png")
plt.close(fig)

# ====== Figure 6: shuffle null control ======
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
ax[0].hist(df["M_real_same_n"], bins=30, color=C_EXC, alpha=0.6,
           label="real (%d cells)" % len(df))
ax[0].hist(df["M_null_median"], bins=30, color=C_NULL, alpha=0.7,
           label="shuffle null (median)")
ax[0].set_xlabel("MRL |M|")
ax[0].set_ylabel("cells")
ax[0].set_title("Real phase-locking vs random-time shuffle null")
ax[0].legend(frameon=False)
ax[1].scatter(df["M_null_median"], df["M_real_same_n"], s=9, color="#444", alpha=0.6)
ax[1].plot([0, max(df["M_null_median"].max(), 0.05)], [0, max(df["M_null_median"].max(), 0.05)], "k--", lw=1)
n_sig = int((df["p_shuffle"] < 0.01).sum())
ax[1].set_xlabel("null M (median)")
ax[1].set_ylabel("real M")
ax[1].set_title("Paired real vs null (shuffle-significant: %d/%d)" % (n_sig, len(df)))
fig.suptitle("Control: phase-locking exceeds chance from random firing times")
plt.tight_layout()
plt.savefig("fig06_null_control.png")
plt.close(fig)

print("All figures saved.")