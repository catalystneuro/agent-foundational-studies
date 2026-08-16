# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# This notebook demonstrates **theta phase precession**, the phenomenon in which
# hippocampal place cells fire at progressively earlier phases of the theta
# oscillation as an animal runs through the cell's place field
# (O'Keefe & Recce, 1993; Skaggs et al., 1996).
#
# **Dataset**: DANDI Archive dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural firing
# dynamics supports both rigid and learned hippocampal sequences", Buzsáki lab),
# session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: rat CA1
# tetrode recordings (137 sorted units) with 128-channel LFP at 1250 Hz while the
# animal runs on a 1.6 m linear maze.
#
# **Approach**:
# 1. Stream the NWB file from DANDI with `remfile` (no full download).
# 2. Pick a theta reference LFP channel data-driven (theta/delta PSD ratio).
# 3. Bandpass-filter the LFP at 6-12 Hz and extract instantaneous phase (Hilbert).
# 4. Compute per-direction place fields on the linearized track coordinate and
#    identify place cells.
# 5. For each place cell, regress spike theta phase on position within the field
#    (circular-linear regression; Kempter et al., 2012) and test against
#    phase-permutation shuffles.

# %% [markdown]
# ## Setup and streaming data access
#
# The DANDI `/download/` endpoint redirects to a presigned S3 URL; we capture the
# redirect target and hand it to `remfile` with a local disk cache, so only the
# data chunks we actually read are fetched.

# %%
import numpy as np
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng(42)

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

api = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
r = requests.get(api, params={"path": ASSET_PATH})
r.raise_for_status()
asset = r.json()["results"][0]
print("asset:", asset["asset_id"], f"({asset['size']/1e9:.2f} GB)")

dl = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
s3_url = requests.get(dl, allow_redirects=False).headers["Location"]

disk_cache = remfile.DiskCache("/tmp/remfile_cache_precession")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
print(nwb)

units = nwb["units"]
epochs = nwb["epochs"]
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin_ts = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print(epochs)

# %% [markdown]
# ## Theta reference channel selection
#
# The electrodes table has no anatomical coordinates, so the theta reference
# channel is chosen data-driven: on a 100 s chunk of the maze epoch we compute
# the Welch PSD of every channel and pick the one with the largest
# theta (6-12 Hz) / delta (1-4 Hz) power ratio.

# %%
FS = 1250.0
chunk = nwb["LFP"].get(18500.0, 18600.0)
data = np.asarray(chunk.values, dtype=np.float64)

f, psd = signal.welch(data, fs=FS, nperseg=4096, axis=0)
theta_band = (f >= 6) & (f <= 12)
delta_band = (f >= 1) & (f <= 4)
ratio = psd[theta_band].mean(axis=0) / psd[delta_band].mean(axis=0)
best_ch = int(np.argmax(ratio))
theta_f = f[theta_band]
peak_f = theta_f[np.argmax(psd[theta_band, best_ch])]
print(f"theta reference channel: {best_ch} (theta/delta = {ratio[best_ch]:.2f}, "
      f"peak {peak_f:.2f} Hz)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(np.arange(128), ratio, color="k")
axes[0].axvline(best_ch, color="r", ls="--", label=f"ch {best_ch}")
axes[0].set_xlabel("LFP channel")
axes[0].set_ylabel("theta (6-12 Hz) / delta (1-4 Hz)")
axes[0].set_title("Theta/delta ratio per channel")
axes[0].legend()
axes[1].semilogy(f, psd[:, best_ch], color="k")
axes[1].set_xlim(0, 40)
axes[1].axvspan(6, 12, color="r", alpha=0.15, label="theta band")
axes[1].axvline(peak_f, color="r", ls="--", label=f"peak {peak_f:.2f} Hz")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD")
axes[1].set_title(f"PSD of channel {best_ch} (maze epoch chunk)")
axes[1].legend()
fig.tight_layout()
fig.savefig("fig_channel_selection.png", dpi=150)

# %% [markdown]
# ## Theta phase extraction
#
# We load the full maze epoch (18079.5-20147 s) for the reference channel only,
# bandpass-filter at 6-12 Hz (zero-phase Butterworth) and take the angle of the
# Hilbert transform as the instantaneous theta phase (0 = peak of the filtered
# waveform).

# %%
t0, t1, pad = 18079.5, 20147.0, 2.0
es_data = h5py_file["processing/ecephys/LFP/LFP/data"]
i0, i1 = int((t0 - pad) * FS), int((t1 + pad) * FS)
t_lfp = np.arange(i0, i1) / FS
x = es_data[i0:i1, best_ch].astype(np.float64) * 3.815e-7  # int16 -> volts

b, a = signal.butter(4, [6, 12], btype="band", fs=FS)
x_filt = signal.filtfilt(b, a, x)
phase = np.angle(signal.hilbert(x_filt))

valid = (t_lfp >= t0) & (t_lfp <= t1)
t_lfp, x, x_filt, phase = t_lfp[valid], x[valid], x_filt[valid], phase[valid]
print(f"maze-epoch LFP: {len(t_lfp)} samples ({t_lfp[-1]-t_lfp[0]:.1f} s)")

# validation: 2 s of raw/filtered LFP, spike raster, and phase
win_t0 = 19000.0
win = (t_lfp >= win_t0) & (t_lfp <= win_t0 + 2.0)
tt = t_lfp[win]
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw=dict(height_ratios=[2, 2, 1]))
axes[0].plot(tt, x[win] * 1e6, color="0.5", lw=0.6, label="raw LFP")
axes[0].plot(tt, x_filt[win] * 1e6, color="k", lw=1.2, label="theta-filtered (6-12 Hz)")
axes[0].set_ylabel("LFP (µV)")
axes[0].legend(loc="upper right")
axes[0].set_title(f"Theta extraction, LFP channel {best_ch}")
keys = list(units.keys())[:20]
for i, k in enumerate(keys):
    spk = units[k].restrict(nap.IntervalSet(win_t0, win_t0 + 2.0)).index.values
    axes[1].vlines(spk, i + 0.5, i + 1.5, color="C0", lw=0.8)
axes[1].set_ylabel("unit #")
axes[1].set_ylim(0.5, len(keys) + 0.5)
axes[1].set_title("Example unit spike rasters (first 20 units)")
axes[2].plot(tt, phase[win], color="darkred", lw=0.8)
axes[2].set_ylabel("theta phase (rad)")
axes[2].set_xlabel("time (s)")
axes[2].set_yticks([-np.pi, 0, np.pi], ["-π", "0", "π"])
fig.tight_layout()
fig.savefig("fig_theta_extraction.png", dpi=150)

# %% [markdown]
# ## Run epochs and place fields
#
# Speed is computed from the smoothed 2D position. Run epochs are bouts with
# speed > 10 cm/s where the linearized track coordinate is defined (the animal
# is on the track arm), split by running direction (sign of x velocity).
# Tuning curves (50 bins over the 1.6 m track, 1.5-bin Gaussian smoothing) are
# computed separately for each direction with pynapple. Excitatory units with a
# peak rate >= 1 Hz in either direction are kept as place cells; the place field
# is the contiguous region around the peak above 20% of the peak rate.

# %%
t_pos = pos2d.index.values
xy = np.asarray(pos2d.values)
lin = np.asarray(lin_ts.values)[:, 0]
dt = np.median(np.diff(t_pos))
print(f"position: {len(t_pos)} samples at {1/dt:.1f} Hz; "
      f"linearized valid {np.mean(~np.isnan(lin))*100:.1f}% of maze epoch")

# interpolate over NaN gaps for the speed estimate only
xyi = xy.copy()
for d in range(2):
    x1 = xy[:, d]
    x1[np.isnan(x1)] = np.interp(t_pos[np.isnan(x1)], t_pos[~np.isnan(x1)],
                                 x1[~np.isnan(x1)])
    xyi[:, d] = x1
kern = signal.windows.gaussian(int(0.5 / dt) // 2 * 2 + 1, std=int(0.25 / dt))
kern /= kern.sum()
vx = signal.convolve(np.gradient(xyi[:, 0], dt), kern, mode="same")
vy = signal.convolve(np.gradient(xyi[:, 1], dt), kern, mode="same")
speed = np.sqrt(vx**2 + vy**2)
direction = np.sign(vx)

def mask_to_epochs(mask, t, min_dur=0.5, max_gap=0.5):
    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[splits + 1]])
    ends = np.concatenate([idx[splits], [idx[-1]]])
    eps = np.stack([t[starts], t[ends]], axis=1)
    merged = [eps[0]]
    for e in eps[1:]:
        if e[0] - merged[-1][1] < max_gap:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    eps = np.array(merged)
    return eps[(eps[:, 1] - eps[:, 0]) >= min_dur]

run_eps_arr = mask_to_epochs((speed > 0.10) & (~np.isnan(lin)), t_pos)
print(f"run epochs: {len(run_eps_arr)} bouts, "
      f"{np.sum(run_eps_arr[:, 1] - run_eps_arr[:, 0]):.1f} s total")

dir_epochs = {+1: [], -1: []}
for s, e in run_eps_arr:
    m = (t_pos >= s) & (t_pos <= e)
    frac_pos = np.mean(direction[m] > 0)
    if frac_pos >= 0.8:
        dir_epochs[+1].append([s, e])
    elif frac_pos <= 0.2:
        dir_epochs[-1].append([s, e])
for d in dir_epochs:
    arr = np.array(dir_epochs[d])
    dir_epochs[d] = nap.IntervalSet(arr[:, 0], arr[:, 1])
    print(f"direction {d:+d}: {len(arr)} bouts, {np.sum(arr[:, 1] - arr[:, 0]):.1f} s")

NBINS = 50
exc_keys = [k for k in units.keys() if units.get_info("cell_type")[k] == "excitatory"]
lin_tsd = nap.Tsd(t=t_pos, d=lin)
tuning = {}
for d, lab in [(+1, "pos"), (-1, "neg")]:
    tc = nap.compute_tuning_curves(units[exc_keys], lin_tsd, bins=NBINS,
                                   range=(0, 1.6), epochs=dir_epochs[d])
    bin_centers = tc.coords[tc.dims[1]].values
    sm = signal.windows.gaussian(9, std=1.5)
    sm /= sm.sum()
    tc_smooth = np.apply_along_axis(lambda r_: signal.convolve(r_, sm, mode="same"),
                                    1, tc.values)
    tuning[lab] = dict(bins=bin_centers, smooth=tc_smooth)

place_info = {}
for i, k in enumerate(exc_keys):
    pk_pos = tuning["pos"]["smooth"][i, :].max()
    pk_neg = tuning["neg"]["smooth"][i, :].max()
    pref = "pos" if pk_pos >= pk_neg else "neg"
    peak = max(pk_pos, pk_neg)
    if peak >= 1.0:
        tc_pref = tuning[pref]["smooth"][i, :]
        peak_bin = int(np.argmax(tc_pref))
        above = tc_pref >= 0.2 * peak
        lo = peak_bin
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = peak_bin
        while hi < NBINS - 1 and above[hi + 1]:
            hi += 1
        place_info[k] = dict(pref_dir=pref, peak_rate=peak,
                             peak_pos=bin_centers[peak_bin],
                             field_lo=bin_centers[lo], field_hi=bin_centers[hi])
print(f"place cells (peak >= 1 Hz): {len(place_info)} / {len(exc_keys)} excitatory units")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
pkeys = np.array(list(place_info.keys()))
for ax, lab, title in [(axes[0], "pos", "+ direction runs"),
                       (axes[1], "neg", "- direction runs")]:
    peaks = [bin_centers[np.argmax(tuning[lab]["smooth"][exc_keys.index(k), :])]
             for k in pkeys]
    mat = []
    for k in pkeys[np.argsort(peaks)]:
        r_ = tuning[lab]["smooth"][exc_keys.index(k), :]
        mat.append(r_ / r_.max() if r_.max() > 0 else r_ * 0)
    im = ax.imshow(np.array(mat), aspect="auto", cmap="viridis",
                   extent=[0, 1.6, len(pkeys), 0])
    ax.set_xlabel("linearized position (m)")
    ax.set_ylabel("place cell (sorted)")
    ax.set_title(f"Rate maps, {title}")
    fig.colorbar(im, ax=ax, label="norm. rate")
snip = (t_pos >= 19000) & (t_pos <= 19060)
ax = axes[2]
ax.plot(t_pos[snip] - 19000, lin[snip], color="k", lw=1)
ax.set_xlabel("time (s)")
ax.set_ylabel("linearized position (m)")
ax2 = ax.twinx()
ax2.plot(t_pos[snip] - 19000, speed[snip] * 100, color="C3", lw=0.8, alpha=0.7)
ax2.set_ylabel("speed (cm/s)", color="C3")
ax2.axhline(10, color="C3", ls=":", lw=0.8)
ax.set_title("Position and speed (60 s snippet)")
fig.tight_layout()
fig.savefig("fig_place_fields.png", dpi=150)

# %% [markdown]
# ## Theta phase precession
#
# For each place cell we take spikes fired during runs in the cell's preferred
# direction, look up the linearized position and the theta phase at each spike
# time, and keep spikes inside the place field (>= 30 spikes required).
#
# The phase-position relationship is quantified by **circular-linear
# regression**: the slope k (cycles/m) is the value maximizing the resultant
# length R(k) = |mean(exp(i(phi - 2 pi k x)))| over a grid, and R at the best
# slope is the circular-linear correlation (Kempter et al., 2012). Significance
# is tested against 500 shuffles in which the phases are permuted across the
# cell's spikes. Slopes are referenced to field progress (negative = phase
# advances to earlier values as the animal runs through the field), so cells
# recorded in either running direction are directly comparable.

# %%
K_GRID = np.linspace(-6, 6, 241)  # cycles per meter
N_SHUFF = 500
MIN_SPIKES = 30

def circ_lin_fit(x, phi):
    E = np.exp(1j * (phi[None, :] - 2 * np.pi * K_GRID[:, None] * x[None, :]))
    R = np.abs(E.mean(axis=1))
    i = int(np.argmax(R))
    return K_GRID[i], R[i], np.angle(E[i].mean())

results = []
spike_cache = {}
for k in tqdm(place_info, desc="cells"):
    info = place_info[k]
    pref, lo, hi = info["pref_dir"], info["field_lo"], info["field_hi"]
    ep = dir_epochs[+1 if pref == "pos" else -1]
    spk = units[int(k)].restrict(ep).index.values
    if len(spk) < MIN_SPIKES:
        continue
    ip = np.searchsorted(t_pos, spk).clip(0, len(t_pos) - 1)
    xsp = lin[ip]
    il = np.searchsorted(t_lfp, spk).clip(0, len(t_lfp) - 1)
    phsp = phase[il]
    ok = ~np.isnan(xsp)
    xsp, phsp = xsp[ok], phsp[ok]
    infield = (xsp >= lo) & (xsp <= hi)
    xf, phif = xsp[infield], phsp[infield]
    if len(xf) < MIN_SPIKES:
        continue

    k_best, R_best, phi0 = circ_lin_fit(xf, phif)
    R_sh = np.empty(N_SHUFF)
    for s in range(N_SHUFF):
        phi_s = phif[rng.permutation(len(phif))]
        E = np.exp(1j * (phi_s[None, :] - 2 * np.pi * K_GRID[:, None] * xf[None, :]))
        R_sh[s] = np.abs(E.mean(axis=1)).max()
    p_val = (np.sum(R_sh >= R_best) + 1) / (N_SHUFF + 1)

    sign = 1.0 if pref == "pos" else -1.0
    prog = (xf - lo) / (hi - lo) if pref == "pos" else (hi - xf) / (hi - lo)
    results.append(dict(unit=int(k), pref=pref, lo=lo, hi=hi,
                        n_spikes=len(xf), k=k_best, k_prog=k_best * sign,
                        R=R_best, phi0=phi0, p=p_val,
                        R_sh_med=np.median(R_sh)))
    spike_cache[int(k)] = (xf, phif, prog)

n_sig = sum(1 for r_ in results if r_["p"] < 0.05)
sig_neg = [r_ for r_ in results if r_["p"] < 0.05 and r_["k_prog"] < 0]
kv = np.array([r_["k_prog"] for r_ in results])
print(f"\ncells analyzed: {len(results)}")
print(f"significant circular-linear correlation (p<0.05): {n_sig}/{len(results)}")
print(f"  of which negative (precessing) slope: {len(sig_neg)}")
print(f"negative slopes overall: {np.sum(kv < 0)}/{len(kv)} "
      f"(binomial p = {binomtest(int(np.sum(kv < 0)), len(kv)).pvalue:.2e})")
print(f"median progress-referenced slope: {np.median(kv):.2f} cycles/m")
print(f"median R: {np.median([r_['R'] for r_ in results]):.3f} vs "
      f"shuffle {np.median([r_['R_sh_med'] for r_ in results]):.3f}")

# %% [markdown]
# ### Example cells
#
# Spike theta phase versus track position for the six strongest significant
# cells (phase plotted over two cycles; black line = circular-linear fit).
# Cells from the negative running direction slope upward in track coordinates,
# which is the same physical precession seen from the other direction; the
# titles report the progress-referenced slope.

# %%
sig_neg.sort(key=lambda r_: r_["R"], reverse=True)
examples = sig_neg[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, r_ in zip(axes.flat, examples):
    xf, phif, prog = spike_cache[r_["unit"]]
    ax.scatter(xf, phif, s=6, color="C0", alpha=0.6)
    ax.scatter(xf, phif + 2 * np.pi, s=6, color="C0", alpha=0.6)
    xg = np.linspace(r_["lo"], r_["hi"], 50)
    yg = r_["phi0"] + 2 * np.pi * r_["k"] * xg
    yg_wr = (yg + np.pi) % (2 * np.pi) - np.pi
    for off in (0, 2 * np.pi):
        yline = yg_wr + off
        jumps = np.where(np.abs(np.diff(yline)) > np.pi)[0]
        xs = np.insert(xg, jumps + 1, np.nan)
        ys = np.insert(yline, jumps + 1, np.nan)
        ax.plot(xs, ys, color="k", lw=1.5)
    ax.set_ylim(-np.pi, 3 * np.pi)
    ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi],
                  ["-π", "0", "π", "2π", "3π"])
    ax.set_xlabel("position on track (m)")
    ax.set_ylabel("theta phase (rad)")
    ax.set_title(f"unit {r_['unit']} ({r_['pref']} dir): slope {r_['k_prog']:.2f} cyc/m, "
                 f"R={r_['R']:.2f}, p={r_['p']:.3f}", fontsize=9)
fig.suptitle("Theta phase precession, example place cells", y=0.995)
fig.tight_layout()
fig.savefig("fig_precession_examples.png", dpi=150)

# %% [markdown]
# ### Single passes
#
# Phase precession is visible on individual traversals of the field, not just in
# the pooled scatter. For one strong cell we show every qualifying pass with
# phase unwrapped within the pass.

# %%
def cell_pass_stats(r_):
    ep = dir_epochs[+1 if r_["pref"] == "pos" else -1]
    spk = units[r_["unit"]].restrict(ep).index.values
    ip = np.searchsorted(t_pos, spk).clip(0, len(t_pos) - 1)
    xsp = lin[ip]
    ok = ~np.isnan(xsp)
    spk_inf = spk[ok][(xsp[ok] >= r_["lo"]) & (xsp[ok] <= r_["hi"])]
    return sum(1 for s, e in ep.values
               if np.sum((spk_inf >= s) & (spk_inf <= e)) >= 5)

best = None
for r_ in sig_neg[:15]:
    if cell_pass_stats(r_) >= 6:
        best = r_
        break
if best is None:
    best = sig_neg[0]
u = best["unit"]
print(f"single-pass example: unit {u}, R={best['R']:.2f}, slope={best['k_prog']:.2f} cyc/m")

ep = dir_epochs[+1 if best["pref"] == "pos" else -1]
spk = units[u].restrict(ep).index.values
ip = np.searchsorted(t_pos, spk).clip(0, len(t_pos) - 1)
x_all = lin[ip]
il = np.searchsorted(t_lfp, spk).clip(0, len(t_lfp) - 1)
phi_all = phase[il]
ok = ~np.isnan(x_all)
spk, x_all, phi_all = spk[ok], x_all[ok], phi_all[ok]
infield = (x_all >= best["lo"]) & (x_all <= best["hi"])
bout_idx = np.full(len(spk), -1)
for bi, (s, e) in enumerate(ep.values):
    bout_idx[(spk >= s) & (spk <= e)] = bi

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(t_pos - t_pos[0], lin, color="0.8", lw=0.5)
ax.plot(spk[infield] - t_pos[0], x_all[infield], "o", color="C0", ms=3)
ax.set_xlim(0, 2068)
ax.set_xlabel("time in maze epoch (s)")
ax.set_ylabel("linearized position (m)")
ax.set_title(f"unit {u}: spikes on track ({best['pref']} dir runs)")

ax = axes[1]
cmap = plt.get_cmap("tab10")
n_plotted = 0
for bi in np.unique(bout_idx[infield]):
    if bi < 0 or n_plotted >= 8:
        continue
    m = (bout_idx == bi) & infield
    if m.sum() >= 5:
        order = (np.argsort(x_all[m]) if best["pref"] == "pos"
                 else np.argsort(-x_all[m]))
        ax.plot(x_all[m][order], np.unwrap(phi_all[m][order]), "o-",
                color=cmap(n_plotted % 10), ms=4, lw=1, label=f"pass {n_plotted + 1}")
        n_plotted += 1
for y in (np.pi, 0, -np.pi):
    ax.axhline(y, color="0.7", lw=0.5)
ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi], ["-2π", "-π", "0", "π"])
ax.set_xlabel("position on track (m)")
ax.set_ylabel("theta phase (rad, unwrapped per pass)")
ax.set_title(f"unit {u}: first {n_plotted} passes through the field")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig_single_pass.png", dpi=150)

# %% [markdown]
# ### Population summary
#
# Left: distribution of progress-referenced slopes across all analyzed place
# cells. Middle: circular-linear correlation R versus the per-cell shuffle
# median. Right: pooled phase versus normalized field progress for all
# significant precessing cells, after removing each cell's constant phase
# offset (the fitted phase at mid-field); the white line is the circular mean
# phase in progress bins. The band descends from late to early phases as the
# animal advances through the field: theta phase precession.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
ax.hist(kv, bins=30, color="0.4")
ax.axvline(0, color="k", lw=1)
ax.axvline(np.median(kv), color="C3", ls="--", label=f"median {np.median(kv):.2f}")
ax.set_xlabel("slope (cycles/m, progress-referenced)")
ax.set_ylabel("# place cells")
ax.set_title(f"Precession slopes ({np.sum(kv < 0)}/{len(kv)} negative)")
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
all_prog, all_phi = [], []
for r_ in sig_neg:
    xf, phif, prog = spike_cache[r_["unit"]]
    phi_mid = r_["phi0"] + 2 * np.pi * r_["k"] * 0.5 * (r_["lo"] + r_["hi"])
    phi_al = (phif - phi_mid + np.pi) % (2 * np.pi)
    all_prog.append(prog)
    all_phi.append(phi_al)
all_prog = np.concatenate(all_prog)
all_phi = np.concatenate(all_phi)
hb = ax.hexbin(np.concatenate([all_prog, all_prog]),
               np.concatenate([all_phi - np.pi, all_phi + np.pi]),
               gridsize=25, cmap="viridis", mincnt=1, extent=[0, 1, -np.pi, 3 * np.pi])
pbins = np.linspace(0, 1, 11)
pmid = 0.5 * (pbins[:-1] + pbins[1:])
cmean = np.array([np.angle(np.mean(np.exp(1j * all_phi[(all_prog >= a) & (all_prog < b)])))
                  for a, b in zip(pbins[:-1], pbins[1:])]) - np.pi
ax.plot(pmid, cmean, color="white", lw=2)
ax.plot(pmid, cmean + 2 * np.pi, color="white", lw=2)
ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi], ["-π", "0", "π", "2π", "3π"])
ax.set_xlabel("normalized field progress")
ax.set_ylabel("theta phase (rad, offset-removed)")
ax.set_title(f"Pooled precession ({len(sig_neg)} sig. cells)")
fig.colorbar(hb, ax=ax, label="# spikes")
fig.tight_layout()
fig.savefig("fig_precession_population.png", dpi=150)

# %% [markdown]
# ## Summary
#
# In the Achilles linear-track session (DANDI 000044), CA1 place cells show
# robust theta phase precession: the large majority of place cells with enough
# in-field spikes have a significant negative circular-linear correlation
# between theta phase and position, the median slope is about -1.1 cycles/m,
# and the effect is visible on single passes through the field. This replicates
# the classic finding of O'Keefe & Recce (1993) directly from publicly
# archived data streamed from the DANDI Archive.
