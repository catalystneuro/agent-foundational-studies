# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# **Phenomenon.** As a rat runs through a place cell's spatial field, the cell's spikes
# occur at progressively earlier phases of the ongoing theta oscillation (6-12 Hz) of the
# hippocampal local field potential (LFP). This "theta phase precession" (O'Keefe & Recce,
# 1993) compresses the spatial trajectory into theta-cycle-time-scale spike sequences and
# is one of the canonical temporal codes of the hippocampus.
#
# **Dataset.** DANDI Archive dandiset
# [000044](https://dandiarchive.org/dandiset/000044) (Buzsaki lab, "Diversity in neural
# firing dynamics..."), session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`:
# rat CA1 tetrode recordings (137 sorted units) with 128-channel LFP at 1250 Hz and
# 2D/linearized position tracking on a 1.6 m linear maze (~42 laps in a 34.5 min epoch).
#
# **Approach.**
# 1. Stream the NWB file with remfile (no full download).
# 2. Extract run bouts on the linear track from the linearized position signal.
# 3. Extract theta phase from the LFP (bandpass 6-12 Hz + Hilbert transform) on the
#    reference channel with the strongest theta rhythm.
# 4. Detect direction-specific place fields (smoothed rate maps + Skaggs spatial
#    information with a circular time-shift shuffle).
# 5. For each place cell and run direction, correlate spike theta phase with travel
#    progress through the field using the circular-linear correlation of Kempter et al.
#    (2012), with the sign and slope from a circular-linear regression; significance from
#    a phase-permutation shuffle. Repeat per single traversal.
#
# All figures are saved as PNG. Runtime is dominated by the place-field shuffle (~3 min).

# %% [markdown]
# ## Setup and streaming data access

# %%
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from tqdm import tqdm

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
CACHE_DIR = "/tmp/remfile_cache"
THETA_CH = 117          # strongest-theta LFP channel (verified by PSD below)
LFP_FS = 1250.0
THETA_BAND = (6, 12)
TWO_PI = 2 * np.pi
RNG = np.random.default_rng(7)

# Resolve the asset's presigned S3 URL through the DANDI API. The /download/ endpoint
# 403s on HEAD, so grab the Location header from a GET without following redirects.
api_url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
r = requests.get(api_url, params={"path": ASSET_PATH})
r.raise_for_status()
asset = r.json()["results"][0]
dl = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/"
      f"assets/{asset['asset_id']}/download/")
s3_url = requests.get(dl, allow_redirects=False).headers["Location"]

rem_file = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
h5py_file = h5py.File(rem_file, "r")
nwbfile = NWBHDF5IO(file=h5py_file).read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## Behavioral epochs, position, and run bouts
#
# The session has PRE / Maze / POST epochs; we use the maze epoch. The linearized
# position (`1.6mLinearMazeLinearizedTimeSeries`, 0-1.6 m) is NaN except during on-track
# runs, so contiguous valid stretches define the run bouts directly. Each bout gets a
# direction from the sign of the median position derivative.

# %%
epochs = nwb["epochs"]
labels = np.asarray(epochs.label)
maze_mask = labels == "MazeEpoch"
maze_start = float(np.asarray(epochs.start)[maze_mask][0])
maze_end = float(np.asarray(epochs.end)[maze_mask][0])
print(f"MazeEpoch: {maze_start:.1f} - {maze_end:.1f} s")

pos = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
pos_t = np.asarray(pos.t)
pos_xy = np.asarray(pos.values)
lin_v = np.asarray(lin.values).ravel()
dt = np.median(np.diff(pos_t))
print(f"position: {len(pos_t)} samples at {1 / dt:.1f} Hz; "
      f"linearized valid fraction {np.mean(np.isfinite(lin_v)):.2f}")

# 2D speed (for the overview plot)
valid2d = np.isfinite(pos_xy[:, 0]) & np.isfinite(pos_xy[:, 1])
xy = pos_xy.copy()
for c in range(2):
    xy[:, c] = np.interp(pos_t, pos_t[valid2d], pos_xy[valid2d, c])
speed2d = np.hypot(np.gradient(xy[:, 0], dt), np.gradient(xy[:, 1], dt))
gw = signal.windows.gaussian(int(round(0.25 / dt)), std=int(round(0.25 / dt)) / 6)
speed2d = np.convolve(speed2d, gw / gw.sum(), mode="same")

# run bouts = merged contiguous valid stretches of the linearized coordinate
lin_ok = np.isfinite(lin_v)
d = np.diff(lin_ok.astype(int))
starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
if lin_ok[0]:
    starts = [0] + starts
if lin_ok[-1]:
    ends = ends + [len(lin_ok)]
segs = list(zip(starts, ends))
merged = []
for s, e in segs:
    if merged and (pos_t[s] - pos_t[merged[-1][1] - 1]) < 0.3:
        merged[-1] = (merged[-1][0], e)
    else:
        merged.append((s, e))

bouts, bout_dir = [], []
for s, e in merged:
    if (e - s) * dt < 1.0:
        continue
    seg_lin = lin_v[s:e]
    ok = np.isfinite(seg_lin)
    if ok.mean() < 0.8:
        continue
    med_speed = np.median(np.abs(np.diff(seg_lin[ok]))) / dt
    span = seg_lin[ok].max() - seg_lin[ok].min()
    if med_speed < 0.15 or span < 0.3:
        continue
    bouts.append((pos_t[s], pos_t[e - 1]))
    bout_dir.append(1 if np.median(np.diff(seg_lin[ok])) > 0 else -1)
bouts = np.array(bouts)
bout_dir = np.array(bout_dir)
print(f"run bouts: {len(bouts)} ({(bout_dir == 1).sum()} +dir, {(bout_dir == -1).sum()} -dir), "
      f"total {(bouts[:, 1] - bouts[:, 0]).sum():.0f} s")

def in_bouts(t, d_bouts):
    """vectorized membership test of times t in bouts (N,2 array)"""
    starts_, ends_ = d_bouts[:, 0], d_bouts[:, 1]
    idx = np.searchsorted(starts_, t, side="right") - 1
    inb = np.zeros(len(t), dtype=bool)
    ok = idx >= 0
    inb[ok] = t[ok] <= ends_[idx[ok]]
    return inb

def spikes_lin_pos(spk_t):
    """linearized position at each spike time (NaN if position invalid)"""
    idx = np.clip(np.searchsorted(pos_t, spk_t) - 1, 0, len(pos_t) - 1)
    return lin_v[idx]

# %% [markdown]
# ## Theta phase from the LFP
#
# The NWB file carries a 128-channel LFP ElectricalSeries at 1250 Hz. The electrodes
# table has no anatomical coordinates, so the theta reference channel is chosen
# data-driven by theta/delta PSD ratio (channel 117; verified below by the clear
# ~9 Hz peak during running). Theta phase is the angle of the analytic signal after
# 6-12 Hz bandpass; phase 0 corresponds to the peak of the filtered LFP.

# %%
lfp_es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
i0, i1 = int(maze_start * LFP_FS), int(maze_end * LFP_FS)
n = i1 - i0
lfp_raw = np.empty(n, dtype=np.float32)
chunk = int(100 * LFP_FS)
for s in tqdm(range(0, n, chunk), desc="reading LFP ch117"):
    e = min(s + chunk, n)
    lfp_raw[s:e] = lfp_es.data[i0 + s:i0 + e, THETA_CH]
lfp_raw = lfp_raw.astype(np.float64) * lfp_es.conversion * 1e6  # uV
lfp_t = maze_start + np.arange(n) / LFP_FS

b, a = signal.butter(4, [THETA_BAND[0] / (LFP_FS / 2), THETA_BAND[1] / (LFP_FS / 2)],
                     btype="band")
lfp_filt = signal.filtfilt(b, a, lfp_raw)
theta_phase = np.mod(np.angle(signal.hilbert(lfp_filt)), TWO_PI)  # 0 = LFP peak
cos_ph, sin_ph = np.cos(theta_phase), np.sin(theta_phase)

# PSD on a run-heavy window to verify the theta peak
m = (lfp_t >= 18500) & (lfp_t < 18700)
psd_f, psd = signal.welch(lfp_raw[m], fs=LFP_FS, nperseg=int(4 * LFP_FS))
theta_pow = psd[(psd_f >= 6) & (psd_f <= 12)].mean()
delta_pow = psd[(psd_f >= 1) & (psd_f <= 4)].mean()
peak_f = psd_f[(psd_f >= 4) & (psd_f <= 12)][np.argmax(psd[(psd_f >= 4) & (psd_f <= 12)])]
print(f"theta/delta = {theta_pow / delta_pow:.2f}, peak {peak_f:.2f} Hz")

def theta_phase_at(t):
    """theta phase at times t by linear interpolation of the analytic signal"""
    x = (t - maze_start) * LFP_FS
    i0_ = np.clip(np.floor(x).astype(int), 0, len(theta_phase) - 2)
    frac = x - np.floor(x)
    c = cos_ph[i0_] * (1 - frac) + cos_ph[i0_ + 1] * frac
    s = sin_ph[i0_] * (1 - frac) + sin_ph[i0_ + 1] * frac
    return np.mod(np.arctan2(s, c), TWO_PI)

# %% [markdown]
# ### Validation figure: behavior, LFP, and the theta rhythm

# %%
fig, axes = plt.subplots(4, 1, figsize=(12, 11))
ax = axes[0]
ax.plot(pos_t, lin_v, lw=0.5, color="k")
ax.set_ylabel("linearized pos (m)")
ax.set_xlim(maze_start, maze_end)
ax.set_title("Linearized position over maze epoch (NaN except on-track runs)")

ax = axes[1]
ax.plot(pos_t, speed2d, lw=0.5, color="tab:blue")
for (s, e), ddir in zip(bouts, bout_dir):
    ax.axvspan(s, e, color="tab:green" if ddir > 0 else "tab:orange", alpha=0.25, lw=0)
ax.set_ylabel("2D speed (m/s)")
ax.set_xlim(maze_start, maze_end)
ax.set_ylim(0, 1.5)
ax.set_title(f"Run bouts from linearized position (n={len(bouts)}; green=+dir, orange=-dir)")

ax = axes[2]
t0 = 18500.0
m2 = (lfp_t >= t0) & (lfp_t < t0 + 1.0)
ax.plot(lfp_t[m2], lfp_raw[m2], lw=0.4, color="gray", label="raw")
ax.plot(lfp_t[m2], lfp_filt[m2], lw=1.2, color="tab:purple", label="6-12 Hz")
ax.set_ylabel("LFP (uV)")
ax.set_title(f"Theta reference channel {THETA_CH}, 1 s snippet at {t0:.0f} s")
ax.legend(loc="upper right")

ax = axes[3]
ax.semilogy(psd_f, psd, color="k")
ax.axvspan(6, 12, color="tab:purple", alpha=0.2, label="theta band")
ax.axvline(peak_f, color="tab:purple", ls="--", lw=0.8)
ax.set_xlim(0, 30)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (uV$^2$/Hz)")
ax.set_title(f"LFP PSD during running: theta/delta = {theta_pow / delta_pow:.1f}, "
             f"peak {peak_f:.1f} Hz")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("fig01_data_overview.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Direction-specific place fields
#
# Smoothed firing-rate maps (50 bins, 1.5-bin Gaussian) are computed separately for the
# two run directions. A cell-direction is a place field if: peak rate >= 1 Hz, Skaggs
# spatial information exceeds a circular time-shift shuffle (500 shifts, p < 0.05), at
# least 30 in-direction spikes, and a contiguous field (rate > 20% of peak) between
# 0.2 and 1.2 m wide.

# %%
NBINS, TRACK_LEN, SMOOTH_SD = 50, 1.6, 1.5
N_SHUFFLE_SI = 500
edges = np.linspace(0, TRACK_LEN, NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
binw = TRACK_LEN / NBINS
gw2 = signal.windows.gaussian(int(round(SMOOTH_SD * 6)) | 1, std=SMOOTH_SD)
gw2 /= gw2.sum()

units = nwb["units"]
cell_type = units.get_info("cell_type")
exc_ids = [u for u in units.keys() if cell_type[u] == "excitatory"]
print(f"{len(units)} units, {len(exc_ids)} excitatory")

def rate_map(spk_pos, occ):
    cnt = np.histogram(spk_pos[np.isfinite(spk_pos)], bins=edges)[0].astype(float)
    cnt_s = np.convolve(cnt, gw2, mode="same")
    occ_s = np.convolve(occ, gw2, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = cnt_s / occ_s
    rate[occ_s < 0.05] = np.nan
    return rate

def skaggs_si(rate, occ):
    p = occ / occ.sum()
    ok = np.isfinite(rate) & (rate > 0) & (p > 0)
    mean_rate = np.nansum(rate[ok] * p[ok])
    if mean_rate <= 0:
        return 0.0
    return float(np.nansum(p[ok] * (rate[ok] / mean_rate)
                           * np.log2(rate[ok] / mean_rate)))

occ_by_dir = {}
for ddir in (+1, -1):
    occ = np.zeros(NBINS)
    for s, e in bouts[bout_dir == ddir]:
        mm = (pos_t >= s) & (pos_t <= e) & np.isfinite(lin_v)
        occ += np.histogram(lin_v[mm], bins=edges)[0] * dt
    occ_by_dir[ddir] = occ

fields = {}   # (uid, dir) -> dict(rate, si, p_si, peak, n_spikes, lo, hi, width, peak_pos)
epoch_len = maze_end - maze_start
for uid in tqdm(exc_ids, desc="place fields"):
    spk_t = np.asarray(units[uid].t)
    spk_pos = spikes_lin_pos(spk_t)
    for ddir in (+1, -1):
        d_bouts = bouts[bout_dir == ddir]
        sp = spk_pos[in_bouts(spk_t, d_bouts)]
        sp = sp[np.isfinite(sp)]
        occ = occ_by_dir[ddir]
        rate = rate_map(sp, occ)
        si = skaggs_si(rate, occ)
        peak = np.nanmax(rate) if np.isfinite(rate).any() else 0.0
        # circular time-shift shuffle null for spatial information
        si_null = np.empty(N_SHUFFLE_SI)
        for i in range(N_SHUFFLE_SI):
            tau = RNG.uniform(20, epoch_len - 20)
            t_sh = maze_start + (spk_t - maze_start + tau) % epoch_len
            sp_sh = spikes_lin_pos(t_sh)[in_bouts(t_sh, d_bouts)]
            si_null[i] = skaggs_si(rate_map(sp_sh[np.isfinite(sp_sh)], occ), occ)
        p_si = float(np.mean(si_null >= si))
        field = None
        if np.isfinite(rate).any() and peak >= 1.0:
            pk = int(np.nanargmax(rate))
            thr = 0.2 * peak
            lo_b, hi_b = pk, pk
            while lo_b > 0 and np.isfinite(rate[lo_b - 1]) and rate[lo_b - 1] > thr:
                lo_b -= 1
            while hi_b < NBINS - 1 and np.isfinite(rate[hi_b + 1]) and rate[hi_b + 1] > thr:
                hi_b += 1
            field = dict(lo=lo_b * binw, hi=(hi_b + 1) * binw,
                         width=(hi_b - lo_b + 1) * binw, peak_pos=centers[pk])
        fields[(uid, ddir)] = dict(rate=rate, si=si, p_si=p_si, peak=peak,
                                   n_spikes=len(sp), field=field)

place = {k: v for k, v in fields.items()
         if v["field"] is not None and v["p_si"] < 0.05 and v["n_spikes"] >= 30
         and 0.2 <= v["field"]["width"] <= 1.2}
print(f"place cell-directions: {len(place)} "
      f"(unique cells: {len(set(u for u, _ in place))})")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 6), width_ratios=[1, 1, 1.2])
for ax, ddir, ttl in zip(axes[:2], (+1, -1), ["+ direction", "- direction"]):
    keys = sorted([k for k in place if k[1] == ddir],
                  key=lambda k: place[k]["field"]["peak_pos"])
    M = np.array([place[k]["rate"] / np.nanmax(place[k]["rate"]) for k in keys])
    im = ax.imshow(M, aspect="auto", cmap="viridis", extent=[0, TRACK_LEN, len(keys), 0])
    ax.set_title(f"{ttl}: {len(keys)} place fields")
    ax.set_xlabel("linearized position (m)")
    ax.set_ylabel("cell (sorted by peak)")
    fig.colorbar(im, ax=ax, label="norm. rate", shrink=0.8)
ax = axes[2]
si_arr = np.array([fields[k]["si"] for k in fields])
is_place = np.array([k in place for k in fields])
ax.hist(si_arr[is_place], bins=30, alpha=0.7, label=f"place (n={is_place.sum()})")
ax.hist(si_arr[~is_place], bins=30, alpha=0.7, label=f"other (n={(~is_place).sum()})")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("cell-directions")
ax.set_title("Place-cell selection (SI shuffle p<0.05)")
ax.legend()
fig.tight_layout()
fig.savefig("fig02_place_fields.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Theta phase precession
#
# For each place cell-direction, spikes fired during runs through the field are assigned
# a theta phase and a *field progress* coordinate (0 at field entry, 1 at exit, oriented
# by the direction of travel so that precession has the same expected sign in both
# directions). The phase-position relationship is quantified by:
#
# - the circular-linear correlation of Kempter et al. (2012),
#   $r = \sqrt{(r_{xc}^2 + r_{xs}^2 - 2 r_{xc} r_{xs} r_{cs}) / (1 - r_{cs}^2)}$,
# - with sign and slope from a circular-linear regression (grid search over slopes
#   maximizing the resultant length of phase minus slope times position),
# - significance from 1000 phase-permutation shuffles (two-sided on |r|),
# - and the same correlation computed per single traversal (>= 4 spikes spanning at
#   least a quarter of the field).

# %%
N_SHUFFLE = 1000
SLOPES = np.linspace(-6 * np.pi, 6 * np.pi, 4321)  # rad per normalized field length

def circ_lin_r(x, ph):
    xc = np.corrcoef(x, np.cos(ph))[0, 1]
    xs = np.corrcoef(x, np.sin(ph))[0, 1]
    cs = np.corrcoef(np.cos(ph), np.sin(ph))[0, 1]
    if not np.isfinite(xc + xs + cs) or 1 - cs**2 <= 0:
        return np.nan
    return float(np.sqrt((xc**2 + xs**2 - 2 * xc * xs * cs) / (1 - cs**2)))

def fit_slope(xn, ph):
    z = np.exp(1j * (ph[:, None] - xn[:, None] * SLOPES[None, :]))
    R = np.abs(z.mean(axis=0))
    return float(SLOPES[int(np.argmax(R))])

records = []
spike_store = {}
for (uid, ddir), v in tqdm(place.items(), desc="precession"):
    lo, hi = v["field"]["lo"], v["field"]["hi"]
    spk_t = np.asarray(units[uid].t)
    d_bouts = bouts[bout_dir == ddir]
    st = spk_t[in_bouts(spk_t, d_bouts)]
    sp = spikes_lin_pos(st)
    ok = np.isfinite(sp) & (sp >= lo) & (sp <= hi)
    st, sp = st[ok], sp[ok]
    if len(st) < 20:
        continue
    ph = theta_phase_at(st)
    xn = (sp - lo) / (hi - lo) if ddir == 1 else (hi - sp) / (hi - lo)

    r_cl = circ_lin_r(xn, ph)
    slope = fit_slope(xn, ph)
    signed_r = np.sign(slope) * r_cl
    null = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        null[i] = circ_lin_r(xn, RNG.permutation(ph))
    p_val = float((np.sum(null >= r_cl) + 1) / (N_SHUFFLE + 1))

    trav_r = []
    for s, e in d_bouts:
        mm = (st >= s) & (st <= e)
        if mm.sum() < 4 or xn[mm].max() - xn[mm].min() < 0.25:
            continue
        r_t = circ_lin_r(xn[mm], ph[mm])
        if np.isfinite(r_t):
            trav_r.append(np.sign(fit_slope(xn[mm], ph[mm])) * r_t)
    trav_r = np.array(trav_r)

    records.append(dict(uid=int(uid), d=int(ddir), n_spikes=len(st), r_cl=r_cl,
                        signed_r=float(signed_r), slope_cycles=slope / TWO_PI,
                        p_val=p_val, lo=float(lo), hi=float(hi),
                        peak_pos=float(v["field"]["peak_pos"]),
                        trav_mean=float(np.mean(trav_r)) if len(trav_r) else np.nan,
                        trav_n=len(trav_r)))
    spike_store[(uid, ddir)] = (xn, ph)

with open("precession_results.json", "w") as f:
    json.dump(records, f, indent=1)

sig_neg = [r for r in records if r["p_val"] < 0.05 and r["signed_r"] < 0]
sig_pos = [r for r in records if r["p_val"] < 0.05 and r["signed_r"] > 0]
tm = np.array([r["trav_mean"] for r in records if np.isfinite(r["trav_mean"])])
print(f"analyzed cell-directions: {len(records)}")
print(f"significant: {len(sig_neg) + len(sig_pos)} "
      f"({len(sig_neg)} precessing, {len(sig_pos)} positive)")
print(f"median signed r = {np.median([r['signed_r'] for r in records]):.3f}, "
      f"median slope = {np.median([r['slope_cycles'] for r in records]):.2f} cycles/field")
print(f"per-traversal: median {np.median(tm):.3f}, {np.mean(tm < 0):.0%} negative")

# %% [markdown]
# ### Example cells: phase precession over pooled traversals

# %%
def fit_line(xn, ph):
    z = np.exp(1j * (ph[:, None] - xn[:, None] * SLOPES[None, :]))
    R = np.abs(z.mean(axis=0))
    m_ = SLOPES[int(np.argmax(R))]
    phi0 = np.angle(np.mean(np.exp(1j * (ph - m_ * xn))))
    return m_, phi0

def plot_phase_pos(ax, xn, ph, ms=4, alpha=0.5):
    m_, phi0 = fit_line(xn, ph)
    ax.scatter(xn, ph / TWO_PI, s=ms, c="k", alpha=alpha)
    ax.scatter(xn, ph / TWO_PI + 1, s=ms, c="k", alpha=alpha)
    xs = np.linspace(xn.min(), xn.max(), 50)
    yfit = np.mod((phi0 + m_ * xs) / TWO_PI, 1.0)
    for off in (0, 1):
        y = yfit + off
        yseg = y.copy()
        yseg[np.where(np.abs(np.diff(y)) > 0.5)[0]] = np.nan
        ax.plot(xs, yseg, color="tab:red", lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["0", "180", "360", "540", "720"])
    ax.set_xlabel("field progress (0=entry, 1=exit)")
    ax.set_ylabel("theta phase (deg)")

neg = [r for r in records if r["p_val"] < 0.05 and r["signed_r"] < 0 and r["n_spikes"] >= 100]
neg.sort(key=lambda r: r["signed_r"])
picks = [neg[i] for i in [0, 3, 6, 9, 12, 15] if i < len(neg)][:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, r in zip(axes.ravel(), picks):
    xn, ph = spike_store[(r["uid"], r["d"])]
    plot_phase_pos(ax, xn, ph)
    ax.set_title(f"unit {r['uid']}, dir {'+' if r['d'] > 0 else '-'}: "
                 f"r={r['signed_r']:.2f}, p={r['p_val']:.1e}, "
                 f"slope={r['slope_cycles']:.2f} cyc/field", fontsize=10)
fig.suptitle("Theta phase precession: example place cells (spikes duplicated over 2 cycles)",
             y=0.995)
fig.tight_layout()
fig.savefig("fig03_example_precession.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Single traversals: precession within individual laps

# %%
cands = [r for r in records if r["p_val"] < 0.05 and r["signed_r"] < 0
         and r["trav_n"] >= 10 and np.isfinite(r["trav_mean"])]
cands.sort(key=lambda r: r["trav_mean"])
best = cands[0]
u, ddir = best["uid"], best["d"]
lo, hi = best["lo"], best["hi"]
spk_t = np.asarray(units[u].t)
d_bouts = bouts[bout_dir == ddir]
st = spk_t[in_bouts(spk_t, d_bouts)]
sp = spikes_lin_pos(st)
ok = np.isfinite(sp) & (sp >= lo) & (sp <= hi)
st, sp = st[ok], sp[ok]
ph_all = theta_phase_at(st)
xn_all = (sp - lo) / (hi - lo) if ddir == 1 else (hi - sp) / (hi - lo)

travs = []
for s, e in d_bouts:
    mm = (st >= s) & (st <= e)
    if mm.sum() >= 5 and xn_all[mm].max() - xn_all[mm].min() > 0.4:
        travs.append((s, xn_all[mm], ph_all[mm]))
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (s, xn_t, ph_t) in zip(axes.ravel(), travs[:6]):
    plot_phase_pos(ax, xn_t, ph_t, ms=30, alpha=1.0)
    ax.set_title(f"traversal at t={s:.0f} s ({len(xn_t)} spikes)", fontsize=10)
fig.suptitle(f"Single-traversal precession: unit {u}, dir {'+' if ddir > 0 else '-'}",
             y=0.995)
fig.tight_layout()
fig.savefig("fig04_single_traversals.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Raw view: spikes riding the theta wave during one traversal

# %%
trav_spikes = sorted(((s, e, int(((st >= s) & (st <= e)).sum()))
                      for s, e in d_bouts if ((st >= s) & (st <= e)).sum() >= 8),
                     key=lambda x: -x[2])
s0, e0, _ = trav_spikes[0]
pad = 0.35
m_lfp = (lfp_t >= s0 - pad) & (lfp_t <= e0 + pad)
m_pos = (pos_t >= s0 - pad) & (pos_t <= e0 + pad)
m_spk = (st >= s0 - pad) & (st <= e0 + pad)

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.2, 1]))
ax = axes[0]
ax.plot(lfp_t[m_lfp], lfp_filt[m_lfp], color="tab:purple", lw=1.2,
        label="theta-filtered LFP (6-12 Hz)")
for t_spk in st[m_spk]:
    ax.axvline(t_spk, color="k", alpha=0.55, lw=0.9)
ax.set_ylabel("LFP (uV)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title(f"Unit {u} spikes (vertical lines) against theta during one field traversal "
             f"(t = {s0:.2f}-{e0:.2f} s)")
ax = axes[1]
ax.plot(pos_t[m_pos], lin_v[m_pos], color="tab:green", lw=1.5)
for t_spk, p_spk in zip(st[m_spk], sp[m_spk]):
    ax.plot(t_spk, p_spk, "k|", ms=8)
ax.axhspan(lo, hi, color="tab:red", alpha=0.12, label="place field")
ax.set_ylabel("linearized pos (m)")
ax.set_xlabel("time (s)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("fig07_raw_traversal.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Population statistics

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sr = np.array([r["signed_r"] for r in records])
pv = np.array([r["p_val"] for r in records])
sl = np.array([r["slope_cycles"] for r in records])

ax = axes[0, 0]
ax.hist(sr[pv < 0.05], bins=25, alpha=0.8, label=f"p<0.05 (n={np.sum(pv < 0.05)})",
        color="tab:blue")
ax.hist(sr[pv >= 0.05], bins=25, alpha=0.6, label=f"n.s. (n={np.sum(pv >= 0.05)})",
        color="gray")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(sr), color="tab:red", ls="--", lw=1, label=f"median {np.median(sr):.2f}")
ax.set_xlabel("signed circular-linear correlation")
ax.set_ylabel("cell-directions")
ax.set_title("Phase-position correlation (pooled spikes)")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.scatter(sr, -np.log10(pv), c=np.where(pv < 0.05, "tab:blue", "gray"), s=18, alpha=0.7)
ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("signed circular-linear correlation")
ax.set_ylabel("-log10 p (phase-permutation shuffle)")
ax.set_title(f"{np.sum((pv < 0.05) & (sr < 0))} precessing vs "
             f"{np.sum((pv < 0.05) & (sr > 0))} positive (of {len(records)})")

ax = axes[1, 0]
ax.hist(sl[pv < 0.05], bins=25, alpha=0.8, color="tab:blue", label="p<0.05")
ax.hist(sl[pv >= 0.05], bins=25, alpha=0.6, color="gray", label="n.s.")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(sl), color="tab:red", ls="--", lw=1, label=f"median {np.median(sl):.2f}")
ax.set_xlabel("precession slope (theta cycles per field)")
ax.set_ylabel("cell-directions")
ax.set_title("Precession slope")
ax.legend(fontsize=9)

ax = axes[1, 1]
tn = np.array([r["trav_n"] for r in records if np.isfinite(r["trav_mean"])])
ax.hist(tm, bins=25, color="tab:green", alpha=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(tm), color="tab:red", ls="--", lw=1, label=f"median {np.median(tm):.2f}")
ax.set_xlabel("mean per-traversal signed correlation")
ax.set_ylabel("cell-directions")
ax.set_title(f"Single-pass precession ({int(tn.sum())} traversals);\n"
             f"{np.mean(tm < 0):.0%} of {len(tm)} cell-directions negative", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig05_population_stats.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Pooled population phase-position density

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, subset, ttl in zip(
        axes,
        [[r for r in records if r["p_val"] < 0.05 and r["signed_r"] < 0], records],
        ["significant precessing cells", "all place cells"]):
    xs_all = np.concatenate([spike_store[(r["uid"], r["d"])][0] for r in subset])
    ph_all = np.concatenate([spike_store[(r["uid"], r["d"])][1] for r in subset]) / TWO_PI
    H, _, _ = np.histogram2d(xs_all, ph_all % 1.0, bins=[40, 36], range=[[0, 1], [0, 1]])
    ax.imshow(np.vstack([H, H]).T, origin="lower", aspect="auto", extent=[0, 1, 0, 2],
              cmap="magma")
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["0", "180", "360", "540", "720"])
    ax.set_xlabel("field progress (0=entry, 1=exit)")
    ax.set_ylabel("theta phase (deg)")
    ax.set_title(f"Pooled spikes, {ttl} (n={len(subset)}, {len(xs_all)} spikes)")
fig.tight_layout()
fig.savefig("fig06_pooled_population.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Results
#
# In the Achilles linear-maze session, 83 of 120 place cell-directions (from 79 unique
# place cells) show a significant circular-linear correlation between theta phase and
# travel progress through the field (phase-permutation shuffle, p < 0.05): 79 are
# negative (precessing) against only 4 positive. The median signed correlation
# across all place cell-directions is about -0.3 and the median slope about -0.5 theta
# cycles per field: spikes enter the field late in the theta cycle (near and after the
# LFP peak) and advance to earlier phases as the rat crosses the field. The effect is
# visible within single traversals (81% of cell-directions have negative per-traversal
# means over 2075 analyzed laps) and in the pooled population density, which shows the
# classic diagonal band sweeping from ~600 deg at field entry down to ~360 deg at exit.
# This reproduces the theta phase precession of O'Keefe & Recce (1993) in an openly
# available dataset.
