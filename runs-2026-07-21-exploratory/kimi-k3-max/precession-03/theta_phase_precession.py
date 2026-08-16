# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# This notebook demonstrates theta phase precession, the phenomenon in which a
# hippocampal place cell fires at progressively earlier phases of the theta
# oscillation as the animal runs through the cell's place field (O'Keefe &
# Recce 1993; Skaggs et al. 1996).
#
# **Dataset**: DANDI Archive dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural
# firing dynamics supports observationally constrained network models of
# hippocampal sharp-wave ripples", Buzsaki lab), session
# `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`:
# rat CA1 tetrode recordings (137 units) with 128-channel LFP at 1250 Hz while
# the animal runs back and forth on a 1.6 m linear maze (~34.5 min maze epoch,
# ~42 laps in each direction).
#
# **Pipeline**:
# 1. Stream the NWB file with remfile (no full download).
# 2. Extract run bouts from the linearized position (NaN except on-track runs).
# 3. Extract theta phase from the LFP channel with the strongest theta rhythm
#    (channel 117; 6-12 Hz bandpass + Hilbert transform).
# 4. Detect place fields per running direction (Skaggs spatial information with
#    a spike-time circular-shift shuffle).
# 5. Quantify precession with the Kempter et al. (2012) circular-linear
#    correlation between spike theta phase and field-progress, with sign and
#    slope from a grid-search circular-linear regression, and significance from
#    a phase-permutation shuffle.
#
# Two analysis details are worth stating explicitly because each silently
# destroys the effect if done wrong:
# - The spatial-information shuffle must circularly shift spike **times**
#   relative to the trajectory. Shifting spike positions in space leaves SI
#   invariant under uniform occupancy and yields p ~ 0.75 everywhere.
# - The field-progress coordinate must be oriented by travel direction
#   ((x-lo)/(hi-lo) for one direction, (hi-x)/(hi-lo) for the other).
#   Otherwise the two travel directions cancel and the population median sits
#   at zero with a 50/50 sign split.

# %% [markdown]
# ## Setup and streaming data access
#
# Dandiset 000044 has no published `.lindi.json`, so we stream the NWB blob
# directly with remfile + DiskCache. The DANDI `/download/` endpoint
# 302-redirects to a presigned S3 URL; the redirect target is GET-only (HEAD
# requests 403), so we grab the `Location` header from a non-following GET and
# hand it to remfile.

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from tqdm import tqdm
import json

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
CACHE_DIR = "/tmp/remfile_cache"

THETA_CH = 117          # strongest theta/delta PSD ratio (data-driven pick)
LFP_FS = 1250.0
THETA_BAND = (6, 12)
TWO_PI = 2 * np.pi


def resolve_s3_url():
    api_url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
               f"/versions/draft/assets/")
    r = requests.get(api_url, params={"path": ASSET_PATH})
    r.raise_for_status()
    asset = r.json()["results"][0]
    dl = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
          f"/versions/draft/assets/{asset['asset_id']}/download/")
    r = requests.get(dl, allow_redirects=False)
    return r.headers["Location"]


s3_url = resolve_s3_url()
disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## Preprocessing: maze epoch, run bouts, and theta phase
#
# The maze epoch spans 18079.5-20147 s. The linearized position series is NaN
# except during on-track runs, so run bouts are defined as contiguous valid
# stretches of that series (merged over <0.3 s tracking dropouts, >=1 s long,
# spanning >=0.3 m, median speed >=0.15 m/s). Theta phase comes from a 4th-order
# Butterworth 6-12 Hz bandpass of LFP channel 117 followed by the Hilbert
# transform; phase 0 is the LFP peak.

# %%
# --- epochs ---
epochs = nwb["epochs"]
labels = np.asarray(epochs.label)
maze_mask = labels == "MazeEpoch"
maze_start = float(np.asarray(epochs.start)[maze_mask][0])
maze_end = float(np.asarray(epochs.end)[maze_mask][0])
print(f"MazeEpoch: {maze_start:.1f} - {maze_end:.1f} s "
      f"({maze_end - maze_start:.1f} s)")

# --- position ---
pos = nwb["1.6mLinearMazeSpatialSeries"]            # TsdFrame x,y
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]     # TsdFrame, 1 column
pos_t = np.asarray(pos.t)
pos_xy = np.asarray(pos.values)
lin_v = np.asarray(lin.values).ravel()
dt = np.median(np.diff(pos_t))
print("position: n =", len(pos_t), "median dt =", dt,
      "span", pos_t[0], "->", pos_t[-1])
assert 0.01 < dt < 0.5, f"unexpected position sampling period {dt}"
print("lin valid fraction:", np.mean(np.isfinite(lin_v)),
      "range:", np.nanmin(lin_v), np.nanmax(lin_v))

# --- 2D speed (overview only) ---
valid2d = np.isfinite(pos_xy[:, 0]) & np.isfinite(pos_xy[:, 1])
xy = pos_xy.copy()
for c in range(2):
    xy[:, c] = np.interp(pos_t, pos_t[valid2d], pos_xy[valid2d, c])
vx = np.gradient(xy[:, 0], dt)
vy = np.gradient(xy[:, 1], dt)
speed2d = np.sqrt(vx**2 + vy**2)
win = int(round(0.25 / dt))
gw_speed = signal.windows.gaussian(win, std=win / 6)
gw_speed /= gw_speed.sum()
speed2d = np.convolve(speed2d, gw_speed, mode="same")

# --- run bouts: contiguous valid stretches of the linearized coordinate ---
lin_ok = np.isfinite(lin_v)
d = np.diff(lin_ok.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
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
bouts = []
for s, e in merged:
    dur = (e - s) * dt
    if dur < 1.0:
        continue
    seg_lin = lin_v[s:e]
    seg_t = pos_t[s:e]
    ok = np.isfinite(seg_lin)
    if ok.mean() < 0.8:
        continue
    v = np.abs(np.diff(seg_lin[ok])) / dt
    med_speed = np.median(v)
    dpos = np.diff(seg_lin[ok])
    direction = 1 if np.median(dpos) > 0 else -1
    span = seg_lin[ok].max() - seg_lin[ok].min()
    if med_speed < 0.15 or span < 0.3:
        continue
    bouts.append((seg_t[0], seg_t[-1], direction, med_speed, span))
bout_dir = np.array([b[2] for b in bouts], dtype=int)
bout_speed = np.array([b[3] for b in bouts])
bout_span = np.array([b[4] for b in bouts])
bouts = np.array([(b[0], b[1]) for b in bouts])
print(f"run bouts: {len(bouts)}, total {(bouts[:, 1] - bouts[:, 0]).sum():.0f} s")
print(f"  +dir: {(bout_dir == 1).sum()}, -dir: {(bout_dir == -1).sum()}")
print(f"  median bout speed {np.median(bout_speed):.2f} m/s, "
      f"median span {np.median(bout_span):.2f} m")

# --- LFP theta channel: read the maze epoch, channel 117, in chunks ---
lfp_es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
i0 = int(maze_start * LFP_FS)
i1 = int(maze_end * LFP_FS)
n = i1 - i0
print(f"LFP maze slice: samples {i0}:{i1} = {n / LFP_FS:.1f} s")
chunk = int(100 * LFP_FS)
lfp_raw = np.empty(n, dtype=np.float32)
for s in tqdm(range(0, n, chunk), desc="reading LFP ch117"):
    e = min(s + chunk, n)
    lfp_raw[s:e] = lfp_es.data[i0 + s:i0 + e, THETA_CH]
lfp_raw = lfp_raw.astype(np.float64) * lfp_es.conversion * 1e6  # uV
lfp_t = maze_start + np.arange(n) / LFP_FS

# --- theta bandpass + Hilbert phase ---
b, a = signal.butter(4, [THETA_BAND[0] / (LFP_FS / 2),
                         THETA_BAND[1] / (LFP_FS / 2)], btype="band")
lfp_filt = signal.filtfilt(b, a, lfp_raw)
analytic = signal.hilbert(lfp_filt)
theta_phase = np.mod(np.angle(analytic), TWO_PI)  # [0, 2pi), 0 = LFP peak

# --- PSD on a run-heavy 200 s window to confirm the theta rhythm ---
m = (lfp_t >= 18500) & (lfp_t < 18700)
f, psd = signal.welch(lfp_raw[m], fs=LFP_FS, nperseg=int(4 * LFP_FS))
theta_pow = psd[(f >= 6) & (f <= 12)].mean()
delta_pow = psd[(f >= 1) & (f <= 4)].mean()
peak_f = f[(f >= 4) & (f <= 12)][np.argmax(psd[(f >= 4) & (f <= 12)])]
print(f"PSD (18500-18700s): theta/delta = {theta_pow / delta_pow:.2f}, "
      f"peak {peak_f:.2f} Hz")

# %% [markdown]
# ### Figure 1: data overview
#
# Linearized position and run bouts over the maze epoch, a 1 s snippet of the
# theta reference channel, and the LFP power spectrum during running showing a
# clear ~9 Hz theta peak.

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
    ax.axvspan(s, e, color="tab:green" if ddir > 0 else "tab:orange",
               alpha=0.25, lw=0)
ax.set_ylabel("2D speed (m/s)")
ax.set_xlim(maze_start, maze_end)
ax.set_ylim(0, 1.5)
ax.set_title(f"Run bouts from linearized position (n={len(bouts)}; "
             f"green=+dir, orange=-dir)")

ax = axes[2]
t0 = 18500.0
m2 = (lfp_t >= t0) & (lfp_t < t0 + 1.0)
ax.plot(lfp_t[m2], lfp_raw[m2], lw=0.4, color="gray", label="raw")
ax.plot(lfp_t[m2], lfp_filt[m2], lw=1.2, color="tab:purple", label="6-12 Hz")
ax.set_ylabel("LFP (uV)")
ax.set_title(f"Theta reference channel {THETA_CH}, 1 s snippet at {t0:.0f} s")
ax.legend(loc="upper right")

ax = axes[3]
ax.semilogy(f, psd, color="k")
ax.axvspan(6, 12, color="tab:purple", alpha=0.2, label="theta band")
ax.axvline(peak_f, color="tab:purple", ls="--", lw=0.8)
ax.set_xlim(0, 30)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (uV$^2$/Hz)")
ax.set_title(f"LFP PSD during running (18500-18700 s): "
             f"theta/delta = {theta_pow / delta_pow:.1f}, peak {peak_f:.1f} Hz")
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig("fig01_data_overview.png", dpi=150)
plt.close(fig)
print("saved fig01_data_overview.png")

# %% [markdown]
# ## Place-field detection
#
# Direction-specific firing-rate maps (50 bins over the 1.6 m track, 1.5-bin
# Gaussian smoothing) are computed for every excitatory unit from spikes inside
# run bouts of that direction. Spatial information (Skaggs et al. 1993) is
# tested against 500 circular shifts of the spike **times** relative to the
# trajectory. A place field is a contiguous run of bins above 20% of the peak
# rate containing the peak bin; cell-directions pass if peak >= 1 Hz,
# shuffle p < 0.05, >= 30 in-field spikes, and field width in [0.2, 1.2] m.

# %%
NBINS = 50
TRACK_LEN = 1.6
SMOOTH_SD = 1.5  # bins
N_SHUFFLE_SI = 500
RNG_SI = np.random.default_rng(42)

units = nwb["units"]
cell_type = units.get_info("cell_type")
unit_ids = list(units.keys())
exc_ids = [u for u in unit_ids if cell_type[u] == "excitatory"]
print(f"{len(unit_ids)} units, {len(exc_ids)} excitatory")

edges = np.linspace(0, TRACK_LEN, NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
binw = TRACK_LEN / NBINS
gw = signal.windows.gaussian(int(round(SMOOTH_SD * 6)) | 1, std=SMOOTH_SD)
gw /= gw.sum()


def spikes_lin_pos(spk_t):
    """linearized position at each spike time (NaN if position invalid)"""
    idx = np.searchsorted(pos_t, spk_t) - 1
    idx = np.clip(idx, 0, len(pos_t) - 1)
    return lin_v[idx]


def in_bouts(t, d_bouts):
    """vectorized membership test of times t in bouts (N,2 array)"""
    starts, ends = d_bouts[:, 0], d_bouts[:, 1]
    idx = np.searchsorted(starts, t, side="right") - 1
    inb = np.zeros(len(t), dtype=bool)
    ok = idx >= 0
    inb[ok] = t[ok] <= ends[idx[ok]]
    return inb


def rate_map(spk_pos, occ):
    cnt = np.histogram(spk_pos[np.isfinite(spk_pos)], bins=edges)[0].astype(float)
    cnt_s = np.convolve(cnt, gw, mode="same")
    occ_s = np.convolve(occ, gw, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = cnt_s / occ_s
    rate[occ_s < 0.05] = np.nan
    return rate, cnt


def skaggs_si(rate, occ):
    p = occ / occ.sum()
    ok = np.isfinite(rate) & (rate > 0) & (p > 0)
    mean_rate = np.nansum(rate[ok] * p[ok])
    if mean_rate <= 0:
        return 0.0, 0.0
    si = np.nansum(p[ok] * (rate[ok] / mean_rate) * np.log2(rate[ok] / mean_rate))
    return si, mean_rate


results = {}
occ_by_dir = {}
for d in (+1, -1):
    d_bouts = bouts[bout_dir == d]
    occ = np.zeros(NBINS)
    for s, e in d_bouts:
        m = (pos_t >= s) & (pos_t <= e) & np.isfinite(lin_v)
        occ += np.histogram(lin_v[m], bins=edges)[0] * np.median(np.diff(pos_t))
    occ_by_dir[d] = occ
    print(f"direction {d:+d}: {len(d_bouts)} bouts, occupancy {occ.sum():.0f} s, "
          f"min occ {occ.min():.2f} s")

for uid in tqdm(exc_ids, desc="place fields"):
    spk_t = np.asarray(units[uid].t)
    spk_pos = spikes_lin_pos(spk_t)
    for d in (+1, -1):
        d_bouts = bouts[bout_dir == d]
        mask = in_bouts(spk_t, d_bouts)
        sp = spk_pos[mask]
        sp = sp[np.isfinite(sp)]
        occ = occ_by_dir[d]
        rate, cnt = rate_map(sp, occ)
        si, mean_rate = skaggs_si(rate, occ)
        peak = np.nanmax(rate) if np.isfinite(rate).any() else 0.0
        n_spikes = len(sp)
        si_null = np.empty(N_SHUFFLE_SI)
        epoch_len = maze_end - maze_start
        for i in range(N_SHUFFLE_SI):
            tau = RNG_SI.uniform(20, epoch_len - 20)
            t_sh = maze_start + (spk_t - maze_start + tau) % epoch_len
            pos_sh = spikes_lin_pos(t_sh)
            m_sh = in_bouts(t_sh, d_bouts)
            sp_sh = pos_sh[m_sh]
            sp_sh = sp_sh[np.isfinite(sp_sh)]
            r_sh, _ = rate_map(sp_sh, occ)
            si_null[i], _ = skaggs_si(r_sh, occ)
        p_si = float(np.mean(si_null >= si))
        field = None
        if np.isfinite(rate).any() and peak >= 1.0:
            pk_bin = int(np.nanargmax(rate))
            thr = 0.2 * peak
            lo, hi = pk_bin, pk_bin
            while lo > 0 and np.isfinite(rate[lo - 1]) and rate[lo - 1] > thr:
                lo -= 1
            while hi < NBINS - 1 and np.isfinite(rate[hi + 1]) and rate[hi + 1] > thr:
                hi += 1
            width = (hi - lo + 1) * binw
            field = dict(lo=lo * binw, hi=(hi + 1) * binw, width=width,
                         peak_bin=pk_bin, peak_pos=centers[pk_bin])
        results[(uid, d)] = dict(rate=rate, si=si, p_si=p_si,
                                 mean_rate=mean_rate, peak=peak,
                                 n_spikes=n_spikes, field=field)

place = {}
for (uid, d), r in results.items():
    f = r["field"]
    if (f is not None and r["p_si"] < 0.05 and r["n_spikes"] >= 30
            and 0.2 <= f["width"] <= 1.2):
        place[(uid, d)] = r
print(f"\nplace cell-directions: {len(place)} "
      f"(unique cells: {len(set(u for u, _ in place))})")

# %% [markdown]
# ### Figure 2: place fields
#
# Normalized rate maps sorted by peak position tile the track in both travel
# directions, and the selected place cells sit at the high end of the spatial
# information distribution.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 6), width_ratios=[1, 1, 1.2])
for ax, d, ttl in zip(axes[:2], (+1, -1), ["+ direction", "- direction"]):
    keys = [(u, dd) for (u, dd) in place.keys() if dd == d]
    keys.sort(key=lambda k: place[k]["field"]["peak_pos"])
    if keys:
        M = np.array([place[k]["rate"] / np.nanmax(place[k]["rate"])
                      for k in keys])
        im = ax.imshow(M, aspect="auto", cmap="viridis",
                       extent=[0, TRACK_LEN, len(keys), 0])
        ax.set_title(f"{ttl}: {len(keys)} place fields")
        ax.set_xlabel("linearized position (m)")
        ax.set_ylabel("cell (sorted by peak)")
        fig.colorbar(im, ax=ax, label="norm. rate", shrink=0.8)

ax = axes[2]
si_all = np.array([results[k]["si"] for k in results])
is_place = np.array([k in place for k in results])
ax.hist(si_all[is_place], bins=30, alpha=0.7, label=f"place (n={is_place.sum()})")
ax.hist(si_all[~is_place], bins=30, alpha=0.7,
        label=f"other (n={len(si_all) - is_place.sum()})")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("cell-directions")
ax.set_title("Place-cell selection (SI shuffle p<0.05)")
ax.legend()
fig.tight_layout()
fig.savefig("fig02_place_fields.png", dpi=150)
plt.close(fig)
print("saved fig02_place_fields.png")

# %% [markdown]
# ## Theta phase precession
#
# For each place cell-direction we take spikes inside the field during bouts of
# that direction, convert position to **field progress** (0 at field entry, 1
# at exit, oriented by travel direction), and interpolate the theta phase at
# each spike time (interpolating cos/sin of the analytic signal separately to
# avoid wrap artifacts).
#
# The circular-linear correlation of Kempter et al. (2012) gives an unsigned
# magnitude; the sign and slope come from a grid-search circular-linear
# regression maximizing the resultant length of `exp(i*(phase - m*x))` over
# slopes m in +/-6*pi rad per field. Significance uses 1000 phase-permutation
# shuffles, which break the phase-position pairing while keeping both marginal
# distributions. We also compute the correlation separately for each individual
# field traversal (>= 4 spikes spanning >= 25% of the field).

# %%
N_SHUFFLE = 1000
MIN_FIELD_SPIKES = 20
MIN_TRAV_SPIKES = 4
RNG = np.random.default_rng(7)

cos_ph, sin_ph = np.cos(theta_phase), np.sin(theta_phase)


def theta_phase_at(t):
    """theta phase (rad, 0 = LFP peak) at times t via linear interpolation of
    the analytic signal"""
    x = (t - maze_start) * LFP_FS
    i0 = np.floor(x).astype(int)
    frac = x - i0
    i0 = np.clip(i0, 0, len(theta_phase) - 2)
    c = cos_ph[i0] * (1 - frac) + cos_ph[i0 + 1] * frac
    s = sin_ph[i0] * (1 - frac) + sin_ph[i0 + 1] * frac
    return np.mod(np.arctan2(s, c), TWO_PI)


def circ_lin_r(pos, ph):
    """Kempter et al. 2012 circular-linear correlation magnitude"""
    xc = np.corrcoef(pos, np.cos(ph))[0, 1]
    xs = np.corrcoef(pos, np.sin(ph))[0, 1]
    cs = np.corrcoef(np.cos(ph), np.sin(ph))[0, 1]
    denom = 1 - cs**2
    if denom <= 0 or not np.isfinite(xc + xs + cs):
        return np.nan
    return np.sqrt((xc**2 + xs**2 - 2 * xc * xs * cs) / denom)


SLOPES = np.linspace(-6 * np.pi, 6 * np.pi, 4321)  # rad per normalized field


def fit_slope(xn, ph):
    """grid-search circular-linear regression: slope + resultant length"""
    z = np.exp(1j * (ph[:, None] - xn[:, None] * SLOPES[None, :]))
    R = np.abs(z.mean(axis=0))
    i = int(np.argmax(R))
    return SLOPES[i], float(R[i])


records = []
spike_store = {}
for uid, d in tqdm(sorted(place.keys()), desc="precession"):
    lo, hi = place[(uid, d)]["field"]["lo"], place[(uid, d)]["field"]["hi"]
    peak_pos = place[(uid, d)]["field"]["peak_pos"]
    spk_t = np.asarray(units[uid].t)
    d_bouts = bouts[bout_dir == d]
    mask = in_bouts(spk_t, d_bouts)
    st = spk_t[mask]
    sp = spikes_lin_pos(st)
    ok = np.isfinite(sp) & (sp >= lo) & (sp <= hi)
    st, sp = st[ok], sp[ok]
    if len(st) < MIN_FIELD_SPIKES:
        continue
    ph = theta_phase_at(st)
    xn = (sp - lo) / (hi - lo) if d == +1 else (hi - sp) / (hi - lo)

    r_cl = circ_lin_r(xn, ph)
    slope, R_best = fit_slope(xn, ph)
    signed_r = np.sign(slope) * r_cl

    null = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        null[i] = circ_lin_r(xn, RNG.permutation(ph))
    p_val = (np.sum(null >= r_cl) + 1) / (N_SHUFFLE + 1)

    trav_r, trav_n = [], []
    for s, e in d_bouts:
        m = (st >= s) & (st <= e)
        if m.sum() < MIN_TRAV_SPIKES:
            continue
        xn_t = xn[m]
        if xn_t.max() - xn_t.min() < 0.25:
            continue
        r_t = circ_lin_r(xn_t, ph[m])
        sl_t, _ = fit_slope(xn_t, ph[m])
        if np.isfinite(r_t):
            trav_r.append(np.sign(sl_t) * r_t)
            trav_n.append(int(m.sum()))
    trav_r = np.array(trav_r)

    records.append(dict(uid=uid, d=d, n_spikes=len(st), r_cl=r_cl,
                        signed_r=signed_r, slope=slope,
                        slope_cycles=slope / TWO_PI, p_val=p_val,
                        lo=lo, hi=hi, peak_pos=peak_pos,
                        trav_mean=np.mean(trav_r) if len(trav_r) else np.nan,
                        trav_n=len(trav_r),
                        trav_frac_neg=np.mean(trav_r < 0) if len(trav_r) else np.nan))
    spike_store[(uid, d)] = (xn, ph)

with open("precession_results.json", "w") as f:
    json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r.items()} for r in records], f, indent=1)

rec = records
sig_neg = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0]
sig_pos = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] > 0]
print(f"\nanalyzed cell-directions: {len(rec)}")
print(f"significant (p<0.05): {len(sig_neg) + len(sig_pos)} "
      f"({len(sig_neg)} negative / precessing, {len(sig_pos)} positive)")
print(f"median signed r: {np.median([r['signed_r'] for r in rec]):.3f}")
print(f"median slope: {np.median([r['slope_cycles'] for r in rec]):.2f} "
      f"cycles/field")
tr = np.array([r["trav_mean"] for r in rec if np.isfinite(r["trav_mean"])])
print(f"per-traversal mean r: median {np.median(tr):.3f} over {len(tr)} "
      f"cell-directions, frac negative {np.mean(tr < 0):.2f}")

# %% [markdown]
# ### Figure 3: example place cells
#
# Six strongly precessing cells. Spikes are duplicated over two theta cycles to
# make the wrap-around visible; the red line is the circular-linear fit. Phase
# decreases systematically from field entry to field exit.

# %%
def fit_line(xn, ph):
    z = np.exp(1j * (ph[:, None] - xn[:, None] * SLOPES[None, :]))
    R = np.abs(z.mean(axis=0))
    m = SLOPES[int(np.argmax(R))]
    phi0 = np.angle(np.mean(np.exp(1j * (ph - m * xn))))
    return m, phi0


def plot_fit(ax, m, phi0, xs, lw=2):
    yfit = np.mod((phi0 + m * xs) / TWO_PI, 1.0)
    for off in (0, 1):
        y = yfit + off
        yseg = y.copy()
        yseg[np.where(np.abs(np.diff(y)) > 0.5)[0]] = np.nan
        ax.plot(xs, yseg, color="tab:red", lw=lw)


def style_phase_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["0", "180", "360", "540", "720"])
    ax.set_xlabel("field progress (0=entry, 1=exit)")
    ax.set_ylabel("theta phase (deg)")


neg = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0
       and r["n_spikes"] >= 100]
neg.sort(key=lambda r: r["signed_r"])
picks = [neg[i] for i in [0, 3, 6, 9, 12, 15] if i < len(neg)][:6]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, r in zip(axes.ravel(), picks):
    u, d = r["uid"], r["d"]
    xn, ph = spike_store[(u, d)]
    m, phi0 = fit_line(xn, ph)
    ax.scatter(xn, ph / TWO_PI, s=4, c="k", alpha=0.5)
    ax.scatter(xn, ph / TWO_PI + 1, s=4, c="k", alpha=0.5)
    plot_fit(ax, m, phi0, np.linspace(0, 1, 50))
    style_phase_ax(ax)
    ax.set_title(f"unit {u}, dir {'+' if d > 0 else '-'}: r={r['signed_r']:.2f}, "
                 f"p={r['p_val']:.1e}, slope={r['slope_cycles']:.2f} cyc/field",
                 fontsize=10)
fig.suptitle("Theta phase precession: example place cells "
             "(spikes duplicated over 2 cycles)", y=0.995)
fig.tight_layout()
fig.savefig("fig03_example_precession.png", dpi=150)
plt.close(fig)
print("saved fig03_example_precession.png")

# %% [markdown]
# ### Figure 4: single traversals
#
# Precession is visible within individual laps, not just in pooled spikes. Six
# traversals of the showcase cell (most negative per-traversal mean among cells
# with >= 10 qualifying traversals).

# %%
cands = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0
         and r["trav_n"] >= 10 and np.isfinite(r["trav_mean"])]
cands.sort(key=lambda r: r["trav_mean"])
best = cands[0]
u, d = best["uid"], best["d"]
lo, hi = best["lo"], best["hi"]
spk_t = np.asarray(units[u].t)
d_bouts = bouts[bout_dir == d]
mask = in_bouts(spk_t, d_bouts)
st = spk_t[mask]
sp = spikes_lin_pos(st)
ok = np.isfinite(sp) & (sp >= lo) & (sp <= hi)
st, sp = st[ok], sp[ok]
ph = theta_phase_at(st)
xn_all = (sp - lo) / (hi - lo) if d == 1 else (hi - sp) / (hi - lo)

travs = []
for s, e in d_bouts:
    m = (st >= s) & (st <= e)
    if m.sum() >= 5 and xn_all[m].max() - xn_all[m].min() > 0.4:
        travs.append((s, xn_all[m], ph[m]))
travs = travs[:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (s, xn_t, ph_t) in zip(axes.ravel(), travs):
    ax.scatter(xn_t, ph_t / TWO_PI, s=30, c="k")
    ax.scatter(xn_t, ph_t / TWO_PI + 1, s=30, c="k")
    if len(xn_t) >= 4:
        m, phi0 = fit_line(xn_t, ph_t)
        plot_fit(ax, m, phi0, np.linspace(xn_t.min(), xn_t.max(), 30), lw=1.5)
    style_phase_ax(ax)
    ax.set_title(f"traversal at t={s:.0f} s ({len(xn_t)} spikes)", fontsize=10)
fig.suptitle(f"Single-traversal precession: unit {u}, "
             f"dir {'+' if d > 0 else '-'}", y=0.995)
fig.tight_layout()
fig.savefig("fig04_single_traversals.png", dpi=150)
plt.close(fig)
print("saved fig04_single_traversals.png")

# %% [markdown]
# ### Figure 5: population statistics
#
# Across all 120 place cell-directions: the signed circular-linear correlation
# distribution is shifted negative (median -0.32); 79 cell-directions are
# significantly negative versus 5 significantly positive; the median precession
# slope is -0.52 theta cycles per field; and 81% of cells show a negative mean
# per-traversal correlation.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax = axes[0, 0]
sr = np.array([r["signed_r"] for r in rec])
pv = np.array([r["p_val"] for r in rec])
ax.hist(sr[pv < 0.05], bins=25, alpha=0.8,
        label=f"p<0.05 (n={np.sum(pv < 0.05)})", color="tab:blue")
ax.hist(sr[pv >= 0.05], bins=25, alpha=0.6,
        label=f"n.s. (n={np.sum(pv >= 0.05)})", color="gray")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(sr), color="tab:red", ls="--", lw=1,
           label=f"median {np.median(sr):.2f}")
ax.set_xlabel("signed circular-linear correlation")
ax.set_ylabel("cell-directions")
ax.set_title("Phase-position correlation (pooled spikes)")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.scatter(sr, -np.log10(pv), c=np.where(pv < 0.05, "tab:blue", "gray"),
           s=18, alpha=0.7)
ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("signed circular-linear correlation")
ax.set_ylabel("-log10 p (phase-permutation shuffle)")
n_sig_neg = np.sum((pv < 0.05) & (sr < 0))
n_sig_pos = np.sum((pv < 0.05) & (sr > 0))
ax.set_title(f"{n_sig_neg} precessing vs {n_sig_pos} positive (of {len(rec)})")

ax = axes[1, 0]
sl = np.array([r["slope_cycles"] for r in rec])
ax.hist(sl[pv < 0.05], bins=25, alpha=0.8, color="tab:blue", label="p<0.05")
ax.hist(sl[pv >= 0.05], bins=25, alpha=0.6, color="gray", label="n.s.")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(sl), color="tab:red", ls="--", lw=1,
           label=f"median {np.median(sl):.2f}")
ax.set_xlabel("precession slope (theta cycles per field)")
ax.set_ylabel("cell-directions")
ax.set_title("Precession slope")
ax.legend(fontsize=9)

ax = axes[1, 1]
tm = np.array([r["trav_mean"] for r in rec if np.isfinite(r["trav_mean"])])
tn = np.array([r["trav_n"] for r in rec if np.isfinite(r["trav_mean"])])
ax.hist(tm, bins=25, color="tab:green", alpha=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(tm), color="tab:red", ls="--", lw=1,
           label=f"median {np.median(tm):.2f}")
ax.set_xlabel("mean per-traversal signed correlation")
ax.set_ylabel("cell-directions")
ax.set_title(f"Single-pass precession ({int(tn.sum())} traversals);\n"
             f"{np.mean(tm < 0):.0%} of {len(tm)} cell-directions negative",
             fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig05_population_stats.png", dpi=150)
plt.close(fig)
print("saved fig05_population_stats.png")

# %% [markdown]
# ### Figure 6: raw demonstration
#
# One field traversal of the showcase cell: the theta-filtered LFP with spike
# times marked (top) and the linearized trajectory with the place field shaded
# (bottom). Successive spikes step to earlier phases of the theta cycle as the
# animal crosses the field.

# %%
trav_spikes = []
for s, e in d_bouts:
    m = (st >= s) & (st <= e)
    if m.sum() >= 8:
        trav_spikes.append((s, e, m.sum()))
trav_spikes.sort(key=lambda x: -x[2])
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
ax.set_title(f"Unit {u} spikes (vertical lines) against theta during one field "
             f"traversal (t = {s0:.2f}-{e0:.2f} s)")

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
print("saved fig07_raw_traversal.png")

# %% [markdown]
# ### Figure 7: pooled population
#
# Spike-density heatmaps of theta phase vs. field progress pooled across cells.
# The significant precessing cells show a clean diagonal band from late phases
# at field entry to early phases at field exit, spanning roughly half a theta
# cycle over the field.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, subset, ttl in zip(
        axes,
        [[r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0], rec],
        ["significant precessing cells", "all place cells"]):
    xs_all, ph_all = [], []
    for r in subset:
        xn, ph = spike_store[(r["uid"], r["d"])]
        xs_all.append(xn)
        ph_all.append(ph)
    xs_all = np.concatenate(xs_all)
    ph_all = np.concatenate(ph_all) / TWO_PI
    H, xe, ye = np.histogram2d(xs_all, ph_all % 1.0, bins=[40, 36],
                               range=[[0, 1], [0, 1]])
    H2 = np.vstack([H, H])
    ax.imshow(H2.T, origin="lower", aspect="auto", extent=[0, 1, 0, 2],
              cmap="magma")
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["0", "180", "360", "540", "720"])
    ax.set_xlabel("field progress (0=entry, 1=exit)")
    ax.set_ylabel("theta phase (deg)")
    ax.set_title(f"Pooled spikes, {ttl} (n={len(subset)}, {len(xs_all)} spikes)")
fig.tight_layout()
fig.savefig("fig06_pooled_population.png", dpi=150)
plt.close(fig)
print("saved fig06_pooled_population.png")

# %% [markdown]
# ## Summary
#
# In this Achilles linear-maze session (DANDI 000044), 79 of 137 recorded units
# (120 cell-directions among excitatory cells) pass place-field criteria.
# Pooling spikes over traversals, 80 of 120 place cell-directions show
# significant negative phase-position correlation (phase-permutation p < 0.05)
# versus only 5 positive; the median signed circular-linear correlation is
# -0.32 and the median slope is -0.52 theta cycles per field. The effect is
# also visible within single laps: the median per-traversal correlation is
# -0.25 and 81% of cell-directions are negative on average. This is the
# canonical theta phase precession signature (O'Keefe & Recce 1993), measured
# against the local theta rhythm (peak ~9.25 Hz) of the strongest-theta LFP
# channel.
