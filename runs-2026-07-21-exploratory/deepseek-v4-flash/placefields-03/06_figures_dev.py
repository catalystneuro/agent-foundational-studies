"""Generate all figures for the hippocampal place-cell demonstration (DANDI 000044)."""
import json
import numpy as np
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; MAGENTA = "#e87ba4"
RED = "#e34948"; GRAY = "#52514e"; LIGHTGRAY = "#c9c8c0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
    "axes.labelsize": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "xtick.color": "#3a3a38", "ytick.color": "#3a3a38",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
FS = 39.0625
T0 = 18079.5
MAZE_END = 20147.0
NBINS = 50
LFP_FS = 1250.0
LFP_CH = 117
LFP_UV = 3.815e-7 * 1e6  # conversion to uV
WIN = (18500.0, 18600.0)

# ---------------- load ----------------
with open("_asset.json") as f:
    info = json.load(f)
rem_file = remfile.File(info["s3_url"], disk_cache=remfile.DiskCache("/tmp/remfile_cache_000044"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f)
nwb = nap.NWBFile(io.read())
units = nwb["units"]
ct = units.get_info("cell_type")
all_keys = list(units.keys())

pos2d_raw = np.asarray(nwb["1.6mLinearMazeSpatialSeries"].values)
lin_raw = np.asarray(nwb["1.6mLinearMazeLinearizedTimeSeries"].values).ravel()
t = T0 + np.arange(len(lin_raw)) / FS
maze_mask = (t >= T0) & (t < MAZE_END)
pos2d = pos2d_raw[maze_mask].copy()
lin2 = lin_raw[maze_mask].copy()
t2 = t[maze_mask]

# ---------------- run bouts ----------------
good = np.flatnonzero(np.isfinite(lin_raw))
gaps = np.diff(good)
be = np.where(gaps > 0.3 * FS)[0]
starts = np.concatenate([[good[0]], good[be + 1]])
ends = np.concatenate([good[be], [good[-1]]])
bouts = []
for s, e in zip(starts, ends):
    ix = good[(good >= s) & (good <= e)]
    seg = lin_raw[ix]
    if (t[e] - t[s]) >= 1.0 and abs(seg[-1] - seg[0]) > 0.3 \
            and np.median(np.abs(np.diff(seg))) * FS > 0.15:
        bouts.append((s, e, 1 if seg[-1] > seg[0] else 0))
b_start_i = np.array([b[0] for b in bouts])
b_end_i = np.array([b[1] for b in bouts])
b_dir = np.array([b[2] for b in bouts])
b_start = t[b_start_i]
b_end = t[b_end_i]
nb = len(bouts)
sizes = b_end_i - b_start_i + 1
bout_offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
tau_run = np.concatenate([np.arange(s) / FS + bout_offsets[i] / FS for i, s in enumerate(sizes)])
run_lin = np.concatenate([lin_raw[s:e + 1] for s, e in zip(b_start_i, b_end_i)])
run_dir = np.repeat(b_dir, sizes)
TOTAL = len(run_lin)
run_total = tau_run[-1] + 1 / FS
edges = np.linspace(0, 1.6, NBINS + 1)
occ_bin = np.histogram(run_lin, bins=edges)[0] * (1 / FS)
occ_pos = np.histogram(run_lin[run_dir == 1], bins=edges)[0] * (1 / FS)
occ_neg = np.histogram(run_lin[run_dir == 0], bins=edges)[0] * (1 / FS)
smooth_w = gaussian(NBINS, 1.5)
smooth_w /= smooth_w.sum()

# ---------------- per-unit helpers ----------------
_spk = {}
def spike_at(u):
    if u in _spk:
        return _spk[u]
    st_ = np.asarray(units[u].t)
    if len(st_) == 0:
        _spk[u] = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        return _spk[u]
    bi = np.clip(np.searchsorted(b_end, st_, side="right"), 0, nb - 1)
    inside = (st_ >= b_start[bi]) & (st_ <= b_end[bi])
    tw = st_[inside]
    tau = bout_offsets[bi[inside]] / FS + (tw - b_start[bi[inside]])
    idx = np.clip(np.searchsorted(tau_run, tau, side="left"), 0, TOTAL - 1)
    _spk[u] = (run_lin[idx], run_dir[idx], tau, tw)
    return _spk[u]


def rate1d_from_spikes(u, occ):
    lp = spike_at(u)[0]
    cnt = np.histogram(lp, bins=edges)[0]
    r = cnt / np.maximum(occ, 1e-12)
    r = np.convolve(r, smooth_w, mode="same")
    r[occ <= 0] = 0.0
    r[r < 0] = 0.0
    return r


xy_ok = np.all(np.isfinite(pos2d), axis=1)
xy = pos2d[xy_ok]
tx = t2[xy_ok]

def spikes_2d(u):
    tw = spike_at(u)[3]
    if len(tw) == 0:
        return np.zeros((0, 2)), tw
    idx = np.clip(np.searchsorted(tx, tw, side="right") - 1, 0, len(tx) - 1)
    near = np.abs(tx[idx] - tw) <= 0.026
    return np.column_stack([xy[idx[near], 0], xy[idx[near], 1]]), tw[near]


def rate_map_2d(spxy, sigma=1.1, nx=40, ny=20):
    """2D rate map on the maze geometry (only bins with enough occupancy)."""
    xb = np.linspace(xy[:, 0].min(), xy[:, 0].max(), nx + 1)
    yb = np.linspace(xy[:, 1].min(), xy[:, 1].max(), ny + 1)
    Hocc, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xb, yb])
    occ = Hocc / FS
    if len(spxy) > 0:
        Hsp, _, _ = np.histogram2d(spxy[:, 0], spxy[:, 1], bins=[xb, yb])
    else:
        Hsp = np.zeros_like(Hocc)
    rate = Hsp / np.maximum(occ, 1e-9)
    rate = gaussian_filter(rate, sigma=gamma)
    good = (occ >= 5 * dt := 1 / FS)
    rate[~good] = np.nan
    return rate

# ---------------- stats ----------------
pf = np.load("_placefields.npz")
glm = np.load("_glm.npz")
is_place = (pf["cell_type"].astype(str) == "excitatory") & (pf["peak"] >= 1.0) \
    & (pf["rate"] > 0.1) & (pf["pvals"] < 0.05)
place_pos = np.where(is_place)[0]
order = place_pos[np.argsort(-pf["si_pool"][place_pos])]
example_pos = [int(p) for p in order[:4]]
example_keys = [int(pf["keys"][p]) for p in example_pos]
print("place cells:", len(place_pos), "example:", example_keys)

# ===================================================================
# FIG 1: raw activity
# ===================================================================
fig = plt.figure(figsize=(10.5, 7.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.55, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
sc = ax.scatter(xy[:, 0], xy[:, 1], s=2.5, c=tx, cmap="viridis", alpha=0.85)
ax.set_title("Head trajectory in the maze")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
plt.colorbar(sc, ax=ax, label="time (s)", fraction=0.046, pad=0.03)

ax = fig.add_subplot(gs[0, 1])
for (s, e, d) in bouts:
    ax.plot(t[s:e + 1] - T0, lin_raw[s:e + 1], lw=1.2,
            color=BLUE if d else ORANGE)
ax.set_title("Linearized position, all laps")
ax.set_xlabel("t in maze (s)"); ax.set_ylabel("pos along track (m)")

ax = fig.add_subplot(gs[0, 2])
i0, i1 = int(WIN[0] * LFP), int(WIN[1] * LFP)
tr = h5f["processing/ecephys/LFP/LFP/data"][i0:i1, LFP_CH]
tf = np.arange(i0, i1) / LFP - WIN[0]
ax.plot(tf, tr * LFP_UV, lw=0.4, color=BLUE)
ax.set_title(f"LFP, channel {LFP_CH} (theta)")
ax.set_xlabel("t in window (s)"); ax.set_ylabel("uV"); ax.set_xlim(0, 20)

ax = fig.add_subplot(gs[1, :])
rates = np.array([len(np.asarray(units[u].t)[(np.asarray(units[u].t) >= -np.inf)]) for u in all_keys])
# pick 20 units with most spikes in the % maze epoch
counts = [np.sum((np.asarray(units[u].t) >= T0) & (np.asarray(units[u].t) < MAZE_END)) for u in all_keys]
sel = np.argsort(counts)[-20:]
for i, k in enumerate(sel):
    u = all_keys[k]
    sp = np.asarray(units[u].t)
    sp = sp[(sp >= WIN[0]) & (sp < WIN[1])]
    ax.scatter(sp - WIN[0], np.full(len(sp), i), s=1.5, color=BLUE, lw=0, alpha=0.7)
ax.set_xlabel("time in window (s)"); ax.set_ylabel("unit (top 20 rate)")
ax.set_title("Spike raster of the 20 highest-rate units across the session")
ax.set_ylim(-0.5, len(sel) - 0.5)
h5f2 = None
ax = fig.add_subplot(gs[2, :])
ax.annotate("", xy=(0.01, 0.01))
fig.delaxes(ax)
fig.text(0.02, 0.02, "Dataset: DANDI 000044, sub-Achilles/ses-Achilles-10252013; CA1 tetrodes on a 1.6 m linear maze. "
         "Position timestamps reconstructed at 39.0625 Hz.", fontsize=8, color=GRAY)

fig.savefig("fig01_raw_activity.png")
print("fig01 done")

# fix the accidental assignment above
import matplotlib.pyplot as _plt

# ===================================================================="
# FIG 2: example cells
"# ===================================================================="
fig, axes = plt.subplots(4, 4, figsize=(13, 12), squeeze=False)
for j, u in enumerate(example_keys):
    spxy, tw = spikes_2d(u)
    lp = spike_at(u)[0]
    ax = axes[j, 0]
    ax.plot(xy[:, 0], xy[:, 1], color=LIGHTGRAY, lw=0.8, zorder=1)
    ax.scatter(spxy[:, 0], spxy[:, 1], s=3, color=RED, zorder=3, lw=0)
    ax.set_title(f"unit {u}: path + spikes")
    ax.set_xlabel("x (m)");
    #    ax.set_ylabel("y (m)") if j == 2 else None
    ax = axes[j, 1]
    rp = rate1d_from_spikes(u, occ_pos)
    rn = rate1d_from_spikes(u, occ_neg)
    xc = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(xc, rp, color=BLUE, lw=1.5, label="pos dir")
    ax.plot(xc, rn, color=ORANGE, lw=1.5, label="neg dir")
    ax.set_title("directional tuning")
    ax.set_xlabel("position (m)")
    if j == 0:
        ax.legend(fontsize=7, loc="upper right")
    ax = axes[j, 2]
    rc = rate1d(u, occ_bin)
    ax.fill_between(xc, occ_bin / occ_bin.max(), alpha=0.25, color=GRAY, label="occupancy")
    ax.plot(xc, rc, color=BLUE, lw=1.6)
    ax2 = ax.twinx()
    ax2.fill_between(xc, occ_bin / occ_bin.max, alpha=0.25, color=GRAY)
    ax.set_title(f"pooled map")
    ax.set_xlabel("position (m)")
    ax2.set_ylabel("occupancy (norm)")
    ax2.set_yticks([])
    ax = axes[j, 3]
    rm = rate_map_2d(spxy, sp)
    im = ax.imshow(rm.T, origin="lower", extent=[xy[:, 0].min(), xy[:, 0].max(),
                    xy[:, 1].min(), xy[:, 1].max()], cmap="magma")
    ax.set_title("2D rate map")
    ax.set_xlabel("x (m)")

for j in range(4):
    axes[j, 0].set_ylabel("y (m)" if j == 0 else "")
    axes[j, 1].set_ylabel("rate (Hz)" if j == 0 else "")
    axes[j, 2].set_ylabel("rate (Hz)" if j == 0 else "")
plt.tight_layout()
fig.savefig("fig02_example_cells.png")
print("fig02 written")

# ===================================================================="
# FIG 3: population SI
# ===================================================================="
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax = axes[0, 0]
si_exc = pf["si_pool"][[i for i, c in enumerate(pf["cell_type"].astype(str)) if c == "excitatory"]]
si_inh = pf["si_pool"][[i for i, c in enumerate(pf["cell_type"].astype(str)) if c == "inhibitory"]]
ax.hist(si_exc, bins=40, color=BLUE, alpha=0.7, label=f"excitatory (n={len(si_exc)})")
ax.hist(si_inh, bins=40, color=ORANGE, alpha=0.7, label=f"inhibitory (n={len(si_inh)})")
ax.axvline(np.median(si_exc), color=BLUE, ls="--")
ax.axvline(np.median(si_inh), color=ORANGE, ls="--")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("units")
ax.set_title("Place-field spatial information")
ax.legend(fontsize=8)

ax = axes[0, 1]
nul = pf["null"]
ex = np.where(pf["cell_type"].astype(str) == "excitatory")[0]
nulls = nul[ex].ravel()
ax.hist(nulls, bins=40, color=LIGHTGRAY, label="shuffle null (exc)")
ax.hist(pf["si_pool"][ex], bins=40, color=BLUE, alpha=0.5, label="real (exc)")
ax.set_xlabel("SI (bits/spike)")
ax.set_title("Real vs circular-shift null")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax = axes[1, 0]
pvals = pf["pvals"]
ax.scatter(pf["si_pool"], -np.log10(pvals + 1e-9), s=8, c=["#e34948" if c == "excitatory" else "#eb6834"
         for c in pf["cell_type"].astype(str)])
ax.axhline(-np.log10(0.05), color=GRAY, ls="--", lw=1)
ax.set_xlabel("SI (bits/spike)")
ax.set_ylabel("-log10(p)")
ax.set_title("SI significance (circular shift)")

ax = axes[1, 1]
place_pos = np.where(is_place)[0]
ax.hist(pf["si_pool"][place_pos], bins=40, color=AQUA)
ax.axvline(np.median(pf["si_pool"][place_pos]), color=AQUA, ls="--")
ax.set_xlabel("SI (bits/spike)")
ax.set_title(f"Place cells ({len(place_pos)}/120)")
plt.tight_layout()
fig.savefig("fig03_population_si.png")
print("fig03 written")

# ======================================================================"
# FIG 4: gallery of place-field maps
# ======================================================================"
fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), gridspec_kw={"width_ratios": [4, 1], "wspace": 0.03})
gx = axes[0]
gx.imshow(rate_mat.T, aspect="auto", cmap="magma", origin="lower",
          extent=[0, NBINS, 0, len(where_place) ])
gx.set_yticks([])
gx.set_title("Cell × position firing-rate map (sorted by peak)")

occ_ax = axes[1]
occ_ax.barh(np.arange(len(where_place)), occ_at_peak, height=0.8)
occ_ax.set_title("peak position")
occ_ax.set_xlabel("rate (Hz)")
occ_ax.invert_yaxis()
# (a second column: occupancy profile)
fig.tight_layout()
fig.savefig("fig04_place_field_gallery.png")
print("fig04 written")

# ======================================================================"
# FIG 5: direction selectivity
# ======================================================================"
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
ax = axes[0, 0]
corr = pf["corr"]
ax.hist(corr[np.isfinite(corr)], bins=40, color=BLUE)
ax.axvline(np.nanmedian(corr), color="#2a78d6", ls="--")
ax.set_xlabel("corr(pos map, neg map)")
ax.set
ax.set_title("Directional selectivity (excitatory)")
ax = axes[0, 1]
ax.scatter(pf["si_pool"], corr, s=8, color=BLUE, alpha=0.6)
ax.set_xlabel("SI pooled (bits/spike)")
ax.set_ylabel("pos-neg map corr")
ax.set_title("SI vs directional selectivity")
ax = axes[1, :]
axs = axes[1].flat:
plt.delaxes(axs)
ax = axes[1, 0]
bc = np.array([pf["si_pool"][p] for p in np.where(pf["cell_type"].astype(str) == "excitatory")[0]])
ax that[:, 0]) = None
ax = axes[1, 0]
for i, u in enumerate(example_keys[:2]):
    rp = rate1d_from_spikes(u, occ_pos)
    rn = rate1d_from_spikes(u, occ_neg)
    ax.plot(xc, rp / rp.max(), color=BLUE, ls="-", lw=1.5)
    ax.plot(xc, rn / rn.max(), color=ORANGE, ls="-", lw=1.5)
ax.set_title("Normalized directional curves (2 example cells)")
ax.set_xlabel("position (m)")
plt.tight_layout()
fig.savefig("fig05_direction_selectivity.png")
print("fig05 written")

# ======================================================================"
# FIG 6: GLM encoding
# ======================================================================"
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
cmap = plt.get_cmap("magma")
for j, pos in enumerate(example_pos, start=1):
    ax = axes[0, 0]
    rr = glm["emp_maps"][j]
    gg = glm["pred_maps"][j]
    ax.plot(xc, em, color=GRAY, lw=1.2, label="empirical" if j == 1 else None)
# fix to example_pos
axes[0, 0].plot(x, eg_emp[0], color=LIGHTGRAY, lw=2)
axes[0, 0].plot(x, glm_pred[0], color=BLUE, lw=2)
axes[0, 0].set_title("Example: empirical vs GLM fit")
ax.scatter(glm["si"], glm["r2"], s=12, color=BLUE, alpha=0.7)
r, p = spearmanr(glm["r2"], glm["si"])
ax.text(0.05, 0.9, f"Spearman rho={r:.2f}", transform=ax.transAxes)
ax.set_xlabel("Skaggs SI (bits/spike)")
ax.set_ylabel("McFadden pseudo-R2")
plt.tight_layout()
fig.savefig("fig06_glm_encoding.png")
print("fig06 written")