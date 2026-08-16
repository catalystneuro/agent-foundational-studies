"""Poisson GLM encoding of place fields (nemos): raised-cosine basis over position.

Each valid run sample (dt = 25.6 ms) is one time bin with the linearized
position as predictor; spike counts per bin are the target. The GLM-predicted
binned rate reproduces the empirical tuning curve (a smooth parametric place
field). Pseudo-R2 (McFadden) is computed manually against the intercept-only
Poisson null.
"""
import json
import numpy as np
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import jax

jax.config.update("jax_enable_x64", True)
import nemos as nmo
from scipy.signal.windows import gaussian
from scipy.special import gammaln

NBINS = 50
FS = 39.0625
T0 = 18079.5
RIDGE = 1e-4

with open("_asset.json") as f:
    info = json.load(f)
rem_file = remfile.File(info["s3_url"], disk_cache=remfile.DiskCache("/tmp/remfile_cache_000044"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f)
nwb = nap.NWBFile(io.read())
units = nwb["units"]

lin_raw = np.asarray(nwb["1.6mLinearMazeLinearizedTimeSeries"].values).ravel()
t = T0 + np.arange(len(lin_raw)) / FS

# ---------- run bins (valid run samples, dt = 1/FS) ----------
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
        bouts.append((s, e))
b_start_i = np.array([b[0] for b in bouts])
b_end_i = np.array([b[1] for b in bouts])
b_start = t[b_start_i]
b_end = t[b_end_i]
nb = len(bouts)

sizes = b_end_i - b_start_i + 1
bout_offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
tau_run = np.concatenate([np.arange(s) / FS + bout_offsets[i] / FS for i, s in enumerate(sizes)])
run_lin = np.concatenate([lin_raw[s:e + 1] for s, e in zip(b_start_i, b_end_i)])
run_total = tau_run[-1] + 1 / FS
bin_edges_tau = np.concatenate([tau_run - 0.5 / FS, [run_total]])

# ---------- per-unit spike tau ----------
all_keys = list(units.keys())
spk_tau = {}
for u in all_keys:
    st_ = np.asarray(units[u].t)
    if len(st_) == 0:
        spk_tau[u] = np.zeros(0)
        continue
    bi = np.clip(np.searchsorted(b_end, st_, side="right"), 0, nb - 1)
    inside = (st_ >= b_start[bi]) & (st_ <= b_end[bi])
    spk_tau[u] = bout_offsets[bi[inside]] / FS + (st_[inside] - b_start[bi[inside]])

# ---------- design matrix ----------
basis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=12, bounds=(0.0, 1.6), label="pos")
X = np.asarray(basis.compute_features(run_lin))

# ---------- place cells ----------
pf = np.load("_placefields.npz")
keys = pf["keys"]
cell_positions = np.where((pf["cell_type"].astype(str) == "excitatory")
                          & (pf["peak"] >= 1.0) & (pf["rate"] > 0.1) & (pf["pvals"] < 0.05))[0]
cell_keys = [int(keys[i]) for i in cell_positions]
print(f"fitting GLM on {len(cell_keys)} excitatory place cells")

edges = np.linspace(0, 1.6, NBINS + 1)
smooth_w = gaussian(NBINS, 1.5)
smooth_w /= smooth_w.sum()
occ_bin = np.histogram(run_lin, bins=edges)[0] * (1 / FS)


def binned_map(rate_per_bin):
    cnt_binned = np.histogram(run_lin, bins=edges, weights=rate_per_bin)[0]
    m = cnt_binned / np.maximum(occ_bin, 1e-12)
    m = np.convolve(m, smooth_w, mode="same")
    m[occ_bin <= 0] = 0.0
    return m


def empirical_map(spike_pos):
    return binned_map(np.histogram(spike_pos, bins=edges)[0])  # counts -> rate via occ normalization


def pois_ll(y, rate):
    return float(np.sum(y * np.log(np.maximum(rate, 1e-12)) - rate - gammaln(y + 1)))


glm_pred_maps = []
glm_emp_maps = []
r2_list = []
for i, u in enumerate(cell_keys):
    y = np.histogram(spk_tau[u], bins=bin_edges_tau)[0]
    glmer = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=RIDGE,
                        solver_name="LBFGS", solver_kwargs={"maxiter": 5000})
    glmer.fit(X, y)
    pred = glmer.predict(X)
    ok = np.isfinite(pred)  # basis is undefined at exact track ends -> NaN bins
    y_ok = np.asarray(y[ok], dtype=float)
    ll_full = pois_ll(y_ok, pred[ok])
    mu = float(y_ok.mean())
    ll_null = pois_ll(y_ok, np.full(y_ok.shape[0], mu, dtype=float))
    r2_list.append(1.0 - ll_full / ll_null if ll_null < 0 else np.nan)
    glm_pred_maps.append(binned_map(pred))
    glm_emp_maps.append(binned_map(y))  # empirical binned map from per-bin counts
    if i % 20 == 0:
        print(f"fit {i + 1}/{len(cell_keys)}")

np.savez("_glm.npz",
         keys=np.array(cell_keys, dtype=int),
         r2=np.array(r2_list),
         si=pf["si_pool"][cell_positions],
         pred_maps=np.array(glm_pred_maps),
         emp_maps=np.array(glm_emp_maps))
print("saved _glm.npz")
print(f"median McFadden pseudo-R2: {np.nanmedian(r2_list):.3f}")
corr_list = [np.corrcoef(pm, em)[0, 1] for pm, em in zip(glm_pred_maps, glm_emp_maps)]
print(f"median pred-vs-emp map corr: {np.nanmedian(corr_list):.3f}")