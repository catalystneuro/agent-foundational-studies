# GLM encoding analysis with nemos: spike counts ~ velocity direction + speed
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import pickle
from tqdm import tqdm
import nemos as nmo
from scipy.stats import poisson

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
hv = nwb["hand_vel"]

onset_t = trials["move_onset_time"].values
BIN = 0.01
MOVE_EPOCH = nap.IntervalSet(start=onset_t, end=onset_t + 0.6)

print("binning spikes and kinematics ...")
counts = units.count(BIN, ep=MOVE_EPOCH)          # (n_bins, n_units)
hv_bin = hv.bin_average(BIN, ep=MOVE_EPOCH)       # (n_bins, 2)
n_b = min(len(counts), len(hv_bin))
C = counts.values[:n_b]
V = hv_bin.values[:n_b]
speed = np.linalg.norm(V, axis=1)
vang = np.arctan2(V[:, 1], V[:, 0])

# design matrix: [cos, sin, speed]
SPEED_SCALE = 500.0
X_full = np.column_stack([np.cos(vang), np.sin(vang), speed / SPEED_SCALE])
X_dir = X_full[:, :2]
X_spd = X_full[:, 2:3]

# train/val mask from the dataset's own split (per trial, expanded to bins)
nbins_per_trial = int(round(0.6 / BIN))
is_train = (trials["split"].values == "train")
trial_mask = np.repeat(is_train, nbins_per_trial)[:n_b]
print(f"bins: {n_b}, train frac: {trial_mask.mean():.2f}")

def poisson_ll(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return poisson.logpmf(y, mu).sum()

n_units = C.shape[1]
results = np.zeros((n_units, 4))  # pseudo-R2: full, dir, spd, and null LL
ll_full = np.zeros(n_units); ll_dir = np.zeros(n_units); ll_spd = np.zeros(n_units); ll_null = np.zeros(n_units)
models = {}
for u in tqdm(range(n_units), desc="GLM fits"):
    y = C[:, u].astype(float)
    ytr, yte = y[trial_mask], y[~trial_mask]
    ll_null[u] = poisson_ll(yte, np.full_like(yte, ytr.mean()))
    for k, X in enumerate([X_full, X_dir, X_spd]):
        m = nmo.glm.GLM(solver_name="LBFGS")
        # explicit init: intercept = log(mean count), avoids failure when a unit
        # has (near-)zero spikes in the training bins
        init_inter = np.log(np.clip(ytr.mean(), 1e-6, None))
        init_params = (np.zeros(X.shape[1]), np.array([init_inter]))
        m.fit(X[trial_mask], ytr, init_params=init_params)
        mu = m.predict(X[~trial_mask])
        ll = poisson_ll(yte, np.asarray(mu))
        if k == 0:
            ll_full[u] = ll
            if u == 0:
                models["example_pred"] = np.asarray(mu)
                models["example_y"] = yte
        elif k == 1:
            ll_dir[u] = ll
        else:
            ll_spd[u] = ll

def pseudo_r2(ll_model, ll_null):
    return 1 - ll_model / ll_null

pr2_full = pseudo_r2(ll_full, ll_null)
pr2_dir = pseudo_r2(ll_dir, ll_null)
pr2_spd = pseudo_r2(ll_spd, ll_null)
print("median pseudo-R2 full:", np.median(pr2_full).round(4))
print("median pseudo-R2 dir: ", np.median(pr2_dir).round(4))
print("median pseudo-R2 spd: ", np.median(pr2_spd).round(4))
print("frac full>0:", (pr2_full > 0).mean().round(2))

# unique contribution: ΔLL
dll_dir = ll_full - ll_spd   # direction contribution beyond speed
dll_spd = ll_full - ll_dir   # speed contribution beyond direction
np.savez("glm_results.npz", pr2_full=pr2_full, pr2_dir=pr2_dir, pr2_spd=pr2_spd,
         dll_dir=dll_dir, dll_spd=dll_spd, ll_null=ll_null,
         example_pred=models["example_pred"], example_y=models["example_y"])
print("saved glm_results.npz")
