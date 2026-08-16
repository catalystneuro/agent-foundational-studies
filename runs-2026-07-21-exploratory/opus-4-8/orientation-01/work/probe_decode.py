import numpy as np, warnings, os, sys
sys.path.insert(0, ".")
os.environ.setdefault("MPLBACKEND", "Agg")
import runpy
# import just the helpers we need by exec'ing the top of the script is messy; re-implement load
import pandas as pd, h5py, remfile, pynapple as nap
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

CACHE_DIR = os.path.expanduser("~/.cache/dandi_000021")
tag = "715093703"
z = np.load(os.path.join(CACHE_DIR, f"spikes_{tag}.npz"))
sel = pd.read_parquet(os.path.join(CACHE_DIR, f"meta_{tag}.parquet"))
spike_dict = {int(k): z[f"u{k}"] for k in sel.index}
t_end = max(v[-1] for v in spike_dict.values() if len(v))
spikes = nap.TsGroup({k: nap.Ts(t=np.sort(v)) for k, v in spike_dict.items()},
                     time_support=nap.IntervalSet(start=0.0, end=float(t_end)+1.0),
                     metadata=sel.loc[list(spike_dict.keys()), ["area"]])

url_cache = os.path.join(CACHE_DIR, "remfile")
with DandiAPIClient() as c:
    a = c.get_dandiset("000021","draft").get_asset_by_path("sub-699733573/sub-699733573_ses-715093703.nwb")
    url = a.get_content_url(follow_redirects=1, strip_query=True)
h5 = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(url_cache)), "r")
nwbfile = NWBHDF5IO(file=h5, load_namespaces=True).read()
dg = nwbfile.intervals["drifting_gratings_presentations"].to_dataframe()
for cc in ["orientation","temporal_frequency"]:
    dg[cc] = pd.to_numeric(dg[cc], errors="coerce")
dg["duration"] = dg["stop_time"] - dg["start_time"]
driven = dg[dg["orientation"].notna()]

start = driven["start_time"].values
stop = start + driven["duration"].values
cnt = spikes.count(ep=nap.IntervalSet(start=start, end=stop))
rates = np.asarray(cnt.values).T / (stop-start)[None,:]
print("rates shape", rates.shape)
print("finite:", np.isfinite(rates).all(), " max:", np.nanmax(rates), " min:", np.nanmin(rates))
print("durations: min", (stop-start).min(), "max", (stop-start).max())
counts = rates.T * 2.0
print("counts finite:", np.isfinite(counts).all(), "max", counts.max(), "dtype", counts.dtype)

# now emulate the decoder inner math
labels = driven["orientation"].values
levels = np.unique(labels)
from sklearn.model_selection import StratifiedKFold
areas = np.asarray(spikes.metadata["area"])
idx = np.where(areas=="VISp")[0]
rng = np.random.default_rng(0)
sub = rng.choice(idx, 30, replace=False)
c2 = rates[sub].T * 2.0
with warnings.catch_warnings():
    warnings.simplefilter("error")
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(c2, labels):
        lam = np.clip(np.stack([c2[tr][labels[tr]==lv].mean(axis=0) for lv in levels]), 1e-3, None)
        print("lam finite", np.isfinite(lam).all(), "loglam min", np.log(lam).min())
        try:
            s = c2[te] @ np.log(lam).T
            print("  ok, max", s.max())
        except Warning as w:
            print("  WARN:", w)
        break

# --- verify matmul result correctness despite the warning ---
lam = np.clip(np.stack([c2[labels!=labels[0]][:0].mean(axis=0)]),1e-3,None) if False else None
tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(c2, labels)))
lam = np.clip(np.stack([c2[tr][labels[tr]==lv].mean(axis=0) for lv in levels]), 1e-3, None)
L = np.log(lam).T
with np.errstate(all="ignore"):
    A = c2[te] @ L
B = np.einsum("ij,jk->ik", c2[te], L, optimize=False)
C = np.array([[float(np.dot(r, L[:,k])) for k in range(L.shape[1])] for r in c2[te]])
print("matmul vs einsum  allclose:", np.allclose(A,B), " max abs diff:", np.max(np.abs(A-B)))
print("matmul vs pyloop  allclose:", np.allclose(A,C), " max abs diff:", np.max(np.abs(A-C)))
print("A finite:", np.isfinite(A).all())
