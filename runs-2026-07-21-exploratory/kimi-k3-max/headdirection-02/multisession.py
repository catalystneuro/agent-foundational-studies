"""Run HD-cell pipeline across 5 sessions of DANDI 000056 and save per-session results."""
import numpy as np
import pynapple as nap
from tqdm import tqdm

from hd_utils import (
    SESSIONS, load_session, compute_head_direction, get_wake_epochs,
    tuning_curves_hd, classify_hd_cells, mean_vector_length,
)


def state_epochs(nwb, label):
    states = nwb["states"]
    return states[states["label"] == label]


def pair_correlations(group, epochs, bin_size=0.1, min_bins=50):
    cnt = group.count(bin_size, ep=epochs)
    X = cnt.values
    if X.shape[0] < min_bins:
        return None
    C = np.corrcoef(X.T)
    iu = np.triu_indices(X.shape[1], k=1)
    return C[iu]


rng = np.random.default_rng(7)
for name, asset_id in SESSIONS.items():
    print(f"\n=== {name} ===", flush=True)
    nwb, io = load_session(asset_id)
    units = nwb["units"]
    hd = compute_head_direction(nwb)
    wake = get_wake_epochs(nwb, "Awake")
    print(f"{len(units)} units, wake {wake.tot_length():.0f} s", flush=True)

    result = classify_hd_cells(units, hd, wake, n_shuffles=1000, rng=rng)
    keys = result["keys"]
    is_hd = result["is_hd"]
    hd_keys = [k for k, h in zip(keys, is_hd) if h]
    print(f"HD cells: {is_hd.sum()}/{len(keys)} -> {hd_keys}", flush=True)

    # occupancy resultant (expected null MVL scale)
    hd_wake = hd.restrict(wake)
    occ_mvl = mean_vector_length(np.asarray(hd_wake.values))

    # state pairwise correlations among HD cells
    pairs = {}
    if len(hd_keys) >= 4:
        group = units[hd_keys]
        for label in ["Awake", "REM", "Non-REM"]:
            ep = wake if label == "Awake" else state_epochs(nwb, label)
            pairs[label] = pair_correlations(group, ep)
    else:
        print("too few HD cells for state correlations", flush=True)

    out = {
        "session": name,
        "keys": np.array(keys),
        "mvl": result["mvl"],
        "p_value": result["p_value"],
        "pref_angle": result["pref_angle"],
        "null_median": result["null_median"],
        "is_hd": is_hd,
        "occ_mvl": occ_mvl,
    }
    for label, v in pairs.items():
        if v is not None:
            out[f"pairs_{label}"] = v
    np.savez(f"cache/results_{name}.npz", **out)
    io.close()
    print(f"saved cache/results_{name}.npz", flush=True)

print("\nall sessions done")
