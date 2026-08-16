"""Multi-session orientation selectivity analysis on DANDI 000021.

For each session: load good units from visual cortical areas, compute firing
rates per drifting-grating sweep, direction tuning curves, selectivity metrics
(gOSI, gDSI, OSI), and a permutation test for significant orientation tuning.
Saves per-session CSVs for figure generation.
"""
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import pandas as pd
from tqdm import tqdm

SESSIONS = {
    "715093703": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json"),
    "719161530": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "02291b99-e583-498b-9929-b68bba2c50e2/nwb.lindi.json"),
    "721123822": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "224b57e5-c9a3-46ef-85db-966713f3ccbe/nwb.lindi.json"),
}

VISUAL_AREAS = ["VISp", "VISl", "VISpm", "VISam", "VISrl"]

N_SHUFFLES = 500
rng = np.random.default_rng(42)


def intervals_to_df(tab, cols):
    return pd.DataFrame({c: np.asarray(tab[c].data[:]) for c in cols})


def gosi_matrix(mean_rates, dirs_deg):
    """Vectorized global OSI for a (n_units x n_dirs) rate matrix."""
    th = np.deg2rad(dirs_deg)
    vec = mean_rates @ np.exp(2j * th)
    denom = mean_rates.sum(axis=1)
    out = np.zeros(len(mean_rates))
    np.divide(np.abs(vec), denom, out=out, where=denom > 0)
    return out


def gdsi_matrix(mean_rates, dirs_deg):
    th = np.deg2rad(dirs_deg)
    vec = mean_rates @ np.exp(1j * th)
    denom = mean_rates.sum(axis=1)
    out = np.zeros(len(mean_rates))
    np.divide(np.abs(vec), denom, out=out, where=denom > 0)
    return out


def analyze_session(session_id, lindi_url):
    print(f"\n=== session {session_id} ===")
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)

    units = nwb["units"]
    elec = nwbfile.electrodes.to_dataframe()
    meta = units.metadata.copy()
    meta["structure"] = meta["peak_channel_id"].map(elec["location"])
    keep = meta[(meta["quality"] == "good") & (meta["structure"].isin(VISUAL_AREAS))]
    unit_ids = list(keep.index)
    print(f"good visual-cortex units: {len(unit_ids)}")
    print(keep["structure"].value_counts().to_dict())

    tab = nwbfile.intervals["drifting_gratings_presentations"]
    dg = intervals_to_df(tab, ["start_time", "stop_time", "orientation", "temporal_frequency"])
    dg.columns = ["start", "stop", "orientation", "temporal_frequency"]
    blank = dg[dg["orientation"].isna()].reset_index(drop=True)
    dg = dg.dropna(subset=["orientation"]).reset_index(drop=True)
    dg["orientation"] = dg["orientation"].astype(float)
    dg["duration"] = dg["stop"] - dg["start"]

    starts, stops = dg["start"].to_numpy(), dg["stop"].to_numpy()
    n = len(dg)
    counts = np.zeros((len(unit_ids), n))
    for i, uid in enumerate(tqdm(unit_ids, desc="reading spike trains")):
        st = np.asarray(nwb["units"][uid].times())
        counts[i] = np.searchsorted(st, stops) - np.searchsorted(st, starts)
    rates = counts / dg["duration"].to_numpy()[None, :]

    # spontaneous rate from blank sweeps
    if len(blank):
        bcounts = np.zeros((len(unit_ids), len(blank)))
        bs, be = blank["start"].to_numpy(), blank["stop"].to_numpy()
        for i, uid in enumerate(unit_ids):
            st = np.asarray(nwb["units"][uid].times())
            bcounts[i] = np.searchsorted(st, be) - np.searchsorted(st, bs)
        spont = (bcounts / (be - bs)[None, :]).mean(axis=1)
    else:
        spont = np.full(len(unit_ids), np.nan)

    # tuning curves
    directions = np.sort(dg["orientation"].unique())
    labels = dg["orientation"].to_numpy()
    oh = np.zeros((n, len(directions)))
    for j, d in enumerate(directions):
        oh[labels == d, j] = 1.0
    n_per = oh.sum(axis=0)
    mean_rates = (rates @ oh) / n_per[None, :]

    # metrics
    gOSI = gosi_matrix(mean_rates, directions)
    gDSI = gdsi_matrix(mean_rates, directions)

    # permutation test on gOSI
    exceed = np.zeros(len(unit_ids), dtype=int)
    for s in range(N_SHUFFLES):
        perm = rng.permutation(n)
        m = (rates @ oh[perm]) / n_per[None, :]
        exceed += gosi_matrix(m, directions) >= gOSI
    pvals = (1 + exceed) / (1 + N_SHUFFLES)

    # preferred direction/orientation
    pref_idx = np.argmax(mean_rates, axis=1)
    pref_dir = directions[pref_idx]

    out = pd.DataFrame({
        "unit_id": unit_ids,
        "session": session_id,
        "structure": keep["structure"].to_numpy(),
        "mean_rate": rates.mean(axis=1),
        "spont_rate": spont,
        "max_rate": mean_rates.max(axis=1),
        "pref_dir": pref_dir,
        "pref_ori": pref_dir % 180,
        "gOSI": gOSI,
        "gDSI": gDSI,
        "p_gOSI": pvals,
    })
    tuning_df = pd.DataFrame(mean_rates, index=unit_ids, columns=directions)
    tuning_df.index.name = "unit_id"
    return out, tuning_df


all_metrics, all_tuning = [], []
for sid, url in SESSIONS.items():
    m, t = analyze_session(sid, url)
    m.to_csv(f"metrics_{sid}.csv", index=False)
    t.to_csv(f"tuning_{sid}.csv")
    all_metrics.append(m)
    all_tuning.append(t)

metrics = pd.concat(all_metrics, ignore_index=True)
metrics.to_csv("metrics_all_sessions.csv", index=False)
print("\n=== pooled ===")
print("total units:", len(metrics))
print("fraction p<0.01:", (metrics["p_gOSI"] < 0.01).mean())
print("fraction p<0.05:", (metrics["p_gOSI"] < 0.05).mean())
print(metrics.groupby("structure").agg(n=("unit_id", "count"),
                                       frac_sig=("p_gOSI", lambda x: (x < 0.01).mean()),
                                       median_gOSI=("gOSI", "median")))
