"""Loading utilities for DANDI dandiset 000939 (Duszkiewicz et al., postsubiculum HD cells)."""
import json, os
import h5py, remfile, numpy as np, pandas as pd
import pynapple as nap
from pynwb import NWBHDF5IO

ASSETS_JSON = "assets_939.json"
CACHE_DIR = "/tmp/remfile_cache"


def get_assets():
    if os.path.exists(ASSETS_JSON):
        return json.load(open(ASSETS_JSON))
    import requests
    r = requests.get("https://api.dandiarchive.org/api/dandisets/000939/versions/draft/assets/",
                     params={"page_size": 100})
    out = {a["path"]: f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
           for a in r.json()["results"]}
    json.dump(out, open(ASSETS_JSON, "w"), indent=1)
    return out


def load_session(path):
    """Stream one NWB session and return a dict of pynapple objects."""
    url = get_assets()[path]
    h5 = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR)), "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()

    # --- units -------------------------------------------------------------
    ut = nwbfile.units.to_dataframe()
    spikes = {i: np.asarray(ut["spike_times"].iloc[i]) for i in range(len(ut))}
    meta = pd.DataFrame({
        "is_hd": ut["is_head_direction"].values.astype(bool),
        "is_exc": ut["is_excitatory"].values.astype(bool),
        "is_fs": ut["is_fast_spiking"].values.astype(bool),
        "channel": np.asarray(ut["electrode_index"].values, dtype=int),
    }, index=np.arange(len(ut)))
    units = nap.TsGroup({k: nap.Ts(v) for k, v in spikes.items()},
                        is_hd=meta["is_hd"].values, is_exc=meta["is_exc"].values,
                        is_fs=meta["is_fs"].values, channel=meta["channel"].values)

    # --- head direction ----------------------------------------------------
    hd_ts = nwbfile.processing["behavior"]["CompassDirection"]["head-direction"]
    hd = nap.Tsd(t=np.asarray(hd_ts.timestamps[:]), d=np.asarray(hd_ts.data[:]))
    pos_ts = nwbfile.processing["behavior"]["Position"]["position"]
    pos = nap.TsdFrame(t=np.asarray(pos_ts.timestamps[:]), d=np.asarray(pos_ts.data[:]),
                       columns=["x", "y"])

    # --- epochs / sleep states --------------------------------------------
    ep = nwbfile.epochs.to_dataframe()
    epochs = {}
    for _, row in ep.iterrows():
        tag = row["tags"][0] if isinstance(row["tags"], (list, tuple, np.ndarray)) else str(row["tags"])
        iset = nap.IntervalSet(start=row["start_time"], end=row["stop_time"])
        epochs[tag] = iset if tag not in epochs else epochs[tag].union(iset)

    ss = nwbfile.intervals["sleep_states"].to_dataframe()
    states = {}
    for s in sorted(set(ss["state"])):
        sub = ss[ss["state"] == s]
        states[s] = nap.IntervalSet(start=sub["start_time"].values.astype(float),
                                    end=sub["stop_time"].values.astype(float))

    return dict(units=units, hd=hd, pos=pos, epochs=epochs, states=states,
                nwbfile=nwbfile, io=io, session=path.split("ses-")[1].split("_")[0],
                subject=path.split("sub-")[1].split("/")[0])
