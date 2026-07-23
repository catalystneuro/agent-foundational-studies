"""Pull the pieces of a DANDI:000021 session that the orientation analysis needs.

Only three things are read out of each ~2.5 GB NWB file: the units table
metadata, the spike times of units in visual structures, and the drifting /
static grating presentation tables. Everything is cached to a local .npz so
repeated runs do not re-stream.
"""

import os

import numpy as np
from tqdm import tqdm

import dandi_io

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_cache")

# Structures kept for the region comparison, ordered thalamus -> primary -> higher.
VISUAL_STRUCTURES = ["LGd", "VISp", "VISl", "VISal", "VISrl", "VISpm", "VISam"]


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def _stim_table(f, name):
    grp = f[f"intervals/{name}_presentations"]
    out = {"start_time": grp["start_time"][:], "stop_time": grp["stop_time"][:]}
    for col in ["orientation", "temporal_frequency", "spatial_frequency", "contrast"]:
        if col in grp:
            v = grp[col][:]
            if v.dtype.kind in "OS":
                v = np.array(
                    [np.nan if _d(x) in ("null", "N/A", "") else float(_d(x)) for x in v]
                )
            out[col] = v
    return out


def _d(x):
    return x.decode() if isinstance(x, bytes) else x


def extract_session(asset_id, session_name, structures=VISUAL_STRUCTURES):
    """Return a dict of arrays for one session, caching to disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{session_name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    f = dandi_io.open_h5(asset_id)

    el = f["general/extracellular_ephys/electrodes"]
    chan_to_loc = dict(zip(el["id"][:], _decode(el["location"][:])))
    chan_to_depth = dict(zip(el["id"][:], el["probe_vertical_position"][:]))

    u = f["units"]
    peak_chan = u["peak_channel_id"][:]
    location = np.array([chan_to_loc.get(c, "") for c in peak_chan])
    depth = np.array([chan_to_depth.get(c, np.nan) for c in peak_chan], dtype=float)
    quality = _decode(u["quality"][:])

    # Allen's recommended unit-quality filter.
    keep = (
        np.isin(location, structures)
        & (quality == "good")
        & (u["amplitude_cutoff"][:] < 0.1)
        & (u["isi_violations"][:] < 0.5)
        & (u["presence_ratio"][:] > 0.9)
    )
    rows = np.where(keep)[0]

    spikes = {}
    index = u["spike_times_index"][:]
    starts = np.concatenate([[0], index[:-1]])
    data = u["spike_times"]
    for r in tqdm(rows, desc=f"{session_name}: spike times"):
        spikes[int(r)] = data[starts[r] : index[r]]

    dg = _stim_table(f, "drifting_gratings")
    sg = _stim_table(f, "static_gratings")

    out = {
        "session": np.array(session_name),
        "unit_id": u["id"][:][rows],
        "location": location[rows],
        "depth": depth[rows],
        "snr": u["snr"][:][rows],
        "firing_rate": u["firing_rate"][:][rows],
        "waveform_duration": u["waveform_duration"][:][rows],
        "spike_times": np.array([spikes[int(r)] for r in rows], dtype=object),
    }
    for k, v in dg.items():
        out["dg_" + k] = v
    for k, v in sg.items():
        out["sg_" + k] = v

    np.savez(path, **out)
    return out


if __name__ == "__main__":
    d = extract_session(*dandi_io.SESSION_ASSETS[0])
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in d.items()})
    import collections

    print(collections.Counter(d["location"].tolist()))
