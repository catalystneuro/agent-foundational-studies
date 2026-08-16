"""Stream MC_Maze (DANDI:000128, sub-Jenkins) from the archive and cache what we need locally.

The NWB file is ~690 MB, so we stream it with remfile + a disk cache rather than
downloading it, then persist the small set of arrays the analysis actually touches.
"""
import os
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO

ASSET_URL = "https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
CACHE_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "mc_maze_jenkins.npz")


def open_nwb():
    f = remfile.File(ASSET_URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    return NWBHDF5IO(file=h5py.File(f, "r"), load_namespaces=True).read()


def build_cache(path=CACHE_NPZ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nwbfile = open_nwb()

    beh = nwbfile.processing["behavior"].data_interfaces
    hp, hv = beh["hand_pos"], beh["hand_vel"]
    t = np.asarray(hp.timestamps[:], dtype=np.float64)
    # Stored in mm / (mm/s); `conversion` maps to SI. Keep cm and cm/s for readability.
    pos = np.asarray(hp.data[:], dtype=np.float32) * (hp.conversion * 100.0)
    vel = np.asarray(hv.data[:], dtype=np.float32) * (hv.conversion * 100.0)

    st = nwbfile.units["spike_times"]
    spikes = [np.asarray(st[i], dtype=np.float64) for i in range(len(nwbfile.units))]
    spike_lens = np.array([len(s) for s in spikes])

    elec = nwbfile.electrodes.to_dataframe()
    group = elec["group_name"].values
    unit_elec = nwbfile.units["electrodes"]
    area = []
    for i in range(len(nwbfile.units)):
        idx = np.asarray(unit_elec[i].index.values)
        area.append(str(group[idx[0]]) if len(idx) else "unknown")

    obs = np.asarray(nwbfile.units["obs_intervals"][0], dtype=np.float64)

    tr = nwbfile.trials
    target_pos = tr["target_pos"][:]
    active = np.asarray(tr["active_target"][:])
    reach_target = np.array(
        [np.asarray(target_pos[i], dtype=float)[active[i]] for i in range(len(active))]
    )

    out = dict(
        t=t,
        pos=pos,
        vel=vel,
        spikes_flat=np.concatenate(spikes),
        spike_lens=spike_lens,
        unit_area=np.array(area),
        obs_intervals=obs,
        trial_start=np.asarray(tr["start_time"][:]),
        trial_stop=np.asarray(tr["stop_time"][:]),
        trial_target_on=np.asarray(tr["target_on_time"][:]),
        trial_go_cue=np.asarray(tr["go_cue_time"][:]),
        trial_move_onset=np.asarray(tr["move_onset_time"][:]),
        trial_num_barriers=np.asarray(tr["num_barriers"][:]),
        trial_maze_id=np.asarray(tr["maze_id"][:]),
        trial_split=np.asarray(tr["split"][:]).astype(str),
        trial_target_xy=reach_target,  # in mm, same units as raw hand_pos data
        subject=str(nwbfile.subject.subject_id),
        session_description=str(nwbfile.session_description),
    )
    np.savez(path, **out)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.0f} MB)")
    return path


def load_cache(path=CACHE_NPZ):
    if not os.path.exists(path):
        build_cache(path)
    d = np.load(path, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    edges = np.concatenate([[0], np.cumsum(data.pop("spike_lens"))])
    flat = data.pop("spikes_flat")
    data["spike_times"] = [flat[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)]
    data["subject"] = str(data["subject"])
    data["session_description"] = str(data["session_description"])
    return data


if __name__ == "__main__":
    d = load_cache()
    print("subject:", d["subject"])
    print("behavior samples:", d["t"].shape, "dt =", np.median(np.diff(d["t"])))
    print("t range:", d["t"][0], d["t"][-1])
    print("obs interval:", d["obs_intervals"])
    print("units:", len(d["spike_times"]), "areas:", np.unique(d["unit_area"], return_counts=True))
    print("mean rate:", np.mean([len(s) for s in d["spike_times"]]) / (d["t"][-1] - d["t"][0]), "Hz")
    print("trials:", len(d["trial_start"]), "barrier-free:", int((d["trial_num_barriers"] == 0).sum()))
    print("pos range (cm):", np.nanmin(d["pos"], 0), np.nanmax(d["pos"], 0))
    print("speed pct (cm/s):", np.nanpercentile(np.linalg.norm(d["vel"], axis=1), [50, 95, 99.9]))
    print("NaNs pos/vel:", np.isnan(d["pos"]).sum(), np.isnan(d["vel"]).sum())
