"""Load MC_Maze (DANDI 000128) via remfile streaming, cache arrays locally."""
import os, numpy as np, h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap

URL = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"
CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)


def open_nwb():
    rf = remfile.File(URL, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()


def load_all(nwbfile=None):
    npz = os.path.join(CACHE, "mc_maze.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        return {k: d[k] for k in d.files}
    n = nwbfile or open_nwb()
    beh = n.processing['behavior'].data_interfaces
    t = beh['hand_vel'].timestamps[:]
    hand_vel = beh['hand_vel'].data[:]
    hand_pos = beh['hand_pos'].data[:]
    units = n.units.to_dataframe()
    # unit -> brain area via electrode table region
    ed = n.electrodes.to_dataframe()
    area = []
    for _, row in units.iterrows():
        eidx = np.atleast_1d(np.asarray(row['electrodes'].index))
        loc = ed.loc[eidx[0], 'group_name'].replace('electrode_group_', '')
        area.append(loc)
    spikes = np.array([np.asarray(s) for s in units['spike_times']], dtype=object)
    tr = n.trials.to_dataframe()
    out = dict(
        t=t, hand_vel=hand_vel, hand_pos=hand_pos,
        spikes=spikes, area=np.array(area), heldout=units['heldout'].values,
        start_time=tr['start_time'].values, stop_time=tr['stop_time'].values,
        go_cue_time=tr['go_cue_time'].values, move_onset_time=tr['move_onset_time'].values,
        target_on_time=tr['target_on_time'].values,
        num_barriers=tr['num_barriers'].values, num_targets=tr['num_targets'].values,
        delay=tr['delay'].values, rt=tr['rt'].values,
        active_target=tr['active_target'].values,
        target_pos=np.array([np.asarray(p) for p in tr['target_pos']], dtype=object),
        split=tr['split'].values.astype(str),
    )
    np.savez(npz, **out)
    return out


def to_pynapple(d):
    """Wrap cached arrays in pynapple objects."""
    vel = nap.TsdFrame(t=d['t'], d=d['hand_vel'], columns=['vx', 'vy'])
    pos = nap.TsdFrame(t=d['t'], d=d['hand_pos'], columns=['x', 'y'])
    spk = nap.TsGroup({i: nap.Ts(t=s) for i, s in enumerate(d['spikes'])},
                      area=d['area'], heldout=d['heldout'])
    return spk, vel, pos


if __name__ == "__main__":
    d = load_all()
    print({k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in d.items()})
    spk, vel, pos = to_pynapple(d)
    print(spk)
    print(vel)
    sp = np.hypot(d['hand_vel'][:, 0], d['hand_vel'][:, 1])
    print("speed pct", np.percentile(sp, [50, 95, 99, 100]))
    print("pos range", d['hand_pos'].min(0), d['hand_pos'].max(0))
    print("rates hz", np.round(np.asarray(spk.rates)[:10], 2))
