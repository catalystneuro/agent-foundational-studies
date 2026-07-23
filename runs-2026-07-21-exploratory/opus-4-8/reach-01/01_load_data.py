"""Load MC_Maze (DANDI:000128) session into pynapple objects, cache locally."""
import os, numpy as np, lindi, pynapple as nap
from pynwb import NWBHDF5IO

URL = ('https://lindi.neurosift.org/dandi/dandisets/000128/assets/'
       '26e85f09-39b7-480f-b337-278a8f034007/nwb.lindi.json')
CACHE = './cache'

def load_nwb():
    os.makedirs(CACHE, exist_ok=True)
    f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=lindi.LocalCache(cache_dir=CACHE + '/lindi'))
    return NWBHDF5IO(file=f, mode='r').read()

def load_session():
    """Returns dict with pynapple objects: units (TsGroup), hand_pos/hand_vel (TsdFrame), trials (DataFrame)."""
    npz = CACHE + '/session.npz'
    nwbfile = load_nwb()
    trials = nwbfile.trials.to_dataframe()
    units_df = nwbfile.units.to_dataframe()

    if os.path.exists(npz):
        d = np.load(npz)
        t, hp, hv = d['t'], d['hp'], d['hv']
    else:
        beh = nwbfile.processing['behavior'].data_interfaces
        t = np.asarray(beh['hand_pos'].timestamps[:])
        hp = np.asarray(beh['hand_pos'].data[:], dtype=np.float32)
        hv = np.asarray(beh['hand_vel'].data[:], dtype=np.float32)
        cp = np.asarray(beh['cursor_pos'].data[:], dtype=np.float32)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(npz, t=t, hp=hp, hv=hv)
        # cursor_pos shares the target coordinate frame (hand_pos is offset by ~35 mm in y),
        # so reach geometry is computed from the cursor.
        np.savez_compressed(CACHE + '/cursor.npz', t=t, cp=cp)

    hand_pos = nap.TsdFrame(t=t, d=hp, columns=['x', 'y'])
    hand_vel = nap.TsdFrame(t=t, d=hv, columns=['vx', 'vy'])
    spikes = nap.TsGroup({i: np.asarray(s) for i, s in enumerate(units_df['spike_times'])},
                         heldout=units_df['heldout'].values.astype(bool))
    return dict(spikes=spikes, hand_pos=hand_pos, hand_vel=hand_vel, trials=trials, nwbfile=nwbfile)

if __name__ == '__main__':
    d = load_session()
    print(d['spikes'])
    print(d['hand_pos'])
    print(d['hand_vel'])
    print('duration (s)', d['hand_pos'].index[-1])
    hv = d['hand_vel'].values
    print('speed percentiles', np.percentile(np.hypot(hv[:,0], hv[:,1]), [50,90,99,100]))
    print('NaNs pos/vel', np.isnan(d['hand_pos'].values).sum(), np.isnan(hv).sum())
