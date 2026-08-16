import remfile, h5py, time
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
from scipy.stats import f_oneway

FREQUENCIES = [2000.0, 4000.0, 8000.0, 16000.0, 32000.0]
RESP_WIN = (0.0, 0.05)
BASE_WIN = (-0.05, 0.0)

def load_session(url, cache_dir='/tmp/remfile_cache'):
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, 'r')
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, nwbfile

def compute_tuning(nwb, frequencies=FREQUENCIES):
    units = nwb['units']
    idx = list(units.index)
    trials = nwb['trials'].as_dataframe()
    trials['start'] = nwb['trials'].start
    n_units = len(idx)
    n_freq = len(frequencies)
    tuning = np.zeros((n_units, n_freq))
    baseline = np.zeros((n_units, n_freq))
    trial_evoked = {ui: [] for ui in idx}

    for fi, f in enumerate(frequencies):
        sub = trials[trials.stim_frequency == f]
        events = nap.Ts(t=sub['start'].values)
        peri = nap.compute_perievent(units, events, window=(BASE_WIN[0], RESP_WIN[1]))
        for ui in idx:
            grp = peri[ui]
            resp = grp.restrict(nap.IntervalSet(*RESP_WIN)).get_info('rate').values
            base = grp.restrict(nap.IntervalSet(*BASE_WIN)).get_info('rate').values
            tuning[idx.index(ui), fi] = np.mean(resp)
            baseline[idx.index(ui), fi] = np.mean(base)
            trial_evoked[ui].append(resp - base)

    pvals = np.ones(n_units)
    for i, ui in enumerate(idx):
        groups = trial_evoked[ui]
        if all(len(g) > 1 for g in groups):
            _, p = f_oneway(*groups)
            pvals[i] = p

    evoked = tuning - baseline
    best_freq_idx = np.argmax(evoked, axis=1)
    return dict(evoked=evoked, tuning=tuning, baseline=baseline, pvals=pvals,
                best_freq_idx=best_freq_idx, unit_ids=idx)

if __name__ == '__main__':
    url = 'https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f'
    t0 = time.time()
    nwb, nwbfile = load_session(url)
    print('load', time.time() - t0)
    t0 = time.time()
    res = compute_tuning(nwb)
    print('tuning compute', time.time() - t0)
    print('n units', len(res['unit_ids']))
    print('pvals < 0.01:', np.sum(res['pvals'] < 0.01))
    print('max evoked per unit (first 10):', np.max(res['evoked'], axis=1)[:10])
    np.save('/tmp/dev_evoked.npy', res['evoked'])
    np.save('/tmp/dev_pvals.npy', res['pvals'])
