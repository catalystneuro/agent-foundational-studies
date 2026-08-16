"""Extract per-unit spike counts per stimulus trial (Allen Visual Coding Neuropixels).

Strategy:
- Read the static/drifting gratings interval tables (excluding the ragged `timeseries` col).
- Exclude trials overlapping `invalid_times` (recorded data-quality intervals).
- Map each unit to a brain region via peak_channel_id -> electrodes.id -> location.
- For every unit, lazily read spike times and count spikes within each stimulus window.
- Save compact npz per stimulus for downstream analysis.
"""
import argparse
import numpy as np
import pandas as pd
import lindi
from pynwb import NWBHDF5IO
from tqdm import tqdm

DANDISET = '000021'


def load_session(asset_id, cache_dir='/tmp/lindi_cache_orient'):
    url = f'https://lindi.neurosift.org/dandi/dandisets/{DANDISET}/assets/{asset_id}/nwb.lindi.json'
    local_cache = lindi.LocalCache(cache_dir=cache_dir)
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    return io.read()


def interval_df(nwbfile, name):
    df = nwbfile.intervals[name].to_dataframe(exclude={'timeseries'})
    return df.reset_index(drop=True)


def build_stimulus(nwbfile, name, post=0.0):
    """Return onset/offset windows and condition labels for a stimulus table."""
    df = interval_df(nwbfile, name)
    onset = df['start_time'].values
    offset = df['stop_time'].values + post
    keep = np.ones(len(df), dtype=bool)
    if 'invalid_times' in nwbfile.intervals:
        inv = interval_df(nwbfile, 'invalid_times')
        for _, r in inv.iterrows():
            keep &= ~((onset < r.stop_time) & (offset >= r.start_time))
    idx = np.where(keep)[0]
    out = {'onset': onset[idx], 'offset': offset[idx]}
    col_short = {'orientation': 'orientation',
                 'spatial_frequency': 'spatial_freq',
                 'temporal_frequency': 'temporal_freq'}
    for col, short in col_short.items():
        if col in df.columns:
            out[short] = pd.to_numeric(df[col].values[idx], errors='coerce').astype(float)
    return out


def count_spikes(sp, onsets, offsets):
    b = np.searchsorted(sp, onsets)
    e = np.searchsorted(sp, offsets)
    return e - b


def extract_session(asset_id, session_id, cache_dir, outdir, do_drifting=True,
                    do_static=True):
    nwbfile = load_session(asset_id, cache_dir)
    ut = nwbfile.units
    quality = ut['quality'][:]
    peak = ut['peak_channel_id'][:]
    id2loc = {int(i): l for i, l in zip(nwbfile.electrodes.id[:],
                                        nwbfile.electrodes['location'][:])}
    regions = np.array([id2loc.get(int(p), 'NA') for p in peak])
    n_units = len(ut)
    good = quality == 'good'
    print(f'session {session_id}: {n_units} units, {good.sum()} good; regions:',
          dict(zip(*np.unique(regions, return_counts=True))))

    def spike_access(u):
        return np.asarray(ut['spike_times'][u]).astype(np.float64)

    keys = []
    if do_drifting:
        keys.append('drifting')
    if do_static:
        keys.append('static')

    for key in keys:
        tab = key + '_gratings_presentations'
        post = 0.1 if key == 'drifting' else 0.0
        try:
            stim = build_stimulus(nwbfile, tab, post=post)
        except KeyError:
            print(f'{key} gratings not present; skipping')
            continue
        n_trials = len(stim['onset'])
        counts = np.zeros((n_units, n_trials), dtype=np.int32)
        ons, offs = stim['onset'], stim['offset']
        for u in tqdm(range(n_units), desc=f'{key} spike counts'):
            counts[u] = count_spikes(spike_access(u), ons, offs)
        out_path = f'{outdir}/{key}_s{session_id}.npz'

        def arr_or_nan(v):
            if v is None:
                return np.full(n_trials, np.nan)
            return np.asarray(v, dtype=np.float64)

        np.savez_compressed(out_path, counts=counts, onset=ons, offset=offs,
                            orientation=arr_or_nan(stim.get('orientation')),
                            spatial_freq=arr_or_nan(stim.get('spatial_freq')),
                            temporal_freq=arr_or_nan(stim.get('temporal_freq')))
        print(f'saved {key}: units={n_units}, trials={n_trials} -> {out_path}')

    meta_path = f'{outdir}/meta_s{session_id}.npz'
    # vlen HDF5 string columns come back as object arrays of bytes; cast to
    # fixed-width unicode so np.load(allow_pickle=False) works downstream.
    np.savez_compressed(meta_path,
                        region=np.asarray(regions).astype('U'),
                        quality=np.asarray(quality).astype('U'))
    print(f'saved meta -> {meta_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='715093703')
    ap.add_argument('--asset', default='58703c97-c0a9-4736-b684-73c85c1a444a')
    ap.add_argument('--cache', default='/tmp/lindi_cache_orient')
    ap.add_argument('--outdir', default='analysis')
    args = ap.parse_args()
    extract_session(args.asset, args.session, args.cache, args.outdir)