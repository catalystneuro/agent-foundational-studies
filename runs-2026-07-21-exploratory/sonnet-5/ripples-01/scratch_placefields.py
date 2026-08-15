import numpy as np
import pynapple as nap
from scratch_load import load_nwb


def get_linearized_position(nwbfile):
    beh = nwbfile.processing['behavior']
    lpos = beh['1.6mLinearMazeLinearizedPosition']
    ss = lpos.spatial_series['1.6mLinearMazeLinearizedTimeSeries']
    dt = ss.rate  # NB: this field actually stores the sample period (s), not Hz
    t = ss.starting_time + np.arange(ss.data.shape[0]) * dt
    d = ss.data[:, 0]
    valid = ~np.isnan(d)
    t_valid = t[valid]
    d_valid = d[valid]

    # build run-segment IntervalSet from contiguous valid samples
    idx = np.where(valid)[0]
    gap = np.where(np.diff(idx) > 1)[0]
    starts_i = np.r_[0, gap + 1]
    ends_i = np.r_[gap, len(idx) - 1]
    seg_start_t = t[idx[starts_i]]
    seg_end_t = t[idx[ends_i]]
    run_ep = nap.IntervalSet(start=seg_start_t, end=seg_end_t)

    pos = nap.Tsd(t=t_valid, d=d_valid, time_support=run_ep)
    return pos, run_ep


if __name__ == '__main__':
    nwbfile, io = load_nwb()
    pos, run_ep = get_linearized_position(nwbfile)
    print('n run segments', len(run_ep))
    print('total run duration (s)', run_ep.tot_length())
    print(pos)

    nwb = nap.NWBFile(nwbfile)
    units = nwb['units']
    meta = units.metadata
    exc_l = meta[(meta['location'] == 'lCA1') & (meta['cell_type'] == 'excitatory')].index.values
    print('n lCA1 excitatory units', len(exc_l))
    pyr = units[exc_l]

    tc = nap.compute_tuning_curves(pyr, pos, bins=40, epochs=run_ep, return_pandas=True)
    print(tc.shape)
    print(tc.head())
    tc.to_pickle('scratch_tuning_curves.pkl')
