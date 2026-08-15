import time
import numpy as np
import pynapple as nap
from core_analysis import (load_nwb, select_ripple_channel, detect_ripples,
                            compute_place_fields, analyze_replay)

URL = 'https://api.dandiarchive.org/api/assets/82714afb-724f-4e2b-b102-c9c47b5cba73/download/'  # Buddy

rng = np.random.default_rng(0)
t00 = time.time()
nwbfile, io = load_nwb(URL)
nwb = nap.NWBFile(nwbfile)

epochs = nwbfile.epochs.to_dataframe()
pre_end = epochs.loc[epochs['label'] == 'PREEpoch', 'stop_time'].values[0]
post_start = epochs.loc[epochs['label'] == 'POSTEpoch', 'start_time'].values[0]
pre_win = (max(0, pre_end - 1800), pre_end)
post_win = (post_start, post_start + 2700)

lfp = nwbfile.processing['ecephys']['LFP']['LFP']
fs = lfp.rate
ch, power = select_ripple_channel(lfp, fs, pre_win[0] + 500, 120)
print('selected channel', ch, 'in', time.time() - t00)

t0 = time.time()
i0, i1 = int(pre_win[0] * fs), int(pre_win[1] * fs)
raw_pre = lfp.data[i0:i1, ch].astype(np.float32)
ripples_pre = detect_ripples(raw_pre, fs, t_offset=pre_win[0])
i0, i1 = int(post_win[0] * fs), int(post_win[1] * fs)
raw_post = lfp.data[i0:i1, ch].astype(np.float32)
ripples_post = detect_ripples(raw_post, fs, t_offset=post_win[0])
print('ripples PRE/POST', len(ripples_pre), len(ripples_post), 'fetch+detect time', time.time() - t0)

tc, place_cells, template_rank, pos, run_ep = compute_place_fields(nwbfile, nwb)
print('n place cells', len(place_cells))

units = nwb['units']
df_pre, shuf_pre = analyze_replay(units, ripples_pre, template_rank, rng)
df_post, shuf_post = analyze_replay(units, ripples_post, template_rank, rng)
print('PRE frac sig', (df_pre['pval'] < 0.05).mean(), 'n=', len(df_pre))
print('POST frac sig', (df_post['pval'] < 0.05).mean(), 'n=', len(df_post))
print('total time', time.time() - t00)
