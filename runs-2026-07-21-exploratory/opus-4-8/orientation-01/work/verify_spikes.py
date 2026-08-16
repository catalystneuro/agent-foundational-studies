import numpy as np, ov_data as ov
spikes, info = ov.load_spikes(ov.SESSIONS[0])
ut = info['nwbfile'].units
sel = info['selected']
rng = np.random.default_rng(0)
for uid in rng.choice(sel.index.values, 5, replace=False):
    row = int(sel.loc[uid, 'row'])
    ref = np.asarray(ut['spike_times'][row])
    got = spikes[int(uid)].t
    assert len(ref) == len(got), (uid, len(ref), len(got))
    assert np.allclose(ref, got), uid
    print(f'unit {uid} row {row}: {len(ref)} spikes OK  (first {ref[0]:.4f}, last {ref[-1]:.4f})')
print('ALL MATCH')
