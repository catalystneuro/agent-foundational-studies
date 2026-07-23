import h5py, remfile, numpy as np, pandas as pd, json, time
from dandi.dandiapi import DandiAPIClient
from tqdm import tqdm

AREAS = ['VISp','VISl','VISal','VISrl','VISam','VISpm','LGd','LP']
c = DandiAPIClient()
d = c.get_dandiset('000021')
assets = [a for a in d.get_assets() if 'probe' not in a.path]
rows = []
for a in tqdm(sorted(assets, key=lambda x: x.path)):
    url = a.get_content_url(follow_redirects=1, strip_query=True)
    rf = remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
    h5 = h5py.File(rf, 'r')
    # electrodes
    eid = h5['general/extracellular_ephys/electrodes/id'][:]
    loc = np.array([x.decode() if isinstance(x, bytes) else x for x in h5['general/extracellular_ephys/electrodes/location'][:]])
    loc_by_id = dict(zip(eid, loc))
    pc = h5['units/peak_channel_id'][:]
    ulocs = np.array([loc_by_id.get(p, '') for p in pc])
    qual = np.array([x.decode() if isinstance(x, bytes) else x for x in h5['units/quality'][:]])
    good = qual == 'good'
    n_spikes = h5['units/spike_times'].shape[0]
    r = {'path': a.path, 'n_units': len(pc), 'n_good': int(good.sum()), 'n_spikes': n_spikes,
         'size_gb': round(a.size/1e9, 2)}
    for ar in AREAS:
        r[ar] = int(((ulocs == ar) & good).sum())
    r['has_dg'] = 'drifting_gratings_presentations' in h5['intervals']
    r['has_sg'] = 'static_gratings_presentations' in h5['intervals']
    rows.append(r)
    h5.close()

df = pd.DataFrame(rows)
df['VIS_total'] = df[['VISp','VISl','VISal','VISrl','VISam','VISpm']].sum(axis=1)
df.to_csv('session_survey.csv', index=False)
pd.set_option('display.width', 250)
print(df[['path','n_units','n_good','VISp','VISl','VISal','VISrl','VISam','VISpm','LGd','LP','VIS_total','has_dg','has_sg','n_spikes']].to_string())
