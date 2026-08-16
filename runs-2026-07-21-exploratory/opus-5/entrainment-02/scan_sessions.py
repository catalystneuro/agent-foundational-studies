"""Screen DANDI 000059 sessions for: curated 'good' units, maze behavior, cooling trials, LFP w/ theta reference."""
import remfile, h5py, json, urllib.request, numpy as np, collections

API = "https://api.dandiarchive.org/api/dandisets/000059/versions/draft/assets/"
res = json.load(urllib.request.urlopen(API + "?page_size=200"))['results']
def sess(p): return p.split('_ses-')[1].split('_desc')[0]
proc = {sess(x['path']): x for x in res if 'processed' in x['path']}
raw  = {sess(x['path']): x for x in res if 'raw' in x['path']}

def open_asset(a):
    url = API + a['asset_id'] + "/download/"
    return h5py.File(remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache')), 'r')

rows = []
cands = sorted(proc, key=lambda s: -proc[s]['size'])[:14]
for s in cands:
    fp = open_asset(proc[s])
    q = [x.decode() if isinstance(x, bytes) else x for x in fp['units/quality'][:]]
    ngood = sum(1 for x in q if x.startswith('good'))
    has_beh = 'behavior' in fp.get('processing', {})
    cool = None
    if 'intervals' in fp and 'trials' in fp['intervals'] and 'cooling state' in fp['intervals/trials']:
        cool = collections.Counter([x.decode() if isinstance(x, bytes) else x
                                    for x in fp['intervals/trials/cooling state'][:]])
    dur = fp['units/spike_times'][-1] if len(fp['units/spike_times']) else 0
    fr = open_asset(raw[s])
    lfp = fr['processing/ecephys/LFP/LFP/data'] if 'processing/ecephys/LFP/LFP/data' in fr else None
    tref = np.where(fr['general/extracellular_ephys/electrodes/theta_reference'][:])[0]
    rows.append(dict(session=s, ngood=ngood, beh=has_beh, cool=dict(cool) if cool else None,
                     dur=round(float(dur)), lfp=None if lfp is None else lfp.shape,
                     chunks=None if lfp is None else lfp.chunks, theta_ref=tref.tolist()))
    print(rows[-1], flush=True)

json.dump(rows, open('session_scan.json', 'w'), indent=1)
