import pickle, numpy as np, collections
d = pickle.load(open('cache_715093703.pkl','rb'))
for k,v in d['stim'].items():
    print('===',k, len(v['start_time']))
    for c in ['orientation','temporal_frequency','spatial_frequency','phase','contrast','stimulus_block']:
        if c in v:
            print(' ', c, collections.Counter(v[c].astype(str)).most_common(10))
    dur = v['stop_time']-v['start_time']
    print('  dur mean/std', dur.mean(), dur.std())
print('regions', collections.Counter(d['meta']['region']))
print('running', d['running']['t'][:3], d['running']['v'][:5])
