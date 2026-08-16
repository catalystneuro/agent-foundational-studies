"""Run the DANDI:000986 tuning pipeline over all 15 sessions and cache the results."""
import pickle
import numpy as np
from tqdm import tqdm
import loaders, pipeline986

assets = loaders.list_assets('000986')
paths = sorted(assets)
results = {}
for p in tqdm(paths, desc='000986 sessions'):
    results[p] = pipeline986.analyze_session(p)
    s = results[p]['stats']
    tqdm.write(f"{p}: {len(s)} units, {int(s.responsive.sum())} responsive, "
               f"{int(s.tuned.sum())} freq-tuned, {results[p]['n_trials']} trials")
with open('results_000986.pkl', 'wb') as fh:
    pickle.dump(results, fh)
print('saved results_000986.pkl')
