import time
import numpy as np
from _dev_tuning import load_session, compute_tuning, FREQUENCIES

SESSIONS = {
    'LA11': 'https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f',
    'LA12': 'https://dandiarchive.s3.amazonaws.com/blobs/bfd/661/bfd66119-8cd8-4ae7-863a-da43bafdcc58',
    'LA3':  'https://dandiarchive.s3.amazonaws.com/blobs/cac/52e/cac52ee7-20d9-4f7d-a234-a04c22a94083',
    'LA8':  'https://dandiarchive.s3.amazonaws.com/blobs/ea8/b2d/ea8b2d92-31d0-45ab-a300-5e844b1f2a57',
    'LA9':  'https://dandiarchive.s3.amazonaws.com/blobs/cda/b38/cdab388d-11fd-4a5a-b1cf-1e686234c777',
}

all_evoked = []
all_pvals = []
all_subject = []
for subj, url in SESSIONS.items():
    t0 = time.time()
    nwb, nwbfile = load_session(url)
    res = compute_tuning(nwb)
    n = len(res['unit_ids'])
    all_evoked.append(res['evoked'])
    all_pvals.append(res['pvals'])
    all_subject += [subj] * n
    print(subj, 'n_units', n, 'time', round(time.time() - t0, 1),
          'n_sig', np.sum(res['pvals'] < 0.01))

evoked = np.concatenate(all_evoked, axis=0)
pvals = np.concatenate(all_pvals, axis=0)
subject = np.array(all_subject)
np.savez('/tmp/dev_pooled.npz', evoked=evoked, pvals=pvals, subject=subject)
print('TOTAL units', evoked.shape[0], 'sig', np.sum(pvals < 0.01))
