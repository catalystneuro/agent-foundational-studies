"""Multi-session grid-cell analysis on DANDI:000582. Saves results to results/."""
import os, numpy as np, pandas as pd, pickle
from tqdm import tqdm
import pynapple as nap
import gridlib

os.makedirs('results', exist_ok=True)

# Top sessions by unit count (2 rats), scanned earlier.
SESSIONS = [
    ('sub-11265_ses-06020601', '60453c7c-79c0-4bc5-8484-7322ae99c4b2'),
    ('sub-11207_ses-18060501', 'edd1a8b4-1d9b-4c9a-9079-e6b006ed455b'),
    ('sub-11207_ses-27060501', '1043a415-4f66-4cdb-b369-cbc0076917f2'),
    ('sub-11265_ses-16030604', 'c18096f7-d802-4813-ba95-7141b58f947a'),
    ('sub-11207_ses-21060503', '11643a0c-f746-4adb-b8b3-e3d9cf90790f'),
    ('sub-11265_ses-09020601', 'f6dc02e9-4917-4a92-b5f7-aadb33364a2e'),
]
N_SHUFFLE = 30  # per cell; pooled into a single null

def spatial_info(rm, pos, nbins=40, extent=50.0):
    """Skaggs spatial information (bits/spike)."""
    x = pos[:, 0].values; y = pos[:, 1].values; t = pos.index.values
    dt = np.median(np.diff(t))
    occ, _, _ = np.histogram2d(x, y, bins=nbins, range=[[-extent, extent]]*2)
    occ = occ.T * dt
    p = occ / occ.sum()
    r = rm.copy(); r[np.isnan(r)] = 0
    mean_r = np.nansum(p * r)
    if mean_r <= 0:
        return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        term = p * (r / mean_r) * np.log2(r / mean_r)
    return np.nansum(term[np.isfinite(term)])

rows = []
maps = {}  # keyed by unit label -> (rate_map, autocorr)
pooled_null = []

for sess, aid in SESSIONS:
    print('loading', sess)
    nwbfile, nwb = gridlib.load_session(aid)
    units = nwb['units']; pos = nwb['SpatialSeriesLED1']
    subject = nwbfile.subject.subject_id
    keys = list(units.keys())
    for i, k in enumerate(tqdm(keys, desc=sess)):
        ut = units[k]
        rm, xe, ye = gridlib.rate_map(ut, pos)
        ac = gridlib.spatial_autocorr(rm)
        gs, params = gridlib.grid_score(ac)
        spacing, orient = gridlib.field_com_stats(ac)
        si = spatial_info(rm, pos)
        label = f'{sess}_u{i}'
        maps[label] = (rm, ac)
        rows.append(dict(session=sess, subject=subject, unit=i, label=label,
                         grid_score=gs, spacing_cm=spacing, orient_deg=orient,
                         peak_rate=np.nanmax(rm), mean_rate=float(ut.rate),
                         spatial_info=si, n_spikes=len(ut)))
        null = gridlib.shuffle_grid_scores(ut, pos, n_shuffles=N_SHUFFLE, seed=i)
        pooled_null.extend(null.tolist())

df = pd.DataFrame(rows)
pooled_null = np.array(pooled_null)
thr = np.percentile(pooled_null, 95)
df['is_grid'] = df['grid_score'] > thr
df.to_csv('results/grid_metrics.csv', index=False)
np.save('results/pooled_null.npy', pooled_null)
with open('results/maps.pkl', 'wb') as f:
    pickle.dump(maps, f)
with open('results/threshold.txt', 'w') as f:
    f.write(str(thr))

print('=== DONE ===')
print(f'total units: {len(df)}  from {df.session.nunique()} sessions, {df.subject.nunique()} rats')
print(f'shuffle 95th-pct threshold: {thr:.3f}')
print(f'grid cells: {df.is_grid.sum()} ({100*df.is_grid.mean():.0f}%)')
print(df[df.is_grid].sort_values('grid_score', ascending=False)[
    ['label','grid_score','spacing_cm','peak_rate','spatial_info']].head(12).to_string())
