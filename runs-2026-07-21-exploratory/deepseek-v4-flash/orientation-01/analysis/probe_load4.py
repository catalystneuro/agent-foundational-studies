"""Check presentation timing structure for static and drifting gratings."""
import lindi
from pynwb import NWBHDF5IO
import numpy as np

local_cache = lindi.LocalCache(cache_dir='/tmp/lindi_cache_orient')
url = 'https://lindi.neurosift.org/dandi/dandisets/000021/assets/58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json'
f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()

sg = nwbfile.intervals['static_gratings_presentations'].to_dataframe(exclude={'timeseries'})
dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe(exclude={'timeseries'})

for name, df in [('static', sg), ('drifting', dg)]:
    on = df.start_time.values
    gaps = np.diff(on)
    print(f'{name}: n={len(df)}')
    print(f'  gap between onsets: min={gaps.min():.4f}, median={np.median(gaps):.4f}, max={gaps.max():.4f}')
    print(f'  stop-start: min={np.min(df.stop_time.values-df.start_time.values):.4f}, '
          f'max={np.max(df.stop_time.values-df.start_time.values):.4f}')
    # orientation-specific gaps around orientation change boundaries
    print(f'  orientations in {name}: {sorted(np.unique(df["orientation"].dropna()))}')
    print(f'  first 10 starts: {np.round(starts[:10], 3)}')

import numpy as np