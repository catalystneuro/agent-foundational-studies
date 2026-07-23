import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _dev_tuning import load_session

url = 'https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f'
nwb, nwbfile = load_session(url)

units = nwb['units']
trials = nwb['trials'].as_dataframe()
trials['start'] = nwb['trials'].start
trials['end'] = nwb['trials'].end

# pick a window spanning end of spontaneous block + start of stimulus block
spont = nwb['spontaneous_blocks']
print(spont)
print(trials.head())
print('first trial start', trials['start'].iloc[0])

# pick 20 units by highest rate for clean raster
rates = units.get_info('rate') if hasattr(units,'get_info') else None
print(rates.sort_values(ascending=False).head(20))
