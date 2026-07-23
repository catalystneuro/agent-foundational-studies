import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
from _dev_tuning import load_session, FREQUENCIES

url = 'https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f'
nwb, nwbfile = load_session(url)

units = nwb['units']
trials = nwb['trials'].as_dataframe()
trials['start'] = nwb['trials'].start
trials['end'] = nwb['trials'].end

rates = units.get_info('rate')
top_units = rates.sort_values(ascending=False).index[:20].tolist()

cmap = plt.get_cmap('viridis')
freq_colors = {f: cmap(i/(len(FREQUENCIES)-1)) for i, f in enumerate(FREQUENCIES)}

fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))

# Panel A: wide view spanning spontaneous -> stimulus block transition
t_start, t_end = 380, 440
ax = axes[0]
ep = nap.IntervalSet(t_start, t_end)
for yi, u in enumerate(top_units):
    ts = units[u].restrict(ep)
    ax.vlines(ts.t, yi, yi+0.9, color='k', linewidth=0.6)
ax.axvspan(t_start, 407.124, color='0.85', label='spontaneous (no sound)')
trial_win = trials[(trials.start >= 407.124) & (trials.start <= t_end)]
for _, row in trial_win.iterrows():
    ax.axvline(row['start'], color=freq_colors[row['stim_frequency']], alpha=0.5, linewidth=1.2, ymin=0, ymax=1)
ax.set_xlim(t_start, t_end)
ax.set_ylim(0, len(top_units))
ax.set_ylabel('Unit # (top 20 by rate)')
ax.set_xlabel('Time (s)')
ax.set_title('A. Spike raster: transition from spontaneous silence to pure-tone presentation')
ax.legend(loc='upper left', fontsize=8)

# Panel B: zoomed view of a handful of trials, colored tone markers
ax = axes[1]
t_start2, t_end2 = 408.0, 418.0
ep2 = nap.IntervalSet(t_start2, t_end2)
for yi, u in enumerate(top_units):
    ts = units[u].restrict(ep2)
    ax.vlines(ts.t, yi, yi+0.9, color='k', linewidth=0.8)
trial_win2 = trials[(trials.start >= t_start2) & (trials.start <= t_end2)]
for _, row in trial_win2.iterrows():
    ax.axvspan(row['start'], row['end'], color=freq_colors[row['stim_frequency']], alpha=0.35)
ax.set_xlim(t_start2, t_end2)
ax.set_ylim(0, len(top_units))
ax.set_ylabel('Unit # (top 20 by rate)')
ax.set_xlabel('Time (s)')
ax.set_title('B. Zoomed view: each shaded band is one 25 ms pure tone (color = frequency)')

# legend for frequencies
handles = [plt.Rectangle((0,0),1,1, color=freq_colors[f], alpha=0.5) for f in FREQUENCIES]
labels = [f'{int(f/1000)} kHz' for f in FREQUENCIES]
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9, title='Tone frequency', frameon=True)

plt.tight_layout(rect=[0,0.08,1,1])
fig.subplots_adjust(bottom=0.14)
plt.savefig('figures/01_raw_raster.png', dpi=150)
print('saved')
