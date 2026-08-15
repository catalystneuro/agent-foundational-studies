"""Figure 1: verify each data stream from one session of DANDI:000986."""
import numpy as np, pynapple as nap, matplotlib.pyplot as plt
import loaders

SESSION = 'sub-LA9/sub-LA9_ses-1_behavior.nwb'
assets = loaders.list_assets('000986')
nwbfile = loaders.open_nwb('000986', assets[SESSION])
nwb = nap.NWBFile(nwbfile)
units = nwb['units']
trials = nwbfile.trials.to_dataframe()
pupil, run = nwb['pupil_diameter'], nwb['running_speed']

ufreq = np.sort(trials.stim_frequency.unique())
cmap = plt.get_cmap('viridis')
fcol = {f: cmap(i / (len(ufreq) - 1)) for i, f in enumerate(ufreq)}

t0 = 500.0
win = nap.IntervalSet(t0, t0 + 12)
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 1, 1], hspace=0.12))

ax = axes[0]
for i, u in enumerate(units.index):
    st = units[u].restrict(win).t
    ax.plot(st, np.full_like(st, i), '|', color='k', ms=3, mew=0.6)
sel = trials[(trials.start_time > win.start[0]) & (trials.start_time < win.end[0])]
for _, r in sel.iterrows():
    ax.axvspan(r.start_time, r.start_time + r.stim_duration, color=fcol[r.stim_frequency],
               alpha=0.85, lw=0, ymin=0.0, ymax=1.0, zorder=0)
ax.set_ylabel('unit #')
ax.set_title(f'DANDI:000986  {SESSION}  —  raw spiking, {len(units)} units\n'
             'shaded bars = 25 ms pure tones (colour = frequency, 60 dB SPL)')
handles = [plt.Line2D([], [], color=fcol[f], lw=6, label=f'{f/1000:g} kHz') for f in ufreq]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.005, 1.0),
          frameon=False, fontsize=9, title='tone')

axes[1].plot(run.restrict(win).t, np.asarray(run.restrict(win)), color='tab:green', lw=1)
axes[1].set_ylabel('running\n(cm/s)')
axes[2].plot(pupil.restrict(win).t, np.asarray(pupil.restrict(win)), color='tab:purple', lw=1)
axes[2].set_ylabel('pupil\ndiameter')
axes[2].set_xlabel('time in session (s)')
for a in axes[1:]:
    a.margins(x=0)
axes[0].margins(x=0)
fig.savefig('figures/fig01_raw_streams.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# session-level overview: firing-rate raster over full session + behaviour + block structure
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.2, 1, 1], hspace=0.15))
bin_size = 1.0
cnt = units.count(bin_size)
z = (np.asarray(cnt) - np.asarray(cnt).mean(0)) / (np.asarray(cnt).std(0) + 1e-9)
im = axes[0].imshow(z.T, aspect='auto', origin='lower', vmin=-2, vmax=4, cmap='magma',
                    extent=[cnt.t[0], cnt.t[-1], 0, len(units)])
axes[0].set_ylabel('unit #')
axes[0].set_title('Whole-session overview: z-scored firing rate (1 s bins)')
plt.colorbar(im, ax=axes[0], pad=0.01, label='z')
axes[1].plot(run.t[::50], np.asarray(run)[::50], color='tab:green', lw=0.5)
axes[1].set_ylabel('running\n(cm/s)')
axes[2].plot(pupil.t[::50], np.asarray(pupil)[::50], color='tab:purple', lw=0.5)
axes[2].set_ylabel('pupil\ndiameter')
axes[2].set_xlabel('time in session (s)')
for _, r in nwbfile.intervals['spontaneous_blocks'].to_dataframe().iterrows():
    for a in axes:
        a.axvspan(r.start_time, r.stop_time, color='0.6', alpha=0.25, lw=0, zorder=0)
axes[2].text(0.01, 0.9, 'grey = spontaneous (no tones) blocks', transform=axes[2].transAxes,
             fontsize=9, va='top')
fig.savefig('figures/fig02_session_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('trials:', len(trials), 'units:', len(units),
      'pupil NaN frac: %.3f' % np.isnan(np.asarray(pupil)).mean())
