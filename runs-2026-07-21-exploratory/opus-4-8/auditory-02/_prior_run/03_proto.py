import numpy as np, matplotlib.pyplot as plt
from loader import list_assets, load_session
from psthlib import psth, window_counts

assets = list_assets()
nwb, raw = load_session(assets[0])
units = nwb['units']
trials = raw.trials.to_dataframe()
onsets = trials.start_time.values
freqs = np.unique(trials.stim_frequency.values)
print('min ITI', np.diff(onsets).min())

bs = 0.005; win = (-0.2, 0.4)
allpsth = np.array([psth(units[k].t, onsets, win, bs)[1] for k in units.keys()])
t = psth(units[list(units.keys())[0]].t, onsets, win, bs)[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(t, allpsth.mean(0), 'k')
ax[0].axvspan(0, 0.025, color='orange', alpha=.3, label='tone')
ax[0].set_xlabel('time from onset (s)'); ax[0].set_ylabel('rate (Hz)'); ax[0].legend()
base = allpsth[:, t < 0].mean(1)
resp = allpsth[:, (t > 0.005) & (t < 0.105)].mean(1)
ax[1].scatter(base, resp, s=8); lim=[0,max(resp.max(),base.max())]
ax[1].plot(lim, lim, 'k--'); ax[1].set_xlabel('baseline (Hz)'); ax[1].set_ylabel('evoked 5-105 ms (Hz)')
plt.tight_layout(); plt.savefig('proto_psth.png', dpi=130)
print('pop peak time', t[np.argmax(allpsth.mean(0))])
print('frac increased', (resp > base*1.2).mean())

# per-frequency population psth
fig, ax = plt.subplots(figsize=(6,4))
for f in freqs:
    on = trials.start_time.values[trials.stim_frequency.values == f]
    p = np.array([psth(units[k].t, on, win, bs)[1] for k in units.keys()]).mean(0)
    ax.plot(t, p, label=f'{f/1000:.0f} kHz')
ax.legend(); ax.set_xlabel('time from onset (s)'); ax.set_ylabel('rate (Hz)')
plt.tight_layout(); plt.savefig('proto_psth_byfreq.png', dpi=130)
