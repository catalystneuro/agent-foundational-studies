"""Detect sharp-wave ripple (SWR) events from CA1 LFP during PRE and POST sleep."""
import numpy as np, pynapple as nap, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
C='/tmp/replay_cache'
fs=float(np.load(f'{C}/lfp_fs.npy')[0])
lfp=np.load(f'{C}/lfp_ch2.npy').astype(float)
t=np.arange(lfp.size)/fs
ep=np.load(f'{C}/epochs.npy', allow_pickle=True)
epd={str(r['label']):(float(r['start_time']),float(r['stop_time'])) for r in ep}
pre=nap.IntervalSet(*epd['PREEpoch']); post=nap.IntervalSet(*epd['POSTEpoch'])
sleep=nap.IntervalSet(start=[epd['PREEpoch'][0],epd['POSTEpoch'][0]],
                      end=[epd['PREEpoch'][1],epd['POSTEpoch'][1]])
print('fs',fs,'lfp dur',t[-1])

# ripple band 150-250 Hz
b,a=butter(4,[150/(fs/2),250/(fs/2)],btype='band')
filt=filtfilt(b,a,lfp)
env=np.abs(hilbert(filt))
# smooth envelope (Gaussian ~ 8ms)
from scipy.ndimage import gaussian_filter1d
env=gaussian_filter1d(env, int(0.008*fs))
env_tsd=nap.Tsd(t=t, d=env)

# z-score envelope using sleep-period statistics
env_sleep=env_tsd.restrict(sleep).values
mu,sd=env_sleep.mean(), env_sleep.std()
z=(env-mu)/sd
z_tsd=nap.Tsd(t=t,d=z)

# detect: peak>5 SD, boundaries at 2 SD, dur 20-250ms, restrict to sleep
LOW,HIGH=2.0,5.0
cand=z_tsd.restrict(sleep).threshold(LOW,'above').time_support
# keep only candidates whose peak z exceeds HIGH
starts,ends,peaks,pk_t=[],[],[],[]
for s,e in zip(cand.start, cand.end):
    seg=z_tsd.get(s,e)
    if len(seg)==0: continue
    pmax=float(np.max(seg.values))
    dur=e-s
    if pmax>=HIGH and 0.02<=dur<=0.25:
        starts.append(s); ends.append(e); peaks.append(pmax)
        pk_t.append(float(seg.index[np.argmax(seg.values)]))
ripples=nap.IntervalSet(start=np.array(starts), end=np.array(ends))
peaks=np.array(peaks); pk_t=np.array(pk_t)
# label epoch
inpost=np.array([post.start[0]<=p<=post.end[0] for p in pk_t])
print(f'Total ripples: {len(ripples)} | PRE {int((~inpost).sum())} | POST {int(inpost.sum())}')
print('POST duration min', (post.end[0]-post.start[0])/60)
print('POST ripple rate (Hz)', inpost.sum()/(post.end[0]-post.start[0]))
np.savez(f'{C}/ripples.npz', start=ripples.start, end=ripples.end, peak=peaks,
         peak_t=pk_t, inpost=inpost)

# ---- figure: example ripple + summary ----
fig=plt.figure(figsize=(14,8))
gs=fig.add_gridspec(2,3, height_ratios=[1.1,1])
# pick a strong POST ripple
pidx=np.where(inpost)[0]
best=pidx[np.argsort(peaks[pidx])[-6]]
tc0=pk_t[best]; w=0.15
sl=slice(int((tc0-w)*fs), int((tc0+w)*fs))
tt=(t[sl]-tc0)*1000
axr=fig.add_subplot(gs[0,:])
axr.plot(tt, lfp[sl], color='0.3', lw=0.7, label='raw LFP')
axr.plot(tt, filt[sl]-lfp[sl].std()*3, color='C3', lw=0.7, label='150-250 Hz')
axr.axvspan((ripples.start[best]-tc0)*1000,(ripples.end[best]-tc0)*1000, color='gold', alpha=0.3, label='detected ripple')
axr.set_xlabel('Time from ripple peak (ms)'); axr.set_ylabel('LFP (a.u.)')
axr.set_title(f'Example POST sharp-wave ripple (peak z={peaks[best]:.1f})'); axr.legend(loc='upper right', fontsize=8)
# duration hist
axd=fig.add_subplot(gs[1,0])
axd.hist((ripples.end-ripples.start)[inpost]*1000, bins=30, color='C0')
axd.set_xlabel('Duration (ms)'); axd.set_ylabel('# ripples'); axd.set_title('POST ripple durations')
# peak amplitude hist
axp=fig.add_subplot(gs[1,1])
axp.hist(peaks[inpost], bins=30, color='C1')
axp.set_xlabel('Peak envelope (SD)'); axp.set_ylabel('# ripples'); axp.set_title('POST ripple amplitudes')
# rate over POST (per 5 min)
axt=fig.add_subplot(gs[1,2])
pt=pk_t[inpost]-post.start[0]
axt.hist(pt/60, bins=40, color='C2')
axt.set_xlabel('Time in POST (min)'); axt.set_ylabel('# ripples'); axt.set_title('Ripple occurrence in POST sleep')
plt.tight_layout(); plt.savefig('fig2_ripples.png', dpi=130); print('saved fig2')
