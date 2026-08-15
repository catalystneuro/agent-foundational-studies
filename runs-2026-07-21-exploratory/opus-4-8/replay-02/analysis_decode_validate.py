"""Validate Bayesian decoder: reconstruct the animal's position during running."""
import numpy as np, pynapple as nap, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C='/tmp/replay_cache'
ep=np.load(f'{C}/epochs.npy',allow_pickle=True)
epd={str(r['label']):(float(r['start_time']),float(r['stop_time'])) for r in ep}
maze=nap.IntervalSet(*epd['MazeEpoch'])
sp=np.load(f'{C}/spikes.npy',allow_pickle=True); loc=np.load(f'{C}/unit_loc.npy')
units=nap.TsGroup({int(uid):nap.Ts(t=np.asarray(times)) for (uid,times),l in zip(sp,loc)})
tl=np.load(f'{C}/lin_t.npy'); xl=np.load(f'{C}/lin_x.npy'); valid=~np.isnan(xl)
pos=nap.Tsd(t=tl[valid],d=xl[valid]).restrict(maze)
tc_R=pd.read_pickle(f'{C}/tc_R.pkl'); tc_L=pd.read_pickle(f'{C}/tc_L.pkl')
pc=np.load(f'{C}/pc_union.npy'); uids=np.load(f'{C}/tc_unit_ids.npy')[pc]
tcR=tc_R.iloc[:,pc]; tcL=tc_L.iloc[:,pc]
grp=units[[int(u) for u in uids]]
xbins=np.load(f'{C}/xbins.npy')
runR=np.load(f'{C}/run_R.npy'); runL=np.load(f'{C}/run_L.npy')
run_R_ep=nap.IntervalSet(start=runR[:,0],end=runR[:,1])
run_L_ep=nap.IntervalSet(start=runL[:,0],end=runL[:,1])

# decode running periods with matching directional template, 250 ms bins
true_all,dec_all,com_all=[],[],[]
best=None
for tc,rep,name in [(tcR,run_R_ep,'R'),(tcL,run_L_ep,'L')]:
    dec,proba=nap.decode_1d(tc, grp, rep, bin_size=0.25)
    P=proba.values                       # (t, xbin)
    com=(P*xbins[None,:]).sum(1)/P.sum(1) # posterior mean
    ti=np.interp(dec.index, pos.restrict(rep).index, pos.restrict(rep).values)
    m=(ti>=0.15)&(ti<=1.45)
    true_all.append(ti[m]); dec_all.append(dec.values[m]); com_all.append(com[m])
    if name=='L':  # keep leftward decode for the example panel
        best=(dec.index, ti, com)
true=np.concatenate(true_all); decm=np.concatenate(dec_all); comm=np.concatenate(com_all)
err=np.abs(decm-true)
print(f'RUN decoding (interior): median error {np.median(err)*100:.1f} cm, '
      f'mean {np.mean(err)*100:.1f} cm; corr(true,COM)={np.corrcoef(true,comm)[0,1]:.2f}')

fig,axes=plt.subplots(1,2,figsize=(13,5.2))
# confusion matrix
H,xe,ye=np.histogram2d(true,decm,bins=[np.linspace(0.1,1.5,29)]*2)
H=H/(H.sum(1,keepdims=True)+1e-9)
im=axes[0].imshow(H.T,origin='lower',extent=[0.1,1.5,0.1,1.5],aspect='auto',cmap='magma')
axes[0].plot([0.1,1.5],[0.1,1.5],'w--',lw=1,alpha=.7)
axes[0].set_xlabel('True position (m)'); axes[0].set_ylabel('Decoded position (m)')
axes[0].set_title(f'Decoder confusion (running)\nmedian err = {np.median(err)*100:.0f} cm')
plt.colorbar(im,ax=axes[0],label='P(decoded | true)')
# continuous example (leftward)
ti_,true_,com_=best[0]-best[0][0], best[1], best[2]
sl=slice(0,120)
axes[1].plot(ti_[sl],true_[sl],'k',lw=2,label='true position')
axes[1].plot(ti_[sl],com_[sl],'C3.-',ms=5,lw=0.8,label='decoded (posterior mean)')
axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Position (m)'); axes[1].set_ylim(0,1.6)
axes[1].set_title('Example continuous decoding (leftward runs)')
axes[1].legend(loc='upper right',fontsize=9)
plt.tight_layout(); plt.savefig('fig3_decode_validation.png',dpi=130); print('saved fig3')
