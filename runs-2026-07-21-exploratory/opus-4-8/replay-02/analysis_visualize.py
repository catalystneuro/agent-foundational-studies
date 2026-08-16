"""Final replay figures: example decoded sweeps + spike rasters, and population summary."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
C='/tmp/replay_cache'
sp=np.load(f'{C}/spikes.npy',allow_pickle=True); loc=np.load(f'{C}/unit_loc.npy')
spikes={int(uid):np.asarray(times) for (uid,times),l in zip(sp,loc)}
tc_R=pd.read_pickle(f'{C}/tc_R.pkl'); tc_L=pd.read_pickle(f'{C}/tc_L.pkl')
pc=np.load(f'{C}/pc_union.npy'); uids=np.load(f'{C}/tc_unit_ids.npy')[pc].astype(int)
TR=tc_R.iloc[:,pc].values+1e-3; TL=tc_L.iloc[:,pc].values+1e-3
X,U=TR.shape; xbins=np.load(f'{C}/xbins.npy')
df=pd.read_pickle(f'{C}/replay_results.pkl')

def counts(s,e,nb):
    edges=np.linspace(s,e,nb+1); N=np.zeros((nb,U))
    for j,u in enumerate(uids):
        st=spikes[u]; st=st[(st>=s)&(st<e)]
        if st.size: N[:,j]=np.histogram(st,bins=edges)[0]
    return N,edges
def decode(N,tau,T):
    def nrm(lp): lp=lp-lp.max(1,keepdims=True); p=np.exp(lp); return p/p.sum(1,keepdims=True)
    return nrm(N@np.log(TR).T-tau*TR.sum(1)), nrm(N@np.log(TL).T-tau*TL.sum(1))

# choose 4 strong, significant replay events (high |r|, well activated), spanning directions
sig=df[(df.pval<0.05)&(df.ncells>=8)].copy()
sig['score']=sig.r.abs()*np.log(sig.nspikes)
sig=sig.sort_values('score',ascending=False)
# pick 2 rightward + 2 leftward
picks=pd.concat([sig[sig.direction=='R'].head(2), sig[sig.direction=='L'].head(2)])
picks=picks.sort_values('start')

# peak place-field position for raster ordering (per direction)
pfR=xbins[np.argmax(TR,0)]; pfL=xbins[np.argmax(TL,0)]

fig=plt.figure(figsize=(16,9))
gs=GridSpec(2,4,figure=fig,height_ratios=[1,1],hspace=0.45,wspace=0.35)
for k,(_,ev) in enumerate(picks.iterrows()):
    s,e=ev.start,ev.end; d=ev.direction; dur=e-s
    nb=max(5,int(round(dur/0.02))); N,edges=counts(s,e,nb); tau=dur/nb
    PR,PL=decode(N,tau,nb); P=PR if d=='R' else PL; pf=pfR if d=='R' else pfL
    # posterior + MAP + line fit
    axP=fig.add_subplot(gs[0,k])
    axP.imshow(P.T,aspect='auto',origin='lower',extent=[0,dur*1000,xbins[0],xbins[-1]],cmap='hot')
    tc_=np.linspace(0,dur*1000,nb); mapx=xbins[np.argmax(P,1)]
    # weighted line fit
    tv=np.arange(nb); w=P; sw=w.sum()
    mt=(w*tv[:,None]).sum()/sw; mx=(w*xbins[None,:]).sum()/sw
    b=(w*(tv[:,None]-mt)*(xbins[None,:]-mx)).sum()/ (w*(tv[:,None]-mt)**2).sum()
    axP.plot(tc_, mx+b*(tv-mt),'c-',lw=2,label=f'fit r={ev.r:+.2f}')
    axP.set_title(f'{"Forward" if (ev.r>0)==(d=="R") else "Reverse"} replay ({d})\np={ev.pval:.3f}',fontsize=10)
    axP.set_xlabel('Time in ripple (ms)'); axP.set_ylabel('Decoded pos (m)')
    axP.legend(fontsize=8,loc='upper right')
    # raster sorted by place-field position
    axR=fig.add_subplot(gs[1,k])
    order=np.argsort(pf)
    for row,j in enumerate(order):
        st=spikes[uids[j]]; st=st[(st>=s)&(st<e)]
        if st.size: axR.plot((st-s)*1000, np.full(st.size,row),'|',color='k',ms=5,mew=1)
    axR.set_xlim(0,dur*1000); axR.set_ylim(-1,len(order))
    axR.set_xlabel('Time in ripple (ms)'); axR.set_ylabel('Cell (by field pos)')
    axR.set_title('Spike raster (place-field order)',fontsize=9)
fig.suptitle('Hippocampal replay: decoded spatial trajectories during POST-sleep sharp-wave ripples',fontsize=13)
plt.savefig('fig4_replay_examples.png',dpi=130); print('saved fig4')

# ---- population summary ----
fig2,ax=plt.subplots(1,3,figsize=(15,4.6))
# fraction significant vs chance
frac=100*(df.pval<0.05).mean()
ax[0].bar(['observed','chance'],[frac,5],color=['C3','0.6'])
ax[0].axhline(5,ls='--',color='k',lw=0.8)
ax[0].set_ylabel('% ripples with significant replay')
ax[0].set_title(f'Replay prevalence\n{int((df.pval<0.05).sum())}/{len(df)} events, {frac:.0f}% (chance 5%)')
for i,v in enumerate([frac,5]): ax[0].text(i,v+0.5,f'{v:.0f}%',ha='center')
# distribution of |r|: significant vs non
ax[1].hist(df[df.pval>=0.05].r.abs(),bins=25,density=True,alpha=0.6,label='n.s.',color='0.6')
ax[1].hist(df[df.pval<0.05].r.abs(),bins=25,density=True,alpha=0.7,label='p<0.05',color='C3')
ax[1].set_xlabel('|weighted correlation|'); ax[1].set_ylabel('density')
ax[1].set_title('Sequence score distribution'); ax[1].legend()
# replay trajectory span (distance represented) for significant events
sigd=df[df.pval<0.05]
ax[2].hist(sigd.span*100,bins=25,color='C0')
ax[2].axvline(sigd.span.median()*100,color='k',ls='--',lw=1,label=f'median {sigd.span.median()*100:.0f} cm')
ax[2].set_xlabel('Decoded trajectory span (cm)'); ax[2].set_ylabel('# replay events')
ax[2].set_title('Distance represented per replay'); ax[2].legend()
plt.tight_layout(); plt.savefig('fig5_replay_summary.png',dpi=130); print('saved fig5')
print(f'SUMMARY: {int((df.pval<0.05).sum())}/{len(df)} = {frac:.1f}% significant; '
      f'median span {sigd.span.median()*100:.0f} cm; dir {sigd.direction.value_counts().to_dict()}')
