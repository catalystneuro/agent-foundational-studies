"""Decode spatial trajectories during POST ripples; test replay via place-field circular-shift shuffle."""
import numpy as np, pandas as pd, sys
from tqdm import tqdm
C='/tmp/replay_cache'
rng=np.random.default_rng(0)
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 0   # 0 = all

sp=np.load(f'{C}/spikes.npy',allow_pickle=True); loc=np.load(f'{C}/unit_loc.npy')
spikes={int(uid):np.asarray(times) for (uid,times),l in zip(sp,loc)}
tc_R=pd.read_pickle(f'{C}/tc_R.pkl'); tc_L=pd.read_pickle(f'{C}/tc_L.pkl')
pc=np.load(f'{C}/pc_union.npy'); uids=np.load(f'{C}/tc_unit_ids.npy')[pc].astype(int)
TR=tc_R.iloc[:,pc].values+1e-3; TL=tc_L.iloc[:,pc].values+1e-3   # (X,U)
X,U=TR.shape; xbins=np.load(f"{C}/xbins.npy")
logTR,logTL=np.log(TR),np.log(TL); sumTR,sumTL=TR.sum(1),TL.sum(1)
R=np.load(f'{C}/ripples.npz'); sel=R['inpost'].astype(bool)
rs,re,rpk=R['start'][sel],R['end'][sel],R['peak'][sel]

TARGET_BIN=0.015; MIN_BINS=5; MIN_CELLS=5; MIN_SPIKES=10; NSH=300
def counts(s,e):
    dur=e-s; nb=max(MIN_BINS,int(round(dur/TARGET_BIN)))
    edges=np.linspace(s,e,nb+1); N=np.zeros((nb,U)); nact=0
    for j,u in enumerate(uids):
        st=spikes[u]; st=st[(st>=s)&(st<e)]
        if st.size: nact+=1
        if st.size: N[:,j]=np.histogram(st,bins=edges)[0]
    return N,nact
def decode(N,lTR,lTL,sR,sL,tau):
    def nrm(lp): lp=lp-lp.max(1,keepdims=True); p=np.exp(lp); return p/p.sum(1,keepdims=True)
    return nrm(N@lTR.T - tau*sR), nrm(N@lTL.T - tau*sL)
def wcorr(P):
    T=P.shape[0]; tv=np.arange(T); w=P; sw=w.sum()
    mt=(w*tv[:,None]).sum()/sw; mx=(w*xbins[None,:]).sum()/sw
    ct=(w*(tv[:,None]-mt)*(xbins[None,:]-mx)).sum()/sw
    vt=(w*(tv[:,None]-mt)**2).sum()/sw; vx=(w*(xbins[None,:]-mx)**2).sum()/sw
    return 0.0 if vt<=0 or vx<=0 else ct/np.sqrt(vt*vx)

idxs=range(len(rs)) if LIMIT==0 else range(LIMIT)
res=[]
for i in tqdm(idxs, desc='replay', mininterval=5):
    N,nact=counts(rs[i],re[i])
    if nact<MIN_CELLS or N.sum()<MIN_SPIKES: continue
    T=N.shape[0]; tau=(re[i]-rs[i])/T
    PR,PL=decode(N,logTR,logTL,sumTR,sumTL,tau)
    rR,rL=wcorr(PR),wcorr(PL)
    if abs(rR)>=abs(rL): P,robs,d=PR,rR,'R'
    else: P,robs,d=PL,rL,'L'
    # place-field circular-shift shuffle
    shuf=np.empty(NSH)
    base=np.arange(X)[:,None]
    for k in range(NSH):
        off=rng.integers(0,X,U)
        ii=(base-off[None,:])%X
        TRs=np.take_along_axis(TR,ii,0); TLs=np.take_along_axis(TL,ii,0)
        pr,pl=decode(N,np.log(TRs),np.log(TLs),TRs.sum(1),TLs.sum(1),tau)
        shuf[k]=max(abs(wcorr(pr)),abs(wcorr(pl)))
    pval=(1+np.sum(shuf>=abs(robs)))/(NSH+1)
    mapx=xbins[np.argmax(P,1)]
    res.append(dict(idx=i,start=rs[i],end=re[i],peak=rpk[i],ncells=nact,
                    nspikes=int(N.sum()),T=T,r=robs,direction=d,pval=pval,
                    span=float(mapx.max()-mapx.min())))
df=pd.DataFrame(res)
if LIMIT==0: df.to_pickle(f'{C}/replay_results.pkl')
nsig=(df.pval<0.05).sum()
print(f'\nCandidates {len(df)} | sig(p<0.05) {nsig} = {100*nsig/max(len(df),1):.1f}% | chance 5%')
print(f'median |r| sig {df[df.pval<0.05].r.abs().median():.3f} | dir {df[df.pval<0.05].direction.value_counts().to_dict()}')
