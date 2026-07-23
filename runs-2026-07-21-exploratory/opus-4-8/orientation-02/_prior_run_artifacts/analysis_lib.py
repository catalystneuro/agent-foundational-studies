"""Core orientation-selectivity analysis on cached Allen Visual Coding data."""
import pickle, numpy as np, pynapple as nap
from scipy import stats

def load(path='cache_715093703.pkl'):
    d = pickle.load(open(path,'rb'))
    meta = d['meta']
    spikes = {int(k): nap.Ts(t=v) for k,v in d['spikes'].items()}
    tsg = nap.TsGroup(spikes, metadata={'region': meta['region'],
                                        'snr': meta['snr'],
                                        'waveform_duration': meta['waveform_duration']})
    return d, tsg

def trial_rates(tsg, tbl, mask):
    """Spike counts / rates per trial (rows) x unit (cols) for selected trials."""
    st, sp = tbl['start_time'][mask], tbl['stop_time'][mask]
    ep = nap.IntervalSet(start=st, end=sp)
    counts = tsg.count(ep=ep)          # TsdFrame trials x units
    dur = (sp - st)[:, None]
    return np.asarray(counts.values) / dur

def ori_metrics(rates_by_dir, dirs_deg):
    """rates_by_dir: mean rate per direction. Returns dict of tuning metrics."""
    r = np.clip(rates_by_dir, 0, None)
    th = np.deg2rad(dirs_deg)
    tot = r.sum()
    if tot <= 0:
        return dict(gOSI=np.nan, gDSI=np.nan, OSI=np.nan, DSI=np.nan,
                    pref_dir=np.nan, pref_ori=np.nan)
    gOSI = np.abs((r*np.exp(2j*th)).sum())/tot
    gDSI = np.abs((r*np.exp(1j*th)).sum())/tot
    pref_ori = np.rad2deg(np.angle((r*np.exp(2j*th)).sum())/2) % 180
    i = int(np.argmax(r))
    pref = dirs_deg[i]
    def at(d):
        j = int(np.argmin(np.abs(((dirs_deg - d + 180) % 360) - 180)))
        return r[j]
    r_pref, r_orth, r_null = r[i], 0.5*(at(pref+90)+at(pref-90)), at(pref+180)
    OSI = (r_pref-r_orth)/(r_pref+r_orth) if (r_pref+r_orth)>0 else np.nan
    DSI = (r_pref-r_null)/(r_pref+r_null) if (r_pref+r_null)>0 else np.nan
    return dict(gOSI=gOSI, gDSI=gDSI, OSI=OSI, DSI=DSI, pref_dir=pref, pref_ori=pref_ori)

def drifting_analysis(tsg, tbl, n_perm=1000, seed=0):
    ori = tbl['orientation'].astype(float)
    tf  = tbl['temporal_frequency'].astype(float)
    valid = ~np.isnan(ori)
    blank = np.isnan(ori)
    dirs = np.unique(ori[valid]); tfs = np.unique(tf[valid])

    R = trial_rates(tsg, tbl, valid)            # trials x units
    Rb = trial_rates(tsg, tbl, blank)
    o, f = ori[valid], tf[valid]
    nU = R.shape[1]

    # mean response per (direction, TF)
    grid = np.zeros((len(dirs), len(tfs), nU))
    for i,dd in enumerate(dirs):
        for j,ff in enumerate(tfs):
            grid[i,j] = R[(o==dd)&(f==ff)].mean(0)

    # preferred TF per unit = TF maximizing mean over directions
    pref_tf_idx = np.argmax(grid.mean(0), axis=0)

    res = {k: np.full(nU, np.nan) for k in
           ['gOSI','gDSI','OSI','DSI','pref_dir','pref_ori','anova_p','perm_p',
            'evoked','baseline','pref_tf','tuning_snr']}
    tc = np.zeros((len(dirs), nU)); tc_sem = np.zeros((len(dirs), nU))
    rng = np.random.default_rng(seed)
    for u in range(nU):
        j = pref_tf_idx[u]
        sub = f == tfs[j]
        groups = [R[sub & (o==dd), u] for dd in dirs]
        m = np.array([g.mean() for g in groups])
        tc[:,u] = m
        tc_sem[:,u] = np.array([g.std(ddof=1)/np.sqrt(len(g)) for g in groups])
        res['anova_p'][u] = stats.f_oneway(*groups).pvalue
        mm = ori_metrics(m, dirs)
        for k,v in mm.items(): res[k][u] = v
        res['pref_tf'][u] = tfs[j]
        res['evoked'][u] = R[sub, u].mean()
        res['baseline'][u] = Rb[:, u].mean()
        # permutation null for gOSI
        y = R[sub, u]; lab = o[sub]
        null = np.empty(n_perm)
        for p in range(n_perm):
            yp = rng.permutation(y)
            mp = np.array([yp[lab==dd].mean() for dd in dirs])
            null[p] = ori_metrics(mp, dirs)['gOSI']
        res['perm_p'][u] = (np.sum(null >= res['gOSI'][u]) + 1)/(n_perm+1)
    return dict(dirs=dirs, tfs=tfs, grid=grid, tc=tc, tc_sem=tc_sem,
                R=R, ori=o, tf=f, metrics=res, blank_rates=Rb)

def static_analysis(tsg, tbl):
    ori = tbl['orientation'].astype(float)
    sf  = tbl['spatial_frequency'].astype(float)
    valid = ~np.isnan(ori)
    oris = np.unique(ori[valid]); sfs = np.unique(sf[valid])
    R = trial_rates(tsg, tbl, valid)
    o, s = ori[valid], sf[valid]
    nU = R.shape[1]
    pref_sf_idx = np.argmax(np.stack([R[s==ss].mean(0) for ss in sfs]), axis=0)
    tc = np.zeros((len(oris), nU)); tc_sem = np.zeros_like(tc)
    gosi = np.full(nU, np.nan); pref = np.full(nU, np.nan); pval = np.full(nU, np.nan)
    for u in range(nU):
        sub = s == sfs[pref_sf_idx[u]]
        groups = [R[sub & (o==oo), u] for oo in oris]
        m = np.array([g.mean() for g in groups])
        tc[:,u] = m
        tc_sem[:,u] = np.array([g.std(ddof=1)/np.sqrt(len(g)) for g in groups])
        pval[u] = stats.f_oneway(*groups).pvalue
        th = np.deg2rad(oris); rr = np.clip(m,0,None)
        if rr.sum() > 0:
            gosi[u] = np.abs((rr*np.exp(2j*th)).sum())/rr.sum()
            pref[u] = np.rad2deg(np.angle((rr*np.exp(2j*th)).sum())/2) % 180
    return dict(oris=oris, sfs=sfs, tc=tc, tc_sem=tc_sem, gOSI=gosi,
                pref_ori=pref, anova_p=pval, R=R, ori=o, sf=s,
                pref_sf=sfs[pref_sf_idx])

if __name__ == '__main__':
    d, tsg = load()
    print(tsg)
    dg = drifting_analysis(tsg, d['stim']['drifting_gratings_presentations'], n_perm=500)
    sg = static_analysis(tsg, d['stim']['static_gratings_presentations'])
    pickle.dump(dict(dg=dg, sg=sg), open('results.pkl','wb'))
    reg = np.array(d['meta']['region'])
    m = dg['metrics']
    for r in ['VISp','VISl','VISrl','VISam','VISpm','LGd']:
        k = reg==r
        sig = (m['anova_p'][k]<0.01)&(m['perm_p'][k]<0.05)
        print(f"{r:6s} n={k.sum():3d} tuned={sig.mean()*100:5.1f}%  "
              f"median gOSI={np.nanmedian(m['gOSI'][k]):.3f}  "
              f"median DSI={np.nanmedian(m['DSI'][k]):.3f}")
