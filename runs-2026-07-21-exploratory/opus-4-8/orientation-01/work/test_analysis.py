import numpy as np, pandas as pd, time
import ov_data as ov, ov_analysis as oa

spikes, info = ov.load_spikes(ov.SESSIONS[0])
dg = ov.stimulus_table(info['nwbfile'], 'drifting_gratings_presentations')
sg = ov.stimulus_table(info['nwbfile'], 'static_gratings_presentations')

t0=time.time(); res = oa.dg_tuning(spikes, dg); print('dg_tuning %.1fs' % (time.time()-t0))
print('tc shape', res['tc'].shape, 'rates', res['rates'].shape)
gosi, pref_ori = oa.circular_selectivity(res['tc'], oa.DG_DIRECTIONS, 2)
gdsi, pref_dir = oa.circular_selectivity(res['tc'], oa.DG_DIRECTIONS, 1)
osi, _ = oa.ratio_osi(res['tc'], oa.DG_DIRECTIONS)
dsi, _ = oa.ratio_dsi(res['tc'], oa.DG_DIRECTIONS)
areas = info['selected'].loc[res['uids'], 'area'].values
df = pd.DataFrame({'area':areas,'gosi':gosi,'osi':osi,'gdsi':gdsi,'dsi':dsi,
                   'peak':res['tc'].max(axis=1),'base':res['baseline']}, index=res['uids'])
print(df.groupby('area')[['gosi','osi','gdsi','peak']].median().round(3))
print('\nresponsive (peak > base+1Hz):', (df.peak > df.base+1).sum(), '/', len(df))

t0=time.time(); sres = oa.sg_tuning(spikes, sg); print('\nsg_tuning %.1fs' % (time.time()-t0))
sgosi, spref = oa.circular_selectivity(sres['tc'], oa.SG_ORIENTATIONS, 2)
df['sg_gosi']=sgosi; df['sg_pref']=spref; df['dg_pref_ori']=pref_ori
print(df.groupby('area')[['gosi','sg_gosi']].median().round(3))
m = (df.peak > df.base+1) & (df.gosi>0.2)
print('\ncirc dist DG vs SG pref ori (deg), tuned units:', np.median(oa.circ_dist_deg(df.dg_pref_ori[m].values, df.sg_pref[m].values)))
print('random expectation ~45')
t0=time.time(); obs,p = oa.permutation_test_gosi(res, n_perm=200); print('perm 200 %.1fs' % (time.time()-t0))
df['p']=p
print('sig fraction by area:'); print(df.groupby('area').apply(lambda g:(g.p<0.05).mean()).round(3))
df.to_csv('proto_metrics.csv')
