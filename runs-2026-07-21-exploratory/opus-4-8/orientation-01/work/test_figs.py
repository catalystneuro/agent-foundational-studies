import numpy as np, pandas as pd
import ov_data as ov, ov_analysis as oa, ov_plots as op

spikes, info = ov.load_spikes(ov.SESSIONS[0])
dg = ov.stimulus_table(info['nwbfile'],'drifting_gratings_presentations')
running = ov.running_speed(info['nwbfile'])
op.fig_raw_overview(spikes, info, dg, running, 'fig_test_raw.png')
res = oa.dg_tuning(spikes, dg)
gosi, pref = oa.circular_selectivity(res['tc'], oa.DG_DIRECTIONS, 2)
gdsi, _ = oa.circular_selectivity(res['tc'], oa.DG_DIRECTIONS, 1)
dsi, _ = oa.ratio_dsi(res['tc'], oa.DG_DIRECTIONS)
osi, _ = oa.ratio_osi(res['tc'], oa.DG_DIRECTIONS)
obs, p = oa.permutation_test_gosi(res, n_perm=500)
areas = info['selected'].loc[res['uids'],'area'].values
df = pd.DataFrame({'area':areas,'gosi':gosi,'osi':osi,'gdsi':gdsi,'dsi':dsi,'p':p,
                   'peak':res['tc'].max(axis=1)}, index=res['uids'])
i = np.where(res['uids']==951877391)[0][0]
op.fig_example_unit(spikes, res, 951877391, 'VISp', 'fig_test_unitV1.png', gosi[i], dsi[i], p[i],
                    note='Primary visual cortex: sharply tuned to ~90/270 deg')
j = np.where(res['uids']==951865312)[0][0]
op.fig_example_unit(spikes, res, 951865312, 'LGd', 'fig_test_unitLGd.png', gosi[j], dsi[j], p[j],
                    note='Thalamic relay (LGd): strongly driven but not orientation tuned')
g = np.array([op.group_label(a) for a in areas])
tcg = {k: res['tc'][g==k] for k in np.unique(g)}
op.fig_population_tuning(tcg, 'fig_test_pop.png')
op.fig_selectivity_by_area(df, 'fig_test_sel.png')
print('done')
