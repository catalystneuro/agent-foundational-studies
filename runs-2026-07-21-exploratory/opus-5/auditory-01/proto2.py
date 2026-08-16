import numpy as np, pynapple as nap, matplotlib.pyplot as plt, loaders, analysis_core as ac

assets = loaders.list_assets('000986')
nwbfile = loaders.open_nwb('000986', assets['sub-LA9/sub-LA9_ses-1_behavior.nwb'])
nwb = nap.NWBFile(nwbfile)
units = nwb['units']
tr = nwbfile.trials.to_dataframe()
onsets = tr.start_time.values; freqs = tr.stim_frequency.values

WIN_EV, WIN_BL = (0.005, 0.105), (-0.105, -0.005)
ev = ac.trial_counts(units, onsets, WIN_EV)
bl = ac.trial_counts(units, onsets, WIN_BL)
print("mean evoked rate", (ev/0.1).mean(), "baseline", (bl/0.1).mean())
curves, sems, stt = ac.tuning_table(ev, bl, freqs, WIN_EV, WIN_BL, np.asarray(units.index))
resp = ac.fdr(stt.p_responsive.values); tun = ac.fdr(stt.p_tuned.values)
print("responsive %d/%d  tuned %d/%d" % (resp.sum(), len(stt), tun.sum(), len(stt)))
print(stt.assign(resp=resp, tuned=tun).sort_values('peak_evoked_hz', ascending=False).head(12).to_string())
print(curves.loc[stt.sort_values('peak_evoked_hz', ascending=False).index[:6]].round(2).to_string())
print("BF counts:", stt[tun].best_frequency.value_counts().to_dict())
# quick PSTH check for the best unit
best = stt.peak_evoked_hz.idxmax()
sub = units[[best]]
t, r = ac.psth(sub, onsets[freqs==stt.loc[best,'best_frequency']])
print("psth peak", r.max(), "at t=", t[r.argmax()])
