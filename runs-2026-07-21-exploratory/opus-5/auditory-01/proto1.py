import numpy as np, pynapple as nap, loaders

assets = loaders.list_assets('000986')
nwbfile = loaders.open_nwb('000986', assets['sub-LA9/sub-LA9_ses-1_behavior.nwb'])
nwb = nap.NWBFile(nwbfile)
units = nwb['units']
tr = nwbfile.trials.to_dataframe()
print("rates pct:", np.round(np.percentile(np.asarray(units.rates),[0,10,50,90,100]),2))
print("trial span", tr.start_time.min(), tr.stop_time.max(), len(tr))
gaps = np.diff(tr.start_time.values)
print("ISI pct", np.round(np.percentile(gaps,[0,1,50,99,100]),3))
print("n blocks (gap>2s):", (gaps>2).sum()+1)
pupil = nwb['pupil_diameter']; run = nwb['running_speed']
print("pupil", pupil.shape, pupil.time_support, "nan:", np.isnan(np.asarray(pupil)).mean())
print("run", run.shape, "nan:", np.isnan(np.asarray(run)).mean())
print("units time_support", units.time_support)
