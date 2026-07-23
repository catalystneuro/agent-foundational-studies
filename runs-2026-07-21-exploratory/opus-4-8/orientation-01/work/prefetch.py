import ov_data as ov, time, ov_analysis as oa, pickle, numpy as np, pandas as pd
for s in ov.SESSIONS:
    t0=time.time()
    spikes, info = ov.load_spikes(s)
    dg = ov.stimulus_table(info['nwbfile'],'drifting_gratings_presentations')
    sg = ov.stimulus_table(info['nwbfile'],'static_gratings_presentations')
    print(s, len(spikes), 'units, %.0fs' % (time.time()-t0), 'DG', len(dg), 'SG', len(sg), flush=True)
