import orientation_lib as ol
import pandas as pd, numpy as np
from tqdm import tqdm

assets = ol.list_session_assets()
print(len(assets), "session files")
rows=[]
for _, a in tqdm(assets.iterrows(), total=len(assets)):
    nwbfile, io = ol.open_session(a.s3_url)
    names = list(nwbfile.intervals.keys())
    has_dg = 'drifting_gratings_presentations' in names
    has_sg = 'static_gratings_presentations' in names
    u = ol.unit_table(nwbfile)
    good = u[ol.passes_qc(u)]
    nvis = good.location.isin(ol.VISUAL_AREAS).sum()
    nlgd = good.location.isin(ol.CONTROL_AREAS).sum()
    rows.append(dict(session_id=a.session_id, s3_url=a.s3_url, has_dg=has_dg, has_sg=has_sg,
                     n_good=len(good), n_vis=nvis, n_lgd=nlgd,
                     areas=",".join(sorted(set(good.location) & set(ol.VISUAL_AREAS)))))
    io.close()
df = pd.DataFrame(rows)
df.to_csv("session_screen.csv", index=False)
print(df[['session_id','has_dg','has_sg','n_good','n_vis','n_lgd']].to_string())
