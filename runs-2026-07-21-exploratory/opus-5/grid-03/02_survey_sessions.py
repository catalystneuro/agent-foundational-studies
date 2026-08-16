"""Survey every session in DANDI:000582: arena geometry, duration, unit count."""
import numpy as np
import pandas as pd
from tqdm import tqdm

import gridlib as gl

rows = []
for a in tqdm(gl.list_assets(), desc="sessions"):
    nwb = gl.open_nwb(a)
    ss = nwb.processing["behavior"]["Position"]["SpatialSeriesLED1"]
    t = np.asarray(ss.timestamps[:])
    d = np.asarray(ss.data[:], dtype=float)
    ok = np.isfinite(d).all(axis=1)
    d = d[ok]
    q = np.percentile(d, [0.5, 99.5], axis=0)
    ext = q[1] - q[0]
    # square vs circular: fraction of the bounding-box area actually visited
    H, _, _ = np.histogram2d(d[:, 0], d[:, 1], bins=20)
    fill = (H > 0).mean()
    rows.append(dict(
        path=a["path"], subject=nwb.subject.subject_id, session=nwb.session_id,
        dur=float(t[-1] - t[0]), n_units=len(nwb.units),
        ext_x=ext[0], ext_y=ext[1], fill=fill,
        histology=str(nwb.units["histology"][0]) if len(nwb.units) else "",
    ))

df = pd.DataFrame(rows)
df["shape"] = np.where(df.fill > 0.9, "square", "circle")
df.to_csv("session_survey.csv", index=False)
print(df.groupby(["shape", pd.cut(df.ext_x, [0, 60, 110, 160, 300])], observed=True)
        .agg(n=("path", "size"), units=("n_units", "sum")))
print(df.describe())
print(df.histology.value_counts())
