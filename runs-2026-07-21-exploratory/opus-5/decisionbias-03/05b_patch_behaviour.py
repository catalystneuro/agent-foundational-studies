"""Second pass over the caches: video regressors and per-unit quality metrics.

Two things the first extraction pass leaves out.  Not every session in
DANDI:000409 carries both cameras, so the first-choice pupil / motion-energy
names miss data that is present under another name.  And we later found that
units which drop out partway through a session (low presence ratio) produce
strongly non-stationary counts that the drift high-pass cannot fully repair, so
the per-unit quality metrics are cached here to allow filtering on them without
re-downloading anything.
"""
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import ibl_io as io
import analysis_lib as al

ALTS = {
    "pupil": [("pupil", "LeftPupilDiameterSmoothed"),
              ("pupil", "RightPupilDiameterSmoothed"),
              ("pupil", "LeftPupilDiameter"),
              ("pupil", "RightPupilDiameter")],
    "motion_energy": [("motion_energy", "LeftCameraMotionEnergy"),
                      ("motion_energy", "RightCameraMotionEnergy"),
                      ("motion_energy", "BodyCameraMotionEnergy")],
}

if __name__ == "__main__":
    assets = {a[0]: a for a in io.list_processed_assets()}
    for p in tqdm(sorted(glob.glob("cache/*.pkl")), desc="patching"):
        r = pd.read_pickle(p)
        missing = [k for k in ALTS if k not in r["beh"]]
        if not missing and "presence_ratio" in r:
            continue
        hf = io.open_h5(assets[r["path"]][2])
        if "presence_ratio" not in r:
            reg = io.read_unit_regions(hf)
            small = io.read_unit_table_small(hf)
            keep = al.unit_mask(reg, small)
            assert keep.sum() == r["X_pre"].shape[1], "unit mask out of sync"
            for k in ["presence_ratio", "firing_rate", "ibl_quality_score",
                      "isi_violations_ratio"]:
                r[k] = small[k][keep]
        for k in missing:
            for grp, name in ALTS[k]:
                g = hf.get(f"/processing/{grp}/{name}")
                if g is not None and "timestamps" in g:
                    r["beh"][k] = al.behaviour_regressor(
                        r["trials"], (g["timestamps"][:], g["data"][:]), al.PRE_WIN)
                    r["beh"][k + "_source"] = name
                    break
        pd.to_pickle(r, p)
        print(r["subject"], {k: (int(np.isfinite(v).sum()) if isinstance(v, np.ndarray)
                                 else v) for k, v in r["beh"].items()}, flush=True)
