"""CLI: compute per-unit orientation-tuning metrics for every cached session."""
import glob
import os

import pandas as pd
from tqdm import tqdm

from pipeline import analyze_session, responsive

paths = sorted(p for p in glob.glob("extracted/*.npz") if "_dg_spikes" not in p)
print("sessions:", [os.path.basename(p) for p in paths], flush=True)
out = pd.concat([analyze_session(p) for p in tqdm(paths, desc="analyzing")], ignore_index=True)
out.to_pickle("units.pkl")
print(f"\n{len(out)} units, {out.session.nunique()} sessions, {out.subject.nunique()} mice")
r = responsive(out)
print(f"visually responsive: {len(r)}/{len(out)}\n")
print(r.groupby("area").apply(lambda g: pd.Series({
    "n": len(g), "frac_sig": (g.p_perm < 0.01).mean(),
    "median_gOSI": g.gosi.median(), "median_gOSI_corr": g.gosi_corrected.median(),
    "median_OSI_classic": g.osi_classic.median(), "median_gDSI": g.gdsi.median(),
}), include_groups=False).to_string())
