"""CLI: cache per-trial firing rates for the given sessions of DANDI:000021."""
import sys

import numpy as np
from tqdm import tqdm

from pipeline import extract_session

for s in tqdm(sys.argv[1:], desc="sessions"):
    d = np.load(extract_session(s), allow_pickle=True)
    areas, counts = np.unique(d["area"], return_counts=True)
    print(f"{s}: {len(d['unit_ids'])} units {dict(zip(areas, counts.tolist()))} "
          f"dg {d['dg_rates'].shape} sg {d['sg_rates'].shape}", flush=True)
