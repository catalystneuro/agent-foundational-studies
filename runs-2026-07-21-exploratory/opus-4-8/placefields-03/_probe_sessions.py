"""Quick check that every session's behaviour/units structure matches the primary session."""
import lindi, numpy as np
from pynwb import NWBHDF5IO

LINDI_BASE = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/{}/nwb.lindi.json"
CACHE_DIR = "/tmp/lindi_cache_000044"
SESSIONS = {
    "Achilles_10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Achilles_11012013": "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a",
    "Cicero_09012014":   "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Cicero_09102014":   "f61dfe09-3db2-464a-b386-2e828b2e7276",
    "Cicero_09172014":   "e381ebb3-128e-4f3f-9517-11277d7aed9b",
    "Gatsby_08022013":   "31ea0aab-4777-424e-9a93-9605b2bdcc29",
    "Gatsby_08282013":   "f7687af7-3bc9-4d20-8d88-ef293d2a3381",
    "Buddy_06272013":    "82714afb-724f-4e2b-b102-c9c47b5cba73",
}

for name, aid in SESSIONS.items():
    f = lindi.LindiH5pyFile.from_lindi_file(LINDI_BASE.format(aid),
                                            local_cache=lindi.LocalCache(cache_dir=CACHE_DIR))
    nwb = NWBHDF5IO(file=f, mode="r").read()
    beh = nwb.processing["behavior"]
    keys = list(beh.data_interfaces)
    lin_key = [k for k in keys if "Maze" in k and "Linearized" in k]
    ss = list(beh[lin_key[0]].spatial_series.values())[0] if lin_key else None
    d = np.asarray(ss.data[:]).squeeze() if ss is not None else None
    ct = np.asarray(nwb.units["cell_type"][:])
    print(f"{name:20s} n_units={len(nwb.units.id):4d} pyr={np.sum(ct=='excitatory'):4d} "
          f"keys={keys} rate={float(ss.rate) if ss is not None else None} "
          f"n={0 if d is None else d.size} frac_finite={0 if d is None else np.isfinite(d).mean():.2f} "
          f"max={np.nan if d is None else np.nanmax(d):.2f}")
