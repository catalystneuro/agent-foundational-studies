"""Check replacement sessions for states table, then run pipeline on them."""
import numpy as np
from hd_analysis import (load_session, get_epochs, analyze_session,
                         state_correlations)
import os

CANDIDATES = {
    "Mouse24-131217": "c6cfc0f2-dda6-4136-be2c-be4a4de5a50e",
    "Mouse20-130515": "f92f2709-4469-4c1d-a883-75eb66ee898a",
}

for name, aid in CANDIDATES.items():
    nwb, io = load_session(aid)
    print(name, "keys:", list(nwb.keys()))
    io.close()
