"""
Extract trial-resolved firing rates for drifting and static gratings across several
sessions of DANDI:000021, and cache them to disk for the analysis stage.
"""
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from os_pipeline import (
    open_session, load_unit_table, load_spikes, load_stimulus_table,
    trial_rates, trial_spike_counts, AREAS_OF_INTEREST,
)

# Sessions chosen from the survey for joint coverage of V1, higher visual areas,
# visual thalamus (LGd/LP) and a non-visual control (hippocampus).
SESSIONS = {
    "755434585": "edf10182-5a4c-454f-ad23-47987a5ca256",
    "791319847": "3adbba7c-3feb-468b-9829-33fcaa27aacd",
    "757970808": "dfc3db15-066a-4a07-b615-a4d7e85c44e1",
    "760345702": "47634abd-db85-48f5-9c33-01887a59d3bc",
    "763673393": "8a17b967-2aa9-4d6a-812c-92b62cf799d7",
    "754312389": "5a58bf3d-a1b9-444b-8ab0-ef5478aa42a6",
}

# Response windows relative to stimulus onset.  Drifting gratings last 2 s; static
# gratings last 0.25 s and are presented back-to-back, so the window is shifted by the
# ~30-50 ms visual response latency.
DG_WINDOW = (0.0, 2.0)
SG_WINDOW = (0.03, 0.28)

OUT = "responses.pkl"


def extract_session(session_id, asset_id):
    h5 = open_session(asset_id)
    units = load_unit_table(h5)
    sel = units["passes_qc"] & units["area"].isin(AREAS_OF_INTEREST)
    rows = np.flatnonzero(sel.values)
    spikes, meta = load_spikes(h5, units, rows)

    meta = meta[["unit_id", "area", "area_group", "snr", "firing_rate", "waveform_duration"]].copy()
    meta["session"] = session_id
    meta["uid"] = session_id + "_" + meta["unit_id"].astype(str)
    assert list(meta["unit_id"]) == list(spikes.index), "metadata must follow TsGroup order"

    dg = load_stimulus_table(h5, "drifting_gratings_presentations")
    sg = load_stimulus_table(h5, "static_gratings_presentations")

    dg_blank = dg[~np.isfinite(dg["orientation"])].reset_index(drop=True)
    sg_blank = sg[~np.isfinite(sg["orientation"])].reset_index(drop=True)
    dg = dg[np.isfinite(dg["orientation"])].reset_index(drop=True)
    sg = sg[np.isfinite(sg["orientation"])].reset_index(drop=True)
    sg["phase"] = sg["phase"].astype(float)

    out = dict(
        session=session_id,
        meta=meta.reset_index(drop=True),
        unit_ids=list(spikes.keys()),
        dg_table=dg,
        sg_table=sg,
        dg_rates=trial_rates(spikes, dg, DG_WINDOW),
        sg_rates=trial_rates(spikes, sg, SG_WINDOW),
        dg_blank_rates=trial_rates(spikes, dg_blank, DG_WINDOW),
        sg_blank_rates=trial_rates(spikes, sg_blank, SG_WINDOW),
    )

    # Peri-onset PSTHs (5 ms bins) for latency validation.  One searchsorted over the
    # full (onset x bin-edge) grid rather than a loop over bins.
    edges = np.arange(-0.1, 0.5001, 0.005)
    for tag, table in (("dg", dg), ("sg", sg)):
        onsets = table["start_time"].values
        grid = (onsets[:, None] + edges[None, :]).ravel()
        psth = np.empty((len(spikes), len(edges) - 1))
        for j, uid in enumerate(spikes.keys()):
            idx = np.searchsorted(spikes[uid].t, grid).reshape(len(onsets), len(edges))
            psth[j] = np.diff(idx, axis=1).sum(axis=0)
        out[f"{tag}_psth"] = psth / (len(onsets) * 0.005)
    out["psth_edges"] = edges

    # Compact peri-event rasters for V1 units: spike times relative to each drifting
    # grating onset, within [-0.5, 2.5] s.
    v1 = [u for u in spikes.keys() if spikes.get_info("area")[u] == "VISp"]
    onsets = dg["start_time"].values
    raster = {}
    for uid in v1:
        t = spikes[uid].t
        lo = np.searchsorted(t, onsets - 0.5)
        hi = np.searchsorted(t, onsets + 2.5)
        trial_idx, rel = [], []
        for k, (a, b) in enumerate(zip(lo, hi)):
            if b > a:
                rel.append(t[a:b] - onsets[k])
                trial_idx.append(np.full(b - a, k))
        raster[uid] = (
            np.concatenate(trial_idx).astype(np.int32) if trial_idx else np.zeros(0, np.int32),
            np.concatenate(rel).astype(np.float32) if rel else np.zeros(0, np.float32),
        )
    out["v1_dg_raster"] = raster

    h5.close()
    return out


if __name__ == "__main__":
    results = []
    for sid, aid in tqdm(SESSIONS.items(), desc="sessions"):
        print(f"\n=== session {sid} ===", flush=True)
        r = extract_session(sid, aid)
        print(f"  {len(r['unit_ids'])} units, {len(r['dg_table'])} DG trials, {len(r['sg_table'])} SG trials", flush=True)
        print("  ", r["meta"]["area"].value_counts().to_dict(), flush=True)
        results.append(r)

    with open(OUT, "wb") as f:
        pickle.dump(results, f)
    print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
