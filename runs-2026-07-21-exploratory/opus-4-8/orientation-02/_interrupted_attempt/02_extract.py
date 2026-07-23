"""Extract per-trial firing rates for drifting and static gratings, one npz per session."""
import os
import sys

import numpy as np
from tqdm import tqdm

import oslib

OUT = os.path.join(oslib.HERE, "extracted")
os.makedirs(OUT, exist_ok=True)

DG_OFFSET, DG_WINDOW = 0.03, 1.97   # 2.0 s presentation, skip response latency
SG_OFFSET, SG_WINDOW = 0.03, 0.18   # 0.25 s presentation, back to back (one 0.217 s gap)


def extract(session_id):
    out = os.path.join(OUT, f"{session_id}.npz")
    if os.path.exists(out):
        return out
    nwbfile = oslib.open_session(session_id)
    tsg, meta = oslib.good_units(nwbfile)

    dg = oslib.stim_table(nwbfile, "drifting_gratings_presentations")
    sg = oslib.stim_table(nwbfile, "static_gratings_presentations")
    dg_rates = oslib.trial_rates(tsg, dg, DG_OFFSET, DG_WINDOW)
    sg_rates = oslib.trial_rates(tsg, sg, SG_OFFSET, SG_WINDOW)

    # Blank out trials that fall in a probe's invalid-data intervals.
    probe_ids = meta["probe_id"].values
    dg_bad = oslib.invalid_trial_mask(nwbfile, probe_ids, dg["start_time"].values, dg["stop_time"].values)
    sg_bad = oslib.invalid_trial_mask(nwbfile, probe_ids, sg["start_time"].values, sg["stop_time"].values)
    dg_rates[dg_bad] = np.nan
    sg_rates[sg_bad] = np.nan
    print(f"  invalid trials: drifting {dg_bad.mean():.3%}, static {sg_bad.mean():.3%}", flush=True)

    # Running speed averaged over each drifting-grating trial, for a locomotion control.
    run = nwbfile.processing["running"]["running_speed"]
    rt, rv = np.asarray(run.timestamps[:]), np.asarray(run.data[:])
    dg_speed = np.array(
        [np.nanmean(rv[(rt >= a) & (rt < b)]) for a, b in zip(dg["start_time"], dg["stop_time"])]
    )

    np.savez_compressed(
        out,
        session=str(session_id),
        unit_ids=np.asarray(list(tsg.keys())),
        area=meta["area"].values.astype(str),
        snr=meta["snr"].values,
        firing_rate=meta["firing_rate"].values,
        waveform_duration=meta["waveform_duration"].values,
        probe_id=meta["probe_id"].values,
        dg_rates=dg_rates.astype(np.float32),
        dg_ori=dg["orientation"].values,
        dg_tf=dg["temporal_frequency"].values,
        dg_start=dg["start_time"].values,
        dg_stop=dg["stop_time"].values,
        dg_speed=dg_speed,
        sg_rates=sg_rates.astype(np.float32),
        sg_ori=sg["orientation"].values,
        sg_sf=sg["spatial_frequency"].values,
        sg_phase=sg["phase"].values,
        sg_start=sg["start_time"].values,
    )
    return out


if __name__ == "__main__":
    sessions = sys.argv[1:]
    for s in tqdm(sessions, desc="sessions"):
        p = extract(s)
        d = np.load(p, allow_pickle=True)
        areas, counts = np.unique(d["area"], return_counts=True)
        print(
            f"{s}: {len(d['unit_ids'])} units {dict(zip(areas, counts))} "
            f"dg {d['dg_rates'].shape} sg {d['sg_rates'].shape}",
            flush=True,
        )
