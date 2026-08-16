"""Run the full pipeline over every session in DANDI:000044 and aggregate.

Ripple statistics are computed for all eight sessions. Replay is only scored for the
five linear-maze sessions: on the circular mazes the linearised coordinate wraps, and
a weighted correlation against a straight line is not the right statistic there.
"""

import os
import re

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import analysis_lib as A
import swr_lib as L

N_SHUF = 500
BIN_M = 0.04  # 4 cm decoding bins
SPEED_TH = 0.05
POS_DT = 1.0 / 39.06263603480421

OUT = "multi_session"


def run_session(session):
    print(f"\n=== {session} ===", flush=True)
    d = L.load_core(session)
    h5, units, epochs, states = d["h5"], d["units"], d["epochs"], d["states"]
    maze_name = d["maze_name"]
    circular = "Circular" in maze_name
    m = re.match(r"([\d.]+)m", maze_name)
    track_len = float(m.group(1)) if m else 2.90  # circular mazes span ~2.9 m linearised
    if circular:
        track_len = float(np.nanmax(d["position"].d))

    ch, _ = A.select_ripple_channel(h5, states["Non-REM"])
    print(f"  ripple channel {ch}, maze {maze_name} ({track_len:.2f} m)", flush=True)

    raw = L.lfp_dataset(h5)[:, ch].astype(np.float32) * L.lfp_conversion(h5)
    nrem_mask = A.nrem_mask_from(states, raw.size)
    rip, _, _ = A.detect_ripples(raw, nrem_mask)
    rip = A.tag_events(rip, states, epochs)
    rip["session"] = session
    del raw
    print(f"  {len(rip)} ripples", flush=True)

    rates = []
    for lab in ["Non-REM", "REM", "Awake"]:
        for ep in ["PRE", "Maze", "POST"]:
            dur = states[lab].intersect(epochs[ep + "Epoch"]).tot_length()
            if dur < 60:
                continue
            cnt = int(((rip["state"] == lab) & (rip["epoch"] == ep)).sum())
            rates.append(dict(session=session, state=lab, epoch=ep, n=cnt,
                              seconds=float(dur), rate_hz=cnt / dur))

    nrem_rip = rip[rip["state"] == "Non-REM"]
    summary = dict(
        session=session, maze=maze_name, circular=bool(circular),
        n_units=len(units), n_ripples=int(len(rip)),
        nrem_rate=float(len(nrem_rip) / states["Non-REM"].tot_length()),
        median_duration_ms=float(nrem_rip["duration"].median() * 1000),
        median_freq_hz=float(nrem_rip["peak_freq"].median()),
    )

    if circular:
        h5.close()
        return rip, pd.DataFrame(rates), summary, None

    # --------------------------------------------------------------- place fields
    ct = units.get_info("cell_type")
    pyr = units[[i for i in units.index if ct[i] == "excitatory"]]
    pos, vel, support = A.behaviour(d["position"], POS_DT)
    speed = nap.Tsd(t=pos.t, d=np.abs(vel.d), time_support=support)
    run_ep = speed.threshold(SPEED_TH).time_support
    right_ep = vel.threshold(SPEED_TH).time_support
    left_ep = nap.Tsd(t=pos.t, d=-vel.d, time_support=support).threshold(SPEED_TH).time_support
    n_bins = int(round(track_len / BIN_M))

    tc_all = A.place_fields(pyr, pos, run_ep, n_bins, track_len, POS_DT)
    tc_r = A.place_fields(pyr, pos, right_ep, n_bins, track_len, POS_DT)
    tc_l = A.place_fields(pyr, pos, left_ep, n_bins, track_len, POS_DT)
    occ = np.histogram(pos.restrict(run_ep).d, bins=n_bins, range=(0, track_len))[0] * POS_DT
    si, mean_rate = A.skaggs_information(tc_all, occ)
    keep_cell = A.select_template_cells(tc_all, mean_rate)
    place_ids = np.array(tc_all.columns)[keep_cell]
    summary.update(run_seconds=float(run_ep.tot_length()), n_pyr=int(len(pyr)),
                   n_template=int(keep_cell.sum()),
                   n_place_si=int((si > 0.5).sum()), median_si=float(np.median(si)),
                   track_len=track_len)
    print(f"  {keep_cell.sum()} template cells ({(si>0.5).sum()} with SI > 0.5), "
          f"{run_ep.tot_length():.0f} s running", flush=True)
    if keep_cell.sum() < 10:
        h5.close()
        return rip, pd.DataFrame(rates), summary, None

    # ------------------------------------------------------------------- replay
    ok = np.flatnonzero(~np.isnan(d["position"].d))
    blocks = np.split(ok, np.flatnonzero(np.diff(ok) > 2) + 1)
    st, sv = [], []
    for b in blocks:
        if len(b) < 5:
            continue
        st.append(d["position"].t[b])
        sv.append(np.abs(np.gradient(d["position"].d[b], d["position"].t[b])))
    st, sv = np.concatenate(st), np.concatenate(sv)
    j = np.clip(np.searchsorted(st, rip["peak_t"].values), 1, len(st) - 1)
    moving = (np.abs(st[j] - rip["peak_t"].values) < 0.2) & (sv[j] > SPEED_TH)

    cand = rip[
        ((rip["epoch"] == "PRE") & (rip["state"] == "Non-REM"))
        | ((rip["epoch"] == "POST") & (rip["state"] == "Non-REM"))
        | ((rip["epoch"] == "Maze") & (rip["state"] == "Awake") & ~moving)
    ].reset_index(drop=True)

    group = units[list(place_ids)]
    counts, keep, nbins = A.build_event_counts(cand, group, place_ids)
    if len(counts) < 50:
        h5.close()
        return rip, pd.DataFrame(rates), summary, None
    dec = A.ReplayDecoder(
        counts, tc_all.index.values,
        {"right": tc_r[list(place_ids)].values.T,
         "left": tc_l[list(place_ids)].values.T},
    )
    out = dec.run(n_shuffles=N_SHUF, progress=tqdm)
    ev = cand.loc[keep].reset_index(drop=True)
    for k in ["r", "slope", "direction", "p_column", "p_placefield", "p_timebin", "significant"]:
        ev[k] = out[k]
    ev["n_bins"] = nbins
    ev["session"] = session

    for lab in ["PRE", "POST", "Maze"]:
        m = ev["epoch"] == lab
        summary[f"n_{lab}"] = int(m.sum())
        summary[f"sig_{lab}"] = int((m & ev["significant"]).sum())
        summary[f"frac_{lab}"] = float((m & ev["significant"]).sum() / max(m.sum(), 1))
    print(f"  replay: PRE {summary['sig_PRE']}/{summary['n_PRE']}, "
          f"POST {summary['sig_POST']}/{summary['n_POST']}, "
          f"Maze {summary['sig_Maze']}/{summary['n_Maze']}", flush=True)
    h5.close()
    return rip, pd.DataFrame(rates), summary, ev


def sweep(sessions=None):
    """Process every session and write the aggregated tables to ``multi_session/``."""
    os.makedirs(OUT, exist_ok=True)
    all_rip, all_rates, all_sum, all_ev = [], [], [], []
    for s in sessions or list(L.ASSETS):
        rip, rates, summary, ev = run_session(s)
        all_rip.append(rip)
        all_rates.append(rates)
        all_sum.append(summary)
        if ev is not None:
            all_ev.append(ev)
    pd.concat(all_rip).to_csv(f"{OUT}/ripples_all_sessions.csv", index=False)
    pd.concat(all_rates).to_csv(f"{OUT}/ripple_rates_all_sessions.csv", index=False)
    pd.DataFrame(all_sum).to_csv(f"{OUT}/session_summary.csv", index=False)
    pd.concat(all_ev).to_csv(f"{OUT}/replay_events_all_sessions.csv", index=False)
    print("\nwrote", OUT)


if __name__ == "__main__":
    sweep()
