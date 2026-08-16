"""Per-session theta phase-locking computation, cached to disk as a pickle.

For each session this computes, for three brain states (maze running, REM sleep,
non-REM sleep):
  * theta phase at every spike of every unit
  * circular statistics (MRL, preferred phase, Rayleigh p)
  * a circular-shift null distribution of MRL
plus a locking-vs-frequency spectrum and a theta-amplitude split for the running
state, and a short raw-data excerpt for plotting.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import hc11_io as io
import theta as th

N_SHUFFLE = 200
MIN_SPIKES = 100
SPECTRUM_BANDS = [(c - 1.5, c + 1.5) for c in
                  [3, 5, 7, 8, 9, 11, 14, 18, 22, 27, 33, 40]]
SPECTRUM_SECONDS = 1200  # duration-match the states before comparing spectra
CACHE_TMPL = "cache_{}.pkl"


def cap_duration(ep, seconds):
    """Take whole intervals from the start of `ep` up to `seconds` of total time."""
    cum = np.cumsum(ep.end - ep.start)
    k = int(np.searchsorted(cum, seconds)) + 1
    return ep[: min(k, len(ep))]


def session_tag(path):
    return path.split("/")[-1].replace("_behavior+ecephys.nwb", "").replace("sub-", "")


def compute(path, n_shuffle=N_SHUFFLE, use_cache=True, verbose=True):
    tag = session_tag(path)
    cache = CACHE_TMPL.format(tag)
    if use_cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    h5 = io.open_session(path)
    spikes = io.load_spikes(h5)
    epochs = io.load_epochs(h5)
    states = io.load_states(h5)
    pos, lin, speed = io.load_position(h5)
    maze = epochs["MazeEpoch"]
    run_ep = th.run_epochs(speed, maze)

    # Sleep states are scored across the whole session; restrict REM/non-REM to
    # the sleep blocks flanking the task so they never overlap maze running.
    sleep = epochs["PREEpoch"].union(epochs["POSTEpoch"])
    state_eps = {"run": run_ep,
                 "REM": states["REM"].intersect(sleep),
                 "nonREM": states["Non-REM"].intersect(sleep)}

    best_ch, ratios = th.select_theta_channel(h5, run_ep)
    if verbose:
        print(f"[{tag}] theta channel {best_ch}, theta/delta={ratios[best_ch]:.2f}; "
              + ", ".join(f"{k}={float(v.tot_length()):.0f}s"
                          for k, v in state_eps.items()))

    lfp_full = io.load_lfp_channel(h5, best_ch)

    rows, phases, spectra, psds = [], {}, {}, {}
    amp_split = []
    for state, ep in state_eps.items():
        if float(ep.tot_length()) < 60:
            continue
        lfp = lfp_full.restrict(ep)
        psds[state] = th.welch(lfp.values, fs=io.LFP_RATE,
                               nperseg=int(4 * io.LFP_RATE))
        _, phase, amp = th.theta_phase(lfp)
        lookup = th.PhaseLookup(phase, ep)
        for u in tqdm(spikes.keys(), desc=f"[{tag}] {state}", disable=not verbose):
            ph = th.spike_phases(spikes[u], phase, ep)
            st = th.circular_stats(ph)
            phases[(state, int(u))] = ph.astype(np.float32)
            row = dict(session=tag, unit=int(u),
                       cell_type=str(spikes.cell_type[u]),
                       location=str(spikes.location[u]),
                       state=state, n_spikes=st["n"],
                       rate=st["n"] / float(ep.tot_length()),
                       mrl=st["mrl"], pref_phase=st["pref"],
                       rayleigh_p=st["p"], theta_channel=best_ch)
            if n_shuffle and st["n"] >= MIN_SPIKES:
                null = lookup.null_mrl(spikes[u], n_shuffle=n_shuffle,
                                       rng=np.random.default_rng(1000 + int(u)))
                row["null_mrl_mean"] = float(np.nanmean(null))
                row["null_mrl_p95"] = float(np.nanpercentile(null, 95))
                row["shuffle_p"] = (np.sum(null >= st["mrl"]) + 1) / (n_shuffle + 1)
            else:
                row.update(null_mrl_mean=np.nan, null_mrl_p95=np.nan,
                           shuffle_p=np.nan)
            rows.append(row)

        if state == "run":
            # Locking strength as a function of theta amplitude (median split).
            thr = np.median(amp.values)
            hi = amp.threshold(thr, "above").time_support
            lo = amp.threshold(thr, "below").time_support
            for u in spikes.keys():
                r = dict(unit=int(u), session=tag,
                         cell_type=str(spikes.cell_type[u]))
                for lbl, sub in [("high", hi), ("low", lo)]:
                    p = th.spike_phases(spikes[u], phase, ep.intersect(sub))
                    s = th.circular_stats(p)
                    r[f"mrl_{lbl}"] = s["mrl"]
                    r[f"n_{lbl}"] = s["n"]
                amp_split.append(r)

        # Locking spectrum: repeat the pipeline in other bands, on a
        # duration-matched slice so the three states are directly comparable.
        ep_s = cap_duration(ep, SPECTRUM_SECONDS)
        lfp_s = lfp_full.restrict(ep_s)
        for band in tqdm(SPECTRUM_BANDS, desc=f"[{tag}] {state} spectrum",
                         disable=not verbose):
            _, ph_b, _ = th.theta_phase(lfp_s, band=band)
            for u in spikes.keys():
                p = th.spike_phases(spikes[u], ph_b, ep_s)
                s = th.circular_stats(p)
                spectra.setdefault((state, int(u)), []).append(s["mrl"])

    df = pd.DataFrame(rows)

    # MRL is biased upward when a unit has few spikes, and the three states
    # differ greatly in duration, so also report an estimate in which every unit
    # contributes the same number of spikes in every state.
    matched = []
    rng = np.random.default_rng(0)
    for u in {k[1] for k in phases}:
        avail = {s: phases[(s, u)] for s in state_eps if (s, u) in phases}
        n = min(len(v) for v in avail.values())
        for s, v in avail.items():
            sel = rng.choice(v, size=n, replace=False) if len(v) > n else v
            st = th.circular_stats(sel)
            matched.append(dict(unit=u, state=s, n_matched=n,
                                mrl_matched=st["mrl"],
                                pref_matched=st["pref"]))
    df = df.merge(pd.DataFrame(matched), on=["unit", "state"], how="left")

    # A short excerpt of raw data for the illustrative figure: the running
    # interval containing the most population spikes.
    durations = run_ep.end - run_ep.start
    counts = np.array([sum(len(spikes[u].restrict(run_ep[i:i + 1]))
                           for u in spikes.keys())
                       for i in range(len(run_ep))])
    pop_rate = np.where(durations > 3, counts / durations, 0)
    best = int(np.argmax(pop_rate))
    t0 = float(run_ep.start[best])
    ex = nap.IntervalSet(start=t0, end=min(t0 + 3.5, float(run_ep.end[best])))
    lfp_ex = lfp_full.restrict(ex)
    filt_ex, phase_ex, _ = th.theta_phase(lfp_ex)

    out = dict(
        tag=tag, df=df, phases=phases,
        spectrum=pd.DataFrame(spectra, index=[np.mean(b) for b in SPECTRUM_BANDS]),
        amp_split=pd.DataFrame(amp_split),
        cell_type={int(u): str(spikes.cell_type[u]) for u in spikes.keys()},
        best_ch=best_ch, ratios=ratios, psds=psds,
        state_durations={k: float(v.tot_length()) for k, v in state_eps.items()},
        excerpt=dict(t=lfp_ex.t, raw=lfp_ex.values, filt=filt_ex.values,
                     phase=phase_ex.values,
                     spikes={int(u): spikes[u].restrict(ex).t
                             for u in spikes.keys()}),
    )
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


if __name__ == "__main__":
    import sys
    sessions = io.SESSIONS if len(sys.argv) < 2 else [io.SESSIONS[int(sys.argv[1])]]
    for s in sessions:
        res = compute(s)
        d = res["df"]
        run = d[(d.state == "run") & (d.n_spikes >= MIN_SPIKES)]
        print(f"{res['tag']}: {len(run)} units with >={MIN_SPIKES} run spikes, "
              f"{(run.shuffle_p < 0.05).sum()} phase-locked (p<0.05)")
