"""Run the frequency-tuning pipeline over all 15 sessions of DANDI:000986."""
import pickle
import numpy as np
from tqdm import tqdm

from loader import list_assets, load_session
from analysis import session_counts, unit_stats, tuning_metrics, decode_frequency
from psthlib import psth

WIN = (-0.2, 0.4)
BS = 0.005

assets = list_assets()
results = []
for a in tqdm(assets, desc="sessions"):
    nwb, raw = load_session(a)
    units = nwb["units"]
    trials = raw.trials.to_dataframe()
    onsets = trials.start_time.values
    freq = trials.stim_frequency.values
    freqs = np.unique(freq)

    ev, bl, keys = session_counts(units, onsets)
    st = unit_stats(ev, bl, freq, freqs)
    tm = tuning_metrics(st["tc"], st["base_rate"], st["tc_odd"], st["tc_even"])

    # per-frequency population PSTHs (mean over units)
    tvec = None
    pf_psth = []
    for f in freqs:
        on = onsets[freq == f]
        rows = []
        for k in keys:
            tvec, r = psth(units[k].t, on, WIN, BS)
            rows.append(r)
        pf_psth.append(np.array(rows))
    pf_psth = np.stack(pf_psth)          # (n_freq, n_units, n_bins)

    dec = decode_frequency(ev, freq)
    dec_resp = decode_frequency(ev, freq, subset=np.flatnonzero(st["p_resp"] < 0.01))

    results.append(dict(
        path=a["path"], subject=raw.subject.subject_id, session_id=raw.session_id,
        n_units=len(keys), n_trials=len(onsets), freqs=freqs,
        unit_keys=keys, tvec=tvec, pf_psth=pf_psth,
        ev_counts=ev, bl_counts=bl, trial_freq=freq,
        trial_amp=trials.stim_amplitude.values,
        trial_dur=trials.stim_duration.values,
        **st, **tm, decode=dec, decode_resp=dec_resp))
    del nwb, raw

with open("results_all_sessions.pkl", "wb") as fh:
    pickle.dump(results, fh)

print(f"{len(results)} sessions, "
      f"{sum(r['n_units'] for r in results)} units, "
      f"{sum(r['n_trials'] for r in results)} tone trials")
for r in results:
    print(f"  {r['path']:38s} units={r['n_units']:4d} trials={r['n_trials']:5d} "
          f"decode_acc={r['decode']['acc']:.3f}")
