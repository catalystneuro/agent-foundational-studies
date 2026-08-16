"""Run the full tuning analysis on one session and return a per-unit results table."""
import numpy as np
import pandas as pd
import sys
from tqdm.auto import tqdm

sys.path.insert(0, "scripts")
from load_data import load_session, build_trials
import tuning_analysis as ta


def analyze_session(asset_id, local_cache=None):
    nwb, nwbfile = load_session(asset_id, local_cache=local_cache)
    trials = build_trials(nwbfile)
    starts = np.asarray(trials.start, dtype=float)
    freqs = np.asarray(trials.stim_frequency, dtype=float)

    units = nwb["units"]
    rows = []
    for uid in tqdm(units.index, desc="units", leave=False):
        sp = np.asarray(units[uid].t, dtype=float)
        resp, base, order, w_p, kw_p = ta.unit_stats(sp, starts, freqs)
        tuning, base_rate = ta.tuning_and_rate(resp, base, freqs, order)
        if np.max(tuning) > 0:
            bf = ta.FREQS[np.argmax(tuning)]
        else:
            bf = np.nan
        rows.append({
            "unit": uid,
            "n_spikes": len(sp),
            "base_rate": base_rate,
            "wilcoxon_p": w_p,
            "kw_p": kw_p,
            "tone_responsive": w_p < 0.05,
            "frequency_tuned": kw_p < 0.05,
            "best_frequency": bf,
            "mean_rate": base_rate + tuning.mean(),
            **{f"rate_{int(f)}": tuning[i] for i, f in enumerate(ta.FREQS)},
        })
    df = pd.DataFrame(rows)
    df["session"] = asset_id
    # folder/context for parallel offline analyses
    return df, nwb, trials


if __name__ == "__main__":
    asset = sys.argv[1]
    df, nwb, trials = analyze_session(asset)
    out = sys.argv[2] if len(sys.argv) > 2 else f"figures/session_{asset}_results.csv"
    df.to_csv(out, index=False)
    print(f"saved {len(df)} units to {out}")
