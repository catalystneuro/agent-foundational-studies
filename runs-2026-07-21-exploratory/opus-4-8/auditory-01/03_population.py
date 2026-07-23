"""Pool frequency tuning across all 15 sessions (5 mice) of DANDI:000986."""
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import analysis
import common

N_SHUFFLE = 200
rng = np.random.default_rng(0)

rows, tunings, halves = [], [], []
for path in tqdm(common.SESSIONS, desc="sessions"):
    s = common.load_session(path)
    onsets = s["trials"]["start_time"].values
    freq = s["frequency"]
    tuning, sem, driven, st, evoked = analysis.session_tuning(s["units"], onsets, freq)

    # shuffled control: permute frequency labels across trials, recompute sparseness
    freqs = np.unique(freq)
    null_sparse = np.empty((N_SHUFFLE, evoked.shape[1]))
    null_peak = np.empty_like(null_sparse)
    for i in range(N_SHUFFLE):
        perm = rng.permutation(freq)
        tc = np.stack([evoked[perm == f].mean(0) for f in freqs], axis=1)
        null_sparse[i] = analysis.sparseness(tc)
        null_peak[i] = tc.max(1) - tc.min(1)
    # split-half tuning: pick the best frequency on one half of the trials and
    # read out the tuning curve on the other half, so panel a/b are unbiased
    half = np.zeros(len(freq), bool)
    half[::2] = True
    tc_a = np.stack([evoked[half & (freq == f)].mean(0) for f in freqs], axis=1)
    tc_b = np.stack([evoked[~half & (freq == f)].mean(0) for f in freqs], axis=1)

    obs_range = tuning.values.max(1) - tuning.values.min(1)
    st["p_perm"] = (null_peak >= obs_range).mean(0)
    st["null_sparseness"] = null_sparse.mean(0)
    st["tuning_range"] = obs_range
    st["subject"] = s["subject"]
    st["session"] = f"{s['subject']}_ses{s['session_id']}"
    st["unit"] = st.index
    rows.append(st)
    st["best_freq_a"] = freqs[np.argmax(tc_a, axis=1)]
    tunings.append(pd.DataFrame(tuning.values, columns=freqs).assign(
        session=st["session"].values, unit=st.index))
    halves.append((tc_a, tc_b))

pop = pd.concat(rows, ignore_index=True)
tun = pd.concat(tunings, ignore_index=True)
freqs = np.unique(np.array([2000., 4000., 8000., 16000., 32000.]))

pop["sig_sound"] = analysis.fdr(pop["p_sound"].values)
pop["sig_freq"] = analysis.fdr(pop["p_freq"].values)
pop["sig_perm"] = analysis.fdr(np.clip(pop["p_perm"].values, 1 / N_SHUFFLE, None))

print(f"\n{len(pop)} units from {pop.subject.nunique()} mice, "
      f"{pop.session.nunique()} sessions")
print(f"sound-responsive (Wilcoxon, FDR<0.05):      {pop.sig_sound.sum()} "
      f"({100*pop.sig_sound.mean():.1f}%)")
print(f"frequency-selective (Kruskal-Wallis, FDR):  {pop.sig_freq.sum()} "
      f"({100*pop.sig_freq.mean():.1f}%)")
print(f"frequency-selective (label permutation):    {pop.sig_perm.sum()} "
      f"({100*pop.sig_perm.mean():.1f}%)")
resp = pop[pop.sig_sound]
print(f"among sound-responsive units, frequency-selective: "
      f"{100*resp.sig_freq.mean():.1f}%")
print("\nbest-frequency counts (frequency-selective units):")
print(pop[pop.sig_freq].best_freq.value_counts().sort_index())
print("\nmedian sparseness observed vs shuffled: "
      f"{pop[pop.sig_freq].sparseness.median():.3f} vs "
      f"{pop[pop.sig_freq].null_sparseness.median():.3f}")

pop.to_csv("population_stats.csv", index=False)
tun.to_csv("population_tuning.csv", index=False)
tc_a = np.concatenate([h[0] for h in halves])
tc_b = np.concatenate([h[1] for h in halves])
with open("pooled_results.pkl", "wb") as fh:
    pickle.dump(dict(pop=pop, tun=tun, freqs=freqs, tc_a=tc_a, tc_b=tc_b), fh)
print("\nsaved population_stats.csv / population_tuning.csv / pooled_results.pkl")
