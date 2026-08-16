"""Run the frequency-tuning pipeline on every session of DANDI 000986 and pool."""

import numpy as np
from scipy import stats
from tqdm import tqdm

import audlib as A

assets = A.list_assets()
rows = []
per_session = []

for path, url in tqdm(assets, desc="sessions"):
    nwbfile, nap_nwb = A.load_session(url)
    trials = A.trial_table(nwbfile)
    spikes = nap_nwb["units"]
    onsets, flab = trials["onset"], trials["frequency"]
    freqs = np.unique(flab)

    ev, bl = A.evoked_rates(spikes, onsets)
    dev = ev - bl

    _, p_resp = stats.ttest_rel(ev, bl, axis=1)
    responsive = A.bh_fdr(p_resp) < 0.01

    p_tune, f_obs, _ = A.permutation_p(dev, flab, freqs, n_perm=500, seed=1)
    tuned = A.bh_fdr(p_tune) < 0.05

    tc = A.group_means(dev, flab, freqs)
    bf = freqs[np.argmax(tc, axis=1)]
    spars = A.sparseness(tc)
    bf1, bf2 = A.split_half_bf(dev, flab, freqs, seed=2)

    counts = A.window_counts(spikes, onsets, A.EVOKED_WIN)
    glm = A.glm_tuning(counts, flab, freqs)
    print(f"  GLM fit check: r(fitted, empirical mean count) = {glm['fit_corr']:.4f}",
          flush=True)

    keep = tuned & responsive
    conf, acc = A.decode_frequency(counts[keep], flab, freqs) if keep.sum() >= 3 \
        else (np.full((5, 5), np.nan), np.nan)

    per_session.append(dict(
        path=path, freqs=freqs, tc=tc, bf=bf, tuned=tuned, responsive=responsive,
        spars=spars, f_obs=f_obs, bf1=bf1, bf2=bf2, conf=conf, acc=acc,
        glm_rate=glm["rate"], glm_bf_oct=glm["bf_oct"], glm_bw=glm["bandwidth_oct"],
        glm_pr2=glm["pseudo_r2"], grid_oct=glm["grid_oct"],
        n_trials=len(onsets), base_rate=bl.mean(axis=1),
    ))
    rows.append((path, len(spikes), int(responsive.sum()), int(keep.sum()),
                 len(onsets), acc))
    print(f"{path}: {len(spikes)} units, {responsive.sum()} responsive, "
          f"{keep.sum()} tuned, decode acc {acc:.3f}", flush=True)

np.save("results_all_sessions.npy", np.array(per_session, dtype=object),
        allow_pickle=True)

n_units = sum(len(s["tuned"]) for s in per_session)
n_resp = sum(int(s["responsive"].sum()) for s in per_session)
n_tuned = sum(int((s["tuned"] & s["responsive"]).sum()) for s in per_session)
print(f"\npooled: {n_units} units, {n_resp} responsive ({100*n_resp/n_units:.0f}%), "
      f"{n_tuned} frequency tuned ({100*n_tuned/n_units:.0f}%)")
accs = np.array([s["acc"] for s in per_session])
print("decoding accuracy: %.3f +- %.3f (chance 0.2)" % (np.nanmean(accs),
                                                        np.nanstd(accs)))
