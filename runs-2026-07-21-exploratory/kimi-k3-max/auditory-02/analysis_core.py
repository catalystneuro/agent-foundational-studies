"""Core analysis: frequency tuning for all 15 sessions of DANDI 000986.

Computes, per unit:
  - baseline rate (pre-stimulus window) and evoked rates per frequency
  - baseline-subtracted tuning curve, best frequency, selectivity, half-max width
  - Kruskal-Wallis p (frequency modulation), Wilcoxon p (responsiveness)
  - Poisson GLM (B-spline on log2 frequency) smooth tuning curve + McFadden pseudo-R2
Also computes population PSTH per frequency per session.
Saves results_all_sessions.npz
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import lindi
from pynwb import NWBHDF5IO
from scipy import stats as sstats
from scipy.special import gammaln
from tqdm import tqdm
import nemos as nmo

ASSETS = {
    "sub-LA3_ses-3": "5e111970-9331-41d0-81b2-829e1c0f8040",
    "sub-LA8_ses-1": "60303460-38be-44a0-951e-82c7957d1217",
    "sub-LA8_ses-2": "ce06d820-e471-4413-a3a8-9c0b21da8680",
    "sub-LA9_ses-1": "d19d0ca3-7c9a-41fe-bd97-b4ee8899a612",
    "sub-LA9_ses-3": "3d1c45a5-a26a-4083-ab68-57f3ecca1357",
    "sub-LA9_ses-4": "ea13b270-0975-4d88-bdf4-17e71efb009b",
    "sub-LA9_ses-5": "eafb5f48-0ed7-414c-a3cc-3b662a4506a2",
    "sub-LA11_ses-1": "aacd1c8a-73f7-469e-bf08-0afd5c1052f9",
    "sub-LA11_ses-2": "b8d3abca-0e78-4df1-9a51-d122a383be63",
    "sub-LA11_ses-3": "a7c6cce3-442a-4dc4-aa86-300558ae1909",
    "sub-LA11_ses-4": "36bbc777-6708-45e5-85f2-48f56b84496d",
    "sub-LA12_ses-1": "eb82c81a-87a0-40a4-b70e-535ac0909c86",
    "sub-LA12_ses-2": "d0986739-6cc0-4bc7-9d2f-8363c233ed64",
    "sub-LA12_ses-3": "ffb5c0b9-0d5b-418a-ad52-1c786818a7e5",
    "sub-LA12_ses-4": "b35476db-13dc-4569-a4ed-4e839adde857",
}

RESP = (0.005, 0.060)   # response window, s post tone onset
BASE = (-0.050, 0.0)    # baseline window, s pre onset
FREQS = np.array([2000., 4000., 8000., 16000., 32000.])
LOG2F = np.log2(FREQS / 1000.)          # 1..5
PSTH_EDGES = np.arange(-0.1, 0.3 + 0.005, 0.005)
PSTH_CENTERS = 0.5 * (PSTH_EDGES[:-1] + PSTH_EDGES[1:])
GRID = np.linspace(LOG2F[0], LOG2F[-1], 100)  # fine grid for GLM curves


def count_in_windows(spk, starts, w):
    lo = np.searchsorted(spk, starts + w[0])
    hi = np.searchsorted(spk, starts + w[1])
    return hi - lo


def psth_for_unit(spk, ons):
    """Vectorized perievent histogram for one unit."""
    idx = np.searchsorted(ons, spk, side="right") - 1
    valid = idx >= 0
    rel = spk[valid] - ons[idx[valid]]
    keep = (rel >= PSTH_EDGES[0]) & (rel < PSTH_EDGES[-1])
    return np.histogram(rel[keep], bins=PSTH_EDGES)[0]


def analyze_session(name, aid, local_cache):
    url = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{aid}/nwb.lindi.json"
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()

    trials = nwbfile.intervals["trials"].to_dataframe()
    onsets = trials["start_time"].values
    freq_trial = trials["stim_frequency"].values
    n_trials = len(onsets)

    st_idx = f['units/spike_times_index'][:]
    st_all = f['units/spike_times'][:]
    bounds = np.concatenate([[0], st_idx])
    spike_times_list = [st_all[bounds[i]:bounds[i + 1]] for i in range(len(st_idx))]
    n_units = len(spike_times_list)

    # per-trial counts
    resp_counts = np.zeros((n_trials, n_units), dtype=np.float64)
    base_counts = np.zeros((n_trials, n_units), dtype=np.float64)
    for i, spk in enumerate(spike_times_list):
        resp_counts[:, i] = count_in_windows(spk, onsets, RESP)
        base_counts[:, i] = count_in_windows(spk, onsets, BASE)

    resp_rate = resp_counts / (RESP[1] - RESP[0])
    base_rate = base_counts / (BASE[1] - BASE[0])
    baseline_mean = base_rate.mean(axis=0)

    tuning = np.zeros((n_units, len(FREQS)))
    tuning_sem = np.zeros((n_units, len(FREQS)))
    for fi, fr in enumerate(FREQS):
        m = freq_trial == fr
        tuning[:, fi] = resp_rate[m].mean(axis=0)
        tuning_sem[:, fi] = resp_rate[m].std(axis=0) / np.sqrt(m.sum())
    evoked = tuning - baseline_mean[:, None]

    # statistics
    kw_p = np.array([
        sstats.kruskal(*[resp_counts[freq_trial == fr, u] for fr in FREQS]).pvalue
        for u in range(n_units)
    ])
    wilcox_p = np.array([
        sstats.wilcoxon(resp_counts[:, u], base_counts[:, u]).pvalue
        if (resp_counts[:, u] != base_counts[:, u]).any() else 1.0
        for u in range(n_units)
    ])

    evoked_clip = np.clip(evoked, 0, None)
    bf_idx = np.argmax(evoked, axis=1)
    r_max = evoked_clip.max(axis=1)
    # Vinje-Gallant sparseness of the (clipped) evoked tuning curve:
    # 0 = equal response to all frequencies, 1 = response to exactly one frequency
    s1 = evoked_clip.sum(axis=1)
    s2 = (evoked_clip ** 2).sum(axis=1)
    n_f = len(FREQS)
    sparseness = np.zeros(n_units)
    ok = s2 > 0
    sparseness[ok] = (1 - s1[ok] ** 2 / (n_f * s2[ok])) / (1 - 1 / n_f)
    halfmax = 0.5 * r_max
    n_above = (evoked_clip >= halfmax[:, None]).sum(axis=1)
    halfmax_width_oct = (n_above - 1).astype(float)  # spacing is 1 octave

    # population PSTH per frequency
    psth = np.zeros((len(FREQS), len(PSTH_CENTERS)))
    for fi, fr in enumerate(FREQS):
        ons = np.sort(onsets[freq_trial == fr])
        tot = np.zeros(len(PSTH_CENTERS))
        for spk in spike_times_list:
            tot += psth_for_unit(spk, ons)
        psth[fi] = tot / (len(ons) * n_units * 0.005)

    # ---- GLM: Poisson, B-spline basis on log2 frequency ----
    basis = nmo.basis.BSplineEval(n_basis_funcs=4)
    X = basis.compute_features(np.log2(freq_trial / 1000.))
    glm = nmo.glm.PopulationGLM(solver_name="LBFGS",
                                solver_kwargs={"tol": 1e-6, "maxiter": 2000})
    glm.fit(X, resp_counts)
    # per-unit McFadden pseudo-R2 vs intercept-only (null) model.
    # NOTE: the full Poisson log-likelihood including -log(y!) is required here;
    # the term cancels in likelihood-ratio differences but NOT in McFadden's ratio.
    mu_model = np.asarray(glm.predict(X))                 # n_trials x n_units, counts
    y = resp_counts
    mu_null = y.mean(axis=0, keepdims=True) * np.ones_like(y)
    eps = 1e-12
    log_yfac = gammaln(y + 1)                             # log(y!)
    ll_model = (y * np.log(mu_model + eps) - mu_model - log_yfac).sum(axis=0)
    ll_null = (y * np.log(mu_null + eps) - mu_null - log_yfac).sum(axis=0)
    pseudo_r2 = 1 - ll_model / ll_null
    pseudo_r2 = np.where(ll_null < 0, pseudo_r2, 0.0)     # degenerate units -> 0
    pseudo_r2 = np.clip(pseudo_r2, 0, 1)
    Xg = basis.compute_features(GRID)
    glm_curves = np.asarray(glm.predict(Xg)) / (RESP[1] - RESP[0])  # Hz, n_grid x n_units
    glm_bf_log2 = GRID[np.argmax(glm_curves, axis=0)]

    return dict(
        name=name, n_units=n_units, n_trials=n_trials,
        baseline=baseline_mean, tuning=tuning, tuning_sem=tuning_sem, evoked=evoked,
        kw_p=kw_p, wilcox_p=wilcox_p, bf_idx=bf_idx, sparseness=sparseness,
        halfmax_width_oct=halfmax_width_oct, psth=psth,
        pseudo_r2=np.asarray(pseudo_r2), glm_curves=glm_curves, glm_bf_log2=glm_bf_log2,
        overall_rate=np.array([len(s) for s in spike_times_list]) / trials["stop_time"].max(),
    )


def main():
    local_cache = lindi.LocalCache()
    results = {}
    for name, aid in tqdm(ASSETS.items(), desc="sessions"):
        results[name] = analyze_session(name, aid, local_cache)
        r = results[name]
        frac_tuned = np.mean(r["kw_p"] < 0.05)
        frac_resp = np.mean(r["wilcox_p"] < 0.05)
        tqdm.write(f"{name}: {r['n_units']} units, {frac_resp*100:.0f}% responsive, "
                   f"{frac_tuned*100:.0f}% freq-modulated, median pseudo-R2={np.median(r['pseudo_r2']):.3f}")

    # save flat arrays with session index
    sess_names = list(results.keys())
    cat = lambda key: np.concatenate([results[s][key] for s in sess_names])
    np.savez(
        "results_all_sessions.npz",
        session_names=np.array(sess_names),
        session_unit_counts=np.array([results[s]["n_units"] for s in sess_names]),
        session_idx=np.concatenate(
            [[i] * results[sess_names[i]]["n_units"] for i in range(len(sess_names))]),
        freqs=FREQS, grid=GRID, psth_centers=PSTH_CENTERS,
        baseline=cat("baseline"), tuning=cat("tuning"), tuning_sem=cat("tuning_sem"),
        evoked=cat("evoked"), kw_p=cat("kw_p"), wilcox_p=cat("wilcox_p"),
        bf_idx=cat("bf_idx"), sparseness=cat("sparseness"),
        halfmax_width_oct=cat("halfmax_width_oct"), pseudo_r2=cat("pseudo_r2"),
        glm_bf_log2=cat("glm_bf_log2"), overall_rate=cat("overall_rate"),
        glm_curves=np.concatenate([results[s]["glm_curves"].T for s in sess_names]),  # units x grid
        psth=np.stack([results[s]["psth"] for s in sess_names]),  # sessions x freqs x time
    )
    print("saved results_all_sessions.npz")


if __name__ == "__main__":
    main()
