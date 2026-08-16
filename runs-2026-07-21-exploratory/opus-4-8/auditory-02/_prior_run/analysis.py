"""Core frequency-tuning analysis for DANDI:000986 (mouse auditory cortex)."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

BASE_WIN = (-0.105, -0.005)   # pre-tone baseline
RESP_WIN = (0.005, 0.105)     # tone-evoked onset response
WIN_DUR = RESP_WIN[1] - RESP_WIN[0]


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(adj, 0, 1)


def session_counts(units, onsets):
    """(n_trials, n_units) evoked and baseline spike counts."""
    from psthlib import window_counts
    keys = list(units.keys())
    ev = np.stack([window_counts(units[k].t, onsets, RESP_WIN) for k in keys], 1)
    bl = np.stack([window_counts(units[k].t, onsets, BASE_WIN) for k in keys], 1)
    return ev, bl, keys


def unit_stats(ev, bl, freq, freqs):
    """Per-unit tuning curves and significance tests.

    ev, bl : (n_trials, n_units) spike counts
    freq   : (n_trials,) tone frequency of each trial
    """
    n_units = ev.shape[1]
    tc = np.zeros((n_units, len(freqs)))       # evoked rate (Hz) per frequency
    tc_sem = np.zeros_like(tc)
    base_rate = bl.mean(0) / WIN_DUR
    p_resp = np.ones(n_units)
    p_tune = np.ones(n_units)
    tc_odd = np.zeros_like(tc)
    tc_even = np.zeros_like(tc)

    groups = [np.flatnonzero(freq == f) for f in freqs]
    for u in range(n_units):
        e, b = ev[:, u], bl[:, u]
        # sound responsiveness: evoked vs baseline, paired across trials
        if np.any(e != b):
            p_resp[u] = stats.wilcoxon(e, b, zero_method="zsplit").pvalue
        # frequency tuning: does evoked count depend on frequency?
        gs = [e[g] for g in groups]
        if e.sum() > 0:
            p_tune[u] = stats.kruskal(*gs).pvalue
        for i, g in enumerate(groups):
            tc[u, i] = e[g].mean() / WIN_DUR
            tc_sem[u, i] = e[g].std(ddof=1) / np.sqrt(len(g)) / WIN_DUR
            tc_odd[u, i] = e[g[0::2]].mean() / WIN_DUR
            tc_even[u, i] = e[g[1::2]].mean() / WIN_DUR
    return dict(tc=tc, tc_sem=tc_sem, base_rate=base_rate, p_resp=p_resp,
                p_tune=p_tune, tc_odd=tc_odd, tc_even=tc_even)


def tuning_metrics(tc, base_rate, tc_odd, tc_even):
    """Best frequency, modulation depth, sparseness, split-half reliability."""
    delta = tc - base_rate[:, None]            # baseline-subtracted tuning curve
    bf_idx = np.argmax(delta, axis=1)
    depth = np.zeros(len(tc))
    pos = np.clip(delta, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = (pos.max(1) - pos.min(1)) / (pos.max(1) + 1e-12)
        n = tc.shape[1]
        sparseness = (1 - (pos.mean(1) ** 2) / (pos ** 2).mean(1)) / (1 - 1 / n)
    reliab = np.array([
        stats.pearsonr(a, b).statistic if np.std(a) > 0 and np.std(b) > 0 else np.nan
        for a, b in zip(tc_odd - base_rate[:, None], tc_even - base_rate[:, None])])
    return dict(delta=delta, bf_idx=bf_idx, depth=depth,
                sparseness=sparseness, reliability=reliab)


def decode_frequency(ev, freq, n_splits=5, seed=0, subset=None):
    """Cross-validated multinomial decoding of tone frequency from population counts."""
    X = ev if subset is None else ev[:, subset]
    y = freq
    if X.shape[1] == 0:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    # numpy 2 emits spurious fp warnings from BLAS matmul inside the solver
    with np.errstate(all="ignore"):
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.1))
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
    acc = (pred == y).mean()
    labels = np.unique(y)
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    # shuffle control
    rng = np.random.default_rng(seed)
    shuf = np.array([(pred == rng.permutation(y)).mean() for _ in range(200)])
    return dict(acc=acc, cm=cm, labels=labels, pred=pred,
                shuffle_mean=shuf.mean(), shuffle_p=(shuf >= acc).mean())
