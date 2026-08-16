"""Population decoding of grating orientation from single-trial spike counts."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import orilib as O

DIRECTIONS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])


def _clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1),
    )


def decode(X, y, n_splits=5, seed=0, shuffle_null=False, rng=None):
    """Cross-validated accuracy of a multinomial logistic decoder."""
    if shuffle_null:
        rng = rng or np.random.default_rng(seed)
        y = rng.permutation(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc = []
    for tr, te in skf.split(X, y):
        clf = _clf()
        clf.fit(X[tr], y[tr])
        acc.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(acc))


def decode_vs_population_size(
    X, y, sizes, n_boot=8, seed=0, shuffle_null=False
):
    """Accuracy as a function of the number of randomly drawn units."""
    rng = np.random.default_rng(seed)
    out = []
    n_units = X.shape[1]
    for s in sizes:
        if s > n_units:
            break
        accs = []
        for b in range(n_boot):
            cols = rng.choice(n_units, s, replace=False)
            accs.append(
                decode(X[:, cols], y, seed=seed + b, shuffle_null=shuffle_null, rng=rng)
            )
        out.append((s, float(np.mean(accs)), float(np.std(accs))))
    return out


def run_decoding(sessions, sizes=(5, 10, 20, 40, 80), areas=("VISp", "VISl", "VISal",
                                                             "VISrl", "VISam", "LP", "CA1")):
    """Decode drift direction (8-way) and orientation (4-way) per area per session."""
    rows = []
    for res in tqdm(sessions, desc="decoding sessions"):
        u = res["units"]
        X_all = res["dg_rates"]
        y_dir = res["dg_labels"].astype(int)
        y_ori = (y_dir % 180).astype(int)
        for area in areas:
            cols = np.where(u["area"].values == area)[0]
            if len(cols) < 5:
                continue
            X = X_all[:, cols]
            X = X[:, X.std(0) > 0]  # constant units break the z-scoring step
            if X.shape[1] < 5:
                continue
            for task, y, chance in (("direction", y_dir, 1 / 8), ("orientation", y_ori, 1 / 4)):
                for s, m, sd in decode_vs_population_size(X, y, sizes):
                    rows.append(
                        dict(session=res["session_id"], area=area, task=task,
                             n_units=s, acc=m, sd=sd, chance=chance, kind="observed")
                    )
                for s, m, sd in decode_vs_population_size(X, y, sizes, shuffle_null=True):
                    rows.append(
                        dict(session=res["session_id"], area=area, task=task,
                             n_units=s, acc=m, sd=sd, chance=chance, kind="shuffled")
                    )
    return pd.DataFrame(rows)
