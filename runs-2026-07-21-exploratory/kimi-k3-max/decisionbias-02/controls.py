"""Controls: (1) block-level shuffle null for prior decoding (slow-drift control);
(2) choice decoding within 0.5-prior blocks only; (3) time-resolved prior decoding."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
SESSIONS = ["31f22c47", "81169999", "e7fa5ae0", "f791a116"]
WIN = (-0.4, 0.0)

def load(key):
    d = np.load(f"session_{key}.npz", allow_pickle=False)
    return (d["spike_times"], d["spike_times_index"],
            {k[7:]: d[k] for k in d.files if k.startswith("trial__")},
            {k[6:]: d[k] for k in d.files if k.startswith("unit__")})

def cv_acc(X, y, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(np.mean([np.mean(p[y[te] == c] == c) for c in np.unique(y)]))
    return np.mean(accs)

def block_labels(probL_valid):
    """Assign a contiguous-block id to each trial (runs of constant probL)."""
    bid = np.zeros(len(probL_valid), dtype=int)
    b = 0
    for i in range(1, len(probL_valid)):
        if probL_valid[i] != probL_valid[i - 1]:
            b += 1
        bid[i] = b
    return bid

for key in SESSIONS:
    spike_times, spike_times_index, trials, unit_meta = load(key)
    good = np.where(unit_meta["ks2_label"].astype(str) == "good")[0]
    bounds = np.concatenate([[0], spike_times_index])
    unit_spikes = [spike_times[bounds[i]:bounds[i+1]] for i in good]

    def count_matrix(t0, t1):
        X = np.empty((len(t0), len(unit_spikes)))
        for j, st in enumerate(unit_spikes):
            X[:, j] = np.searchsorted(st, t1, side="left") - np.searchsorted(st, t0, side="left")
        return X

    choice = trials["choice"]; stimOn = trials["stimOn_times"]
    probL = trials["probabilityLeft"]
    chose_left = (choice == 1).astype(int)
    valid = (choice != 0) & ~np.isnan(stimOn)

    # ---- (1) block prior decoding with block-level shuffle null ----
    bl = valid & np.isin(probL, [0.2, 0.8])
    X = count_matrix(stimOn[bl] + WIN[0], stimOn[bl] + WIN[1])
    yb = (probL[bl] == 0.8).astype(int)
    bids = block_labels(probL[bl])
    real = cv_acc(X, yb)
    null_trial = np.array([cv_acc(X, rng.permutation(yb)) for _ in range(100)])
    # block-level shuffle: permute labels of whole contiguous blocks
    null_block = []
    for _ in range(100):
        ub = np.unique(bids)
        perm = rng.permutation(ub)
        mapping = dict(zip(ub, yb[np.searchsorted(bids, ub)][perm.argsort()]))
        # simpler: assign each block a random label from the set of block labels
        block_label = {u: yb[bids == u][0] for u in ub}
        shuffled_block_labels = rng.permutation(list(block_label.values()))
        ys = np.array([shuffled_block_labels[list(ub).index(b)] for b in bids])
        null_block.append(cv_acc(X, ys))
    null_block = np.array(null_block)
    print(f"{key} prior decoding: real={real:.3f} | trial-shuffle {null_trial.mean():.3f}±{null_trial.std():.3f} "
          f"| block-shuffle {null_block.mean():.3f}±{null_block.std():.3f} p_block={np.mean(null_block >= real):.3f}")

    # ---- (2) choice decoding within 0.5-prior blocks ----
    nb = valid & (probL == 0.5)
    if nb.sum() > 40:
        X5 = count_matrix(stimOn[nb] + WIN[0], stimOn[nb] + WIN[1])
        y5 = chose_left[nb]
        r5 = cv_acc(X5, y5)
        n5 = np.array([cv_acc(X5, rng.permutation(y5)) for _ in range(100)])
        print(f"   choice decoding in 0.5 blocks: real={r5:.3f} null={n5.mean():.3f}±{n5.std():.3f} p={np.mean(n5 >= r5):.3f} n={nb.sum()}")
    else:
        print(f"   choice decoding in 0.5 blocks: only {nb.sum()} trials, skipped")
