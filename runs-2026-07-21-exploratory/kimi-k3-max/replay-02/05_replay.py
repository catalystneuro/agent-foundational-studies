"""Step 5: Bayesian decoding of SWR events and replay scoring.

Template: direction-pooled place-field rate maps (place cells only), 50 bins.
Each SWR is binned into 20 ms bins; events need >=5 non-empty bins and >=5
active template cells. Position is decoded with a Poisson Bayesian decoder.
Replay score: correlation between time and position across all (time bin,
position bin) pairs weighted by the posterior mass ("weighted correlation").
Significance: 500 cell-ID shuffles per event (rate maps permuted across cells),
p = (1 + #|null| >= |real|) / 501.

Saves cache/replay.npz and figures/fig04_example_replays.png,
figures/fig05_replay_stats.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

RNG = np.random.default_rng(7)
BIN = 0.020          # 20 ms decoding bins
MIN_BINS = 5
MIN_CELLS = 5
NSHUF = 500
RATE_FLOOR = 1e-2    # Hz, avoids log(0)

pf = np.load("cache/placefields.npz")
place_ids = pf["place_ids"]
R = pf["rate_maps"]                      # (C, P) Hz
bin_centers = pf["bin_centers"]
rip = np.load("cache/ripples.npz")
events_all, pre_events, post_events = rip["events"], rip["pre"], rip["post"]
s = np.load("cache/spikes.npz")
spikes = {k: s[f"spikes_{k}"] for k in place_ids}

C, P = R.shape
Rf = np.maximum(R, RATE_FLOOR)
logR = np.log(Rf)                        # (C, P)
sumR = Rf.sum(axis=0)                    # (P,)
print(f"template: {C} place cells x {P} position bins")


def event_counts(ev):
    """Spike-count matrix (T, C) of 20 ms bins for one event."""
    t0, t1 = ev
    n_bins = int((t1 - t0) // BIN)
    if n_bins < MIN_BINS:
        return None
    counts = np.zeros((n_bins, C), dtype=np.float64)
    for ci, k in enumerate(place_ids):
        st = spikes[k]
        i0, i1 = np.searchsorted(st, [t0, t0 + n_bins * BIN])
        if i1 > i0:
            bidx = ((st[i0:i1] - t0) / BIN).astype(int)
            counts[:, ci] = np.bincount(bidx, minlength=n_bins)
    return counts


def posterior_from_counts(counts, logR_, sumR_):
    """Bayesian posterior (T, P) for counts (T, C)."""
    with np.errstate(all="ignore"):  # cosmetic Accelerate BLAS FP warnings
        LL = counts @ logR_ - BIN * sumR_[None, :]
    LL -= LL.max(axis=1, keepdims=True)
    Ppost = np.exp(LL)
    Ppost /= Ppost.sum(axis=1, keepdims=True)
    return Ppost


def weighted_corr(Ppost, t, x):
    """Time-position correlation weighted by posterior mass. Ppost: (S, T, P) -> (S,)."""
    S, T, P = Ppost.shape
    W = float(T)  # each row of the posterior sums to 1
    ex = np.einsum("stp,p->st", Ppost, x)              # E[x | t] per time bin
    mx = ex.sum(axis=1) / W
    mt = t.mean()
    cov = (ex * t[None, :]).sum(axis=1) / W - mt * mx
    var_t = ((t - mt) ** 2).mean()
    dx2 = (x[None, :] - mx[:, None]) ** 2              # (S, P)
    var_x = np.einsum("stp,sp->s", Ppost, dx2) / W
    corr = cov / np.sqrt(var_t * var_x)
    return np.nan_to_num(corr)  # zero-variance posterior -> 0


def decode_event(counts):
    """Real score + shuffle distribution for one event."""
    t = (np.arange(counts.shape[0]) + 0.5) * BIN
    x = bin_centers
    Ppost = posterior_from_counts(counts, logR, sumR)
    real = weighted_corr(Ppost[None], t, x)[0]
    # batched cell-ID shuffles: independent permutations of the cell axis
    perms = RNG.permuted(np.tile(np.arange(C), (NSHUF, 1)), axis=1)  # (NSHUF, C)
    cperm = counts[:, perms].transpose(1, 0, 2)                      # (NSHUF, T, C)
    # macOS Accelerate BLAS emits spurious FP RuntimeWarnings on these shapes;
    # outputs verified NaN-free and bit-identical to a brute-force loop.
    with np.errstate(all="ignore"):
        LL = cperm @ logR - BIN * sumR[None, None, :]                # (NSHUF, T, P)
        LL -= LL.max(axis=2, keepdims=True)
        Pn = np.exp(LL)
        Pn /= Pn.sum(axis=2, keepdims=True)
        null = weighted_corr(Pn, t, x)
    p = (np.sum(np.abs(null) >= abs(real)) + 1) / (NSHUF + 1)
    return real, null, p, Ppost


def run_events(events, label):
    rows = []
    for ev in tqdm(events, desc=f"decoding {label}"):
        counts = event_counts(ev)
        if counts is None:
            continue
        n_active = (counts.sum(axis=0) > 0).sum()
        n_nonempty = (counts.sum(axis=1) > 0).sum()
        if n_active < MIN_CELLS or n_nonempty < MIN_BINS:
            continue
        real, null, p, Ppost = decode_event(counts)
        rows.append({"start": ev[0], "end": ev[1], "score": real, "p": p,
                     "n_bins": counts.shape[0], "n_cells": n_active,
                     "null": null, "posterior": Ppost})
    print(f"{label}: {len(rows)} decodable events")
    return rows


pre_rows = run_events(pre_events, "PRE")
post_rows = run_events(post_events, "POST")

np.savez_compressed(
    "cache/replay.npz",
    pre_scores=np.array([r["score"] for r in pre_rows]),
    pre_ps=np.array([r["p"] for r in pre_rows]),
    pre_starts=np.array([r["start"] for r in pre_rows]),
    pre_ends=np.array([r["end"] for r in pre_rows]),
    post_scores=np.array([r["score"] for r in post_rows]),
    post_ps=np.array([r["p"] for r in post_rows]),
    post_starts=np.array([r["start"] for r in post_rows]),
    post_ends=np.array([r["end"] for r in post_rows]),
)

# --- stats -----------------------------------------------------------------------
def summarize(rows, label):
    scores = np.array([r["score"] for r in rows])
    ps = np.array([r["p"] for r in rows])
    sig = ps < 0.05
    fwd = sig & (scores > 0)
    rev = sig & (scores < 0)
    print(f"{label}: {sig.sum()}/{len(rows)} significant ({100 * sig.mean():.1f}%), "
          f"fwd {fwd.sum()}, rev {rev.sum()}, median |score| {np.median(np.abs(scores)):.3f}")
    return scores, ps, sig

pre_scores, pre_ps, pre_sig = summarize(pre_rows, "PRE")
post_scores, post_ps, post_sig = summarize(post_rows, "POST")

# Fisher exact: significant fraction PRE vs POST
table = [[pre_sig.sum(), len(pre_rows) - pre_sig.sum()],
         [post_sig.sum(), len(post_rows) - post_sig.sum()]]
odds, fisher_p = stats.fisher_exact(table)
mw = stats.mannwhitneyu(np.abs(pre_scores), np.abs(post_scores))
if post_sig.sum() > 0:
    n_fwd = int((post_sig & (post_scores > 0)).sum())
    binom = stats.binomtest(n_fwd, int(post_sig.sum()), 0.5)
    print(f"POST fwd/rev binomial p = {binom.pvalue:.2e}")
print(f"Fisher exact PRE vs POST significant fraction: p = {fisher_p:.2e}")
print(f"Mann-Whitney |score| PRE vs POST: p = {mw.pvalue:.2e}")

# --- figures ----------------------------------------------------------------------
# example replays: top forward and top reverse POST events
post_sorted_fwd = sorted([r for r in post_rows if r["p"] < 0.05 and r["score"] > 0],
                         key=lambda r: -r["score"])[:3]
post_sorted_rev = sorted([r for r in post_rows if r["p"] < 0.05 and r["score"] < 0],
                         key=lambda r: r["score"])[:3]
examples = post_sorted_fwd + post_sorted_rev
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, r in zip(axes.ravel(), examples):
    Ppost = r["posterior"]
    T = Ppost.shape[0]
    ax.imshow(Ppost.T, aspect="auto", origin="lower", cmap="hot",
              extent=[0, T * BIN * 1000, 0, 1.6])
    direction = "forward" if r["score"] > 0 else "reverse"
    ax.set_title(f"{direction}, score={r['score']:.2f}, p={r['p']:.3f}", fontsize=10)
    ax.set_xlabel("time in SWR (ms)")
    ax.set_ylabel("position (m)")
fig.suptitle("Example decoded POST-sleep SWR events (Bayesian posterior)")
fig.tight_layout()
fig.savefig("figures/fig04_example_replays.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
bins = np.linspace(0, 1, 41)
ax.hist(np.abs(pre_scores), bins=bins, alpha=0.6, density=True,
        label=f"PRE (n={len(pre_scores)})")
ax.hist(np.abs(post_scores), bins=bins, alpha=0.6, density=True,
        label=f"POST (n={len(post_scores)})")
ax.set_xlabel("|replay score| (weighted correlation)")
ax.set_ylabel("density")
ax.set_title(f"|score| distributions (MW p={mw.pvalue:.1e})")
ax.legend()
ax = axes[1]
ax.hist(pre_scores, bins=np.linspace(-1, 1, 61), alpha=0.6, density=True, label="PRE")
ax.hist(post_scores, bins=np.linspace(-1, 1, 61), alpha=0.6, density=True, label="POST")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("replay score (signed)")
ax.set_ylabel("density")
ax.set_title("Signed score distributions")
ax.legend()
ax = axes[2]
fracs = [pre_sig.mean(), post_sig.mean()]
bars = ax.bar(["PRE", "POST"], [100 * f for f in fracs], color=["tab:gray", "tab:red"])
ax.set_ylabel("% events significant (p<0.05)")
ax.set_title(f"Significant replay fraction (Fisher p={fisher_p:.1e})")
for bar, rows_, sig_ in zip(bars, [pre_rows, post_rows], [pre_sig, post_sig]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{sig_.sum()}/{len(rows_)}", ha="center", fontsize=10)
ax.set_ylim(0, max(100 * f for f in fracs) * 1.35 + 1)
fig.tight_layout()
fig.savefig("figures/fig05_replay_stats.png", dpi=150)
plt.close(fig)
print("DONE")
