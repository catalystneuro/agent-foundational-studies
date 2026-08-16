"""Reach-direction tuning: per-trial firing rates as a function of reach direction.

Two epochs are analysed: the delay period (target visible, movement withheld) and the
movement period. For each unit a cosine model r(theta) = b0 + b1*cos(theta - theta_pref)
is fit by linear regression on [1, cos theta, sin theta], and significance is assessed
with a permutation test on trial labels.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm

from s02_behavior import build

MOVE_EP = (-0.10, 0.25)    # relative to movement onset; motor cortex leads the hand
DELAY_EP = (-0.30, -0.05)  # relative to the go cue, i.e. the end of the instructed delay
N_PERM = 500


def trial_rates(spikes, starts, ends):
    """Firing rate (Hz) per trial per unit."""
    ep = nap.IntervalSet(start=starts, end=ends)
    counts = np.asarray(spikes.count(ep=ep))
    dur = (ends - starts)[:, None]
    return counts / dur


def fit_cosine(rates, angles):
    """Least-squares cosine fit per unit. Returns dict of arrays over units."""
    X = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    b0, bc, bs = beta
    # This numpy build emits spurious FP warnings from the SIMD matmul path; the assertion
    # below confirms the result is in fact finite.
    with np.errstate(all="ignore"):
        pred = X @ beta
    assert np.isfinite(pred).all() and np.isfinite(beta).all()
    ss_res = ((rates - pred) ** 2).sum(0)
    ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    return dict(
        baseline=b0,
        modulation=np.hypot(bc, bs),
        pref_dir=np.arctan2(bs, bc),
        r2=1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot),
    )


def permutation_test(rates, angles, n_perm=N_PERM, seed=0):
    """p-value for cosine modulation depth under shuffled reach directions.

    The generator is seeded per call so that results do not depend on how many times
    this function has already been called in the same process.
    """
    rng = np.random.default_rng(seed)
    obs = fit_cosine(rates, angles)["modulation"]
    null = np.empty((n_perm, rates.shape[1]))
    for i in tqdm(range(n_perm), desc="permutations", leave=False):
        null[i] = fit_cosine(rates, rng.permutation(angles))["modulation"]
    return obs, (null >= obs).mean(0), null


def circ_corr(a, b):
    """Jammalamadaka circular-circular correlation with a permutation p-value."""
    def _r(x, y):
        xm = np.angle(np.exp(1j * x).sum())
        ym = np.angle(np.exp(1j * y).sum())
        num = (np.sin(x - xm) * np.sin(y - ym)).sum()
        den = np.sqrt((np.sin(x - xm) ** 2).sum() * (np.sin(y - ym) ** 2).sum())
        return num / den
    r = _r(a, b)
    rng = np.random.default_rng(1)
    null = np.array([_r(a, rng.permutation(b)) for _ in range(2000)])
    return r, (np.abs(null) >= abs(r)).mean()


def rayleigh(angles):
    """Rayleigh test for non-uniformity of a circular sample."""
    n = len(angles)
    R = np.abs(np.exp(1j * angles).sum()) / n
    z = n * R ** 2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))
    return R, p


def binned_tuning(rates, angles, n_bins=12):
    """Mean rate per direction bin, with SEM."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(angles, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    mean = np.full((n_bins, rates.shape[1]), np.nan)
    sem = np.full_like(mean, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 2:
            mean[b] = rates[m].mean(0)
            sem[b] = rates[m].std(0) / np.sqrt(m.sum())
    return centers, mean, sem


def run(S):
    tr = S["trials"]
    ang = tr["reach_angle"].values
    mo = tr["move_onset"].values

    move = trial_rates(S["spikes"], mo + MOVE_EP[0], mo + MOVE_EP[1])

    # The instructed delay varies in length; keep trials with a delay long enough to hold
    # the whole analysis window after target onset.
    gc = tr["go_cue"].values
    long_delay = (gc + DELAY_EP[0]) > (tr["target_on"].values + 0.05)
    delay = trial_rates(S["spikes"], gc[long_delay] + DELAY_EP[0], gc[long_delay] + DELAY_EP[1])
    print(f"delay-period analysis uses {long_delay.sum()}/{len(gc)} trials")

    fit_move = fit_cosine(move, ang)
    fit_delay = fit_cosine(delay, ang[long_delay])
    obs, pval, null = permutation_test(move, ang)
    fit_move["p"] = pval
    _, pval_d, _ = permutation_test(delay, ang[long_delay])
    fit_delay["p"] = pval_d

    centers, tc_mean, tc_sem = binned_tuning(move, ang)

    return dict(
        rates_move=move, rates_delay=delay, angles=ang,
        fit_move=fit_move, fit_delay=fit_delay, null=null,
        tc_centers=centers, tc_mean=tc_mean, tc_sem=tc_sem,
    )


def figure_examples(S, R, path="figures/fig02_direction_tuning_examples.png"):
    fit = R["fit_move"]
    depth = fit["modulation"] / (fit["baseline"] + 1e-9)
    ok = (fit["p"] < 0.01) & (fit["baseline"] > 5)
    # one example per preferred-direction quadrant, taking the most strongly tuned unit in each
    units = []
    for lo in [-np.pi, -np.pi / 2, 0, np.pi / 2]:
        m = ok & (fit["pref_dir"] >= lo) & (fit["pref_dir"] < lo + np.pi / 2)
        if m.any():
            units.append(np.where(m)[0][np.argmax(depth[m])])

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 4, hspace=0.22, wspace=0.45, height_ratios=[0.85, 1.3])

    th = np.linspace(-np.pi, np.pi, 200)
    for j, u in enumerate(units):
        ax = fig.add_subplot(gs[0, j], projection="polar")
        m = np.concatenate([R["tc_mean"][:, u], R["tc_mean"][:1, u]])
        e = np.concatenate([R["tc_sem"][:, u], R["tc_sem"][:1, u]])
        c = np.concatenate([R["tc_centers"], R["tc_centers"][:1]])
        ax.plot(c, m, "o-", color="tab:blue", ms=3, lw=1.2)
        ax.fill_between(c, m - e, m + e, color="tab:blue", alpha=0.25)
        pred = fit["baseline"][u] + fit["modulation"][u] * np.cos(th - fit["pref_dir"][u])
        ax.plot(th, pred, color="crimson", lw=1.4, label="cosine fit")
        ax.set_title(
            f"unit {u}\nPD={np.degrees(fit['pref_dir'][u]):.0f}$\\degree$, "
            f"depth={depth[u]:.2f}, p<{max(fit['p'][u],1/N_PERM):.3f}",
            fontsize=9.5, pad=22,
        )
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.25, -0.12))

    # raster + PSTH by direction octant for the single best unit
    u = units[0]
    tr = S["trials"]
    ang = R["angles"]
    oct_edges = np.linspace(-np.pi, np.pi, 9)
    st = S["spikes"][u].t
    lags = np.arange(-0.4, 0.6, 0.01)
    cmap = plt.get_cmap("hsv")

    ax_r = fig.add_subplot(gs[1, :2])
    ax_p = fig.add_subplot(gs[1, 2:])
    y = 0
    for b in range(8):
        m = np.where((ang >= oct_edges[b]) & (ang < oct_edges[b + 1]))[0]
        if len(m) == 0:
            continue
        col = cmap((oct_edges[b] + np.pi + 0.4) / (2 * np.pi))
        psth = np.zeros(len(lags) - 1)
        for i in m[:40]:
            s = st[(st > tr["move_onset"].values[i] - 0.4) & (st < tr["move_onset"].values[i] + 0.6)]
            s = s - tr["move_onset"].values[i]
            ax_r.plot(s, np.full_like(s, y), "|", color=col, ms=2.5, mew=0.6)
            y += 1
        for i in m:
            s = st[(st > tr["move_onset"].values[i] - 0.4) & (st < tr["move_onset"].values[i] + 0.6)]
            psth += np.histogram(s - tr["move_onset"].values[i], bins=lags)[0]
        psth = psth / len(m) / 0.01
        k = np.ones(5) / 5
        ax_p.plot(lags[:-1] + 0.005, np.convolve(psth, k, "same"), color=col,
                  label=f"{np.degrees(oct_edges[b]):.0f}$\\degree$")
        y += 6
    for ax, ttl in [(ax_r, f"unit {u}: spike raster by reach direction"),
                    (ax_p, f"unit {u}: direction-conditioned PSTH")]:
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("time from movement onset (s)")
        ax.set_title(ttl, fontsize=11)
    ax_r.set_ylabel("trials (grouped by direction)")
    ax_p.set_ylabel("firing rate (Hz)")
    ax_p.legend(fontsize=7.5, ncol=2, title="reach direction", title_fontsize=8)

    fig.suptitle("Reach-direction tuning of single motor cortex units", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def figure_population(R, path="figures/fig03_direction_tuning_population.png"):
    fm, fd = R["fit_move"], R["fit_delay"]
    sig = fm["p"] < 0.01
    depth = fm["modulation"] / (fm["baseline"] + 1e-9)

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0], projection="polar")
    ax.hist(fm["pref_dir"][sig], bins=24, color="tab:blue", alpha=0.85)
    Rv, pr = rayleigh(fm["pref_dir"][sig])
    ax.set_title(f"Preferred directions ({sig.sum()}/{len(sig)} tuned, p<0.01)\n"
                 f"Rayleigh R={Rv:.2f}, p={pr:.3f}", fontsize=11, pad=22)
    ax.set_yticklabels([])

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(depth[sig], bins=25, color="tab:blue", alpha=0.8, label="tuned")
    ax.hist(depth[~sig], bins=25, color="0.6", alpha=0.8, label="not tuned")
    ax.set_xlabel("modulation depth  (b$_1$ / b$_0$)")
    ax.set_ylabel("units")
    ax.set_title("Depth of cosine modulation", fontsize=11)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(fm["r2"], bins=25, color="tab:blue", alpha=0.8)
    ax.set_xlabel("$R^2$ of cosine fit (single trials)")
    ax.set_ylabel("units")
    ax.set_title("Variance explained by direction alone", fontsize=11)

    # normalized tuning curves aligned to each unit's preferred direction
    ax = fig.add_subplot(gs[1, 0])
    c, m = R["tc_centers"], R["tc_mean"]
    aligned = []
    for u in np.where(sig)[0]:
        shift = np.argmin(np.abs(((c - fm["pref_dir"][u]) + np.pi) % (2 * np.pi) - np.pi))
        v = np.roll(m[:, u], -shift + len(c) // 2)
        aligned.append((v - v.min()) / (v.max() - v.min() + 1e-9))
    aligned = np.array(aligned)
    rel = np.degrees(c - c[len(c) // 2])
    ax.plot(rel, aligned.mean(0), "o-", color="crimson")
    ax.fill_between(rel, aligned.mean(0) - aligned.std(0), aligned.mean(0) + aligned.std(0),
                    color="crimson", alpha=0.2)
    ax.set_xlabel("direction relative to preferred ($\\degree$)")
    ax.set_ylabel("normalized rate")
    ax.set_title("PD-aligned population tuning curve", fontsize=11)

    ax = fig.add_subplot(gs[1, 1])
    both = sig & (fd["p"] < 0.01)
    diff = np.degrees(((fd["pref_dir"] - fm["pref_dir"]) + np.pi) % (2 * np.pi) - np.pi)
    ax.scatter(np.degrees(fm["pref_dir"][both]), np.degrees(fd["pref_dir"][both]),
               s=18, color="tab:purple", alpha=0.8)
    ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
    r, p = circ_corr(fm["pref_dir"][both], fd["pref_dir"][both])
    ax.set_xlabel("preferred direction, movement ($\\degree$)")
    ax.set_ylabel("preferred direction, delay ($\\degree$)")
    ax.set_title(f"Delay vs movement PD (n={both.sum()} tuned in both)\n"
                 f"circular r={r:.2f}, p={max(p,5e-4):.3f}; "
                 f"median |$\\Delta$|={np.median(np.abs(diff[both])):.0f}$\\degree$", fontsize=11)

    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, np.percentile(fm["modulation"], 99), 40)
    ax.hist(R["null"].ravel(), bins=bins, color="0.6", alpha=0.75, density=True,
            label="shuffled reach directions")
    ax.hist(fm["modulation"], bins=bins, color="tab:blue", alpha=0.75, density=True,
            label="observed")
    ax.set_xlabel("cosine modulation b$_1$ (Hz)")
    ax.set_ylabel("probability density")
    ax.set_title("Observed modulation vs shuffle null", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Population summary of reach-direction tuning (n=%d units)" % len(sig), fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    S = build()
    R = run(S)
    fm = R["fit_move"]
    print("significantly direction-tuned (p<0.01):", int((fm["p"] < 0.01).sum()), "/", len(fm["p"]))
    print("median modulation depth:", float(np.median(fm["modulation"] / (fm["baseline"] + 1e-9))))
    print("median single-trial R2:", float(np.nanmedian(fm["r2"])))
    figure_examples(S, R)
    figure_population(R)
