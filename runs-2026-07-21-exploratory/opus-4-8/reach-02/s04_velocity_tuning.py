"""Velocity tuning: firing rate as a continuous function of instantaneous hand velocity.

Direction tuning measured trial-by-trial (s03) does not distinguish a neuron that codes
*which way* the hand goes from one that codes the full velocity vector. Here spike trains
and hand velocity are binned at 20 ms and treated as continuous signals, which lets us ask
(a) what the 2-D tuning surface over (vx, vy) looks like, (b) whether rate grows with speed
along the preferred direction, and (c) at what lag the relationship is tightest.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm

from s02_behavior import build

BIN = 0.02          # s
SMOOTH_STD = 0.05   # s, Gaussian kernel used to turn spike counts into a rate estimate
MOVE_SPEED = 5.0    # cm/s, threshold defining "moving"
LAGS = np.arange(-0.50, 0.505, 0.02)   # positive lag = neural activity leads the hand


def binned(S, ep):
    """Smoothed firing rate (Hz) and mean velocity in matched BIN-second bins over `ep`."""
    counts = S["spikes"].count(BIN, ep=ep)
    rate = counts.smooth(SMOOTH_STD) / BIN
    vel = S["vel"].bin_average(BIN, ep=ep)
    # bin_average and count share the same bin centers over the same epochs
    assert np.allclose(counts.t, vel.t)
    return counts, rate, vel


def moving_epochs(S):
    ep = S["speed"].threshold(MOVE_SPEED, "above").time_support
    return ep.drop_short_intervals(0.1)


def lag_profile(counts, vel, lags=LAGS):
    """R^2 of rate ~ [1, vx, vy] as a function of the lag applied to velocity."""
    t = np.asarray(counts.t)
    C = np.asarray(counts, dtype=float)
    V = np.asarray(vel, dtype=float)
    step = int(round(BIN * 1e6))
    tk = np.round(t * 1e6).astype(np.int64)
    r2 = np.full((len(lags), C.shape[1]), np.nan)
    for i, lag in enumerate(tqdm(lags, desc="lag scan", leave=False)):
        k = int(round(lag / BIN))
        if k == 0:
            a, b = np.arange(len(t)), np.arange(len(t))
        elif k > 0:
            a, b = np.arange(0, len(t) - k), np.arange(k, len(t))
        else:
            a, b = np.arange(-k, len(t)), np.arange(0, len(t) + k)
        # only keep pairs separated by exactly `lag` (i.e. not spanning a gap between trials)
        good = (tk[b] - tk[a]) == k * step
        a, b = a[good], b[good]
        X = np.column_stack([np.ones(len(a)), V[b, 0], V[b, 1]])
        y = C[a]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        with np.errstate(all="ignore"):
            pred = X @ beta
        ss_tot = ((y - y.mean(0)) ** 2).sum(0)
        r2[i] = 1 - ((y - pred) ** 2).sum(0) / np.where(ss_tot == 0, np.nan, ss_tot)
    return r2


def shift_velocity(counts, vel, lag):
    """Return (counts, velocity) aligned so velocity is `lag` seconds after each count bin."""
    t = np.asarray(counts.t)
    step = int(round(BIN * 1e6))
    tk = np.round(t * 1e6).astype(np.int64)
    k = int(round(lag / BIN))
    a = np.arange(max(0, -k), len(t) - max(0, k))
    b = a + k
    good = (tk[b] - tk[a]) == k * step
    return a[good], b[good]


def run(S):
    move_ep = moving_epochs(S)
    print(f"moving epochs: {len(move_ep)} intervals, {float(move_ep.tot_length()):.0f} s "
          f"({100*float(move_ep.tot_length())/float(S['obs'].tot_length()):.0f}% of recorded time)")

    # Bin over whole trials so that smoothing and lagging have room to work, then select
    # the moving bins afterwards.
    counts, rate, vel = binned(S, S["obs"])
    active = np.asarray(counts, dtype=float).sum(0) >= 500
    print(f"units with >=500 spikes: {active.sum()}/{len(active)}")

    r2 = lag_profile(rate, vel)
    best_lag = np.full(r2.shape[1], np.nan)
    best_lag[active] = LAGS[np.nanargmax(r2[:, active], axis=0)]
    pop_lag = LAGS[np.nanargmax(np.nanmean(r2[:, active], axis=1))]
    print(f"population-optimal lag: {pop_lag*1000:.0f} ms (positive = neural leads hand)")

    # Re-align at the population-optimal lag, then keep only bins where the hand is moving.
    a, b = shift_velocity(rate, vel, pop_lag)
    C = np.asarray(rate, dtype=float)[a]                  # Hz
    V = np.asarray(vel, dtype=float)[b]                   # cm/s
    sp = np.linalg.norm(V, axis=1)
    keep = sp > MOVE_SPEED
    C, V, sp = C[keep], V[keep], sp[keep]
    print(f"{keep.sum()} of {len(keep)} lag-aligned bins have speed > {MOVE_SPEED} cm/s")
    ang = np.arctan2(V[:, 1], V[:, 0])

    # Cosine + speed model:  rate = b0 + speed * (bc cos ang + bs sin ang)
    X = np.column_stack([np.ones(len(sp)), V[:, 0], V[:, 1]])
    beta, *_ = np.linalg.lstsq(X, C, rcond=None)
    pref_dir = np.arctan2(beta[2], beta[1])
    vel_gain = np.hypot(beta[1], beta[2])                 # Hz per (cm/s)

    # Nested model comparison, cross-validated over 5 contiguous blocks so that the extra
    # parameters of the larger model are not rewarded for free.
    #   direction only : rate ~ 1 + cos(ang) + sin(ang)          (speed carries no weight)
    #   dir + speed    : the above plus vx, vy                   (nests the smaller model)
    D = np.column_stack([np.ones(len(sp)), np.cos(ang), np.sin(ang)])
    DS = np.column_stack([D, V[:, 0], V[:, 1]])
    folds = np.array_split(np.arange(len(sp)), 5)
    cv = {}
    for name, Xm in [("dir", D), ("dir+speed", DS), ("speed", np.column_stack([np.ones(len(sp)), sp]))]:
        pred = np.empty_like(C)
        for f in folds:
            tr = np.setdiff1d(np.arange(len(sp)), f)
            bm, *_ = np.linalg.lstsq(Xm[tr], C[tr], rcond=None)
            with np.errstate(all="ignore"):
                pred[f] = Xm[f] @ bm
        cv[name] = 1 - ((C - pred) ** 2).sum(0) / ((C - C.mean(0)) ** 2).sum(0)
    r2_dironly = cv["dir"]
    r2_full = cv["dir+speed"]
    r2_speedonly = cv["speed"]

    # 2-D velocity tuning surfaces via pynapple. Shifting the velocity timestamps back by
    # pop_lag labels each moment of spiking with the velocity it predicts.
    vel_shift = nap.TsdFrame(
        t=S["vel"].t - pop_lag, d=np.asarray(S["vel"]), columns=["vx", "vy"],
        time_support=nap.IntervalSet(start=move_ep.start - pop_lag, end=move_ep.end - pop_lag),
    )
    tc2d = nap.compute_tuning_curves(
        S["spikes"], vel_shift, bins=[14, 14],
        range=[(-60, 60), (-60, 60)], feature_names=["vx", "vy"],
    )
    # Bins visited for less than 1 s give unusable rate estimates.
    occ = tc2d.attrs["occupancy"]
    print(f"velocity bins kept: {int((occ >= 1.0).sum())}/{occ.size}")
    tc2d = tc2d.where(occ >= 1.0)

    # Speed tuning along / against each unit's preferred direction
    speed_bins = np.array([5, 10, 20, 30, 40, 50, 60, 80, 110])
    sc = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    align = np.cos(ang[:, None] - pref_dir[None, :])   # (samples, units)
    sp_pref = np.full((len(sc), C.shape[1]), np.nan)
    sp_anti = np.full((len(sc), C.shape[1]), np.nan)
    for j in range(len(sc)):
        m = (sp >= speed_bins[j]) & (sp < speed_bins[j + 1])
        if m.sum() < 50:
            continue
        for u in range(C.shape[1]):
            pm = m & (align[:, u] > 0.7)
            am = m & (align[:, u] < -0.7)
            if pm.sum() > 20:
                sp_pref[j, u] = C[pm, u].mean()
            if am.sum() > 20:
                sp_anti[j, u] = C[am, u].mean()

    return dict(
        ep=move_ep, r2_lag=r2, best_lag=best_lag, pop_lag=pop_lag,
        pref_dir=pref_dir, vel_gain=vel_gain, baseline=beta[0],
        r2_full=r2_full, r2_dironly=r2_dironly, r2_speedonly=r2_speedonly,
        tc2d=tc2d, active=active,
        speed_centers=sc, sp_pref=sp_pref, sp_anti=sp_anti,
        C=C, V=V, sp=sp, ang=ang,
    )


def speed_tuning_control(S, lag=0.10):
    """Split-half control: preferred directions from the first half of the moving bins,
    speed tuning measured on the second half. This removes the concern that the split
    between preferred and anti-preferred speed tuning is an artifact of estimating each
    unit's preferred direction from the same samples used to measure the split."""
    from scipy.stats import wilcoxon

    counts, rate, vel = binned(S, S["obs"])
    a, b = shift_velocity(rate, vel, lag)
    C = np.asarray(rate, dtype=float)[a]
    V = np.asarray(vel, dtype=float)[b]
    sp = np.linalg.norm(V, axis=1)
    keep = sp > MOVE_SPEED
    C, V, sp = C[keep], V[keep], sp[keep]
    ang = np.arctan2(V[:, 1], V[:, 0])
    h = len(sp) // 2

    X = np.column_stack([np.ones(h), V[:h, 0], V[:h, 1]])
    beta, *_ = np.linalg.lstsq(X, C[:h], rcond=None)
    pref_dir = np.arctan2(beta[2], beta[1])

    C2, ang2, sp2 = C[h:], ang[h:], sp[h:]
    align = np.cos(ang2[:, None] - pref_dir[None, :])
    speed_bins = np.array([5, 10, 20, 30, 40, 50, 60, 80, 110])
    sc = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    curves = {}
    for key, sel in [("pref", lambda u: align[:, u] > 0.7), ("anti", lambda u: align[:, u] < -0.7)]:
        arr = np.full((len(sc), C.shape[1]), np.nan)
        for j in range(len(sc)):
            m = (sp2 >= speed_bins[j]) & (sp2 < speed_bins[j + 1])
            for u in range(C.shape[1]):
                mm = m & sel(u)
                if mm.sum() > 20:
                    arr[j, u] = C2[mm, u].mean()
        slopes = []
        for u in range(arr.shape[1]):
            y = arr[:, u]
            ok = np.isfinite(y)
            slopes.append(np.polyfit(sc[ok], y[ok], 1)[0] if ok.sum() > 3 else np.nan)
        curves[key] = np.array(slopes)

    ok = np.isfinite(curves["pref"]) & np.isfinite(curves["anti"])
    stat, p = wilcoxon(curves["pref"][ok], curves["anti"][ok])
    return dict(slope_pref=curves["pref"], slope_anti=curves["anti"], wilcoxon_p=p,
                frac_pos_pref=float(np.nanmean(curves["pref"] > 0)),
                frac_neg_anti=float(np.nanmean(curves["anti"] < 0)))


def figure_2d(R, path="figures/fig04_velocity_tuning_2d.png"):
    tc = R["tc2d"]
    vals = tc.values  # (units, nvx, nvy)
    order = np.argsort(-R["vel_gain"])
    units = order[:6]

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.55)
    vx = tc.coords["vx"].values
    vy = tc.coords["vy"].values

    for j, u in enumerate(units):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        m = vals[u].T
        im = ax.pcolormesh(vx, vy, m, cmap="viridis", shading="auto")
        ax.arrow(0, 0, 45 * np.cos(R["pref_dir"][u]), 45 * np.sin(R["pref_dir"][u]),
                 color="w", width=1.6, head_width=7, length_includes_head=True)
        ax.set_aspect("equal")
        ax.set_title(f"unit {u}: gain={R['vel_gain'][u]*100:.1f} Hz per (m/s)", fontsize=10)
        ax.set_xlabel("$v_x$ (cm/s)")
        ax.set_ylabel("$v_y$ (cm/s)")
        fig.colorbar(im, ax=ax, label="rate (Hz)", fraction=0.046)

    # PD-aligned population average surface
    ax = fig.add_subplot(gs[:, 3])
    grid = np.stack(np.meshgrid(vx, vy, indexing="ij"), -1)
    gang = np.arctan2(grid[..., 1], grid[..., 0])
    gsp = np.linalg.norm(grid, axis=-1)
    rel_bins = np.linspace(-np.pi, np.pi, 13)
    sp_bins = np.array([0, 15, 30, 45, 60, 75])
    acc = np.zeros((len(rel_bins) - 1, len(sp_bins) - 1))
    cnt = np.zeros_like(acc)
    for u in range(vals.shape[0]):
        v = vals[u]
        if not np.isfinite(v).any() or np.nanmax(v) <= 0:
            continue
        vn = v / np.nanmax(v)
        rel = (gang - R["pref_dir"][u] + np.pi) % (2 * np.pi) - np.pi
        ri = np.digitize(rel, rel_bins) - 1
        si = np.digitize(gsp, sp_bins) - 1
        ok = np.isfinite(vn) & (ri >= 0) & (ri < acc.shape[0]) & (si >= 0) & (si < acc.shape[1])
        np.add.at(acc, (ri[ok], si[ok]), vn[ok])
        np.add.at(cnt, (ri[ok], si[ok]), 1)
    pop = acc / np.where(cnt == 0, np.nan, cnt)
    im = ax.pcolormesh(np.degrees(rel_bins), sp_bins, pop.T, cmap="magma", shading="flat")
    ax.set_xlabel("velocity direction relative to PD ($\\degree$)")
    ax.set_ylabel("speed (cm/s)")
    ax.set_title("PD-aligned population average\n(each unit normalized to its peak)", fontsize=10)
    fig.colorbar(im, ax=ax, label="normalized rate", fraction=0.046)

    fig.suptitle("Firing rate as a function of instantaneous hand velocity", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def figure_speed_and_lag(R, path="figures/fig05_speed_and_lag.png"):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.32)
    sc = R["speed_centers"]

    # population speed tuning, preferred vs anti-preferred direction
    ax = fig.add_subplot(gs[0, 0])
    for arr, col, lab in [(R["sp_pref"], "crimson", "moving toward PD"),
                          (R["sp_anti"], "steelblue", "moving away from PD")]:
        norm = arr / np.nanmax(R["sp_pref"], axis=0)[None, :]
        m = np.nanmean(norm, axis=1)
        s = np.nanstd(norm, axis=1) / np.sqrt(np.sum(np.isfinite(norm), axis=1))
        ax.plot(sc, m, "o-", color=col, label=lab)
        ax.fill_between(sc, m - s, m + s, color=col, alpha=0.25)
    ax.set_xlabel("hand speed (cm/s)")
    ax.set_ylabel("rate (normalized to peak)")
    ax.set_title("Speed tuning splits by direction", fontsize=11)
    ax.legend(fontsize=9)

    # per-unit speed slopes
    ax = fig.add_subplot(gs[0, 1])
    slopes = {}
    for key, arr in [("pref", R["sp_pref"]), ("anti", R["sp_anti"])]:
        sl = []
        for u in range(arr.shape[1]):
            y = arr[:, u]
            m = np.isfinite(y)
            sl.append(np.polyfit(sc[m], y[m], 1)[0] if m.sum() > 3 else np.nan)
        slopes[key] = np.array(sl)
    bins = np.linspace(-0.35, 0.35, 40)
    ax.hist(slopes["pref"], bins=bins, color="crimson", alpha=0.65, label="toward PD")
    ax.hist(slopes["anti"], bins=bins, color="steelblue", alpha=0.65, label="away from PD")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("rate vs speed slope (Hz per cm/s)")
    ax.set_ylabel("units")
    ax.set_title("Speed sensitivity per unit", fontsize=11)
    ax.legend(fontsize=9)

    # nested model comparison: does adding speed help out of sample?
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(R["r2_dironly"], R["r2_full"], s=16, color="tab:purple", alpha=0.75)
    lim = [0, max(np.nanmax(R["r2_full"]), np.nanmax(R["r2_dironly"])) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("cross-validated $R^2$, direction only")
    ax.set_ylabel("cross-validated $R^2$, direction + speed")
    win = np.mean(R["r2_full"] > R["r2_dironly"]) * 100
    ax.set_title(f"Adding speed improves prediction\nfor {win:.0f}% of units (held-out blocks)",
                 fontsize=11)

    # lag profiles: population mean, per-unit heatmap, and optimal-lag histogram
    r2 = R["r2_lag"]
    good = R["active"] & (np.nanmax(r2, axis=0) > 0.02)
    norm = r2[:, good] / np.nanmax(r2[:, good], axis=0)

    ax = fig.add_subplot(gs[1, 0])
    m = norm.mean(axis=1)
    e = norm.std(axis=1) / np.sqrt(norm.shape[1])
    ax.plot(LAGS * 1000, m, color="crimson", lw=2)
    ax.fill_between(LAGS * 1000, m - e, m + e, color="crimson", alpha=0.25)
    ax.axvline(R["pop_lag"] * 1000, color="k", ls="--", lw=1,
               label=f"peak at {R['pop_lag']*1000:.0f} ms")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("lag applied to velocity (ms)\npositive = spikes lead the hand")
    ax.set_ylabel("$R^2$ (normalized per unit)")
    ax.set_title(f"Velocity model fit vs lag (n={good.sum()} units)", fontsize=11)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    order = np.argsort(R["best_lag"][good])
    im = ax.imshow(norm[:, order].T, aspect="auto", origin="lower", cmap="magma",
                   extent=[LAGS[0] * 1000, LAGS[-1] * 1000, 0, good.sum()])
    ax.axvline(0, color="w", lw=0.8, ls="--")
    ax.set_xlabel("lag applied to velocity (ms)")
    ax.set_ylabel("units (sorted by optimal lag)")
    ax.set_title("Per-unit lag profiles", fontsize=11)
    fig.colorbar(im, ax=ax, label="$R^2$ (normalized)", fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(R["best_lag"][good] * 1000, bins=np.arange(-510, 520, 20), color="tab:blue", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(np.median(R["best_lag"][good]) * 1000, color="crimson", ls="--",
               label=f"median {np.median(R['best_lag'][good])*1000:.0f} ms")
    ax.set_xlabel("optimal lag (ms)")
    ax.set_ylabel("units")
    ax.set_title(f"Optimal lead time (n={good.sum()} units, $R^2$>0.02)", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Speed sensitivity and neural lead time", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    S = build()
    R = run(S)
    print("median R2 full velocity:", float(np.nanmedian(R["r2_full"])),
          " direction only:", float(np.nanmedian(R["r2_dironly"])))
    print("fraction of units where full > direction-only:",
          float(np.mean(R["r2_full"] > R["r2_dironly"])))
    figure_2d(R)
    figure_speed_and_lag(R)

    ctl = speed_tuning_control(S)
    print(f"split-half control: median slope toward PD = {np.nanmedian(ctl['slope_pref']):.4f}, "
          f"away from PD = {np.nanmedian(ctl['slope_anti']):.4f} Hz per cm/s; "
          f"Wilcoxon p = {ctl['wilcoxon_p']:.1e}")
