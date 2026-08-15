"""Figures for the DANDI:000582 grid-cell analysis."""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import grid_lib as gl

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv("unit_metrics.csv")
shuf = np.load("shuffle_gridness.npy")
maps = np.load("maps.npz")
stab = pd.read_csv("unit_stability.csv")
df = df.merge(stab, on=["path", "unit"], how="left")

inc = df["included"].fillna(False).values
THRESH = float(np.nanpercentile(shuf[inc].ravel(), 95))
df["is_grid"] = inc & (df["gridness"] > THRESH)
df["is_hd"] = inc & (df["hd_mvl"] > df["hd_mvl_shuf_95"])
df["is_conj"] = df["is_grid"] & df["is_hd"]
df.to_csv("unit_metrics_classified.csv", index=False)
print(f"gridness shuffle threshold (95th pct of {shuf[inc].size} shuffles) = {THRESH:.3f}")
print(f"{int(inc.sum())} units analysed, {int(df.is_grid.sum())} grid cells "
      f"({100 * df.is_grid.sum() / inc.sum():.1f}%)")


def key(row, suf):
    return f"{row['path']}|{row['unit']}|{suf}"


def show_map(ax, m, title, cmap="jet", extent=None):
    v = np.nanmax(m)
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=0, vmax=v, extent=extent,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, pad=3)
    return im


# ---------------------------------------------------------------- figure 1
# Raw data from one session: behaviour, spike rasters, firing-rate traces.
def figure_raw():
    ex = df[df.is_grid].sort_values("gridness", ascending=False).iloc[0]
    sess = [s for s in gl.list_assets() if s["path"] == ex["path"]][0]
    d = gl.load_session(sess["asset_id"])
    half, center = gl.session_extent(d["position"])
    edges = gl.map_edges(half)
    occ = gl.occupancy_map(d["position"], d["dt"], edges, center)
    xy = np.asarray(d["position_all"].values) - center
    t = d["position_all"].t

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 0.6, 0.9, 1.2], hspace=0.55, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xy[:, 0], xy[:, 1], lw=0.25, color="0.55")
    ax.set_aspect("equal"); ax.set_title("Foraging trajectory (10 min)")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(occ.T, origin="lower", extent=[edges[0], edges[-1]] * 2, cmap="viridis")
    ax.set_title("Occupancy (s per 2.5 cm bin)"); ax.set_xlabel("x (cm)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    sp = np.asarray(d["speed"].values)
    ax.hist(sp[np.isfinite(sp)], bins=60, color="0.4")
    ax.axvline(gl.SPEED_THRESH, color="crimson", ls="--")
    ax.set_xlabel("running speed (cm/s)"); ax.set_ylabel("samples")
    ax.set_title(f"Speed (samples < {gl.SPEED_THRESH} cm/s excluded)")

    win = nap.IntervalSet(start=t[0] + 120, end=t[0] + 240)
    ax = fig.add_subplot(gs[1, :])
    p = d["position_all"].restrict(win)
    ax.plot(p.t, np.asarray(p.values)[:, 0] - center[0], label="x", lw=0.8)
    ax.plot(p.t, np.asarray(p.values)[:, 1] - center[1], label="y", lw=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.set_ylabel("position (cm)"); ax.set_title("Raw behaviour and spiking, 2-minute window")
    ax.set_xlim(win.start[0], win.end[0]); ax.set_xticklabels([])

    ax = fig.add_subplot(gs[2, :])
    for j, u in enumerate(d["units"]):
        st = np.asarray(d["units"][u].restrict(win).t)
        ax.plot(st, np.full_like(st, j), "|", ms=4, color="k", mew=0.6)
    ax.set_ylabel("unit"); ax.set_yticks(range(len(d["units"])))
    ax.set_yticklabels([str(n) for n in d["units"].unit_name], fontsize=6)
    ax.set_xlim(win.start[0], win.end[0]); ax.set_xticklabels([])

    ax = fig.add_subplot(gs[3, :])
    ui = list(d["units"].keys())[int(np.flatnonzero(
        np.asarray([str(n) for n in d["units"].unit_name]) == ex["unit"])[0])]
    rate = d["units"][ui].count(0.25, win) / 0.25
    rate = rate.smooth(0.5)
    ax.plot(rate.t, np.asarray(rate.values), color="crimson", lw=1, zorder=3)

    # shade the times the animal was inside a grid field of this cell
    rm = gl.analyze_unit(np.asarray(d["units"][ui].restrict(d["run_ep"]).t),
                         d["position"], occ, edges, center)["rate_map"]
    field = rm > 0.4 * np.nanmax(rm)
    pw = d["position_all"].restrict(win)
    pxy = np.asarray(pw.values) - center
    ix = np.clip(np.digitize(pxy[:, 0], edges) - 1, 0, len(edges) - 2)
    iy = np.clip(np.digitize(pxy[:, 1], edges) - 1, 0, len(edges) - 2)
    inside = field[iy, ix]
    edges_i = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
    for s, e in zip(np.flatnonzero(edges_i == 1), np.flatnonzero(edges_i == -1) - 1):
        ax.axvspan(pw.t[s], pw.t[e], color="steelblue", alpha=0.25, lw=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("rate (Hz)")
    ax.set_title(f"Firing rate of grid cell {ex['unit']} ({ex['layer']}, gridness "
                 f"{ex['gridness']:.2f}). Shading marks the times the animal was inside "
                 f"one of this cell's grid fields.")
    ax.set_xlim(win.start[0], win.end[0])

    fig.suptitle(f"Raw data: {ex['path']}", y=0.98)
    fig.savefig("fig01_raw_session.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig01_raw_session.png")


# ---------------------------------------------------------------- figure 2
def figure_examples(n=6):
    # examples are drawn from grid cells that are also well sampled and stable
    ex = df[df.is_grid & (df.n_spikes >= 500) & (df.stability_r >= 0.5)]
    ex = ex.sort_values("gridness", ascending=False)
    # spread examples over subjects for a fairer sample
    picked, seen = [], {}
    for _, r in ex.iterrows():
        c = seen.get(r["subject"], 0)
        if c < 2:
            picked.append(r); seen[r["subject"]] = c + 1
        if len(picked) == n:
            break
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7.8))
    for j, r in enumerate(picked):
        traj = maps[r["path"] + "|traj"]
        sxy = maps[key(r, "sxy")]
        a = axes[0, j]
        a.plot(traj[:, 0], traj[:, 1], lw=0.2, color="0.75")
        a.plot(sxy[:, 0], sxy[:, 1], ".", ms=1.6, color="crimson")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"rat {r['subject']}, ses {r['session']}\n{r['unit']}, {r['layer']}, "
                    f"{r['n_spikes']} spikes", pad=4)
        show_map(axes[1, j], maps[key(r, "rm")], f"peak {r['peak_rate']:.1f} Hz")
        ac = maps[key(r, "ac")]
        axes[2, j].imshow(ac, origin="lower", cmap="jet", vmin=-0.6, vmax=1,
                          interpolation="nearest")
        axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
        axes[2, j].set_title(f"g = {r['gridness']:.2f}\n{r['spacing_cm']:.0f} cm, "
                             f"{r['orientation_deg']:.0f}°", pad=4)
    for lab, row in zip(["spikes on path", "rate map", "autocorrelogram"], range(3)):
        axes[row, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle("Grid cells in medial entorhinal cortex (DANDI:000582, Sargolini et al. 2006)",
                 y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.2)
    fig.savefig("fig02_example_grid_cells.png", dpi=140)
    plt.close(fig)
    print("fig02_example_grid_cells.png")


# ---------------------------------------------------------------- figure 3
def figure_population():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    obs = df.loc[inc, "gridness"].values
    null = shuf[inc].ravel()
    null = null[np.isfinite(null)]
    ax = axes[0, 0]
    bins = np.linspace(-1.2, 1.8, 60)
    ax.hist(null, bins=bins, density=True, color="0.7", label=f"shuffled (n={null.size:,})")
    ax.hist(obs, bins=bins, density=True, histtype="step", lw=1.8, color="crimson",
            label=f"observed (n={len(obs)})")
    ax.axvline(THRESH, color="k", ls="--", lw=1)
    ax.text(THRESH, ax.get_ylim()[1] * 0.92, f" 95th pct = {THRESH:.2f}", fontsize=8)
    ax.set_xlabel("gridness score"); ax.set_ylabel("probability density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Observed gridness exceeds the shuffled null")

    ax = axes[0, 1]
    frac_obs = float(np.mean(obs > THRESH))
    ax.bar([0, 1], [frac_obs * 100, 5], color=["crimson", "0.7"], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["observed", "chance (by\nconstruction)"])
    ax.set_ylabel("% of units above threshold")
    ax.set_title(f"{int(df.is_grid.sum())}/{int(inc.sum())} units classified as grid cells")
    for x, v in zip([0, 1], [frac_obs * 100, 5]):
        ax.text(x, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)

    ax = axes[1, 0]
    layers = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]
    sub = df[inc & df.layer.isin(layers)]
    frac = sub.groupby("layer")["is_grid"].agg(["mean", "count"]).reindex(layers)
    ax.bar(range(len(layers)), frac["mean"] * 100, color="steelblue")
    for i, (m, c) in enumerate(zip(frac["mean"], frac["count"])):
        ax.text(i, m * 100 + 1, f"{m * 100:.0f}%\n(n={int(c)})", ha="center", fontsize=8)
    ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers)
    ax.set_ylabel("% grid cells"); ax.set_title("Grid cells by MEC layer")
    ax.set_ylim(0, max(frac["mean"] * 100) * 1.35)

    ax = axes[1, 1]
    g = df[df.is_grid]; ng = df[inc & ~df.is_grid]
    ax.scatter(ng["gridness"], ng["stability_r"], s=8, color="0.6", label="other units")
    ax.scatter(g["gridness"], g["stability_r"], s=10, color="crimson", label="grid cells")
    ax.axvline(THRESH, color="k", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("gridness score"); ax.set_ylabel("split-half map correlation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    med_g = g["stability_r"].median(); med_n = ng["stability_r"].median()
    ax.set_title(f"Grid firing is stable within session\n(median r: grid {med_g:.2f}, "
                 f"other {med_n:.2f})")

    fig.tight_layout()
    fig.savefig("fig03_gridness_population.png", dpi=140)
    plt.close(fig)
    print("fig03_gridness_population.png")


# ---------------------------------------------------------------- figure 4
def figure_geometry():
    g = df[df.is_grid]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = axes[0, 0]
    ax.hist(g["spacing_cm"].dropna(), bins=np.arange(20, 90, 5), color="steelblue")
    ax.set_xlabel("grid spacing (cm)"); ax.set_ylabel("cells")
    ax.set_title(f"Spacing: median {g['spacing_cm'].median():.0f} cm "
                 f"(IQR {g['spacing_cm'].quantile(.25):.0f}-{g['spacing_cm'].quantile(.75):.0f})")

    ax = axes[0, 1]
    ax.hist(g["orientation_deg"].dropna(), bins=np.arange(0, 61, 5), color="steelblue")
    ax.set_xlabel("grid orientation (deg, mod 60)"); ax.set_ylabel("cells")
    ax.set_title("Orientation relative to the box walls")

    ax = axes[0, 2]
    ax.scatter(df.loc[inc, "spatial_info"], df.loc[inc, "gridness"], s=8, color="0.6")
    ax.scatter(g["spatial_info"], g["gridness"], s=10, color="crimson")
    ax.axhline(THRESH, color="k", ls="--", lw=1)
    ax.set_xlabel("spatial information (bits/spike)"); ax.set_ylabel("gridness")
    ax.set_title("Gridness vs spatial information")

    ax = axes[1, 0]
    layers = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]
    data = [g.loc[g.layer == L, "spacing_cm"].dropna().values for L in layers]
    ax.boxplot(data, labels=layers, showfliers=False)
    for i, d_ in enumerate(data):
        ax.plot(np.random.default_rng(0).normal(i + 1, 0.06, len(d_)), d_, ".", ms=4,
                color="crimson", alpha=0.6)
    ax.set_ylabel("grid spacing (cm)"); ax.set_title("Spacing by layer")

    ax = axes[1, 1]
    ax.scatter(g["depth_mm"], g["spacing_cm"], s=12, color="crimson")
    ok = g[["depth_mm", "spacing_cm"]].dropna()
    if len(ok) > 3:
        r = np.corrcoef(ok["depth_mm"], ok["spacing_cm"])[0, 1]
        b = np.polyfit(ok["depth_mm"], ok["spacing_cm"], 1)
        xs = np.linspace(ok["depth_mm"].min(), ok["depth_mm"].max(), 10)
        ax.plot(xs, np.polyval(b, xs), "k--", lw=1)
        ax.set_title(f"Spacing vs electrode depth (r = {r:.2f}, n = {len(ok)})")
    ax.set_xlabel("depth below dura (mm)"); ax.set_ylabel("grid spacing (cm)")

    ax = axes[1, 2]
    ax.scatter(df.loc[inc, "mean_rate"], df.loc[inc, "peak_rate"], s=8, color="0.6",
               label="other units")
    ax.scatter(g["mean_rate"], g["peak_rate"], s=10, color="crimson", label="grid cells")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean rate (Hz)"); ax.set_ylabel("peak rate (Hz)")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Firing-rate range")

    fig.tight_layout()
    fig.savefig("fig04_grid_geometry.png", dpi=140)
    plt.close(fig)
    print("fig04_grid_geometry.png")


# ---------------------------------------------------------------- figure 5
def figure_conjunctive():
    has_hd = inc & np.isfinite(df["hd_mvl"].values)
    fig = plt.figure(figsize=(12.5, 7.5))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, :2])
    d_ = df[has_hd]
    ax.scatter(d_["gridness"], d_["hd_mvl"], s=10, color="0.6")
    c = d_[d_.is_conj]
    ax.scatter(c["gridness"], c["hd_mvl"], s=14, color="darkorange", label="conjunctive")
    gonly = d_[d_.is_grid & ~d_.is_hd]
    ax.scatter(gonly["gridness"], gonly["hd_mvl"], s=14, color="crimson", label="pure grid")
    ax.axvline(THRESH, color="k", ls="--", lw=1)
    ax.set_xlabel("gridness score"); ax.set_ylabel("head-direction mean vector length")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Position and head direction are conjunctively encoded")

    ax = fig.add_subplot(gs[0, 2:])
    cats = {
        "grid only": int((d_.is_grid & ~d_.is_hd).sum()),
        "conjunctive\n(grid x HD)": int(d_.is_conj.sum()),
        "HD only": int((d_.is_hd & ~d_.is_grid).sum()),
        "neither": int((~d_.is_grid & ~d_.is_hd).sum()),
    }
    ax.bar(range(len(cats)), list(cats.values()),
           color=["crimson", "darkorange", "seagreen", "0.7"])
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(list(cats), fontsize=8)
    ax.set_ylabel("units")
    for i, v in enumerate(cats.values()):
        ax.text(i, v + 1, str(v), ha="center", fontsize=8)
    ax.set_title(f"Classification of {int(has_hd.sum())} units with head-direction tracking")

    fig.text(0.5, 0.47, "Head-direction tuning curves of four conjunctive grid cells "
                        "(radius = firing rate)", ha="center", fontsize=9)
    conj = d_[d_.is_conj].sort_values("gridness", ascending=False).head(4)
    for j, (_, r) in enumerate(conj.iterrows()):
        a = fig.add_subplot(gs[1, j], projection="polar")
        tc = maps[key(r, "hdtc")]
        # three-bin circular smoothing, for display only
        tcs = np.convolve(np.r_[tc[-2:], tc, tc[:2]], np.ones(3) / 3, "same")[2:-2]
        ang = np.linspace(0, 2 * np.pi, len(tc), endpoint=False)
        a.plot(np.append(ang, ang[0]), np.append(tcs, tcs[0]), color="darkorange")
        a.set_title(f"rat {r['subject']} {r['unit']}\ng={r['gridness']:.2f}, "
                    f"MVL={r['hd_mvl']:.2f}", fontsize=8, pad=18)
        a.set_yticklabels([])
        a.set_thetagrids([0, 90, 180, 270], labels=["0°", "", "180°", ""], fontsize=6)
    fig.savefig("fig05_conjunctive_cells.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig05_conjunctive_cells.png")


# ---------------------------------------------------------------- figure 6
def figure_single_cell_control():
    """One cell in detail: real map vs three shuffles, and its own null."""
    ex = df[df.is_grid].sort_values("gridness", ascending=False).iloc[1]
    sess = [s for s in gl.list_assets() if s["path"] == ex["path"]][0]
    d = gl.load_session(sess["asset_id"])
    pos, ep = d["position"], d["run_ep"]
    half, center = gl.session_extent(pos)
    edges = gl.map_edges(half)
    occ = gl.occupancy_map(pos, d["dt"], edges, center)
    names = [str(n) for n in d["units"].unit_name]
    ui = list(d["units"].keys())[names.index(ex["unit"])]
    st = np.asarray(d["units"][ui].restrict(ep).t)
    t0, t1 = float(ep.start[0]), float(ep.end[-1])
    rng = np.random.default_rng(1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    r = gl.analyze_unit(st, pos, occ, edges, center)
    show_map(axes[0, 0], r["rate_map"], f"observed\npeak {r['peak_rate']:.1f} Hz")
    axes[1, 0].imshow(r["autocorr"], origin="lower", cmap="jet", vmin=-0.6, vmax=1)
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    axes[1, 0].set_title(f"g = {r['gridness']:.2f}")
    for j in range(1, 3):
        sst = np.sort(gl.shift_spikes(st, t0, t1, rng.uniform(20, t1 - t0 - 20)))
        rs = gl.analyze_unit(sst, pos, occ, edges, center)
        show_map(axes[0, j], rs["rate_map"], f"shuffle {j}\npeak {rs['peak_rate']:.1f} Hz")
        axes[1, j].imshow(rs["autocorr"], origin="lower", cmap="jet", vmin=-0.6, vmax=1)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        axes[1, j].set_title(f"g = {rs['gridness']:.2f}")

    row = df[(df.path == ex["path"]) & (df.unit == ex["unit"])].index[0]
    ax = axes[0, 3]
    ax.hist(shuf[row], bins=25, color="0.7")
    ax.axvline(ex["gridness"], color="crimson", lw=2)
    ax.set_xlabel("gridness"); ax.set_ylabel("shuffles")
    ax.set_title(f"this cell's null (p = {ex['gridness_p']:.3f})")
    ax.set_xticks(ax.get_xticks())

    ax = axes[1, 3]
    traj = maps[ex["path"] + "|traj"]
    ax.plot(traj[:, 0], traj[:, 1], lw=0.2, color="0.75")
    sxy = maps[key(ex, "sxy")]
    ax.plot(sxy[:, 0], sxy[:, 1], ".", ms=1.6, color="crimson")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("spikes on path")
    fig.suptitle(f"Shuffling control — rat {ex['subject']}, unit {ex['unit']} ({ex['layer']}): "
                 f"circularly shifting the spike train destroys the hexagonal pattern", y=1.0,
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig06_shuffle_control.png", dpi=140)
    plt.close(fig)
    print("fig06_shuffle_control.png")


# ---------------------------------------------------------------- figure 7
def figure_gallery():
    g = df[df.is_grid].sort_values("gridness", ascending=False)
    n = min(24, len(g))
    ncol = 8
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(1.7 * ncol, 3.5 * nrow))
    for i in range(nrow * ncol):
        rr, cc = divmod(i, ncol)
        a1, a2 = axes[2 * rr, cc], axes[2 * rr + 1, cc]
        if i >= n:
            a1.axis("off"); a2.axis("off"); continue
        r = g.iloc[i]
        show_map(a1, maps[key(r, "rm")],
                 f"rat {r['subject']} {r['session']}\n{r['unit']} ({r['layer']})", cmap="jet")
        a1.title.set_fontsize(6)
        a2.imshow(maps[key(r, "ac")], origin="lower", cmap="jet", vmin=-0.6, vmax=1)
        a2.set_xticks([]); a2.set_yticks([])
        a2.set_title(f"g={r['gridness']:.2f}", fontsize=6, pad=2)
    fig.suptitle("Rate maps (top) and spatial autocorrelograms (bottom) of the 24 "
                 "highest-gridness cells", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig("fig07_grid_gallery.png", dpi=140)
    plt.close(fig)
    print("fig07_grid_gallery.png")


# ---------------------------------------------------------------- figure 8
def crosscorrelogram(a_map, b_map):
    """NaN-aware 2-D spatial cross-correlogram of two rate maps."""
    ma, mb = np.isfinite(a_map).astype(float), np.isfinite(b_map).astype(float)
    a = np.where(ma > 0, a_map, 0.0)
    b = np.where(mb > 0, b_map, 0.0)
    n = gl._corr_full(ma, mb)
    num = n * gl._corr_full(a, b) - gl._corr_full(a, mb) * gl._corr_full(ma, b)
    d1 = n * gl._corr_full(a * a, mb) - gl._corr_full(a, mb) ** 2
    d2 = n * gl._corr_full(ma, b * b) - gl._corr_full(ma, b) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.sqrt(np.clip(d1, 0, None) * np.clip(d2, 0, None))
    r[n < gl.MIN_OVERLAP_BINS] = np.nan
    return np.clip(r, -1, 1)


def figure_modules():
    """Simultaneously recorded grid cells share spacing and orientation."""
    from scipy import stats

    g = df[df.is_grid].dropna(subset=["spacing_cm", "orientation_deg"])
    rng = np.random.default_rng(3)

    def dorient(a, b):
        d = np.abs(a - b) % 60
        return np.minimum(d, 60 - d)

    within, across, pairs = [], [], []
    for path, sub in g.groupby("path"):
        idx = list(sub.index)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                r1, r2 = df.loc[idx[i]], df.loc[idx[j]]
                within.append((dorient(r1.orientation_deg, r2.orientation_deg),
                               abs(r1.spacing_cm - r2.spacing_cm) /
                               (0.5 * (r1.spacing_cm + r2.spacing_cm))))
                pairs.append((r1, r2))
    others = g.index.values
    for _ in range(4000):
        i, j = rng.choice(len(others), 2, replace=False)
        r1, r2 = df.loc[others[i]], df.loc[others[j]]
        if r1["path"] == r2["path"]:
            continue
        across.append((dorient(r1.orientation_deg, r2.orientation_deg),
                       abs(r1.spacing_cm - r2.spacing_cm) /
                       (0.5 * (r1.spacing_cm + r2.spacing_cm))))
    within = np.array(within); across = np.array(across)

    fig = plt.figure(figsize=(12.5, 9.5))
    gsp = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.62, wspace=0.3)
    for col, (lab, lim) in enumerate([("|Δ orientation| (deg)", (0, 30)),
                                      ("relative |Δ spacing|", (0, 1.0))]):
        ax = fig.add_subplot(gsp[0, col])
        bins = np.linspace(*lim, 16)
        ax.hist(across[:, col], bins=bins, density=True, color="0.75",
                label=f"different sessions (n={len(across)})")
        ax.hist(within[:, col], bins=bins, density=True, histtype="step", lw=1.8,
                color="crimson", label=f"same session (n={len(within)})")
        u, p = stats.mannwhitneyu(within[:, col], across[:, col])
        ax.set_xlabel(lab); ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"median {np.median(within[:, col]):.2f} vs "
                     f"{np.median(across[:, col]):.2f}, p = {p:.1e}")

    ax = fig.add_subplot(gsp[0, 2])
    ax.axis("off")
    ax.text(0, 0.5,
            "Grid cells recorded simultaneously in one\nanimal share grid orientation far "
            "more closely\nthan cells from different sessions (left), the\nsignature of a "
            "common orientation reference.\nGrid spacing is only weakly clustered\nwithin "
            "a session: several spacing modules can\nbe recorded on the same tetrode "
            "array.\n\nBottom rows: three grid cells recorded together\nand their spatial "
            "cross-correlograms with the\nfirst cell. The cross-correlograms are "
            "periodic\nbut have no central peak, so the grids are\nshifted in phase "
            "relative to one another.",
            fontsize=8.5, va="center")

    # an example set of co-recorded grid cells
    counts = g.groupby("path").size().sort_values(ascending=False)
    path = counts.index[0]
    sub = g[g.path == path].sort_values("gridness", ascending=False).head(3)
    r0 = sub.iloc[0]
    for j, (_, r) in enumerate(sub.iterrows()):
        ax = fig.add_subplot(gsp[1, j])
        show_map(ax, maps[key(r, "rm")],
                 f"{r['unit']}  g = {r['gridness']:.2f}\nspacing {r['spacing_cm']:.0f} cm, "
                 f"orientation {r['orientation_deg']:.0f}°")
        ax2 = fig.add_subplot(gsp[2, j])
        cc = crosscorrelogram(maps[key(r0, "rm")], maps[key(r, "rm")])
        ax2.imshow(cc, origin="lower", cmap="jet", vmin=-0.5, vmax=1)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title("autocorrelogram" if j == 0
                      else f"cross-correlogram with {r0['unit']}", pad=3)
    fig.suptitle(f"Simultaneously recorded grid cells — example session "
                 f"{path.split('/')[-1]}", y=0.98, fontsize=10)
    fig.savefig("fig08_grid_modules.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig08_grid_modules.png")


if __name__ == "__main__":
    figure_raw()
    figure_examples()
    figure_population()
    figure_geometry()
    figure_conjunctive()
    figure_single_cell_control()
    figure_gallery()
    figure_modules()
