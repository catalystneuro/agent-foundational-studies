"""Core place-field analysis for one session: rate maps, spatial information,
circular-shift significance test, and the classic population figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
import pf_lib

N_BINS = 40
N_SHUFFLES = 1000
SMOOTH = 1.0          # gaussian sigma in bins (bin = 4 cm -> sigma = 4 cm)
MIN_PEAK_RATE = 1.0   # Hz
MIN_SPIKES = 50
ALPHA = 0.01

nwbfile = pf_lib.open_session(pf_lib.PROTOTYPE)
pf_lib.check_timebase(nwbfile)
pos2d, lin, fs, maze = pf_lib.load_behavior(nwbfile)
epochs = pf_lib.load_epochs(nwbfile)
lin = lin.restrict(epochs["MazeEpoch"])
lin_valid, vel, run_eps = pf_lib.make_run_epochs(lin, fs)
units = pf_lib.load_units(nwbfile)
exc = units[np.where(units.cell_type == "excitatory")[0]]
rng_track = pf_lib.track_range(lin_valid)

print(f"session Achilles_10252013: {len(units)} units, {len(exc)} excitatory")
for k in ("rightward", "leftward"):
    print(f"  {k}: {len(run_eps[k])} laps, {run_eps[k].tot_length():.0f} s")

DIRS = ["rightward", "leftward"]
res = {}
for d in DIRS:
    rm = pf_lib.DirectionalRateMaps(lin_valid, run_eps[d], fs, N_BINS,
                                    rng_track, smooth_bins=SMOOTH)
    maps = rm.rate_maps(exc)
    si, sparsity, mean_rate = rm.si(maps)
    null = rm.shuffle_si(exc, n_shuffles=N_SHUFFLES, seed=1)
    # one-sided p: fraction of shuffles with SI >= observed
    pval = (np.sum(null >= si[None, :], axis=0) + 1) / (N_SHUFFLES + 1)
    zval = (si - null.mean(axis=0)) / null.std(axis=0)
    peak_rate, peak_pos, width = pf_lib.field_metrics(maps, rm.centers, rm.bin_size)
    n_spk = np.array([len(exc[u].restrict(run_eps[d])) for u in exc.keys()])
    odd, even = rm.lap_split_maps(exc)
    ok = rm.valid_bins
    stability = np.array([
        np.corrcoef(odd[i, ok], even[i, ok])[0, 1]
        if np.isfinite(odd[i, ok]).all() and np.isfinite(even[i, ok]).all()
        and odd[i, ok].std() > 0 and even[i, ok].std() > 0 else np.nan
        for i in range(len(exc))])

    is_place = (pval < ALPHA) & (peak_rate >= MIN_PEAK_RATE) & (n_spk >= MIN_SPIKES)
    res[d] = dict(rm=rm, maps=maps, si=si, null=null, pval=pval, zval=zval,
                  peak_rate=peak_rate, peak_pos=peak_pos, width=width,
                  n_spk=n_spk, odd=odd, even=even, stability=stability,
                  is_place=is_place, mean_rate=mean_rate, sparsity=sparsity)
    print(f"\n[{d}] place cells: {is_place.sum()}/{len(exc)} "
          f"({100*is_place.mean():.0f}%)")
    print(f"  SI (place cells): median {np.median(si[is_place]):.2f} bits/spike, "
          f"null median {np.median(null):.2f}")
    print(f"  peak rate median {np.median(peak_rate[is_place]):.2f} Hz, "
          f"field width median {np.median(width[is_place]):.2f} m")
    print(f"  odd/even stability median r={np.nanmedian(stability[is_place]):.2f}")

either = res["rightward"]["is_place"] | res["leftward"]["is_place"]
both = res["rightward"]["is_place"] & res["leftward"]["is_place"]
print(f"\nplace cell in at least one direction: {either.sum()}/{len(exc)} "
      f"({100*either.mean():.0f}%);  in both: {both.sum()}")

np.savez("results_single_session.npz",
         centers=res["rightward"]["rm"].centers,
         valid=res["rightward"]["rm"].valid_bins,
         **{f"{d}_{k}": res[d][k] for d in DIRS
            for k in ("maps", "si", "pval", "zval", "peak_rate", "peak_pos",
                      "width", "n_spk", "odd", "even", "stability", "is_place",
                      "mean_rate", "sparsity")})

# ---------------------------------------------------------------------------
# Figure 3: example place cells, spike-position raster over laps + rate maps
# ---------------------------------------------------------------------------
# Choose six well-isolated, highly stable examples whose fields tile the track:
# split the track into six segments and take the highest-SI place cell in each.
r = res["rightward"]
cand = np.where(r["is_place"] & (r["stability"] > 0.85) & (r["width"] <= 0.6))[0]
seg_edges = np.linspace(rng_track[0], rng_track[1], 7)
pick = []
for lo, hi in zip(seg_edges[:-1], seg_edges[1:]):
    inseg = cand[(r["peak_pos"][cand] >= lo) & (r["peak_pos"][cand] < hi)]
    if len(inseg):
        pick.append(inseg[np.argmax(r["si"][inseg])])
pick = np.array(pick)
print(f"example units: {[list(exc.keys())[i] for i in pick]}")

fig, axes = plt.subplots(3, len(pick), figsize=(3.2 * len(pick), 9),
                         gridspec_kw={"height_ratios": [2, 2, 1.4]})
unit_ids = list(exc.keys())
for c, i in enumerate(pick):
    u = unit_ids[i]
    for row, d in enumerate(DIRS):
        ax = axes[row, c]
        ep = run_eps[d]
        for lap in range(len(ep)):
            iv = nap.IntervalSet(start=ep.start[lap], end=ep.end[lap])
            st = exc[u].restrict(iv)
            if len(st) == 0:
                continue
            sp = lin_valid.restrict(iv).interpolate(st).values
            ax.plot(sp, np.full(len(sp), lap), "|", ms=3,
                    color="tab:red" if d == "rightward" else "tab:blue", mew=0.9)
        ax.set_xlim(rng_track)
        ax.set_ylim(-1, len(ep))
        if c == 0:
            ax.set_ylabel(f"{d}\nlap #")
        if row == 0:
            ax.set_title(f"unit {u}\nSI {res['rightward']['si'][i]:.2f} / "
                         f"{res['leftward']['si'][i]:.2f} bits/spk",
                         fontsize=10, pad=6)
        ax.tick_params(labelbottom=False)

    ax = axes[2, c]
    for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
        ax.plot(res[d]["rm"].centers, res[d]["maps"][i], color=col, lw=2, label=d)
    ax.set_xlim(rng_track)
    ax.set_xlabel("position on track (m)")
    if c == 0:
        ax.set_ylabel("rate (Hz)")
        ax.legend(fontsize=8, frameon=False)

fig.suptitle("CA1 place cells, DANDI:000044 sub-Achilles ses-10252013 — "
             "spikes plotted at the animal's position on each traversal",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("figures/fig03_example_place_cells.png", dpi=150)
plt.close(fig)
print("saved figures/fig03_example_place_cells.png")

# ---------------------------------------------------------------------------
# Figure 4: population rate maps, cross-validated ordering
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, d in enumerate(DIRS):
    R = res[d]
    sel = np.where(R["is_place"])[0]
    ok = R["rm"].valid_bins
    cen = R["rm"].centers

    def norm(m):
        mx = np.nanmax(m, axis=1, keepdims=True)
        return m / np.where(mx > 0, mx, np.nan)

    # order by peak of the ODD laps, then display the EVEN laps: the diagonal
    # in the right-hand panel is therefore cross-validated
    order = sel[np.argsort(np.nanargmax(R["odd"][sel][:, ok], axis=1))]
    for col, (M, title) in enumerate([
            (R["maps"], "all laps"),
            (R["odd"], "odd laps (used for ordering)"),
            (R["even"], "even laps (held out)")]):
        ax = axes[row, col]
        im = ax.imshow(norm(M[order][:, ok]), aspect="auto", origin="lower",
                       cmap="viridis", vmin=0, vmax=1,
                       extent=[cen[ok][0], cen[ok][-1], 0, len(order)])
        ax.set_title(f"{d} — {title}", fontsize=11)
        ax.set_xlabel("position on track (m)")
        if col == 0:
            ax.set_ylabel("place cell # (ordered by odd-lap peak)")
        fig.colorbar(im, ax=ax, label="normalised rate", fraction=0.046)

fig.suptitle("Place fields tile the track, and the ordering holds on held-out laps",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("figures/fig04_population_maps.png", dpi=150)
plt.close(fig)
print("saved figures/fig04_population_maps.png")

# ---------------------------------------------------------------------------
# Figure 5: spatial information vs the circular-shift null
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
R = res["rightward"]
ax = axes[0, 0]
ax.hist(R["null"].ravel(), bins=60, density=True, color="0.7",
        label=f"circular-shift null ({N_SHUFFLES} shuffles x {len(exc)} units)")
ax.hist(R["si"], bins=30, density=True, alpha=0.7, color="tab:red",
        label="observed (rightward)")
ax.set(xlabel="spatial information (bits/spike)", ylabel="density",
       title="Observed spatial information exceeds the shuffled null")
ax.legend(fontsize=9)

ax = axes[0, 1]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.scatter(R["null"].mean(axis=0), R["si"], s=18, alpha=0.6, color=col,
               label=f"{d} ({R['is_place'].sum()} place cells)")
lim = [0, max(np.max(res[d]["si"]) for d in DIRS) * 1.05]
ax.plot(lim, lim, "k--", lw=1, label="unity")
ax.set(xlabel="mean SI of that unit's own shuffles (bits/spike)",
       ylabel="observed SI (bits/spike)", xlim=lim, ylim=lim,
       title="Every unit compared with its own null")
ax.legend(fontsize=9)

ax = axes[1, 0]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    ax.hist(res[d]["zval"], bins=40, alpha=0.55, color=col, label=d)
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="SI z-score relative to own shuffle null", ylabel="units",
       title="Spatial information z-scores")
ax.legend(fontsize=9)

ax = axes[1, 1]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.scatter(R["si"], R["stability"], s=18, alpha=0.6, color=col, label=d)
ax.axhline(0, color="k", lw=1)
ax.set(xlabel="spatial information (bits/spike)",
       ylabel="odd vs even lap map correlation",
       title="Informative cells also have stable fields")
ax.legend(fontsize=9)

fig.suptitle("Statistical validation of spatial tuning", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("figures/fig05_spatial_information.png", dpi=150)
plt.close(fig)
print("saved figures/fig05_spatial_information.png")

# ---------------------------------------------------------------------------
# Figure 6: field properties and directionality
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.hist(R["peak_rate"][R["is_place"]], bins=np.arange(0, 30, 1.5),
            alpha=0.55, color=col, label=d)
ax.set(xlabel="in-field peak rate (Hz)", ylabel="place cells",
       title="Peak firing rate")
ax.legend(fontsize=9)

ax = axes[0, 1]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.hist(R["width"][R["is_place"]], bins=np.arange(0, 1.0, 0.06),
            alpha=0.55, color=col, label=d)
ax.set(xlabel="field width at half maximum (m)", ylabel="place cells",
       title="Place field width")
ax.legend(fontsize=9)

ax = axes[1, 0]
for d, col in zip(DIRS, ["tab:red", "tab:blue"]):
    R = res[d]
    ax.hist(R["peak_pos"][R["is_place"]], bins=np.linspace(*rng_track, 17),
            alpha=0.55, color=col, label=d)
ax.set(xlabel="field peak position (m)", ylabel="place cells",
       title="Fields cover the whole track (ends over-represented)")
ax.legend(fontsize=9)

ax = axes[1, 1]
ok = res["rightward"]["rm"].valid_bins
sel = np.where(either)[0]
cc = np.array([np.corrcoef(res["rightward"]["maps"][i, ok],
                           res["leftward"]["maps"][i, ok])[0, 1]
               if np.isfinite(res["rightward"]["maps"][i, ok]).all()
               and np.isfinite(res["leftward"]["maps"][i, ok]).all()
               and res["rightward"]["maps"][i, ok].std() > 0
               and res["leftward"]["maps"][i, ok].std() > 0 else np.nan
               for i in sel])
ax.hist(cc[np.isfinite(cc)], bins=np.linspace(-1, 1, 33), color="tab:purple")
ax.axvline(np.nanmedian(cc), color="k", ls="--",
           label=f"median r = {np.nanmedian(cc):.2f}")
ax.set(xlabel="correlation between rightward and leftward rate maps",
       ylabel="place cells", title="Fields are directional on a linear track")
ax.legend(fontsize=9)

fig.suptitle("Place field properties, sub-Achilles ses-10252013", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("figures/fig06_field_properties.png", dpi=150)
plt.close(fig)
print("saved figures/fig06_field_properties.png")
print(f"\ndirectional map correlation: median r = {np.nanmedian(cc):.3f}")
