# Population figures for gerbil AN STRFs
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from harvest_population import analyze_fiber

m = pd.read_csv("population_metrics.csv")
print(m.shape)

# ---------- 1. Example STRF gallery: pick 6 fibers spanning CF range ----------
m_sorted = m.sort_values("cf")
# good examples: high peak rate, resolved across CF range
good = m[(m["peak_rate"] > 80)].sort_values("cf")
targets_cf = [800, 2000, 4000, 7000, 10000, 14000]
picks = []
for tc_ in targets_cf:
    i = (good["cf"] - tc_).abs().idxmin()
    if i not in picks:
        picks.append(i)
ex = good.loc[picks]
print(ex[["path", "cf", "latency_ms", "peak_rate", "level"]].to_string())

examples = []
for _, row in ex.iterrows():
    res = analyze_fiber(row["asset_id"], row["path"])
    examples.append(res)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, res in zip(axes.flat, examples):
    net_s = res["net_map"]
    freqs = res["freqs"]
    centers = res["centers"]
    vmax = np.percentile(net_s, 99.5)
    im = ax.imshow(net_s, aspect="auto", origin="lower",
                   extent=[centers[0] * 1e3, centers[-1] * 1e3,
                           freqs[0] / 1e3, freqs[-1] / 1e3],
                   cmap="viridis", vmin=0, vmax=vmax)
    ax.axvline(0, color="w", lw=0.8)
    ax.axvline(res["dur"] * 1e3, color="w", lw=0.8, ls="--")
    ax.axhline(res["cf"] / 1e3, color="r", lw=0.8, ls=":")
    ax.set_title(f"CF={res['cf']:.0f} Hz, lat={res['latency_ms']:.1f} ms, "
                 f"{res['level']:.0f} dB SPL", fontsize=10)
    ax.set_xlabel("time rel tone onset (ms)")
    ax.set_ylabel("frequency (kHz)")
    plt.colorbar(im, ax=ax, label="net sp/s")
fig.suptitle("Example auditory nerve fiber STRFs (tone-pip maps) across the tonotopic axis")
fig.tight_layout()
fig.savefig("figures/an_strf_gallery.png", dpi=150)
print("saved figures/an_strf_gallery.png")

# ---------- 2. Population summary figure ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
ax.scatter(m["table_bf"], m["cf"], c="k", s=15)
lims = [300, 20000]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("BF from published analysis table (Hz)")
ax.set_ylabel("CF from STRF (Hz)")
r = np.corrcoef(np.log10(m["table_bf"]), np.log10(m["cf"]))[0, 1]
ax.set_title(f"STRF CF vs published BF (r={r:.3f}, n={len(m)})")

ax = axes[0, 1]
ok = m["latency_ms"].between(1, 25)
ax.scatter(m.loc[ok, "cf"], m.loc[ok, "latency_ms"], c="k", s=15)
ax.set_xscale("log")
logcf = np.log10(m.loc[ok, "cf"])
slope, intercept, rval, pval, _ = stats.linregress(logcf, m.loc[ok, "latency_ms"])
xx = np.linspace(np.log10(400), np.log10(16000), 50)
ax.plot(10 ** xx, intercept + slope * xx, "r--", lw=1,
        label=f"slope={slope:.2f} ms/decade, p={pval:.1e}")
ax.set_xlabel("CF (Hz)")
ax.set_ylabel("STRF onset latency (ms)")
ax.set_title(f"Latency vs CF (n={ok.sum()})")
ax.legend()

ax = axes[1, 0]
okb = m["bw_oct"].notna()
ax.scatter(m.loc[okb, "cf"], m.loc[okb, "bw_oct"], c="k", s=15)
ax.set_xscale("log")
ax.set_xlabel("CF (Hz)")
ax.set_ylabel("bandwidth (octaves)")
ax.set_title(f"STRF bandwidth at half-max (n={okb.sum()})")

ax = axes[1, 1]
ax.scatter(m["spont_rate"], m["peak_rate"], c="k", s=15)
ax.set_xlabel("spontaneous rate (sp/s)")
ax.set_ylabel("peak evoked net rate (sp/s)")
ax.set_title("Evoked vs spontaneous rate")

fig.suptitle(f"Gerbil auditory nerve STRF population (n={len(m)} fibers, "
             f"{m['subject'].nunique()} animals)")
fig.tight_layout()
fig.savefig("figures/an_population_summary.png", dpi=150)
print("saved figures/an_population_summary.png")

# ---------- 3. CF-aligned average STRF ----------
# recompute maps for all fibers would be expensive; use examples only? No:
# use stored maps? maps were lost. Skip full average; instead do a
# composite from the metrics: tuning width in octaves vs CF done above.
print("done")
