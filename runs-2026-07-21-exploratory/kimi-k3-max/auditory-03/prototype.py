"""Prototype: auditory frequency tuning in one session of DANDI 000986."""
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import stats

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/cac/52e/cac52ee7-20d9-4f7d-a234-a04c22a94083"
FIG = "figures"

disk_cache = remfile.DiskCache('/tmp/remfile_cache_auditory03')
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
trials = nwb["trials"]
freqs = np.array(sorted(np.unique(trials["stim_frequency"])))
print("frequencies:", freqs)
print("n units:", len(units), " n trials:", len(trials))

# Mean firing rate per unit over the full session
t_end = float(units.time_support.end[0])
t_start = float(units.time_support.start[0])
duration = t_end - t_start
rates = np.array([len(units[i].t) / duration for i in range(len(units))])
print(f"session duration {duration:.1f} s; rate range {rates.min():.3f}–{rates.max():.1f} Hz")
keep = np.where(rates >= 0.5)[0]
print(f"units with rate >= 0.5 Hz: {len(keep)}")

# ---- Figure 1: raw data snapshot ----
# Raster of 15 units over 20 s + tone onsets colored by frequency
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 1]))
t0 = trials["start"][100]
win = nap.IntervalSet(start=t0, end=t0 + 20)
ax = axes[0]
shown = keep[:15]
for k, u in enumerate(shown):
    sp = units[int(u)].restrict(win)
    ax.plot(sp.t, np.full(len(sp), k), "|", ms=6, color="k")
ax.set_yticks(range(len(shown)))
ax.set_yticklabels([str(u) for u in shown], fontsize=7)
ax.set_ylabel("unit index")
ax.set_title("Raw spike raster (20 s window) with tone presentations")

ax = axes[1]
cmap = plt.cm.viridis
fcols = {f: cmap(i / (len(freqs) - 1)) for i, f in enumerate(freqs)}
tr = trials.intersect(win)
for j in range(len(tr)):
    f = float(tr["stim_frequency"][j])
    ax.axvline(float(tr["start"][j]), color=fcols[f], lw=1.5)
for f in freqs:
    ax.axvline(np.nan, color=fcols[f], label=f"{int(f/1000)} kHz" if f >= 1000 else f"{int(f)} Hz")
ax.legend(ncol=5, fontsize=8, loc="upper right", frameon=False)
ax.set_yticks([])
ax.set_xlabel("time (s)")
ax.set_ylabel("tones")
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_raw_raster.png", dpi=150)
plt.close()
print("saved fig1")

# ---- Perievent responses ----
tone_onsets = nap.Ts(t=np.array(trials["start"]))
trial_freq = np.array(trials["stim_frequency"])

RESP = (0.005, 0.055)   # response window relative to tone onset
BASE = (-0.050, 0.0)    # baseline window

def evoked_rates(unit_ts):
    """Mean rate in response and baseline windows for each trial."""
    pe = nap.compute_perievent(unit_ts, tone_onsets, window=(BASE[0], RESP[1]))
    # pe is a TsGroup-like: keys are event indices
    n_trials = len(tone_onsets)
    resp = np.zeros(n_trials)
    base = np.zeros(n_trials)
    for j in range(n_trials):
        t = pe[j].t
        resp[j] = np.sum((t >= RESP[0]) & (t < RESP[1])) / (RESP[1] - RESP[0])
        base[j] = np.sum((t >= BASE[0]) & (t < BASE[1])) / (BASE[1] - BASE[0])
    return resp, base

results = []
for u in keep:
    resp, base = evoked_rates(units[int(u)])
    net = resp - base
    groups = [net[trial_freq == f] for f in freqs]
    H, p = stats.kruskal(*groups)
    tuning = np.array([g.mean() for g in groups])
    results.append(dict(unit=int(u), p=p, H=H, tuning=tuning,
                        bf=freqs[np.argmax(tuning)], rate=rates[u]))

sig = [r for r in results if r["p"] < 0.01]
print(f"significantly tuned (KW p<0.01): {len(sig)}/{len(results)}")

# ---- Figure 2: example unit rasters + PSTHs per frequency ----
# pick 4 tuned units with distinct BFs
sig_sorted = sorted(sig, key=lambda r: r["p"])
examples = []
seen_bf = set()
for r in sig_sorted:
    if r["bf"] not in seen_bf:
        examples.append(r)
        seen_bf.add(r["bf"])
    if len(examples) == 4:
        break
print("example units:", [(r["unit"], r["bf"], f'{r["p"]:.1e}') for r in examples])

fig, axes = plt.subplots(4, 2, figsize=(11, 10))
bin_edges = np.arange(-0.05, 0.15, 0.002)
for row, r in enumerate(examples):
    u = r["unit"]
    pe = nap.compute_perievent(units[u], tone_onsets, window=(-0.05, 0.15))
    ax_r, ax_p = axes[row]
    offset = 0
    for f in freqs:
        idx = np.where(trial_freq == f)[0]
        # raster
        for j in idx:
            t = pe[j].t
            ax_r.plot(t, np.full(len(t), offset), "|", ms=2, color=fcols[f])
            offset += 1
        # PSTH
        all_t = np.concatenate([pe[j].t for j in idx]) if len(idx) else np.array([])
        counts, _ = np.histogram(all_t, bins=bin_edges)
        ax_p.plot(bin_edges[:-1] * 1000, counts / (len(idx) * 0.002),
                  color=fcols[f], lw=1.2,
                  label=f"{int(f/1000)} kHz" if f >= 1000 else f"{int(f)} Hz")
    ax_r.axvline(0, color="k", lw=0.8, ls="--")
    ax_p.axvline(0, color="k", lw=0.8, ls="--")
    ax_r.set_ylabel(f"unit {u}\ntrials", fontsize=8)
    ax_p.set_ylabel("rate (Hz)", fontsize=8)
    if row == 0:
        ax_r.set_title("Peri-tone raster (sorted by frequency)")
        ax_p.set_title("PSTH per frequency")
        ax_p.legend(fontsize=7, frameon=False, ncol=2)
    if row == 3:
        ax_r.set_xlabel("time from tone onset (s)")
        ax_p.set_xlabel("time from tone onset (ms)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_example_psth.png", dpi=150)
plt.close()
print("saved fig2")

# ---- Figure 3: tuning curves ----
n = len(results)
ncol = 6
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.2 * nrow), sharex=True)
for k, r in enumerate(results):
    ax = axes.flat[k]
    ax.plot(np.log2(freqs / 1000), r["tuning"], "o-", ms=3, lw=1,
            color="crimson" if r["p"] < 0.01 else "gray")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f'u{r["unit"]} p={r["p"]:.0e}', fontsize=6)
    if k % ncol == 0:
        ax.set_ylabel("evoked − base (Hz)", fontsize=7)
    if k >= (nrow - 1) * ncol:
        ax.set_xticks(np.log2(freqs / 1000))
        ax.set_xticklabels([f"{int(f/1000)}" for f in freqs], fontsize=7)
        ax.set_xlabel("freq (kHz)", fontsize=7)
for k in range(n, nrow * ncol):
    axes.flat[k].axis("off")
plt.suptitle("Frequency tuning curves, all units (red = significantly tuned)", y=1.0)
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_tuning_curves.png", dpi=150)
plt.close()
print("saved fig3")

# ---- Figure 4: population summary ----
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
ax = axes[0]
ps = np.clip(np.array([r["p"] for r in results]), 1e-50, 1)
ax.hist(np.log10(ps), bins=20, color="steelblue")
ax.axvline(np.log10(0.01), color="r", ls="--", label="p = 0.01")
ax.set_xlabel("log10(Kruskal–Wallis p)")
ax.set_ylabel("# units")
ax.legend(frameon=False)
ax.set_title("Tuning significance")

ax = axes[1]
bfs = np.array([r["bf"] for r in sig])
counts = [np.sum(bfs == f) for f in freqs]
ax.bar(range(len(freqs)), counts, color=[fcols[f] for f in freqs])
ax.set_xticks(range(len(freqs)))
ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("# tuned units")
ax.set_title("Best-frequency distribution (tuned units)")

ax = axes[2]
maxresp = np.array([r["tuning"].max() for r in results])
ax.scatter([r["rate"] for r in results], maxresp,
           c=["crimson" if r["p"] < 0.01 else "gray" for r in results], s=12)
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("max evoked − base (Hz)")
ax.set_title("Response magnitude vs rate")
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_population.png", dpi=150)
plt.close()
print("saved fig4")
print("DONE")
