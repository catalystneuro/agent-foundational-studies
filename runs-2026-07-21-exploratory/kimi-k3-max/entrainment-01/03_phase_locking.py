"""03: Spike-theta-phase locking for all CA1 units during maze running."""
import remfile, h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import stats
from tqdm import tqdm

with open("s3_url.txt") as f:
    BARE_URL = f.read().strip().split("?")[0]

d = np.load("results/theta_phase_maze.npz")
t_lfp, phase, fs = d["t_lfp"], d["phase"], float(d["fs"])
t_pos, speed = d["t_pos"], d["speed"]
ref_ch = int(d["ref_ch"])

disk_cache = remfile.DiskCache('/tmp/remfile_cache_theta')
h5py_file = h5py.File(remfile.File(BARE_URL, disk_cache=disk_cache), "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]

# --- running intervals: smoothed speed > 10 cm/s, min 0.5 s, merge gaps < 0.25 s ---
run = (speed > 0.10) & ~np.isnan(speed)
# find runs
edges = np.diff(run.astype(int))
starts = t_pos[:-1][edges == 1]
ends = t_pos[:-1][edges == -1]
if run[0]:
    starts = np.r_[t_pos[0], starts]
if run[-1]:
    ends = np.r_[ends, t_pos[-1]]

def merge_intervals(starts, ends, min_gap):
    if len(starts) == 0:
        return starts, ends
    ms, me = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - me[-1] < min_gap:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
    return np.array(ms), np.array(me)

starts, ends = merge_intervals(starts, ends, 0.25)
dur = ends - starts
starts, ends = starts[dur >= 0.5], ends[dur >= 0.5]
run_ep = nap.IntervalSet(start=starts, end=ends)
print(f"run intervals: {len(starts)} bouts, total {dur[dur>=0.5].sum():.0f} s")

# --- phase interpolation helpers (interpolate cos/sin to avoid wrap artifacts) ---
cos_p, sin_p = np.cos(phase), np.sin(phase)
def phase_at(times):
    return np.arctan2(np.interp(times, t_lfp, sin_p), np.interp(times, t_lfp, cos_p))

def rayleigh_p(n, R):
    """Fisher (1993) approximation for Rayleigh test p-value."""
    z = n * R**2
    p = np.exp(-z)
    if n < 50:
        p *= (1 + (2 * z - z**2) / (4 * n)
              - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2))
    return min(p, 1.0)

# --- per-unit phase locking ---
maze_ep = nap.IntervalSet(start=[18079.5], end=[20147.0])
analysis_ep = maze_ep.intersect(run_ep)

records = []
keys = list(units.keys())
cell_types = np.asarray(units.get_info("cell_type"))
locations = np.asarray(units.get_info("location"))
rng = np.random.default_rng(42)
min_shift = int(fs)  # at least 1 s decorrelation shift
for i, k in enumerate(tqdm(keys, desc="units")):
    spk = units[k].restrict(analysis_ep)
    t_spk = spk.index.values
    n = len(t_spk)
    if n < 100:
        continue
    ph = phase_at(t_spk)
    R = np.abs(np.mean(np.exp(1j * ph)))
    pref = np.angle(np.mean(np.exp(1j * ph)))
    p = rayleigh_p(n, R)
    rate = n / analysis_ep.tot_length("s")
    # time-shift control: roll the phase signal by a random offset, recompute MRL
    shift = rng.integers(min_shift, len(phase) - min_shift)
    sin_s = np.roll(sin_p, shift)
    cos_s = np.roll(cos_p, shift)
    ph_s = np.arctan2(np.interp(t_spk, t_lfp, sin_s), np.interp(t_spk, t_lfp, cos_s))
    R_s = np.abs(np.mean(np.exp(1j * ph_s)))
    records.append(dict(unit=k, cell_type=cell_types[i], location=locations[i],
                        n_spikes=n, rate=rate, mrl=R, pref_phase=pref, rayleigh_p=p,
                        mrl_shuffle=R_s))

import pandas as pd
df = pd.DataFrame(records)
df.to_csv("results/phase_locking_per_unit.csv", index=False)
sig = df["rayleigh_p"] < 0.01
print(f"\nunits analyzed: {len(df)}")
print(f"significantly phase-locked (Rayleigh p<0.01): {sig.sum()} ({100*sig.mean():.1f}%)")
for ct in ["excitatory", "inhibitory"]:
    m = df["cell_type"] == ct
    print(f"  {ct}: {sig[m].sum()}/{m.sum()} ({100*sig[m].mean():.1f}%), "
          f"median MRL {df.loc[m,'mrl'].median():.3f}")
mw = stats.mannwhitneyu(df.loc[df.cell_type=="excitatory","mrl"],
                        df.loc[df.cell_type=="inhibitory","mrl"])
print(f"MRL exc vs inh Mann-Whitney p = {mw.pvalue:.2e}")
mw2 = stats.mannwhitneyu(df["mrl"], df["mrl_shuffle"])
print(f"median MRL observed {df['mrl'].median():.3f} vs time-shift control "
      f"{df['mrl_shuffle'].median():.3f} (Mann-Whitney p = {mw2.pvalue:.2e})")

# --- Figure 3: example cells, polar phase histograms ---
nbins = 18
bins = np.linspace(-np.pi, np.pi, nbins + 1)
centers = (bins[:-1] + bins[1:]) / 2

sig_exc = df[(df.cell_type == "excitatory") & sig].sort_values("mrl", ascending=False)
sig_inh = df[(df.cell_type == "inhibitory") & sig].sort_values("mrl", ascending=False)
examples = list(sig_exc["unit"].iloc[:4]) + list(sig_inh["unit"].iloc[:2])
labels = [f"unit {u} (exc)" for u in sig_exc["unit"].iloc[:4]] + \
         [f"unit {u} (inh)" for u in sig_inh["unit"].iloc[:2]]

fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), subplot_kw={"projection": "polar"})
for ax, u, lab in zip(axes.flat, examples, labels):
    spk = units[u].restrict(analysis_ep)
    ph = phase_at(spk.index.values)
    counts, _ = np.histogram(ph, bins=bins)
    counts = counts / counts.sum()
    ax.bar(centers, counts, width=2 * np.pi / nbins, color="tab:blue",
           edgecolor="k", lw=0.5, alpha=0.8)
    row = df[df.unit == u].iloc[0]
    ax.set_title(lab, fontsize=10, pad=10)
    ax.text(0.5, -0.13, f"MRL={row.mrl:.2f}, p={row.rayleigh_p:.0e}, n={row.n_spikes}",
            transform=ax.transAxes, ha="center", fontsize=8.5)
    ax.set_theta_zero_location("E")
    ax.set_yticks([])
    ax.set_xticklabels([])
fig.suptitle("Spike phase relative to CA1 theta (0° = LFP peak at right, 180° = trough at left)",
             y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.94], h_pad=2.5)
fig.savefig("figures/fig4_example_phase_histograms.png", dpi=150)
print("saved figures/fig4_example_phase_histograms.png")

# --- Figure 4: population summary ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

ax = axes[0]
be = np.linspace(0, 1, 41)
ax.hist(df.loc[df.cell_type=="excitatory","mrl"], bins=be, alpha=0.7,
        color="tab:blue", label=f"excitatory (n={sum(df.cell_type=='excitatory')})")
ax.hist(df.loc[df.cell_type=="inhibitory","mrl"], bins=be, alpha=0.7,
        color="tab:red", label=f"inhibitory (n={sum(df.cell_type=='inhibitory')})")
ax.hist(df["mrl_shuffle"], bins=be, histtype="step", color="k", lw=1.2,
        label="time-shift control")
ax.set_xlabel("Mean resultant length (phase locking)")
ax.set_ylabel("Units")
ax.set_title(f"Phase-locking strength\nMann-Whitney p={mw.pvalue:.1e}")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
fracs = [100 * sig[df.cell_type==ct].mean() for ct in ["excitatory", "inhibitory"]]
bars = ax.bar(["excitatory", "inhibitory"], fracs, color=["tab:blue", "tab:red"], alpha=0.8)
for b, ct in zip(bars, ["excitatory", "inhibitory"]):
    m = df.cell_type == ct
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            f"{sig[m].sum()}/{m.sum()}", ha="center", fontsize=9)
ax.set_ylabel("% significantly locked")
ax.set_title("Fraction phase-locked\n(Rayleigh p<0.01, >=100 spikes)")
ax.set_ylim(0, 105)

ax = axes[2]
for ct, c in [("excitatory", "tab:blue"), ("inhibitory", "tab:red")]:
    m = (df.cell_type == ct) & sig
    ax.hist(np.degrees(df.loc[m, "pref_phase"]), bins=np.linspace(-180, 180, 19),
            alpha=0.7, color=c, label=f"{ct} (n={m.sum()})")
ax.axvline(180, color="k", ls=":", lw=0.8)
ax.axvline(-180, color="k", ls=":", lw=0.8)
ax.set_xlabel("Preferred theta phase (deg; 0=peak, ±180=trough)")
ax.set_ylabel("Units")
ax.set_title("Preferred phase of locked units")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig5_population_summary.png", dpi=150)
print("saved figures/fig5_population_summary.png")

# --- Figure 5: pooled spike-phase histogram of locked pyramidal cells ---
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), subplot_kw={"projection": "polar"})
all_ph = []
for u in df.loc[(df.cell_type=="excitatory") & sig, "unit"]:
    spk = units[u].restrict(analysis_ep)
    all_ph.append(phase_at(spk.index.values))
all_ph = np.concatenate(all_ph)
counts, _ = np.histogram(all_ph, bins=bins)
counts = counts / counts.sum()
ax.bar(centers, counts, width=2*np.pi/nbins, color="tab:blue", edgecolor="k", lw=0.5, alpha=0.8)
# duplicate for continuity
ax.set_theta_zero_location("E")
ax.set_yticks([])
ax.set_xticklabels([])
R_pool = np.abs(np.mean(np.exp(1j*all_ph)))
pref_pool = np.angle(np.mean(np.exp(1j*all_ph)))
ax.set_title(f"Pooled phases, {int(sig[df.cell_type=='excitatory'].sum())} locked pyramidal cells\n"
             f"{len(all_ph):,} spikes, MRL={R_pool:.3f}, pref={np.degrees(pref_pool):.0f}°\n"
             "(0° = LFP peak at right, 180° = trough at left)",
             fontsize=10, pad=16)
fig.tight_layout()
fig.savefig("figures/fig6_pooled_phase.png", dpi=150)
print("saved figures/fig6_pooled_phase.png")
print(f"pooled excitatory: {len(all_ph)} spikes, MRL={R_pool:.4f}, pref={np.degrees(pref_pool):.1f} deg")

# --- Figure 6: spikes of a well-locked unit riding the theta trace ---
# pick a high-rate locked unit and its densest 1.5 s window so spikes are visible
cand = df[(df.cell_type == "excitatory") & sig & (df.n_spikes > 1000)].sort_values(
    "mrl", ascending=False)
best_u = int(cand["unit"].iloc[0])
theta_sig = d["theta"]
spk_all = units[best_u].restrict(analysis_ep).index.values
# densest 1.5 s window via binned spike counts
edges_w = np.arange(spk_all[0], spk_all[-1], 0.25)
c_w, _ = np.histogram(spk_all, bins=np.append(edges_w, edges_w[-1] + 1.5))
# count spikes in sliding 1.5 s windows
idx = np.searchsorted(spk_all, edges_w)
idx2 = np.searchsorted(spk_all, edges_w + 1.5)
counts_w = idx2 - idx
t0 = edges_w[np.argmax(counts_w)]
win = (t_lfp >= t0) & (t_lfp < t0 + 1.5)
spk = units[best_u].get(t0, t0 + 1.5)
spk_t = spk.index.values
spk_ph = phase_at(spk_t)
# place each spike on the filtered trace at its own phase's amplitude
spk_amp = np.interp(spk_t, t_lfp, theta_sig)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_lfp[win], theta_sig[win] * 1e6, color="tab:blue", lw=1.4,
        label=f"theta-filtered LFP (ch {ref_ch})")
ax.plot(spk_t, spk_amp * 1e6, "o", color="tab:red", ms=9, mfc="none", mew=1.8,
        label=f"unit {best_u} spikes (n={len(spk_t)})")
ax.set_ylabel("LFP (µV)")
ax.set_xlabel("Time (s)")
row = df[df.unit == best_u].iloc[0]
ax.set_title(f"Cycle-by-cycle entrainment: unit {best_u}, MRL={row.mrl:.2f}, "
             f"preferred phase {np.degrees(row.pref_phase):.0f}°")
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig7_spikes_on_theta.png", dpi=150)
print("saved figures/fig7_spikes_on_theta.png")
