# 05: peri-ripple firing rates (exc vs inh) + showcase replay event figure
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from common import open_nwb, load_lfp_channel, LFP_FS

nwb, h5 = open_nwb()
units = nwb["units"]
rip = np.load("ripples_post.npz")
pf = np.load("place_fields.npz")
res = np.load("replay_results.npz")
ev_peak = rip["peak_t"]
cell_type = units.get_info("cell_type")
exc_keys = cell_type[cell_type == "excitatory"].index.to_numpy()
inh_keys = cell_type[cell_type == "inhibitory"].index.to_numpy()

# ---------- peri-ripple PSTH ----------
WIN = 0.5
PBIN = 0.010
pedges = np.arange(-WIN, WIN + 1e-9, PBIN)
pc = 0.5 * (pedges[:-1] + pedges[1:])

def peri_rates(keys):
    rates = np.zeros((len(keys), len(pedges) - 1))
    for i, k in enumerate(keys):
        sp = units[k].t
        # spike count in each peri-peak bin, summed over events
        # align: for each event, histogram sp - peak
        counts = np.zeros(len(pedges) - 1)
        # vectorized: collect relative times in chunks
        i0_all = np.searchsorted(sp, ev_peak - WIN)
        i1_all = np.searchsorted(sp, ev_peak + WIN)
        rel = []
        for i0, i1, pk in zip(i0_all, i1_all, ev_peak):
            if i1 > i0:
                rel.append(sp[i0:i1] - pk)
        if rel:
            rel = np.concatenate(rel)
            counts, _ = np.histogram(rel, bins=pedges)
        rates[i] = counts / (len(ev_peak) * PBIN)  # Hz per unit
    return rates

r_exc = peri_rates(exc_keys)
r_inh = peri_rates(inh_keys)

# z-score per unit using baseline [-0.5, -0.3] s
base = (pc >= -0.5) & (pc <= -0.3)
def zscore(r):
    mu = r[:, base].mean(axis=1, keepdims=True)
    sd = r[:, base].std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (r - mu) / sd
z_exc, z_inh = zscore(r_exc), zscore(r_inh)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(pc * 1000, z_exc.mean(axis=0), color="seagreen",
        label=f"excitatory (n={len(exc_keys)})")
ax.plot(pc * 1000, z_inh.mean(axis=0), color="crimson",
        label=f"inhibitory (n={len(inh_keys)})")
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.set_xlabel("time from ripple peak (ms)")
ax.set_ylabel("firing rate (z-scored)")
ax.set_title("Peri-ripple firing: CA1 pyramidal cells and interneurons (POST Non-REM)")
ax.legend()
plt.tight_layout()
plt.savefig("fig_peri_ripple_psth.png", dpi=150)
print("saved fig_peri_ripple_psth.png")
print(f"peak z: exc {z_exc.mean(0).max():.2f} at {pc[z_exc.mean(0).argmax()]*1000:.0f} ms, "
      f"inh {z_inh.mean(0).max():.2f} at {pc[z_inh.mean(0).argmax()]*1000:.0f} ms")

# ---------- showcase event ----------
# pick a long, significant, high-|wc| event
sig = res["p_cellid"] < 0.05
cand = np.where(sig & (res["n_bins"] >= 6))[0]
best = cand[np.argsort(np.abs(res["wc_real"][cand]))[::-1]]
# prefer a forward event with moderate duration for display
fwd_cand = [e for e in best if res["wc_real"][e] > 0][:20]
show = fwd_cand[3] if len(fwd_cand) > 3 else best[0]
ei = res["qualified_idx"][show]
a, b = rip["start"][ei], rip["end"][ei]
print(f"showcase event {ei}: {a:.3f}-{b:.3f} s, wc={res['wc_real'][show]:.2f}, "
      f"p={res['p_cellid'][show]:.3f}")

pad = 0.08
t_lfp, x_lfp = load_lfp_channel(h5, int(rip["ch"]), a - pad, b + pad)
sos = signal.butter(4, [100, 250], btype="band", fs=LFP_FS, output="sos")
x_filt = signal.sosfiltfilt(sos, x_lfp)

place_keys = pf["exc_keys"][pf["is_place"]]
template = np.nan_to_num(pf["rate_sm"][pf["is_place"]])
peak_pos = np.array([pf["centers"][np.argmax(r)] if r.max() > 0 else np.nan
                     for r in template])
cell_order = np.argsort(peak_pos)

BIN = res["bin"]
centers = pf["centers"]
edges = np.arange(a, b + 1e-9, BIN)
n_bins = len(edges) - 1
counts = np.zeros((n_bins, len(place_keys)))
raster = []
for c, k in enumerate(place_keys):
    sp = units[k].t
    i0, i1 = np.searchsorted(sp, a - pad), np.searchsorted(sp, b + pad)
    raster.append(sp[i0:i1])
    j0, j1 = np.searchsorted(sp, a), np.searchsorted(sp, b)
    if j1 > j0:
        counts[:, c] = np.histogram(sp[j0:j1], bins=edges)[0]

tmpl = np.clip(template, 0.01, None)
lp = counts @ np.log(tmpl) - BIN * tmpl.sum(axis=0)[None, :]
lp -= lp.max(axis=1, keepdims=True)
post = np.exp(lp)
post /= post.sum(axis=1, keepdims=True)

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.4, 1.4], hspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax1.plot((t_lfp - a) * 1000, x_lfp, color="gray", lw=0.6, label="raw LFP")
ax1.plot((t_lfp - a) * 1000, x_filt * 4, color="navy", lw=0.8,
         label="ripple band x4")
ax1.axvspan(0, (b - a) * 1000, color="red", alpha=0.08)
ax1.set_ylabel("uV")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title(f"Ripple event at {a:.2f} s (POST Non-REM): wc={res['wc_real'][show]:.2f}, "
              f"p={res['p_cellid'][show]:.3f}")
ax1.set_xlim(-pad * 1000, (b - a + pad) * 1000)

ax2 = fig.add_subplot(gs[1], sharex=ax1)
for row, c in enumerate(cell_order):
    spk = raster[c]
    if len(spk):
        ax2.scatter((spk - a) * 1000, np.full(len(spk), row), s=4, color="black")
ax2.axvspan(0, (b - a) * 1000, color="red", alpha=0.08)
ax2.set_ylabel("place cell (sorted by field position)")
ax2.set_ylim(-1, len(place_keys))

ax3 = fig.add_subplot(gs[2])
ax3.imshow(post.T, aspect="auto", origin="lower",
           extent=[0, n_bins * BIN * 1000, 0, 1.6], cmap="hot")
wm = (post * centers[None, :]).sum(axis=1)
ax3.plot((np.arange(n_bins) + 0.5) * BIN * 1000, wm, color="cyan", lw=1.5)
ax3.set_xlabel("time from event start (ms)")
ax3.set_ylabel("track position (m)")
ax3.set_title("Bayesian decoded position")
plt.savefig("fig_replay_showcase.png", dpi=150)
print("saved fig_replay_showcase.png")
