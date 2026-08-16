# Revised fig4: diverse cortical STRF gallery
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("cortical_strf.npz")
maps, freqs, centers = d["maps"], d["freqs"], d["centers"]
both = d["modulated"] & d["responsive"]
resp_win = (centers >= 0.005) & (centers < 0.060)
tc = maps[:, :, resp_win].mean(axis=2)
lat = d["latency_rw"] if "latency_rw" in d else d["latency"]

idx = np.where(both)[0]
dom_abs = np.argmax(np.abs(tc[idx]), axis=1)

# excitatory/dominant pick per frequency (skip empty pools)
picks = []
for fi in range(len(freqs)):
    pool = idx[dom_abs == fi]
    if len(pool) == 0:
        continue
    strength = np.abs(tc[pool, fi])
    picks.append(pool[np.argmax(strength)])
# add strongest suppressive unit not already picked
minvals = tc[idx].min(axis=1)
for u in idx[np.argsort(minvals)]:
    if u not in picks:
        picks.append(u)
        break
picks = picks[:5]
print("picks:", picks, [f"{freqs[np.argmax(np.abs(tc[u]))]/1000:.0f}kHz" for u in picks])

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, u in zip(axes.flat, picks):
    m = maps[u]
    v = np.abs(m).max()
    im = ax.imshow(m, aspect="auto", origin="lower",
                   extent=[centers[0] * 1e3, centers[-1] * 1e3, 0, len(freqs)],
                   cmap="RdBu_r", vmin=-v, vmax=v)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(25, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f"{f/1000:.0f}" for f in freqs])
    dom_f = freqs[np.argmax(np.abs(tc[u]))] / 1000
    kind = "excitatory" if tc[u].max() > abs(tc[u].min()) else "suppressive"
    ax.set_title(f"unit {u} ({kind}, peak {dom_f:.0f} kHz, "
                 f"latency={lat[u]:.0f} ms)", fontsize=10)
    ax.set_xlabel("time rel tone onset (ms)")
    ax.set_ylabel("frequency (kHz)")
    plt.colorbar(im, ax=ax, label="net sp/s")

# BF-aligned population average over units with excitatory responses
ax = axes.flat[5]
exc = idx[tc[idx].max(axis=1) > 0]
bf_idx = np.argmax(tc[exc], axis=1)
shifted = [np.roll(maps[u], -(b - 2), axis=0) for u, b in zip(exc, bf_idx)]
avg = np.mean(shifted, axis=0)
v = np.abs(avg).max()
im = ax.imshow(avg, aspect="auto", origin="lower",
               extent=[centers[0] * 1e3, centers[-1] * 1e3, 0, len(freqs)],
               cmap="RdBu_r", vmin=-v, vmax=v)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(25, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(freqs)))
ax.set_yticklabels(["-2 oct", "-1 oct", "BF", "+1 oct", "+2 oct"])
ax.set_title(f"population average, BF-aligned (n={len(exc)} excitatory)", fontsize=10)
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency rel BF")
plt.colorbar(im, ax=ax, label="net sp/s")

fig.suptitle("Cortical STRFs (mouse auditory cortex, DANDI:000986, sub-LA8 ses-1)")
fig.tight_layout()
fig.savefig("figures/fig4_redo.png", dpi=150)
print("saved figures/fig4_redo.png")
