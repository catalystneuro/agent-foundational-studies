import numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import place_lib as pl, analysis_lib as al

nwbfile = pl.open_nwb("Achilles_10252013")
beh = pl.load_behavior(nwbfile); units = pl.load_units(nwbfile)
pyr = units[np.flatnonzero(units.cell_type == "excitatory")]
ep = al.running_epochs(beh, "right")
tc = al.rate_maps(pyr, beh, ep)
si = al.spatial_information(tc)
order = np.argsort(si)[::-1][:12]
fig, axes = plt.subplots(3, 4, figsize=(13, 7), sharex=True)
x = tc.coords["position"].values
for ax, i in zip(axes.ravel(), order):
    ax.plot(x, tc.values[i], "k")
    ax.set_title(f"unit {list(pyr.keys())[i]}  SI={si[i]:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig("figures/_proto_tc.png", dpi=110)

# sorted heatmap
lam = np.asarray(tc.values); pk = lam.argmax(1)
norm = lam / np.maximum(lam.max(1, keepdims=True), 1e-9)
srt = np.argsort(pk)
fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(norm[srt], aspect="auto", extent=[0, 1.6, len(srt), 0], cmap="viridis")
ax.set_xlabel("position (m)"); ax.set_ylabel("unit (sorted by peak)")
plt.tight_layout(); plt.savefig("figures/_proto_heat.png", dpi=110)
print("done")
