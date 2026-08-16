import numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import place_lib as pl, analysis_lib as al

nwbfile = pl.open_nwb("Achilles_10252013")
beh = pl.load_behavior(nwbfile); units = pl.load_units(nwbfile)
pyr = units[np.flatnonzero(units.cell_type == "excitatory")]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# 1. derived linear position vs raw camera x, colour = heading
xr = beh["raw_xy"]["x"].interpolate(beh["position"])
h = beh["heading"].d
for v, c, lab in [(1, "tab:blue", "rightward"), (-1, "tab:red", "leftward")]:
    m = h == v
    ax[0].plot(xr.d[m], beh["position"].d[m], ".", ms=1, color=c, label=lab, alpha=.3)
ax[0].set_xlabel("raw camera x (m)"); ax[0].set_ylabel("derived linear position (m)"); ax[0].legend()

# 2. occupancy per bin per direction
for d, c in [("right", "tab:blue"), ("left", "tab:red")]:
    ep = al.running_epochs(beh, d)
    tc = al.rate_maps(pyr, beh, ep)
    occ = np.asarray(tc.attrs["occupancy"]) * beh["period"]
    ax[1].plot(tc.coords["position"].values, occ, color=c, label=d)
ax[1].set_xlabel("position (m)"); ax[1].set_ylabel("occupancy (s)"); ax[1].legend()

# 3. speed vs position
sp = beh["speed"]
ep = al.running_epochs(beh, "right")
ax[2].plot(beh["position"].restrict(ep).d, sp.restrict(ep).d, ".", ms=1, alpha=.2)
ax[2].set_xlabel("position (m)"); ax[2].set_ylabel("speed (m/s)")
plt.tight_layout(); plt.savefig("figures/_check_linearization.png", dpi=110)

# spike counts during running
ep = al.running_epochs(beh, "right")
n = np.array([len(pyr[k].restrict(ep)) for k in pyr.keys()])
print("spike counts during rightward running: median", np.median(n), "min", n.min(),
      "frac<50", np.mean(n < 50))
