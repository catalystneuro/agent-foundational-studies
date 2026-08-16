"""Prototype: HD signal, states, tuning curves, HD-cell selection on Mouse17-130128."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm
from prototype_load import load_session

rng = np.random.default_rng(42)

nwb = load_session("Mouse17-130128")

# --- states ---
states = nwb["states"]
print(states)
for lab in np.unique(states["label"]):
    ep = states[states["label"] == lab]
    print(lab, "total duration (s):", ep.tot_length())

# --- HD from LEDs ---
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
red_v = np.asarray(red.values)
blue_v = np.asarray(blue.values)
t = red.t
print("red shape", red_v.shape, "rate ~", 1 / np.median(np.diff(t[:1000])))

valid = (red_v[:, 0] > 0) & (red_v[:, 1] > 0) & (blue_v[:, 0] > 0) & (blue_v[:, 1] > 0)
hd = np.arctan2(red_v[:, 1] - blue_v[:, 1], red_v[:, 0] - blue_v[:, 0]) % (2 * np.pi)
hd[~valid] = np.nan
hd_tsd = nap.Tsd(t=t, d=hd)
print("fraction valid:", valid.mean())

# --- units: rebuild with sorted spike times (one unit unsorted in this session) ---
units = nwb["units"]
print(units)
spikes = {k: nap.Ts(np.sort(units[k].t)) for k in units.keys()}
group = nap.TsGroup(spikes)
print("n units:", len(group))

wake = states[states["label"] == "Awake"]
rem = states[states["label"] == "REM"]
nrem = states[states["label"] == "Non-REM"]

# --- tuning curves during wake ---
nbins = 60
tc_da = nap.compute_tuning_curves(
    group, hd_tsd, bins=nbins, range=(0, 2 * np.pi), epochs=wake,
    return_counts=True,
)
print("tuning DataArray:", tc_da.dims, tc_da.shape, "attrs:", list(tc_da.attrs))
occupancy = tc_da.attrs["occupancy"] / tc_da.attrs["fs"]
counts = np.asarray(tc_da)  # (unit, bin)
with np.errstate(invalid="ignore", divide="ignore"):
    tc_hz = counts / occupancy[None, :]
tc_hz = np.where(occupancy[None, :] > 0, tc_hz, np.nan)
# bin centers: coordinate of the non-unit dim
bin_dim = [d for d in tc_da.dims if d != "unit"][0]
centers = np.asarray(tc_da.coords[bin_dim])
print("tc shape:", tc_hz.shape, "centers:", centers[:3])

# --- HD cell selection: MVL of spike angles + random-time null ---
def resultant_length(angles):
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan
    return np.abs(np.exp(1j * angles).mean())

# valid wake HD samples
wake_mask = wake.in_interval(hd_tsd) >= 0
valid_wake = valid & wake_mask & ~np.isnan(hd)
hd_valid_t = t[valid_wake]
hd_valid_a = hd[valid_wake]
print("valid wake HD samples:", len(hd_valid_t))

mvl = np.full(len(group), np.nan)
pval = np.full(len(group), np.nan)
nshuf = 500
for i, k in enumerate(tqdm(list(group.keys()), desc="HD stats")):
    spk = group[k].restrict(wake)
    ang = spk.value_from(hd_tsd)  # pynapple 0.11: ts.value_from(tsd)
    ang = ang[~np.isnan(ang)]
    n = len(ang)
    if n < 100:
        continue
    mvl[i] = resultant_length(ang)
    # random-time null: draw n times uniformly from valid wake HD samples
    n_cap = min(n, 100_000)
    null = np.empty(nshuf)
    idx_base = rng.integers(0, len(hd_valid_a), size=n_cap)
    for s in range(nshuf):
        idx = rng.integers(0, len(hd_valid_a), size=n_cap)
        null[s] = np.abs(np.exp(1j * hd_valid_a[idx]).mean())
    pval[i] = (np.sum(null >= mvl[i]) + 1) / (nshuf + 1)

is_hd = (mvl > 0.3) & (pval < 0.05)
print("HD cells:", np.where(is_hd)[0], "of", len(group))
np.savez("proto_hd_stats.npz", mvl=mvl, pval=pval, is_hd=is_hd,
         tc_hz=tc_hz, centers=np.asarray(centers))

# --- figure: tuning curves polar ---
hd_idx = np.where(is_hd)[0]
pref = centers[np.nanargmax(tc_hz[hd_idx], axis=1)]
order = np.argsort(pref)
fig, axes = plt.subplots(2, (len(hd_idx) + 1) // 2, subplot_kw=dict(projection="polar"),
                         figsize=(2.2 * ((len(hd_idx) + 1) // 2), 4.6))
for ax, u in zip(axes.ravel(), hd_idx[order]):
    ax.plot(np.append(centers, centers[0]), np.append(tc_hz[u], tc_hz[u][0]))
    ax.set_title(f"u{u}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("Wake HD tuning curves (Mouse17-130128)")
plt.tight_layout()
plt.savefig("proto_tuning.png", dpi=150)
print("saved proto_tuning.png")
