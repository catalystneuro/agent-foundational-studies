"""Raw-data + tuning figures for Mouse28-140310 (20 HD cells)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from prototype_load import load_session

FIG = "../figures"
os.makedirs(FIG, exist_ok=True)
SESSION = "Mouse28-140310"

nwb = load_session(SESSION)
states = nwb["states"]
wake = states[states["label"] == "Awake"]

red = np.asarray(nwb["SubjectPosition/RedLED"].values)
blue = np.asarray(nwb["SubjectPosition/BlueLED"].values)
t = nwb["SubjectPosition/RedLED"].t
valid = (red[:, 0] > 0) & (red[:, 1] > 0) & (blue[:, 0] > 0) & (blue[:, 1] > 0)
hd = np.arctan2(red[:, 1] - blue[:, 1], red[:, 0] - blue[:, 0]) % (2 * np.pi)
hd[~valid] = np.nan
hd_tsd = nap.Tsd(t=t, d=hd)

units = nwb["units"]
group = nap.TsGroup({k: nap.Ts(np.sort(units[k].t)) for k in units.keys()})
keys = list(group.keys())
d = np.load(f"cache/{SESSION}.npz")
hd_idx = d["hd_idx"]
pref = d["pref"][hd_idx]
order = np.argsort(pref)
tc_hz = d["tc_hz"]
centers = d["centers"]

# ---------- Fig R1: wake raster sorted by preferred angle + HD trace ----------
# pick a 40 s wake window with large HD coverage (animal turning)
hd_wake = hd_tsd.restrict(wake)
span = 40.0
best, best_range = None, -1
ts = hd_wake.t
vs = np.asarray(hd_wake.values)
i0 = 0
for i in range(0, len(ts) - 1, 200):
    j = np.searchsorted(ts, ts[i] + span)
    if j >= len(ts):
        break
    seg = vs[i:j]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 100:
        continue
    seg_un = np.unwrap(seg)  # true angular span, robust to 0/360 wrap
    cov = np.percentile(seg_un, 95) - np.percentile(seg_un, 5)
    if cov > best_range:
        best_range, best = cov, (ts[i], ts[i] + span)
t0, t1 = best
print("window:", t0, t1, "coverage deg:", np.degrees(best_range))

ep = nap.IntervalSet(t0, t1)
fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 2.2]))
ax = axes[0]
seg = hd_tsd.restrict(ep)
ax.plot(seg.t - t0, np.degrees(seg.values), ".", ms=1.5, color="k")
ax.set_ylabel("Head direction (deg)")
ax.set_yticks([0, 180, 360])
ax.set_title(f"Wake exploration ({SESSION})", fontsize=10)

ax = axes[1]
sorted_units = [keys[i] for i in hd_idx[order]]
for row, k in enumerate(sorted_units):
    spk = group[k].restrict(ep)
    ax.plot(spk.t - t0, np.full(len(spk), row), "|", ms=4, color=f"C0")
ax.set_ylabel("HD cells (sorted by\npreferred direction)")
ax.set_xlabel("Time in window (s)")
ax.set_yticks([0, len(sorted_units) - 1])
ax.set_yticklabels(["1", str(len(sorted_units))])
ax.set_ylim(-1, len(sorted_units))
plt.tight_layout()
plt.savefig(f"{FIG}/fig_raw_raster_wake.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_raw_raster_wake.png")

# ---------- Fig R2: polar tuning curves, all HD cells, sorted ----------
n = len(hd_idx)
ncol = 5
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, subplot_kw=dict(projection="polar"),
                         figsize=(2.0 * ncol, 2.0 * nrow))
for ax, u in zip(axes.ravel(), hd_idx[order]):
    tc_u = tc_hz[u]
    ax.plot(np.append(centers, centers[0]), np.append(tc_u, tc_u[0]), color="C0", lw=1.5)
    ax.set_title(f"u{u}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle(f"Wake head-direction tuning curves ({SESSION}, {n} HD cells)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_tuning_polar_M28.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_tuning_polar_M28.png")

# ---------- Fig R3: HD-cell selection stats across all sessions ----------
SESSIONS = ["Mouse17-130128", "Mouse20-130514", "Mouse24-131213",
            "Mouse25-140123", "Mouse28-140310"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
rng = np.random.default_rng(1)
for s in SESSIONS:
    dd = np.load(f"cache/{s}.npz")
    prefs = np.degrees(dd["pref"][dd["hd_idx"]])
    ax.scatter(prefs, rng.normal(0, 0.02, len(prefs)),
               s=12, alpha=0.7, label=s.replace("Mouse", "M"))
ax.set_xlabel("Preferred direction (deg)")
ax.set_yticks([])
ax.set_title(f"Preferred directions tile the ring ({int(sum(np.load(f'cache/{s}.npz')['is_hd'].sum() for s in SESSIONS))} HD cells)")
ax.legend(frameon=False, fontsize=7, markerscale=1.2)

ax = axes[1]
all_mvl, all_hd = [], []
for s in SESSIONS:
    dd = np.load(f"cache/{s}.npz")
    all_mvl.append(dd["mvl"]); all_hd.append(dd["is_hd"])
all_mvl = np.concatenate(all_mvl); all_hd = np.concatenate(all_hd)
b = np.linspace(0, 1, 40)
ax.hist(all_mvl[~all_hd & ~np.isnan(all_mvl)], bins=b, color="0.6", label="non-HD units")
ax.hist(all_mvl[all_hd], bins=b, color="C0", label="HD cells")
ax.axvline(0.3, color="r", ls="--", lw=1, label="MVL threshold 0.3")
ax.set_xlabel("Mean vector length (wake)")
ax.set_ylabel("Units")
ax.set_title("HD-cell selection")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_hdcell_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_hdcell_selection.png")
