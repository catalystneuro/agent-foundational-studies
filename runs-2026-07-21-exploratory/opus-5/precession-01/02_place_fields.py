"""Step 2: direction-specific 1-D place fields on the linear track."""

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

import precession_lib as pl

SESSION = "Achilles-10252013"
N_BINS = 40  # 4 cm bins over the 1.6 m track
MIN_PEAK_RATE = 1.0  # Hz
MIN_SPATIAL_INFO = 0.5  # bits/spike
MIN_FIELD_SPIKES = 50

f, nwbfile = pl.open_session(SESSION)
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
exc = units.getby_category("cell_type")["excitatory"]

pos, fs_pos, track_len, maze_name = pl.load_position(f)
vel, speed = pl.compute_speed(pos, fs_pos)
runs = dict(zip(("right", "left"), pl.find_runs(pos, vel, fs_pos, track_len)))

bins = np.linspace(0, 1.6, N_BINS + 1)
centers = 0.5 * (bins[1:] + bins[:-1])

results = {}
for direction, ep in runs.items():
    tc = nap.compute_1d_tuning_curves(exc, pos, nb_bins=N_BINS, minmax=(0, 1.6), ep=ep)
    # occupancy in seconds per bin
    p = pos.restrict(ep)
    occ, _ = np.histogram(p.values, bins=bins)
    occ = occ / fs_pos
    tc_s = tc.apply(lambda c: gaussian_filter1d(c.values, 1.0, mode="nearest"))
    results[direction] = dict(tc=tc_s, occ=occ, ep=ep)
    print(f"{direction}: {len(ep)} laps, occupancy {occ.sum():.1f} s, "
          f"min/max bin occupancy {occ.min():.2f}/{occ.max():.2f} s")

# ---------------------------------------------------------------------------
# place-cell selection
# ---------------------------------------------------------------------------
rows = []
for direction, r in results.items():
    tc, occ = r["tc"], r["occ"]
    for uid in tc.columns:
        v = tc[uid].values
        if not np.isfinite(v).any() or np.nanmax(v) < MIN_PEAK_RATE:
            continue
        si = pl.spatial_info(v, occ)
        lo, hi, pk = pl.field_bounds(v, centers)
        n_spk = len(exc[uid].restrict(r["ep"]))
        in_field = ((pos.restrict(r["ep"]).values >= lo) &
                    (pos.restrict(r["ep"]).values <= hi))
        rows.append(dict(unit=uid, direction=direction, peak_rate=np.nanmax(v),
                         spatial_info=si, field_lo=lo, field_hi=hi, field_peak=pk,
                         field_width=hi - lo, n_spikes_run=n_spk))

import pandas as pd

df = pd.DataFrame(rows)
df["is_place_cell"] = (
    (df.peak_rate >= MIN_PEAK_RATE)
    & (df.spatial_info >= MIN_SPATIAL_INFO)
    & (df.field_width >= 0.10)
    & (df.field_width <= 1.0)
)
print(f"\n{df.is_place_cell.sum()} / {len(df)} unit-direction pairs pass place-cell criteria")
print(df[df.is_place_cell].sort_values("spatial_info", ascending=False).head(12).to_string(index=False))
df.to_csv("place_fields_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Figure 3: place field population maps + example fields
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.42, wspace=0.32)

for j, direction in enumerate(("right", "left")):
    ax = fig.add_subplot(gs[0, j])
    sel = df[(df.direction == direction) & df.is_place_cell].sort_values("field_peak")
    mat = np.array([results[direction]["tc"][u].values for u in sel.unit])
    mat = mat / np.nanmax(mat, axis=1, keepdims=True)
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 1.6, 0, len(sel)], interpolation="nearest")
    ax.set_xlabel("position on track (m)")
    ax.set_ylabel("place cell (sorted by field peak)")
    ax.set_title(f"{direction}ward runs\n{len(sel)} place fields")
    fig.colorbar(im, ax=ax, label="normalised rate")

ax = fig.add_subplot(gs[0, 2])
ax.hist(df[df.is_place_cell].spatial_info, bins=20, color="0.3")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("# place fields")
ax.set_title("Spatial information")

# three example tuning curves
sel = df[df.is_place_cell].sort_values("spatial_info", ascending=False)
for k in range(3):
    r = sel.iloc[k]
    ax = fig.add_subplot(gs[1, k])
    tc = results[r.direction]["tc"][r.unit].values
    ax.plot(centers, tc, "k", lw=2)
    ax.axvspan(r.field_lo, r.field_hi, color="tab:orange", alpha=0.25, lw=0)
    ax.set_xlabel("position (m)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {r.unit}, {r.direction}ward\n"
                 f"{r.spatial_info:.2f} bits/spike, peak {r.peak_rate:.1f} Hz", fontsize=9)

fig.suptitle(f"{SESSION}: CA1 place fields on the 1.6 m linear track", fontsize=13)
fig.savefig("fig03_place_fields.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nwrote fig03_place_fields.png, place_fields_summary.csv")
