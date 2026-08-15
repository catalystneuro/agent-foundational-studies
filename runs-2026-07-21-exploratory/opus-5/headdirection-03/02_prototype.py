"""Prototype the head-direction analysis on one session."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
import hd_lib

S = "sub-A3701_ses-191119"
d = hd_lib.load_session(S)
hd, units = d["hd"], d["units"]
sq = d["epochs"]["wake_square"]

# --- sanity: sampling of the HD signal -------------------------------------
dt = np.diff(hd.t)
print("HD dt: median %.4f s  (%.1f Hz), max gap %.2f s, n gaps>1s %d"
      % (np.median(dt), 1 / np.median(dt), dt.max(), (dt > 1).sum()))
print("HD support:", hd.time_support)
hd_sq = hd.restrict(sq)
print("in square:", len(hd_sq), "samples,", hd_sq.time_support.tot_length(), "s")
print("value range", hd_sq.values.min(), hd_sq.values.max(),
      "n nan", np.isnan(hd_sq.values).sum())

# --- figure 1: raw data ----------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 2]})
t0 = sq.start[0] + 300
win = nap.IntervalSet(t0, t0 + 60)
h = hd.restrict(win)
axes[0].plot(h.t, np.degrees(h.values), "k.", ms=1.5)
axes[0].set_ylabel("head direction\n(deg)")
p = d["position"].restrict(win)
axes[1].plot(p.t, p["x"].values, lw=0.8, label="x")
axes[1].plot(p.t, p["y"].values, lw=0.8, label="y")
axes[1].set_ylabel("position (cm)")
axes[1].legend(fontsize=7, ncol=2)
hd_units = np.where(units.metadata["is_hd_author"].values)[0][:25]
for k, u in enumerate(hd_units):
    ts = units[u].restrict(win).t
    axes[2].plot(ts, np.full_like(ts, k), "|", ms=4, color="C3")
axes[2].set_ylabel("HD-labelled units")
axes[2].set_xlabel("time (s)")
fig.suptitle(f"{S}: raw head direction, position and spiking")
fig.tight_layout()
fig.savefig("fig_proto_raw.png", dpi=130)
print("saved raw fig")

# --- tuning curves ---------------------------------------------------------
tc, occ, centers = hd_lib.tuning_curves(units, hd, sq, nb_bins=60)
print("tc shape", tc.shape, "occ total", occ.sum())
mvl = np.array([hd_lib.circular_mean_vector(centers, tc[c].values)[0] for c in tc.columns])
pref = np.array([hd_lib.circular_mean_vector(centers, tc[c].values)[1] for c in tc.columns])
info = np.array([hd_lib.hd_information(centers, tc[c].values, occ) for c in tc.columns])
auth = units.metadata["is_hd_author"].values
print("MVL: author-HD median %.3f | non-HD median %.3f" % (np.median(mvl[auth]), np.median(mvl[~auth])))
print("info bits/spk: HD %.2f | non %.2f" % (np.median(info[auth]), np.median(info[~auth])))

order = np.argsort(-mvl)[:12]
fig, axes = plt.subplots(3, 4, figsize=(12, 9), subplot_kw={"projection": "polar"})
for ax, u in zip(axes.ravel(), order):
    r = tc[tc.columns[u]].values
    ax.plot(np.append(centers, centers[0]), np.append(r, r[0]), color="C3")
    ax.fill(np.append(centers, centers[0]), np.append(r, r[0]), color="C3", alpha=0.3)
    ax.set_title(f"u{u}  r={mvl[u]:.2f}", fontsize=9, pad=12)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6)
fig.suptitle("Top-12 directionally tuned units (wake, square arena)")
fig.tight_layout()
fig.savefig("fig_proto_tuning.png", dpi=130)
print("saved tuning fig")
