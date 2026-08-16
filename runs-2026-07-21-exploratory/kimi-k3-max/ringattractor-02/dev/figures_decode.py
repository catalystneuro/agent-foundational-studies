"""Decoding figures from cache/decode_Mouse28-140310.npz (v2 keys)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "../figures"
os.makedirs(FIG, exist_ok=True)
d = np.load("cache/decode_Mouse28-140310.npz")
centers = d["centers"]

def crop_smooth(lab, t0, t1):
    t = d[f"{lab}_smooth_t"]
    m = (t >= t0) & (t <= t1)
    P = d[f"{lab}_smooth_P"][m]
    return t[m], P, d[f"{lab}_smooth_ang"][m]

# ---------- Fig D1: wake validation ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
t = d["wake_t"]
t0 = t[len(t) // 3]
m = (t >= t0) & (t <= t0 + 60)
ax = axes[0]
ax.plot(t[m] - t0, np.degrees(d["wake_actual"][m]), ".", ms=2, color="0.5", label="actual (LED)")
ax.plot(t[m] - t0, np.degrees(d["wake_ang"][m]), ".", ms=2, color="C0", label="decoded")
ax.set_ylabel("Head direction (deg)")
ax.set_xlabel("Time in wake window (s)")
ax.set_yticks([0, 180, 360])
ax.legend(frameon=False, markerscale=3, loc="upper right")
ax.set_title("Wake: decoded vs actual head direction")
ax = axes[1]
err = np.degrees(d["wake_err"])
b = np.linspace(-180, 180, 72)
ax.hist(err, bins=b, color="C0", density=True)
ax.axvline(0, color="k", lw=0.8)
med = np.median(np.abs(err))
ax.axvline(med, color="r", ls="--", lw=1)
ax.axvline(-med, color="r", ls="--", lw=1)
ax.set_title(f"Wake decode error (median |err| = {med:.0f} deg)")
ax.set_xlabel("Decoded - actual HD (deg)")
ax.set_ylabel("Density")
ax.set_xlim(-180, 180)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_decode_wake_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_decode_wake_validation.png")

# ---------- Fig D2: REM posterior heatmap (smoothed), strongest-bump 60 s ----------
rem_ep = d["rem_ep"]
t_r = d["rem_smooth_t"]
P_r = d["rem_smooth_P"]
R_r = (P_r * np.exp(1j * centers[None, :])).sum(axis=1)
R_r = np.abs(R_r)
m_ep = (t_r >= rem_ep[0]) & (t_r <= rem_ep[1])
t_ep, R_ep = t_r[m_ep], R_r[m_ep]
best_t0, best_R = None, -1
for cand in np.arange(rem_ep[0], rem_ep[1] - 60, 5):
    m = (t_ep >= cand) & (t_ep < cand + 60)
    if m.sum() < 100:
        continue
    if R_ep[m].mean() > best_R:
        best_R, best_t0 = R_ep[m].mean(), cand
t0, t1 = best_t0, best_t0 + 60

fig, ax = plt.subplots(figsize=(10, 3.2))
tt, Pn, ang = crop_smooth("rem", t0, t1)
ax.imshow(Pn.T, origin="lower", aspect="auto", cmap="viridis",
          extent=[tt[0] - t0, tt[-1] - t0, 0, 360], interpolation="nearest")
ax.plot(tt - t0, np.degrees(ang), color="r", lw=1.0, alpha=0.9)
ax.set_yticks([0, 180, 360])
ax.set_ylabel("HD (deg)")
ax.set_xlabel("Time in REM episode (s)")
ax.set_title("REM sleep: internally generated bump of HD activity drifts around the ring", fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_decode_rem.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_decode_rem.png")

# ---------- Fig D3: NREM snippet (smoothed) ----------
nrem_ep = d["nrem_ep"]
t_n = d["nrem_smooth_t"]
P_n = d["nrem_smooth_P"]
R_n = np.abs((P_n * np.exp(1j * centers[None, :])).sum(axis=1))
best_t0, best_R = None, -1
for cand in np.arange(nrem_ep[0], nrem_ep[0] + 600, 5):
    m = (t_n >= cand) & (t_n < cand + 60)
    if m.sum() > 0 and R_n[m].mean() > best_R:
        best_R, best_t0 = R_n[m].mean(), cand
t0, t1 = best_t0, best_t0 + 60
fig, ax = plt.subplots(figsize=(10, 3.2))
tt, Pn, ang = crop_smooth("nrem", t0, t1)
ax.imshow(Pn.T, origin="lower", aspect="auto", cmap="viridis",
          extent=[tt[0] - t0, tt[-1] - t0, 0, 360], interpolation="nearest")
ax.plot(tt - t0, np.degrees(ang), color="r", lw=1.0, alpha=0.9)
ax.set_yticks([0, 180, 360])
ax.set_ylabel("HD (deg)")
ax.set_xlabel("Time in NREM episode (s)")
ax.set_title("NREM sleep: the HD bump is maintained and drifts without sensory input", fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_decode_nrem.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_decode_nrem.png")

# ---------- Fig D4: stats — angular speed (real vs bin-shuffle) and R ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
labs = ["wake", "rem", "nrem"]
names = ["Wake", "REM", "NREM"]
cols = ["k", "r", "b"]
speeds = [d[f"{l}_speed"][~np.isnan(d[f"{l}_speed"])] for l in labs]
speeds += [d["rem_binshuf_speed"][~np.isnan(d["rem_binshuf_speed"])],
           d["nrem_binshuf_speed"][~np.isnan(d["nrem_binshuf_speed"])]]
names += ["REM\nbin-shuf", "NREM\nbin-shuf"]
cols += ["0.65", "0.65"]
bp = ax.boxplot([np.degrees(s) for s in speeds], tick_labels=names, showfliers=False,
                patch_artist=True, medianprops=dict(color="k"))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax.set_ylabel("Decoded angular speed (deg/s)")
ax.set_title("Bump drift is temporally continuous in sleep")
ax.set_yscale("log")

ax = axes[1]
Rs = [d[f"{l}_R"] for l in labs]
bp = ax.boxplot(Rs, tick_labels=names[:3], showfliers=False, patch_artist=True,
                medianprops=dict(color="k"))
for patch, c in zip(bp["boxes"], cols[:3]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax.set_ylabel("Posterior concentration R")
ax.set_title("A single localized bump persists in sleep")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_decode_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_decode_stats.png")

for l, n_ in zip(["wake", "rem", "nrem"], ["Wake", "REM", "NREM"]):
    s = d[f"{l}_speed"]; s = s[~np.isnan(s)]
    print(f"{n_:5s} median speed {np.degrees(np.median(s)):7.1f} deg/s   median R {np.median(d[f'{l}_R']):.3f}")
for l, n_ in [("rem", "REM bin-shuf"), ("nrem", "NREM bin-shuf")]:
    s = d[f"{l}_binshuf_speed"]; s = s[~np.isnan(s)]
    print(f"{n_:13s} median speed {np.degrees(np.median(s)):7.1f} deg/s")
