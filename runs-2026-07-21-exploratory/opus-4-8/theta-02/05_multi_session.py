"""Repeat the entrainment / precession analysis across the linear-track sessions."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import hc11
from analysis_core import analyze_session, precession

summary = []
for name in tqdm(hc11.LINEAR_SESSIONS, desc="sessions"):
    res = analyze_session(name, verbose=False)
    prec = precession(res)
    lock = [{k: v for k, v in l.items() if k != "phases"}
            for l in res["locking"] if l["n"] >= 100]
    pooled_phases = np.concatenate(
        [l["phases"] for l in res["locking"]
         if l["n"] >= 100 and l["cell_type"] == "excitatory"])
    summary.append(dict(
        session=name, best_ch=res["best_ch"], theta_delta=res["theta_delta"],
        run_time=float(res["run"].tot_length()),
        n_units=len(res["units"]), locking=lock,
        pooled_hist=np.histogram(pooled_phases,
                                 bins=np.linspace(0, 2 * np.pi, 25))[0],
        prec=[{k: v for k, v in q.items() if k not in ("x", "phi", "tc", "centers")}
              for q in prec],
        prec_x=np.concatenate([q["x"] for q in prec]) if prec else np.array([]),
        prec_phi=np.concatenate([q["phi"] for q in prec]) if prec else np.array([]),
    ))
    s = summary[-1]
    sl = np.array([q["slope"] for q in s["prec"]])
    pv = np.array([q["p"] for q in s["prec"]])
    lp = np.array([l["p"] for l in lock])
    print(f"{name}: {len(lock)} units, {(lp < 0.01).sum()} phase-locked, "
          f"{len(sl)} fields, {((pv < 0.05) & (sl < 0)).sum()} precessing, "
          f"median slope {np.degrees(np.median(sl)):.0f} deg")

with open("results_all_sessions.pkl", "wb") as fh:
    pickle.dump(summary, fh)

# ------------------------------------------------------------------ figure 6
names = [s["session"] for s in summary]
frac_lock = [np.mean([l["p"] < 0.01 for l in s["locking"]]) for s in summary]
frac_prec = [np.mean([(q["p"] < 0.05) & (q["slope"] < 0) for q in s["prec"]])
             for s in summary]
med_slope = [np.degrees(np.median([q["slope"] for q in s["prec"]])) for s in summary]
n_fields = [len(s["prec"]) for s in summary]

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.32)
x = np.arange(len(names))
short = [n.replace("_", "\n") for n in names]

ax = fig.add_subplot(gs[0, 0])
ax.bar(x, frac_lock, color="tab:blue")
ax.set_xticks(x), ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("fraction of units"), ax.set_ylim(0, 1)
ax.set_title("Units significantly theta phase-locked\n(Rayleigh p<0.01)", fontsize=10)

ax = fig.add_subplot(gs[0, 1])
ax.bar(x, frac_prec, color="tab:orange")
for i, n in enumerate(n_fields):
    ax.text(i, frac_prec[i] + 0.02, str(n), ha="center", fontsize=7)
ax.set_xticks(x), ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("fraction of place fields"), ax.set_ylim(0, 1)
ax.set_title("Fields with significant precession\n(p<0.05, negative slope; n above bar)",
             fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.bar(x, med_slope, color="tab:green")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x), ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("median slope (deg per field)")
ax.set_title("Median precession slope per session", fontsize=10)

ax = fig.add_subplot(gs[1, 0])
ctr = np.degrees(0.5 * (np.linspace(0, 2 * np.pi, 25)[:-1]
                        + np.linspace(0, 2 * np.pi, 25)[1:]))
for s in summary:
    h = s["pooled_hist"] / s["pooled_hist"].sum()
    ax.plot(np.r_[ctr, ctr + 360], np.r_[h, h], lw=1.2, label=s["session"])
ax.set_xticks([0, 180, 360, 540, 720])
ax.set_xlabel("theta phase (deg)"), ax.set_ylabel("spike fraction")
ax.set_title("Pooled pyramidal spike-phase distribution", fontsize=10)
ax.legend(fontsize=6, ncol=2)

ax = fig.add_subplot(gs[1, 1])
X = np.concatenate([s["prec_x"] for s in summary])
P = np.concatenate([s["prec_phi"] for s in summary])
H, _, _ = np.histogram2d(X, P, bins=[20, 24], range=[[0, 1], [0, 2 * np.pi]])
H = H / H.sum(axis=1, keepdims=True)
im = ax.imshow(np.tile(H.T, (2, 1)), origin="lower", aspect="auto",
               extent=[0, 1, 0, 720], cmap="magma")
ax.set_yticks([0, 180, 360, 540, 720])
ax.set_xlabel("normalised position in field"), ax.set_ylabel("theta phase (deg)")
ax.set_title(f"All sessions pooled: {sum(n_fields)} fields,\n{len(X):,} spikes",
             fontsize=10)
plt.colorbar(im, ax=ax, label="P(phase | position)")

ax = fig.add_subplot(gs[1, 2])
allslopes = np.degrees(np.concatenate([[q["slope"] for q in s["prec"]]
                                       for s in summary]))
ax.hist(allslopes, bins=np.linspace(-720, 720, 49), color="tab:gray")
ax.axvline(0, color="tab:red", lw=1.5)
ax.axvline(np.median(allslopes), color="tab:blue", ls="--",
           label=f"median {np.median(allslopes):.0f}$\\degree$")
ax.set_xlabel("precession slope (deg per field)"), ax.set_ylabel("count")
ax.set_title(f"All sessions: {(allslopes < 0).mean() * 100:.0f}% negative slopes",
             fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("DANDI 000044 (hc-11): theta entrainment and precession across "
             "the 5 linear-track CA1 sessions", fontsize=13)
fig.savefig("fig06_across_sessions.png", dpi=150, bbox_inches="tight")
print("saved fig06_across_sessions.png")

n_lock = sum(len(s["locking"]) for s in summary)
n_lock_sig = sum(sum(l["p"] < 0.01 for l in s["locking"]) for s in summary)
n_prec_sig = sum(sum((q["p"] < 0.05) and (q["slope"] < 0) for q in s["prec"])
                 for s in summary)
print(f"TOTAL: {n_lock_sig}/{n_lock} units phase-locked, "
      f"{n_prec_sig}/{sum(n_fields)} fields precessing, "
      f"median slope {np.median(allslopes):.0f} deg/field")
