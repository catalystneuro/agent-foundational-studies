# Figure: example unit direction tuning curves (Cartesian + polar)
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("analysis_results.pkl", "rb") as f:
    R = pickle.load(f)

tc = R["tc_move"]; tc_se = R["tc_move_se"]
centers = R["bin_centers"]
tun = R["tun_move"]; sig = R["sig_move"]

# pick 4 example units: significant, high modulation depth, spread of PDs
order = np.argsort(-tun["mod_depth"] * sig)
cands = [u for u in order if sig[u]]
# choose units with distinct PDs
chosen = []
for u in cands:
    if all(abs(np.angle(np.exp(1j * (tun["pd"][u] - tun["pd"][v])))) > 0.6 for v in chosen):
        chosen.append(u)
    if len(chosen) == 4:
        break
print("example units:", chosen, "PDs (deg):", np.round(np.degrees(tun["pd"][chosen]), 0))

th = np.linspace(-np.pi, np.pi, 200)
fig = plt.figure(figsize=(13, 7))
for i, u in enumerate(chosen):
    # Cartesian
    ax = fig.add_subplot(2, 4, i + 1)
    m = ~np.isnan(tc[u])
    ax.errorbar(np.degrees(centers[m]), tc[u, m], yerr=tc_se[u, m],
                fmt="o", color="k", ms=4, capsize=2, label="data (mean ± sem)")
    fit = tun["base"][u] + tun["amp"][u] * np.cos(th - tun["pd"][u])
    ax.plot(np.degrees(th), fit, color="crimson", lw=1.5, label="cosine fit")
    ax.set_xlabel("reach direction (deg)")
    if i == 0:
        ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {u}  (PD={np.degrees(tun['pd'][u]):.0f}°, depth={tun['mod_depth'][u]:.2f})", fontsize=9)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")
    # polar
    axp = fig.add_subplot(2, 4, i + 5, projection="polar")
    th_b = np.concatenate([centers[m], centers[m][:1] + 2 * np.pi])
    r_b = np.concatenate([tc[u, m], tc[u, m][:1]])
    axp.plot(th_b, r_b, "o-", color="k", ms=4, lw=1)
    th_f = np.concatenate([th, th[:1] + 2 * np.pi])
    r_f = np.concatenate([fit, fit[:1]])
    axp.plot(th_f, np.clip(r_f, 0, None), color="crimson", lw=1.5)
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    axp.tick_params(labelsize=7)
    axp.set_rlabel_position(22)
fig.suptitle("Direction tuning of example M1/PMd units (peri-movement rates, −50 to +400 ms around move onset)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig3_example_direction_tuning.png", dpi=150)
print("saved fig3_example_direction_tuning.png")
