"""Re-plot the decoding / GLM figures with a window-matched GLM-vs-empirical comparison."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

d = np.load("decoding_glm.npz")
s = np.load("session1_tuning.npz")
cm, acc, acc_shuf = d["cm"], float(d["acc"]), float(d["acc_shuf"])
sizes, curve = d["sizes"], d["curve"]
kernels, kernel_time, emp_d, ufreq = d["kernels"], d["kernel_time"], d["emp_d"], d["ufreq"]

# Compare the GLM to the window-count analysis over the SAME 10-60 ms window.
kmask = (kernel_time >= 10) & (kernel_time < 60)
glm_gain = kernels[:, :, kmask].max(axis=2)
tuned = s["sig_tune"] & s["sig_resp"] & (emp_d.max(axis=0) > 0)
agree = np.mean(np.argmax(glm_gain[:, tuned], 0) == np.argmax(emp_d[:, tuned], 0))
agree_all = np.mean(np.argmax(glm_gain, 0) == np.argmax(emp_d, 0))
rho = stats.spearmanr(glm_gain[:, tuned].ravel(), emp_d[:, tuned].ravel()).statistic
print("tuned units in this session: %d/%d" % (tuned.sum(), len(tuned)))
print("BF agreement: tuned %.2f | all units %.2f | Spearman rho %.3f"
      % (agree, agree_all, rho))
print("decoding %.3f (shuffled %.3f)" % (acc, acc_shuf))

freq_colors = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(ufreq)))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
im = axes[0].imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
axes[0].set_xticks(range(5), ["%g" % (f / 1000) for f in ufreq])
axes[0].set_yticks(range(5), ["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("decoded frequency (kHz)")
axes[0].set_ylabel("presented frequency (kHz)")
axes[0].set_title("Single-trial decoding: %.0f%% correct\n(chance 20%%, label-shuffled %.0f%%)"
                  % (100 * acc, 100 * acc_shuf))
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, "%.2f" % cm[i, j], ha="center", va="center", fontsize=8,
                     color="w" if cm[i, j] > cm.max() / 2 else "k")
plt.colorbar(im, ax=axes[0], label="P(decoded | presented)")

axes[1].plot(sizes, 100 * curve.mean(axis=1), "o-", color="C0")
axes[1].fill_between(sizes, 100 * curve.min(axis=1), 100 * curve.max(axis=1),
                     alpha=0.25, color="C0")
axes[1].axhline(20, color="k", ls="--", lw=1, label="chance")
axes[1].set_xscale("log")
axes[1].set_xlabel("number of units (random subsets)")
axes[1].set_ylabel("decoding accuracy (%)")
axes[1].set_title("Accuracy vs population size")
axes[1].legend(frameon=False)

axes[2].scatter(emp_d[:, ~tuned].ravel(), glm_gain[:, ~tuned].ravel(), s=5,
                alpha=0.25, color="0.6", label="untuned units")
axes[2].scatter(emp_d[:, tuned].ravel(), glm_gain[:, tuned].ravel(), s=6,
                alpha=0.4, color="C2", label="tuned units")
axes[2].set_xlabel("evoked rate change, window count (Hz)")
axes[2].set_ylabel("GLM kernel peak, 10-60 ms (log gain)")
axes[2].set_title("GLM vs window-count tuning\ntuned units: rho = %.2f, "
                  "BF agreement %.0f%%" % (rho, 100 * agree))
axes[2].legend(fontsize=8, frameon=False, loc="lower right")
for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig07_decoding.png", dpi=140)
print("saved fig07")

sel = np.where(tuned)[0][np.argsort(
    -(glm_gain[:, tuned].max(axis=0) - glm_gain[:, tuned].min(axis=0)))[:6]]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for c, j in enumerate(sel):
    ax = axes.ravel()[c]
    for i in range(len(ufreq)):
        ax.plot(kernel_time, kernels[i, j], color=freq_colors[i],
                label="%g kHz" % (ufreq[i] / 1000))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("unit %d" % j, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if c >= 3:
        ax.set_xlabel("time from tone onset (ms)")
    if c % 3 == 0:
        ax.set_ylabel("GLM kernel (log rate)")
    if c == 0:
        ax.legend(fontsize=7, frameon=False)
fig.suptitle("NeMoS Poisson GLM — frequency-specific temporal response kernels (LA11_ses1)")
fig.tight_layout()
fig.savefig("fig08_glm_kernels.png", dpi=140)
print("saved fig08")
