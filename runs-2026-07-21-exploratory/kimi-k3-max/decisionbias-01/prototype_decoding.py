# Prototype: pre-stimulus choice decoding on session 1
import time
import numpy as np
import decoding_lib as dl

t0 = time.time()
trials, unit_spikes, ks2, regions = dl.load_session("cache_sub-92130c1b.npz")
mask, rates = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True)
print(f"selected {mask.sum()} / {len(mask)} units")

units = [s for s, m in zip(unit_spikes, mask) if m]
stim_on = trials["stimOn_times"].to_numpy()
choice = trials["choice"].to_numpy()  # +1 left, -1 right
cl = trials["contrastLeft"].to_numpy()
cr = trials["contrastRight"].to_numpy()
p_left = trials["probabilityLeft"].to_numpy()

valid = ~np.isnan(choice) & ~np.isnan(stim_on)
zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)
y_all = (choice[valid] == 1).astype(int)
y_zero = (choice[zero_c] == 1).astype(int)
print(f"all trials: {valid.sum()} (P(left)={y_all.mean():.2f}); "
      f"0% contrast: {zero_c.sum()} (P(left)={y_zero.mean():.2f})")

PRE = (-0.4, 0.0)

# --- pre-stimulus decoding, 0% contrast trials ---
X_zero = dl.count_in_windows(units, stim_on[zero_c] + PRE[0], stim_on[zero_c] + PRE[1])
acc, sem = dl.cv_accuracy(X_zero, y_zero, n_repeats=10)
null = dl.shuffle_null(X_zero, y_zero, n_shuffles=200)
p = (np.sum(null >= acc) + 1) / (len(null) + 1)
print(f"[0% contrast] pre-stim choice decoding: acc={acc:.3f}+-{sem:.3f}, "
      f"null={null.mean():.3f} (95% {np.percentile(null,95):.3f}), p={p:.4f}")

# --- pre-stimulus decoding, all trials ---
X_all = dl.count_in_windows(units, stim_on[valid] + PRE[0], stim_on[valid] + PRE[1])
acc_a, sem_a = dl.cv_accuracy(X_all, y_all, n_repeats=10)
null_a = dl.shuffle_null(X_all, y_all, n_shuffles=200)
p_a = (np.sum(null_a >= acc_a) + 1) / (len(null_a) + 1)
print(f"[all trials]  pre-stim choice decoding: acc={acc_a:.3f}+-{sem_a:.3f}, "
      f"null={null_a.mean():.3f} (95% {np.percentile(null_a,95):.3f}), p={p_a:.4f}")

# --- within-block controls on 0% trials (block prior held constant) ---
for pl in [0.2, 0.8]:
    m = zero_c & (p_left == pl)
    yb = (choice[m] == 1).astype(int)
    if m.sum() < 30 or yb.sum() < 8 or (1 - yb).sum() * len(yb) < 8:
        print(f"[0% | pL={pl}] too few trials ({m.sum()})"); continue
    Xb = dl.count_in_windows(units, stim_on[m] + PRE[0], stim_on[m] + PRE[1])
    ab, _ = dl.cv_accuracy(Xb, yb, n_repeats=10)
    nb = dl.shuffle_null(Xb, yb, n_shuffles=200)
    pb = (np.sum(nb >= ab) + 1) / (len(nb) + 1)
    print(f"[0% | pL={pl}] n={m.sum()} acc={ab:.3f}, null95={np.percentile(nb,95):.3f}, p={pb:.4f}")

# --- block-prior decoding (0.2 vs 0.8) from pre-stim activity, all trials ---
mb = valid & (p_left != 0.5)
yb = (p_left[mb] == 0.8).astype(int)
Xb = dl.count_in_windows(units, stim_on[mb] + PRE[0], stim_on[mb] + PRE[1])
ab, _ = dl.cv_accuracy(Xb, yb, n_repeats=10)
nb = dl.shuffle_null(Xb, yb, n_shuffles=200)
pb = (np.sum(nb >= ab) + 1) / (len(nb) + 1)
print(f"[block prior 0.8 vs 0.2] n={mb.sum()} acc={ab:.3f}, null95={np.percentile(nb,95):.3f}, p={pb:.4f}")

# --- history control: decode PREVIOUS trial's choice from same window ---
prev_choice = np.roll(choice, 1)
mh = zero_c & ~np.isnan(prev_choice) & (np.arange(len(choice)) > 0)
yh = (prev_choice[mh] == 1).astype(int)
Xh = dl.count_in_windows(units, stim_on[mh] + PRE[0], stim_on[mh] + PRE[1])
ah, _ = dl.cv_accuracy(Xh, yh, n_repeats=10)
nh = dl.shuffle_null(Xh, yh, n_shuffles=200)
ph = (np.sum(nh >= ah) + 1) / (len(nh) + 1)
print(f"[previous choice] n={mh.sum()} acc={ah:.3f}, null95={np.percentile(nh,95):.3f}, p={ph:.4f}")

print(f"\ntotal time {time.time()-t0:.1f}s")
