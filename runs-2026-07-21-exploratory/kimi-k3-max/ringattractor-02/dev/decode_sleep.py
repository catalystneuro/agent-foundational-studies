"""Decode virtual head direction during sleep on Mouse28-140310 (20 HD cells).

Manual Poisson Bayesian decoder (Zhang et al. 1998), cross-checked against
nap.decode_bayes. Controls: time-bin shuffle within epochs (destroys temporal
continuity, keeps posteriors identical in distribution).

Saves cache/decode_Mouse28-140310.npz
"""
import numpy as np
import pynapple as nap
from prototype_load import load_session

rng = np.random.default_rng(7)
SESSION = "Mouse28-140310"
OUT = "cache/decode_Mouse28-140310.npz"
BIN = 0.1
SW = 5  # 500 ms smoothing window for display decode

nwb = load_session(SESSION)
states = nwb["states"]
wake = states[states["label"] == "Awake"]
rem = states[states["label"] == "REM"]
nrem = states[states["label"] == "Non-REM"]

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
print("HD cells:", len(hd_idx))
hd_group = nap.TsGroup({keys[i]: group[keys[i]] for i in hd_idx})

tc_da = nap.compute_tuning_curves(hd_group, hd_tsd, bins=60, range=(0, 2 * np.pi), epochs=wake)
tc_hz = np.asarray(tc_da)  # (n_units, n_bins), Hz
bin_dim = [x for x in tc_da.dims if x != "unit"][0]
centers = np.asarray(tc_da.coords[bin_dim])


def decode_counts(counts, tc_hz, bin_s):
    """Poisson Bayesian decode. counts (T, U); tc_hz (U, B) -> P (T, B)."""
    lam = np.clip(tc_hz * bin_s, 1e-12, None)          # expected counts per bin
    logL = counts @ np.log(lam) - lam.sum(axis=0)[None, :]
    logL -= logL.max(axis=1, keepdims=True)
    P = np.exp(logL)
    P /= P.sum(axis=1, keepdims=True)
    return P


def circ_stats(P, centers):
    z = (P * np.exp(1j * centers[None, :])).sum(axis=1)
    return np.abs(z), np.angle(z) % (2 * np.pi)


def counts_per_epoch(group, ep, bin_s, smooth_bins=None):
    """Count spikes per epoch; optional uniform smoothing within each epoch.
    Returns concatenated counts, times, and per-epoch row counts."""
    all_c, all_t, lens = [], [], []
    for i in range(len(ep)):
        seg = nap.IntervalSet(ep.start[i], ep.end[i])
        c = group.count(bin_s, seg)
        v = np.asarray(c.values, dtype=float)
        if smooth_bins and smooth_bins > 1:
            k = np.ones(smooth_bins) / smooth_bins
            v = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), 0, v)
        all_c.append(v)
        all_t.append(c.t)
        lens.append(len(v))
    return np.vstack(all_c), np.concatenate(all_t), lens


out = {}
for lab, ep in [("wake", wake), ("rem", rem), ("nrem", nrem)]:
    # unsmoothed decode for statistics
    C, tt, lens = counts_per_epoch(hd_group, ep, BIN)
    P = decode_counts(C, tc_hz, BIN)
    R, ang = circ_stats(P, centers)
    dt = np.diff(tt)
    speed = np.abs(np.angle(np.exp(1j * np.diff(ang)))) / dt
    speed[dt > 2 * BIN] = np.nan
    out.update({f"{lab}_R": R, f"{lab}_ang": ang, f"{lab}_t": tt,
                f"{lab}_speed": speed, f"{lab}_P": P})
    print(f"{lab}: median R={np.median(R):.3f}  median speed={np.degrees(np.nanmedian(speed)):.1f} deg/s")

    # smoothed decode for display
    Cs, tts, _ = counts_per_epoch(hd_group, ep, BIN, smooth_bins=SW)
    Ps = decode_counts(Cs, tc_hz, BIN * 1.0)
    Rs, angs = circ_stats(Ps, centers)
    out.update({f"{lab}_smooth_P": Ps, f"{lab}_smooth_ang": angs, f"{lab}_smooth_t": tts})

    # time-bin shuffle control (sleep states only): permute bins within each epoch
    if lab in ("rem", "nrem"):
        Cs_h = C.copy()
        row0 = 0
        for n_i in lens:
            seg_idx = np.arange(row0, row0 + n_i)
            Cs_h[seg_idx] = Cs_h[seg_idx][rng.permutation(n_i)]
            row0 += n_i
        Psh = decode_counts(Cs_h, tc_hz, BIN)
        Rsh, angsh = circ_stats(Psh, centers)
        speedsh = np.abs(np.angle(np.exp(1j * np.diff(angsh)))) / dt
        speedsh[dt > 2 * BIN] = np.nan
        out.update({f"{lab}_binshuf_speed": speedsh, f"{lab}_binshuf_R": Rsh})
        print(f"{lab} bin-shuffle: median speed={np.degrees(np.nanmedian(speedsh)):.1f} deg/s")

# wake validation against actual HD (circular-safe interpolation of cos/sin)
dec_w_ang = out["wake_ang"]
t_w = out["wake_t"]
cos_i = np.interp(t_w, t[valid], np.cos(hd[valid]))
sin_i = np.interp(t_w, t[valid], np.sin(hd[valid]))
actual = np.angle(cos_i + 1j * sin_i) % (2 * np.pi)
err = np.angle(np.exp(1j * (dec_w_ang - actual)))
print("wake decode median |err| = %.1f deg" % np.degrees(np.median(np.abs(err))))

# cross-check manual decoder vs nap.decode_bayes on a wake subset
ep_check = nap.IntervalSet(t_w[0], t_w[0] + 300)
dec_ref, _ = nap.decode_bayes(tc_da, hd_group, ep_check, BIN)
m = (t_w >= t_w[0]) & (t_w <= t_w[0] + 300)
n_overlap = min(len(dec_ref), m.sum())
agree = np.angle(np.exp(1j * (np.asarray(dec_ref.values)[:n_overlap] - out["wake_ang"][m][:n_overlap])))
print("manual vs decode_bayes: median |diff| = %.2f deg over %d bins"
      % (np.degrees(np.median(np.abs(agree))), n_overlap))

def longest_epoch(ep):
    lengths = ep.end - ep.start
    i = int(np.argmax(lengths))
    return float(ep.start[i]), float(ep.end[i])

out.update(dict(wake_err=err, wake_actual=actual, centers=centers,
                rem_ep=np.array(longest_epoch(rem)), nrem_ep=np.array(longest_epoch(nrem)),
                pref=d["pref"][hd_idx]))
np.savez(OUT, **out)
print("saved", OUT)
