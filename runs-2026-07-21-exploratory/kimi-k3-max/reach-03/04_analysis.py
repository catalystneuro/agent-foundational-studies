# Core analysis prototype: direction & velocity tuning in MC_Maze (DANDI 000128)
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import pandas as pd
import requests
import pickle
from tqdm import tqdm
from scipy import stats

DANDISET = "000128"
VERSION = "0.220113.0400"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

def get_download_url(dandiset, version, path):
    url = f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/{version}/assets/"
    while url:
        r = requests.get(url, params={"page_size": 1000})
        r.raise_for_status()
        d = r.json()
        for a in d["results"]:
            if a["path"] == path:
                return f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
        url = d.get("next")
    raise FileNotFoundError(path)

s3_url = get_download_url(DANDISET, VERSION, ASSET_PATH)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_mcmaze")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
trials = nwbfile.trials.to_dataframe()
hp = nwb["hand_pos"]
hv = nwb["hand_vel"]
n_units = len(units)
print(f"{n_units} units, {len(trials)} trials")

# ---------------- per-trial reach direction ----------------
def active_pos(row):
    p = np.asarray(row["target_pos"])
    return p if p.ndim == 1 else p[int(row["active_target"])]

trials["target_xy"] = trials.apply(active_pos, axis=1)
onset_t = trials["move_onset_time"].values
go_t = trials["go_cue_time"].values
pos_at_onset = np.stack([hp.get(t - 0.005, t + 0.005).values.mean(axis=0)
                         for t in tqdm(onset_t, desc="pos at onset")])
txy = np.stack(trials["target_xy"].values)
reach_vec = txy - pos_at_onset
trials["reach_dir"] = np.arctan2(reach_vec[:, 1], reach_vec[:, 0])
trials["reach_dist"] = np.linalg.norm(reach_vec, axis=1)

# ---------------- trial-based firing rates ----------------
def trial_rates(win, align_t):
    """Spike-count rates per unit per trial in window win around align_t."""
    ep = nap.IntervalSet(start=align_t + win[0], end=align_t + win[1])
    cnt = units.count(win[1] - win[0], ep=ep)  # one row per trial
    assert cnt.shape[0] == len(align_t), (cnt.shape, len(align_t))
    return cnt.values / (win[1] - win[0])  # Hz

MOVE_WIN = (-0.05, 0.40)
rates_move = trial_rates(MOVE_WIN, onset_t)          # (n_trials, n_units)
DELAY_WIN = (-0.40, 0.0)
rates_delay = trial_rates(DELAY_WIN, go_t)
print("rates_move:", rates_move.shape, "mean rate:", rates_move.mean().round(2), "Hz")

# ---------------- direction tuning ----------------
NBINS = 8
bin_edges = np.linspace(-np.pi, np.pi, NBINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
dir_bin = np.digitize(trials["reach_dir"].values, bin_edges) - 1  # 0..7
print("trials per direction bin:", np.bincount(dir_bin, minlength=NBINS))

def tuning_stats(rates, dirs, n_shuf=500, rng=None):
    """Cosine fit + ANOVA + shuffle test for each unit.

    rates: (n_trials, n_units); dirs: (n_trials,) reach direction (rad)
    Returns dict of per-unit arrays.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_trials, n_units = rates.shape
    # cosine fit via linear regression on [cos d, sin d]
    X = np.column_stack([np.ones(n_trials), np.cos(dirs), np.sin(dirs)])
    beta, res, *_ = np.linalg.lstsq(X, rates, rcond=None)  # (3, n_units)
    pred = X @ beta
    ss_res = ((rates - pred) ** 2).sum(axis=0)
    ss_tot = ((rates - rates.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    pd_ = np.arctan2(beta[2], beta[1])          # preferred direction
    amp = np.hypot(beta[1], beta[2])            # cosine amplitude (Hz)
    base = beta[0]
    mod_depth = amp / np.maximum(base, 1e-6)    # modulation depth
    # rate-weighted mean resultant length in [0,1]
    resultant = np.abs((rates * np.exp(1j * dirs[:, None])).sum(axis=0)) / np.maximum(rates.sum(axis=0), 1e-12)
    # ANOVA across populated direction bins
    populated = [b for b in range(NBINS) if (dir_bin == b).sum() > 1]
    p_anova = np.full(n_units, np.nan)
    for u in range(n_units):
        groups = [rates[dir_bin == b, u] for b in populated]
        p_anova[u] = stats.f_oneway(*groups)[1]
    # shuffle test on resultant length
    shuf = np.zeros((n_shuf, n_units))
    for s in range(n_shuf):
        perm = rng.permutation(n_trials)
        rs = rates[perm]
        shuf[s] = np.abs((rs * np.exp(1j * dirs[:, None])).sum(axis=0)) / np.maximum(rs.sum(axis=0), 1e-12)
    p_shuf = (shuf >= resultant).mean(axis=0)
    return dict(pd=pd_, amp=amp, base=base, mod_depth=mod_depth, r2=r2,
                resultant=resultant, p_anova=p_anova, p_shuf=p_shuf)

print("\ncomputing movement-epoch tuning stats...")
tun_move = tuning_stats(rates_move, trials["reach_dir"].values)
print("computing delay-epoch tuning stats...")
tun_delay = tuning_stats(rates_delay, trials["reach_dir"].values)

sig_move = (tun_move["p_anova"] < 0.01) & (tun_move["p_shuf"] < 0.01)
sig_delay = (tun_delay["p_anova"] < 0.01) & (tun_delay["p_shuf"] < 0.01)
print(f"direction-tuned (movement): {sig_move.sum()}/{n_units} = {sig_move.mean()*100:.0f}%")
print(f"direction-tuned (delay):    {sig_delay.sum()}/{n_units} = {sig_delay.mean()*100:.0f}%")
print(f"median modulation depth (tuned, move): {np.median(tun_move['mod_depth'][sig_move]):.2f}")
print(f"median cosine R2 (tuned, move): {np.median(tun_move['r2'][sig_move]):.2f}")

# ---------------- continuous velocity tuning during movement ----------------
MOVE_EPOCH = nap.IntervalSet(start=onset_t, end=onset_t + 0.6)
print("\nloading full hand_vel ...")
hv_t = hv.t
hv_v = hv.values  # (n, 2)
speed = np.linalg.norm(hv_v, axis=1)
vel_ang = np.arctan2(hv_v[:, 1], hv_v[:, 0])
speed_tsd = nap.Tsd(t=hv_t, d=speed)
ang_tsd = nap.Tsd(t=hv_t, d=vel_ang)

print("speed tuning curves ...")
tc_speed = nap.compute_tuning_curves(units, speed_tsd, bins=12, range=(0, np.percentile(speed, 99)),
                                     epochs=MOVE_EPOCH, feature_names=["speed"])
print("velocity-direction tuning curves ...")
tc_vdir = nap.compute_tuning_curves(units, ang_tsd, bins=12, range=(-np.pi, np.pi),
                                    epochs=MOVE_EPOCH, feature_names=["vel_dir"])
print("tc_speed shape:", tc_speed.shape if hasattr(tc_speed, 'shape') else type(tc_speed))

# ---------------- spike-rate / speed cross-correlation ----------------
BIN = 0.01  # 10 ms
counts = units.count(BIN, ep=MOVE_EPOCH)   # TsdFrame within movement epochs
spd_binned = speed_tsd.bin_average(BIN, ep=MOVE_EPOCH)
# align lengths
n_b = min(len(counts), len(spd_binned))
C = counts.values[:n_b]
S = spd_binned.values[:n_b]
# smooth spike counts with 50 ms Gaussian
from scipy.ndimage import gaussian_filter1d
Cs = gaussian_filter1d(C.astype(float), sigma=5, axis=0) / BIN  # Hz
lags = np.arange(-30, 31)  # in 10 ms bins -> -300..300 ms
xcorr = np.full((len(lags), n_units), np.nan)
Sz = (S - S.mean()) / S.std()
Cz = (Cs - Cs.mean(axis=0)) / Cs.std(axis=0)
for li, lag in enumerate(lags):
    if lag < 0:
        a, b = Cz[-lag:], Sz[:lag]
    elif lag > 0:
        a, b = Cz[:-lag], Sz[lag:]
    else:
        a, b = Cz, Sz
    xcorr[li] = (a * b[:, None]).mean(axis=0)
peak_lag = lags[np.nanargmax(xcorr, axis=0)] * BIN * 1000  # ms; positive = spikes lag speed? check convention
peak_r = np.nanmax(xcorr, axis=0)
print(f"median peak |r|: {np.median(peak_r):.3f}, median peak lag: {np.median(peak_lag):.0f} ms")

# per-bin tuning curves (mean +/- sem per direction bin) for plotting
def binned_tc(rates):
    tc = np.full((rates.shape[1], NBINS), np.nan)
    se = np.full((rates.shape[1], NBINS), np.nan)
    for b in range(NBINS):
        m = dir_bin == b
        if m.sum() > 1:
            tc[:, b] = rates[m].mean(axis=0)
            se[:, b] = rates[m].std(axis=0) / np.sqrt(m.sum())
    return tc, se

tc_move, tc_move_se = binned_tc(rates_move)
tc_delay, tc_delay_se = binned_tc(rates_delay)

with open("analysis_results.pkl", "wb") as f:
    pickle.dump(dict(
        trials=trials.drop(columns=["target_xy"]),
        rates_move=rates_move, rates_delay=rates_delay,
        tun_move=tun_move, tun_delay=tun_delay,
        sig_move=sig_move, sig_delay=sig_delay,
        bin_centers=bin_centers, dir_bin=dir_bin,
        tc_move=tc_move, tc_move_se=tc_move_se,
        tc_delay=tc_delay, tc_delay_se=tc_delay_se,
        tc_speed=tc_speed, tc_vdir=tc_vdir,
        xcorr=xcorr, lags=lags, peak_lag=peak_lag, peak_r=peak_r,
    ), f)
print("\nsaved analysis_results.pkl")
