"""Helpers for spectrotemporal receptive fields in gerbil auditory-nerve fibres.

Dataset: DANDI:001262, "Single-unit auditory nerve fibre responses of young-adult
and aging gerbils" (Heeringa & Koeppl; Sci Data 2024, doi:10.1038/s41597-024-03259-3).

Each NWB file holds one single fibre. Stimulus protocols are stored as rows of the
`units` table, one row per stimulus repetition, tagged with the protocol name:
  BF_FREQ<f>_rep<n>      tone at frequency f (frequency-tuning / best-frequency run)
  NOISE_NOISE<k>_rep<n>  frozen broadband noise token at level k, 60 repeats
  RLF_ABI<db>_rep<n>     rate-level function
  PH_ABI<db>_rep<n>      Schroeder-phase run
Spike times in each row are relative to the onset of that repetition. The acoustic
waveforms live in /stimulus/presentation at 48828.125 Hz. The four NOISE_k entries
hold the *same* frozen token; the four presentations differ only in level, which was
set by an attenuator and is therefore not reflected in the stored waveform.
"""

import json
import urllib.request

import h5py
import numpy as np
import pynapple as nap
import remfile
import scipy.signal as ss

DANDISET = "001262"
CACHE = "/tmp/remfile_cache"
FS_STIM = 48828.125          # TDT sampling rate of the stored acoustic waveform

# cochleagram / STRF settings
N_BANDS = 34
F_LO, F_HI = 500.0, 12000.0   # the frozen noise token is flat over ~0.6-12 kHz
# One analysis bin is a whole number of stimulus samples, so that the cochleagram
# rows and the spike histogram share exactly the same time base. Rounding 1 ms to
# 49 samples and then *calling* it 1 ms would drift by 8 ms across the 2.4 s token.
STEP = int(round(FS_STIM * 0.001))     # 49 samples
BIN = STEP / FS_STIM                   # 1.0035 ms
LAG_MIN, LAG_MAX = -0.010, 0.025       # STRF lags; negative lags are an acausal control
NOISE_TOKENS = ["NOISE_NOISE_1", "NOISE_NOISE_2", "NOISE_NOISE_3", "NOISE_NOISE_4"]


# --------------------------------------------------------------------------- IO
def list_assets():
    """All assets of the dandiset as a list of dicts (path, asset_id, size)."""
    rows, url = [], (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
                     "/versions/draft/assets/?page_size=1000")
    while url:
        j = json.load(urllib.request.urlopen(url))
        rows += [{"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]}
                 for a in j["results"]]
        url = j.get("next")
    return rows


def open_h5(asset_id):
    """Stream one NWB asset from the DANDI S3 bucket with a local disk cache."""
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           f"/versions/draft/assets/{asset_id}/download/")
    return h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE)), "r")


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def read_fibre(asset_id, want_stim=True):
    """Read one fibre: trial tags, per-repeat spike times, stimulus waveforms, metadata."""
    h = open_h5(asset_id)
    tags = [_s(t) for t in h["units/tag"][:]]
    idx = h["units/spike_times_index"][:].astype(int)
    st = h["units/spike_times"][:]
    starts = np.concatenate([[0], idx[:-1]])
    spikes = []
    for i in range(len(tags)):
        s = st[starts[i]:idx[i]]
        spikes.append(np.sort(s[~np.isnan(s)]))

    at = h["analysis/analysis_table"]
    exp = [_s(e) for e in at["experiment"][:]]
    res = {}
    for key, col in [("bf", "results_bf"), ("sr", "results_sr"),
                     ("threshold", "results_threshold")]:
        res[key] = {e: float(x) for e, x in zip(exp, at[col][:])}

    sub = h["general/subject"]
    info = dict(
        asset_id=asset_id,
        subject=_s(sub["subject_id"][()]),
        age_days=int("".join(ch for ch in _s(sub["age"][()]) if ch.isdigit())),
        sex=_s(sub["sex"][()]),
        session=_s(h["identifier"][()]),
        bf_hz=res["bf"].get("BF", np.nan),
        spont_rate=float(np.nanmax(list(res["sr"].values()))) if res["sr"] else np.nan,
        threshold_db=res["threshold"].get("RLF", np.nan),
    )

    stim = {}
    if want_stim and "stimulus/presentation" in h:
        for k in h["stimulus/presentation"]:
            if k.startswith("NOISE"):
                stim[k] = h[f"stimulus/presentation/{k}/data"][:]
    h.close()
    return dict(tags=tags, spikes=spikes, stim=stim, info=info)


def noise_summary(fibre):
    """Repeat count and mean driven rate for each of the four noise levels."""
    out = {}
    for tok in NOISE_TOKENS:
        rows = [i for i, t in enumerate(fibre["tags"]) if t.startswith(tok + "_rep")]
        if rows:
            n = np.array([len(fibre["spikes"][i]) for i in rows])
            out[tok] = dict(n_reps=len(rows), rate=float(n.mean() / token_duration(fibre)))
    return out


def token_duration(fibre):
    any_wave = next(iter(fibre["stim"].values()))
    return len(any_wave) / FS_STIM


# ------------------------------------------------------------------ cochleagram
def band_cfs(n=N_BANDS, f_lo=F_LO, f_hi=F_HI):
    """Logarithmically spaced filterbank centre frequencies."""
    return np.logspace(np.log10(f_lo), np.log10(f_hi), n)


def band_group_delay(cfs=None, fs=FS_STIM):
    """Envelope-peak delay of each gammatone analysis filter, in seconds.

    Narrow low-frequency filters ring for longer and peak later, which adds an
    analysis delay of several ms at the low-frequency end. Any claim about how
    neural latency varies with frequency has to subtract this.
    """
    cfs = band_cfs() if cfs is None else cfs
    n = int(0.06 * fs)
    imp = np.zeros(n)
    imp[0] = 1.0
    out = np.zeros(len(cfs))
    for i, cf in enumerate(cfs):
        b, a = ss.gammatone(cf, "iir", fs=fs)
        env = np.abs(ss.hilbert(ss.lfilter(b, a, imp)))
        out[i] = np.argmax(env) / fs
    return out


def cochleagram(wave, fs=FS_STIM, cfs=None, bin_size=BIN, floor_db=-50.0):
    """Gammatone filterbank -> Hilbert envelope -> dB -> `bin_size` bins.

    Returns an (n_bins, n_bands) array of band level in dB relative to the loudest
    band/bin of the token, floored at `floor_db`. Working in dB matters: cochlear
    transduction is compressive, and STRFs estimated on a linear-amplitude
    representation are dominated by the few loudest moments of the stimulus.
    """
    cfs = band_cfs() if cfs is None else cfs
    step = int(round(fs * bin_size))
    n_bins = len(wave) // step
    out = np.zeros((n_bins, len(cfs)), dtype=np.float32)
    for i, cf in enumerate(cfs):
        b, a = ss.gammatone(cf, "iir", fs=fs)
        env = np.abs(ss.hilbert(ss.lfilter(b, a, wave)))
        out[:, i] = env[: n_bins * step].reshape(n_bins, step).mean(1)
    db = 20 * np.log10(np.maximum(out, 1e-12) / out.max())
    return np.maximum(db, floor_db).astype(np.float32)


# ----------------------------------------------------------- pynapple assembly
def build_pynapple(fibre, token, cochlea, bin_size=BIN):
    """Lay the repeats of one frozen-noise token end to end on a common time base.

    Repeat n occupies [n*T, (n+1)*T), so the fibre becomes one spike train driven by
    a stimulus that repeats the token n_reps times. Returns pynapple objects plus the
    (n_reps, n_bins) spike-count matrix used for PSTHs and reliability.
    """
    rows = [i for i, t in enumerate(fibre["tags"]) if t.startswith(token + "_rep")]
    if not rows:
        return None
    n_bins = cochlea.shape[0]
    T = n_bins * bin_size

    times, starts = [], []
    for j, i in enumerate(rows):
        off = j * T
        times.append(fibre["spikes"][i] + off)
        starts.append(off)
    times = np.sort(np.concatenate(times)) if times else np.array([])
    starts = np.array(starts)

    support = nap.IntervalSet(start=0.0, end=len(rows) * T)
    spikes = nap.Ts(t=times, time_support=support)
    reps = nap.IntervalSet(start=starts, end=starts + T - bin_size / 2)
    tvec = (np.arange(n_bins) + 0.5) * bin_size
    stim = nap.TsdFrame(t=tvec, d=cochlea, columns=np.round(band_cfs()).astype(int))

    # per-repeat spike counts, aligned to the token time base
    edges = np.arange(n_bins + 1) * bin_size
    counts = np.zeros((len(rows), n_bins), dtype=np.int32)
    for j, iv in enumerate(reps):
        s = spikes.restrict(iv).t - starts[j]
        counts[j] = np.histogram(s, bins=edges)[0]

    return dict(spikes=spikes, stim=stim, reps=reps, counts=counts,
                n_reps=len(rows), token_dur=T, bin_size=bin_size, tvec=tvec)


# ---------------------------------------------------------------------- STRF
def lag_design(S, bin_size=BIN, lag_min=LAG_MIN, lag_max=LAG_MAX):
    """(T, F) stimulus -> (T, F*L) lagged design; column order is (freq, lag).

    X[t, (f, k)] = S[t - k, f], so a positive weight at lag k means "sound energy in
    band f, k ms ago, increases the firing rate now".
    """
    S = np.asarray(S, dtype=np.float64)
    T, F = S.shape
    k0, k1 = int(round(lag_min / bin_size)), int(round(lag_max / bin_size))
    lags = np.arange(k0, k1 + 1)
    X = np.zeros((T, F, lags.size))
    for j, k in enumerate(lags):
        if k >= 0:
            X[k:, :, j] = S[: T - k]
        else:
            X[: T + k, :, j] = S[-k:]
    return X.reshape(T, F * lags.size), lags * bin_size


def ridge_fit(X, y, alpha=1.0):
    """Centred ridge regression, penalty scaled to the mean eigenvalue of X'X.

    The np.errstate guard silences spurious floating-point warnings that this
    NumPy build raises from the vectorized matmul kernel; the outputs were checked
    to be finite.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    with np.errstate(all="ignore"):
        XtX = Xc.T @ Xc
        lam = alpha * np.trace(XtX) / XtX.shape[0]
        w = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), Xc.T @ yc)
        return w, ym - xm @ w


def sound_on(cochlea, floor_db=-50.0):
    """Boolean mask of bins in which the token actually contains sound."""
    lev = cochlea.mean(1)
    return lev > (lev.max() + floor_db) / 2


def analysis_mask(cochlea, skip_onset_bins=50, lag_min=None, lag_max=None,
                  bin_size=BIN):
    """Bins used for fitting and scoring: ongoing noise, away from the padded edges.

    The stored tokens are 1 s noise bursts separated by silence. Silence and the
    onset transient are trivially predictable from the stimulus envelope and inflate
    any prediction score, so the STRF is fitted and scored on ongoing noise only.
    """
    lag_min = LAG_MIN if lag_min is None else lag_min
    lag_max = LAG_MAX if lag_max is None else lag_max
    on = sound_on(cochlea)
    T = len(on)
    since = np.full(T, T, dtype=int)
    last = -T
    for i in range(T):
        if on[i] and (i == 0 or not on[i - 1]):
            last = i
        since[i] = i - last
    k0, k1 = int(round(lag_min / bin_size)), int(round(lag_max / bin_size))
    idx = np.arange(T)
    edge = (idx >= max(k1, 0)) & (idx < T + min(k0, 0))
    return on & (since >= skip_onset_bins) & edge


def token_period(cochlea, min_bins=200, thresh=0.9):
    """Length in bins of the repeating unit inside a stored stimulus token.

    The noise tokens in this dataset contain the same 1 s noise burst twice, each
    followed by a silent gap. Cross-validation has to know that, or a test segment
    and its identical twin end up on opposite sides of the split.
    """
    x = cochlea.mean(1)
    x = x - x.mean()
    n = len(x)
    best = n
    for lag in range(min_bins, n // 2 + 1):
        r = np.corrcoef(x[:n - lag], x[lag:])[0, 1]
        if r > thresh:
            best = lag
            break
    return best


def blocked_folds(T, n_folds=5, n_blocks=25, period=None):
    """Assign contiguous time blocks to folds, so test data is not adjacent to train.

    With `period` set, blocks are defined on position within the repeating unit, so
    repeats of the same stimulus segment always land in the same fold.
    """
    p = T if period is None else period
    phase = np.arange(T) % p
    block = np.minimum(phase // int(np.ceil(p / n_blocks)), n_blocks - 1)
    return (np.arange(n_blocks) % n_folds)[block]


def cv_strf(X, y, alphas=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0), n_folds=5, fold=None):
    """Cross-validate the ridge penalty on held-out *time segments* of the token.

    Held-out segments are stimulus the filter has never seen, so this is a genuine
    predictive test even though the token itself is frozen across repeats.
    Returns (best_alpha, r_per_alpha, y_pred_at_best).
    """
    fold = blocked_folds(len(y), n_folds=n_folds) if fold is None else fold
    rs, preds = [], []
    for a in alphas:
        yhat = np.zeros_like(y, dtype=float)
        for f in range(n_folds):
            te = fold == f
            w, b = ridge_fit(X[~te], y[~te], alpha=a)
            with np.errstate(all="ignore"):
                yhat[te] = X[te] @ w + b
        rs.append(np.corrcoef(y, yhat)[0, 1])
        preds.append(yhat)
    k = int(np.argmax(rs))
    return alphas[k], np.array(rs), preds[k]


def split_half_ceiling(counts, n_boot=50, rng=None):
    """Noise ceiling: expected correlation between the PSTH and the true rate.

    Split the repeats in half, correlate the two half-PSTHs, and apply the
    Spearman-Brown correction to get the reliability of the full-repeat PSTH.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = counts.shape[0]
    rs = []
    for _ in range(n_boot):
        p = rng.permutation(n)
        a = counts[p[: n // 2]].mean(0)
        b = counts[p[n // 2:]].mean(0)
        if a.std() > 0 and b.std() > 0:
            rs.append(np.corrcoef(a, b)[0, 1])
    r_half = float(np.mean(rs))
    r_full = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
    return float(np.sqrt(max(r_full, 0.0)))    # ceiling on r(pred, PSTH)


def strf_metrics(strf, cfs, lags):
    """Best frequency, latency, bandwidth and post-excitatory suppression of a STRF."""
    causal = lags >= 0
    A = strf[:, causal]
    lg = lags[causal]
    fi, li = np.unravel_index(np.argmax(A), A.shape)
    peak = A[fi, li]
    prof = A[:, li]
    half = peak / 2
    lo = hi = fi
    while lo > 0 and prof[lo - 1] >= half:
        lo -= 1
    while hi < len(prof) - 1 and prof[hi + 1] >= half:
        hi += 1
    row = A[fi]
    onset = np.where(row >= half)[0]
    post = row[li:]
    # acausal control: largest weight before lag 0 in the best-frequency band,
    # relative to the excitatory peak. Band-envelope autocorrelation smears a little
    # weight to negative lags, but a valid STRF should have most of its mass at
    # positive lags.
    acausal = np.abs(strf[fi, ~causal]).max() if (~causal).any() else np.nan
    return dict(
        strf_bf_hz=float(cfs[fi]),
        peak_lag_ms=float(lg[li] * 1e3),
        onset_lag_ms=float(lg[onset[0]] * 1e3) if onset.size else np.nan,
        bw_octaves=float(np.log2(cfs[hi] / cfs[lo])) if hi > lo else 0.0,
        peak=float(peak),
        min_post=float(post.min()),
        inhib_ratio=float(-post.min() / peak) if peak > 0 else np.nan,
        acausal_ratio=float(acausal / peak) if peak > 0 else np.nan,
    )


def revcor(spike_times, wave, fs=FS_STIM, win=(-0.0005, 0.008)):
    """Classic reverse correlation: spike-triggered average of the pressure waveform.

    Uses repeat-folded spike times, so every repeat contributes to the same average.
    For low-CF fibres the result is the phase-locked impulse response of the cochlear
    filter (de Boer & de Jongh, 1978).
    """
    n0, n1 = int(round(win[0] * fs)), int(round(win[1] * fs))
    idx = np.round(spike_times * fs).astype(int)
    ok = (idx + n0 >= 0) & (idx + n1 < len(wave))
    idx = idx[ok]
    offs = np.arange(n0, n1)
    sta = wave[idx[:, None] + offs[None, :]].mean(0)
    return offs / fs, sta, len(idx)


def revcor_spectrum(sta, fs=FS_STIM):
    """Magnitude spectrum of the revcor filter and its peak frequency."""
    w = sta * np.hanning(len(sta))
    n = 1 << int(np.ceil(np.log2(len(w) * 8)))
    mag = np.abs(np.fft.rfft(w, n))
    fr = np.fft.rfftfreq(n, 1 / fs)
    return fr, mag, float(fr[np.argmax(mag)])
