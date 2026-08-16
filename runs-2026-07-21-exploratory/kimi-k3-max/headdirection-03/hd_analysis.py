"""Core functions for head-direction cell analysis of DANDI 000056.

Peyrache et al. (2015), "Internally organized mechanisms of the head
direction sense", Nature Neuroscience. Recordings from anterodorsal
thalamus (ADn) and postsubiculum (PoS) in freely moving mice, with
dual-LED head tracking and sleep-state scoring.
"""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np

DANDI_VERSION = "0.250624.0430"


def asset_url(asset_id):
    return f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"


def load_session(asset_id, cache_dir="/tmp/remfile_cache_hd"):
    """Stream an NWB session from DANDI with a shared disk cache."""
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(asset_url(asset_id), disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, io


def compute_head_direction(nwb):
    """Head direction from the angle of the red-blue LED vector.

    Tracking failures are marked with -1 sentinel values; those samples
    are set to NaN (excluded from occupancy and spike-angle histograms).
    Returns a pynapple Tsd in radians, [0, 2*pi).
    """
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    valid = (red.values[:, 0] > 0) & (red.values[:, 1] > 0) & \
            (blue.values[:, 0] > 0) & (blue.values[:, 1] > 0)
    angle = np.arctan2(red.values[:, 1] - blue.values[:, 1],
                       red.values[:, 0] - blue.values[:, 0]) % (2 * np.pi)
    angle[~valid] = np.nan
    return nap.Tsd(t=red.t, d=angle)


def get_epochs(nwb):
    """Return (wake, rem, nrem) IntervalSets from the scored states."""
    states = nwb["states"]
    labels = np.asarray(states["label"])
    return (states[labels == "Awake"],
            states[labels == "REM"],
            states[labels == "Non-REM"])


def valid_hd_samples(hd, epochs):
    """Valid (non-NaN) HD timestamps and values within epochs."""
    hd_e = hd.restrict(epochs)
    mask = ~np.isnan(hd_e.values)
    return hd_e.t[mask], hd_e.values[mask]


def spike_angles(spike_times, hd_t, hd_d):
    """HD angle at each spike time (linear interpolation)."""
    return np.interp(spike_times, hd_t, hd_d)


def mean_vector_length(angles):
    """Mean resultant vector length and preferred direction."""
    angles = np.asarray(angles)
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan, np.nan
    z = np.exp(1j * angles).mean()
    return np.abs(z), np.angle(z) % (2 * np.pi)


def random_time_null(n_spikes, hd_d_valid, n_shuf=500, rng=None, block=50):
    """Null distribution of MVL from random wake-time resampling.

    Draws n_spikes angles with replacement from the empirical HD
    distribution. This null preserves the non-uniform HD occupancy, so
    significance reflects tuning beyond what occupancy alone produces.
    Computed in blocks to bound memory for high-count units.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.empty(n_shuf)
    for i in range(0, n_shuf, block):
        j = min(i + block, n_shuf)
        draws = rng.choice(hd_d_valid, size=(j - i, n_spikes))
        out[i:j] = np.abs(np.exp(1j * draws).mean(axis=1))
    return out


def analyze_session(nwb, mvl_floor=0.3, n_shuf=500, min_spikes=100,
                    seed=0, verbose=True):
    """Full HD-cell pipeline for one session.

    Returns a dict with per-unit MVL, preferred direction, null 95th
    percentile, HD-cell labels, tuning curves (wake/REM/NREM), and epoch
    summaries.
    """
    units = nwb["units"]
    hd = compute_head_direction(nwb)
    wake, rem, nrem = get_epochs(nwb)

    # filter near-silent units using the session-wide rate metadata
    rates = units.metadata["rate"].values
    units_f = units[rates > 0.1]
    keys = list(units_f.keys())

    hd_t_w, hd_d_w = valid_hd_samples(hd, wake)

    rng = np.random.default_rng(seed)
    mvl = np.full(len(keys), np.nan)
    pref = np.full(len(keys), np.nan)
    null95 = np.full(len(keys), np.nan)
    pval = np.full(len(keys), np.nan)
    n_spikes = np.zeros(len(keys), dtype=int)

    for i, u in enumerate(keys):
        sp = units_f[u].restrict(wake)
        n_spikes[i] = len(sp)
        if n_spikes[i] < min_spikes:
            continue
        ang = spike_angles(sp.t, hd_t_w, hd_d_w)
        mvl[i], pref[i] = mean_vector_length(ang)
        null = random_time_null(n_spikes[i], hd_d_w, n_shuf=n_shuf, rng=rng)
        null95[i] = np.percentile(null, 95)
        pval[i] = (np.sum(null >= mvl[i]) + 1) / (n_shuf + 1)

    is_hd = (pval < 0.05) & (mvl > mvl_floor)
    if verbose:
        print(f"  {len(keys)} units analyzed, {np.sum(is_hd)} HD cells "
              f"(p<0.05 & MVL>{mvl_floor})")

    # tuning curves per state (60 bins over the circle), computed as
    # spike counts first so zero-occupancy bins and per-state spike
    # counts are explicit
    def tc(ep):
        if ep.tot_length() < 10:
            return None, None
        counts = nap.compute_tuning_curves(units_f, hd, bins=60,
                                           range=(0, 2 * np.pi), epochs=ep,
                                           feature_names=["hd"],
                                           return_counts=True)
        occ = np.asarray(counts.attrs["occupancy"], dtype=float) / counts.attrs["fs"]
        with np.errstate(invalid="ignore", divide="ignore"):
            rates_tc = np.where(occ[None, :] > 0,
                                counts.values / occ[None, :], np.nan)
        return rates_tc, counts.values.sum(axis=1)

    tc_wake, cnt_wake = tc(wake)
    tc_rem, cnt_rem = tc(rem)
    tc_nrem, cnt_nrem = tc(nrem)

    # preferred direction per state, from tuning-curve resultant
    # (NaN bins contribute 0; resultant is meaningful only with enough
    # spikes in the state, enforced by the caller via cnt_*)
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False) + 2 * np.pi / 120

    def pref_from_tc(tc_values):
        if tc_values is None:
            return (np.full(len(keys), np.nan), np.full(len(keys), np.nan))
        vals = np.nan_to_num(tc_values, nan=0.0)
        z = (vals * np.exp(1j * theta[None, :])).sum(axis=1)
        denom = vals.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(z) / denom
        r[denom == 0] = np.nan
        return np.angle(z) % (2 * np.pi), r

    pref_wake, mvl_tc_wake = pref_from_tc(tc_wake)
    pref_rem, mvl_tc_rem = pref_from_tc(tc_rem)
    pref_nrem, mvl_tc_nrem = pref_from_tc(tc_nrem)

    return dict(keys=np.array(keys), rates=rates[rates > 0.1],
                n_spikes=n_spikes, mvl=mvl, pref=pref, null95=null95,
                pval=pval, is_hd=is_hd,
                pref_wake=pref_wake, mvl_tc_wake=mvl_tc_wake,
                pref_rem=pref_rem, mvl_tc_rem=mvl_tc_rem,
                pref_nrem=pref_nrem, mvl_tc_nrem=mvl_tc_nrem,
                cnt_wake=cnt_wake, cnt_rem=cnt_rem, cnt_nrem=cnt_nrem,
                tc_wake=tc_wake, tc_rem=tc_rem, tc_nrem=tc_nrem,
                theta=theta,
                wake_dur=wake.tot_length(), rem_dur=rem.tot_length(),
                nrem_dur=nrem.tot_length())


def state_correlations(units_f, hd_cell_keys, epochs, bin_size=0.1,
                       min_rate=0.05):
    """Pairwise Pearson correlation matrix of HD cells within a state.

    Spike trains are binned at bin_size seconds within the epochs and
    correlated across cells. Cells with mean rate below min_rate in the
    state are excluded (returned as NaN rows/columns).
    """
    n = len(hd_cell_keys)
    if n < 2 or epochs.tot_length() < 30:
        return np.full((n, n), np.nan)
    group = units_f[list(hd_cell_keys)]
    counts = group.count(bin_size, epochs)
    rates = np.asarray(counts.values, dtype=float) / bin_size
    keep = rates.mean(axis=0) >= min_rate
    C = np.full((n, n), np.nan)
    if keep.sum() >= 2:
        Ck = np.corrcoef(rates[:, keep].T)
        C[np.ix_(keep, keep)] = Ck
    return C


def circ_corr(ang1, ang2):
    """Circular correlation of paired angles (Jammalamadaka-SenGupta)."""
    mask = ~np.isnan(ang1) & ~np.isnan(ang2)
    a1, a2 = ang1[mask], ang2[mask]
    if len(a1) < 3:
        return np.nan, mask.sum()
    a1c = a1 - np.angle(np.exp(1j * a1).mean())
    a2c = a2 - np.angle(np.exp(1j * a2).mean())
    num = np.sum(np.sin(a1c) * np.sin(a2c))
    den = np.sqrt(np.sum(np.sin(a1c) ** 2) * np.sum(np.sin(a2c) ** 2))
    return num / den, mask.sum()
