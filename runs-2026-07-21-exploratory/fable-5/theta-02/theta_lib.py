"""Shared helpers for the theta entrainment / phase precession analysis of DANDI:000044.

Dataset: Grosmark & Buzsaki (2016), "Diversity in neural firing dynamics supports both
rigid and learned hippocampal sequences" (hc-11). Rats run back and forth on a 1.6 m
linear maze; dorsal CA1 silicon probe recordings with spike-sorted units (excitatory /
inhibitory labels), linearized position, and 1250 Hz LFP on 128 channels.

All access is streamed from the DANDI S3 bucket with remfile + a local disk cache; the
LFP dataset is chunked one channel at a time, so pulling a single channel over the maze
epoch reads only a few MB.
"""

import numpy as np
import h5py
import remfile
import requests
import scipy.signal
import scipy.stats
import scipy.ndimage
import pynapple as nap

nap.nap_config.suppress_conversion_warnings = True

DANDISET = "000044"
CACHE_DIR = "/tmp/remfile_cache_000044"

# Sessions used in this analysis (subject / session id embedded in the asset path).
SESSIONS = [
    "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb",
    "sub-Achilles/sub-Achilles_ses-Achilles-11012013_behavior+ecephys.nwb",
    "sub-Cicero/sub-Cicero_ses-Cicero-09102014_behavior+ecephys.nwb",
    "sub-Gatsby/sub-Gatsby_ses-Gatsby-08022013_behavior+ecephys.nwb",
]

THETA_BAND = (6.0, 10.0)
LFP_FS = 1250.0


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------
def asset_url(path):
    """Resolve a dandiset asset path to a streamable download URL."""
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/",
        params={"path": path, "page_size": 10},
    )
    r.raise_for_status()
    results = r.json()["results"]
    match = [a for a in results if a["path"] == path]
    return f"https://api.dandiarchive.org/api/assets/{match[0]['asset_id']}/download/"


def open_session(path):
    """Open an NWB file from DANDI by streaming. Returns the raw h5py file handle."""
    url = asset_url(path)
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    return h5py.File(rf, "r")


def load_behavior_and_spikes(h5):
    """Pull spikes, linearized position and the maze epoch out of an open NWB file.

    Returns a dict with:
      spikes    : TsGroup with 'cell_type' and 'location' metadata
      position  : Tsd, linearized position along the maze (cm)
      maze_ep   : IntervalSet, the running (maze) epoch
    """
    # --- epochs -----------------------------------------------------------------------
    ep = h5["intervals/epochs"]
    labels = [s.decode() if isinstance(s, bytes) else s for s in ep["label"][:]]
    i_maze = labels.index("MazeEpoch")
    maze_ep = nap.IntervalSet(
        start=float(ep["start_time"][i_maze]), end=float(ep["stop_time"][i_maze])
    )

    # --- linearized position ----------------------------------------------------------
    # Note on timing: the `rate` attribute in these files is actually the sampling
    # *period* (0.0256 s, i.e. 39.06 Hz). len(data) * period reproduces the maze epoch
    # duration exactly, which is how the interpretation is verified below.
    pos_key = [k for k in h5["processing/behavior"] if "LinearizedPosition" in k][0]
    pos_grp = h5[f"processing/behavior/{pos_key}"]
    series = pos_grp[list(pos_grp.keys())[0]]
    pos_data = series["data"][:, 0].astype(float)
    conversion = float(series["data"].attrs.get("conversion", 1.0))
    t0 = float(series["starting_time"][()])
    rate_attr = float(series["starting_time"].attrs["rate"])
    dt = rate_attr if rate_attr < 1.0 else 1.0 / rate_attr
    maze_dur = maze_ep.end[0] - maze_ep.start[0]
    assert abs(len(pos_data) * dt - maze_dur) < 1.0, "position timing does not match maze epoch"
    pos_t = t0 + np.arange(len(pos_data)) * dt
    # SpatialSeries data are in meters here; convert to centimetres.
    position = nap.Tsd(t=pos_t, d=pos_data * conversion * 100.0)

    # --- spikes -----------------------------------------------------------------------
    u = h5["units"]
    idx = u["spike_times_index"][:]
    st = u["spike_times"]
    starts = np.concatenate([[0], idx[:-1]])
    spike_dict = {i: st[a:b] for i, (a, b) in enumerate(zip(starts, idx))}
    cell_type = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in u["cell_type"][:]]
    )
    location = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in u["location"][:]]
    )
    shank = u["shank_id"][:]
    spikes = nap.TsGroup(
        spike_dict,
        cell_type=cell_type,
        location=location,
        shank_id=shank,
    )
    return dict(spikes=spikes, position=position, maze_ep=maze_ep)


def read_lfp_channel(h5, channel, start_time, stop_time):
    """Read a single LFP channel over [start_time, stop_time] as a pynapple Tsd."""
    ds = h5["processing/ecephys/LFP/LFP/data"]
    grp = h5["processing/ecephys/LFP/LFP"]
    t0 = float(grp["starting_time"][()])
    fs = float(grp["starting_time"].attrs["rate"])
    conv = float(ds.attrs.get("conversion", 1.0))
    i0 = max(0, int(np.floor((start_time - t0) * fs)))
    i1 = min(ds.shape[0], int(np.ceil((stop_time - t0) * fs)))
    d = ds[i0:i1, channel].astype(np.float64) * conv
    t = t0 + np.arange(i0, i1) / fs
    return nap.Tsd(t=t, d=d)


def pick_theta_channel(h5, maze_ep, candidate_channels, probe_seconds=180.0):
    """Choose the LFP channel with the strongest theta/delta power ratio.

    A short probe window in the middle of the maze epoch is used so that only a few MB
    are streamed per candidate channel.
    """
    mid = 0.5 * (maze_ep.start[0] + maze_ep.end[0])
    t0, t1 = mid - probe_seconds / 2, mid + probe_seconds / 2
    scores = {}
    for ch in candidate_channels:
        lfp = read_lfp_channel(h5, ch, t0, t1)
        f, pxx = scipy.signal.welch(lfp.values, fs=LFP_FS, nperseg=int(4 * LFP_FS))
        theta = pxx[(f >= 6) & (f <= 10)].mean()
        delta = pxx[(f >= 2) & (f <= 4)].mean()
        scores[ch] = theta / delta
    best = max(scores, key=scores.get)
    return best, scores


# --------------------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------------------
def theta_phase_from_lfp(lfp, band=THETA_BAND, fs=LFP_FS):
    """Bandpass-filter the LFP and return (filtered signal, instantaneous phase, amplitude).

    Phase convention: 0 / 2*pi at the peak of the theta oscillation, pi at the trough.
    """
    filt = nap.apply_bandpass_filter(lfp, cutoff=band, fs=fs)
    analytic = scipy.signal.hilbert(filt.values)
    phase = np.angle(analytic) % (2 * np.pi)
    amp = np.abs(analytic)
    return (
        filt,
        nap.Tsd(t=filt.t, d=phase, time_support=filt.time_support),
        nap.Tsd(t=filt.t, d=amp, time_support=filt.time_support),
    )


# --------------------------------------------------------------------------------------
# Behaviour: laps and running epochs
# --------------------------------------------------------------------------------------
def get_laps(position, min_dur=0.5, min_extent=80.0):
    """Extract single track traversals from the linearized position.

    The linearized SpatialSeries is only defined while the animal is running along the
    track (it is NaN in the reward areas and during off-track behaviour), so every
    contiguous run of finite samples is one candidate traversal. Traversals shorter than
    `min_dur` seconds or covering less than `min_extent` cm of track are discarded.

    Returns (laps, position_clean) where `laps` is an IntervalSet carrying a `direction`
    metadata column ('right' = increasing position, 'left' = decreasing).
    """
    v = position.values
    finite = np.isfinite(v)
    edges = np.diff(finite.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1
    if finite[0]:
        starts = np.r_[0, starts]
    if finite[-1]:
        stops = np.r_[stops, len(v)]

    lap_start, lap_end, direction = [], [], []
    for a, b in zip(starts, stops):
        seg_t, seg_v = position.t[a:b], v[a:b]
        if seg_t[-1] - seg_t[0] < min_dur:
            continue
        if abs(seg_v[-1] - seg_v[0]) < min_extent:
            continue
        lap_start.append(seg_t[0])
        lap_end.append(seg_t[-1])
        direction.append("right" if seg_v[-1] > seg_v[0] else "left")

    laps = nap.IntervalSet(
        start=np.array(lap_start),
        end=np.array(lap_end),
        metadata={"direction": np.array(direction)},
    )
    position_clean = nap.Tsd(t=position.t[finite], d=v[finite])
    return laps, position_clean


def lap_speed(position_clean, laps, smooth_sigma=3):
    """Instantaneous running speed (cm/s), computed independently within each lap.

    Each lap is differentiated on its own so that the gaps between traversals never enter
    the derivative; edge-preserving smoothing avoids the artefactual velocity spikes that
    zero-padded convolution produces at lap boundaries.
    """
    t_all, v_all = [], []
    for a, b in zip(laps.start, laps.end):
        seg = position_clean.get(a, b)
        if len(seg) < 5:
            continue
        smoothed = scipy.ndimage.gaussian_filter1d(seg.values, smooth_sigma, mode="nearest")
        v_all.append(np.abs(np.gradient(smoothed, seg.t)))
        t_all.append(seg.t)
    t, v = np.concatenate(t_all), np.concatenate(v_all)
    t, keep = np.unique(t, return_index=True)  # laps can share a boundary sample
    return nap.Tsd(t=t, d=v[keep], time_support=laps)


# --------------------------------------------------------------------------------------
# Place fields
# --------------------------------------------------------------------------------------
BIN_WIDTH_CM = 4.0


def track_geometry(position, bin_width=BIN_WIDTH_CM):
    """Track extent and bin count for a session.

    Two of the sessions in this dandiset use the 1.6 m linear maze and two use a ~2.9 m
    circular maze, so the range cannot be hard-coded: positions beyond a fixed range would
    be silently dropped by the histogram.
    """
    top = float(np.ceil(np.nanmax(position.values) / bin_width) * bin_width)
    return (0.0, top), int(round(top / bin_width))


def place_fields(spikes, position, ep, track_range, bins, smooth_bins=1.0):
    """1D firing-rate maps over the epochs `ep`, smoothed across position bins."""
    tc = nap.compute_tuning_curves(
        spikes, position, bins=bins, range=track_range, epochs=ep, feature_names=["position"]
    )
    rates = np.asarray(tc.values, dtype=float)
    rates = np.nan_to_num(rates, nan=0.0)
    rates = scipy.ndimage.gaussian_filter1d(rates, smooth_bins, axis=1, mode="nearest")
    centers = np.asarray(tc.coords["position"].values, dtype=float)
    occupancy = np.asarray(tc.attrs["occupancy"], dtype=float)
    return rates, centers, occupancy


def spatial_information(rates, occupancy):
    """Skaggs spatial information in bits per spike, one value per unit."""
    p = occupancy / occupancy.sum()
    mean_rate = (rates * p[None, :]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rates / mean_rate[:, None]
        term = np.where(rates > 0, rates / mean_rate[:, None] * np.log2(ratio), 0.0)
    return np.nansum(term * p[None, :], axis=1), mean_rate


def field_bounds(rate_map, centers, frac=0.2):
    """Boundaries of the main place field: contiguous bins above `frac` of the peak."""
    peak_bin = int(np.argmax(rate_map))
    thresh = frac * rate_map[peak_bin]
    lo = peak_bin
    while lo > 0 and rate_map[lo - 1] >= thresh:
        lo -= 1
    hi = peak_bin
    while hi < len(rate_map) - 1 and rate_map[hi + 1] >= thresh:
        hi += 1
    bw = centers[1] - centers[0]
    return centers[lo] - bw / 2, centers[hi] + bw / 2, centers[peak_bin]


def split_half_stability(spikes, position, laps_dir, run_ep, track_range, bins):
    """Correlation between rate maps built from odd and even laps of one direction."""
    odd = nap.IntervalSet(start=laps_dir.start[::2], end=laps_dir.end[::2]).intersect(run_ep)
    even = nap.IntervalSet(start=laps_dir.start[1::2], end=laps_dir.end[1::2]).intersect(run_ep)
    maps = []
    for sub in (odd, even):
        m, _, _ = place_fields(spikes, position, sub, track_range, bins)
        maps.append(m)
    out = []
    for i in range(maps[0].shape[0]):
        a, b = maps[0][i], maps[1][i]
        if a.std() == 0 or b.std() == 0:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(a, b)[0, 1])
    return np.array(out)


# --------------------------------------------------------------------------------------
# Circular statistics
# --------------------------------------------------------------------------------------
def analyze_session(
    path,
    speed_thresh=10.0,
    si_thresh=0.5,
    peak_thresh=1.0,
    stability_thresh=0.4,
    min_field_spikes=40,
    min_field_width=15.0,
    max_field_width=120.0,
    exclude_edge_fields=True,
    edge_margin=15.0,
    verbose=True,
):
    """Full single-session pipeline: laps, place fields, theta locking, phase precession.

    Returns a dict of everything the figures need. Nothing is cached in memory beyond the
    single selected LFP channel, so this is cheap enough to run over several sessions.
    """
    h5 = open_session(path)
    session_id = h5["general/session_id"][()]
    session_id = session_id.decode() if isinstance(session_id, bytes) else session_id
    d = load_behavior_and_spikes(h5)
    spikes, position, maze_ep = d["spikes"], d["position"], d["maze_ep"]

    laps, pos = get_laps(position)
    speed = lap_speed(pos, laps)
    run_ep = speed.threshold(speed_thresh, method="above").time_support
    run_ep = run_ep.drop_short_intervals(0.2)
    track_range, n_bins = track_geometry(pos)
    # Only directions the animal actually ran are analysed: the circular-maze sessions are
    # traversed in one direction only, so the other direction has no laps.
    dir_ep = {
        dr: laps[laps.direction == dr].intersect(run_ep)
        for dr in ("right", "left")
        if (laps.direction == dr).sum() >= 5
    }

    # --- LFP theta -------------------------------------------------------------------
    groups = np.array(
        [s.decode() for s in h5["general/extracellular_ephys/electrodes/group_name"][:]]
    )
    bad = h5["general/extracellular_ephys/electrodes/bad_electrode"][:]
    candidates = []
    for g in np.unique(groups):
        idx = np.where((groups == g) & (~bad))[0]
        if len(idx):
            candidates.append(int(idx[len(idx) // 2]))
    best_ch, ch_scores = pick_theta_channel(h5, maze_ep, candidates)
    lfp = read_lfp_channel(h5, best_ch, maze_ep.start[0], maze_ep.end[0])
    theta_filt, theta_phase, theta_amp = theta_phase_from_lfp(lfp)

    pyr = spikes.getby_category("cell_type")["excitatory"]

    # --- theta phase locking of every unit during running ----------------------------
    phases_at_spikes = spikes.restrict(run_ep).value_from(theta_phase)
    locking = {}
    for uid in spikes.index:
        ph = phases_at_spikes[uid].values
        mu, r, p = rayleigh(ph)
        locking[uid] = dict(
            mu=mu, mrl=r, p=p, n=len(ph), cell_type=str(spikes.cell_type[uid])
        )

    # --- directional place fields ----------------------------------------------------
    fields = []
    maps = {}
    for direction, ep in dir_ep.items():
        rates, centers, occ = place_fields(pyr, pos, ep, track_range, n_bins)
        si, mean_rate = spatial_information(rates, occ)
        stab = split_half_stability(
            pyr, pos, laps[laps.direction == direction], run_ep, track_range, n_bins
        )
        peak = rates.max(axis=1)
        is_place = (peak >= peak_thresh) & (si >= si_thresh) & (stab >= stability_thresh)
        maps[direction] = dict(
            rates=rates, centers=centers, occ=occ, si=si, stab=stab, peak=peak,
            is_place=is_place, unit_ids=np.array(pyr.index),
        )

        pos_ep = pos.restrict(ep)
        for k, uid in enumerate(pyr.index):
            if not is_place[k]:
                continue
            lo, hi, pk = field_bounds(rates[k], centers)
            width = hi - lo
            if not (min_field_width <= width <= max_field_width):
                continue
            # A field whose peak sits at the very end of the track is cut off by the
            # reward area, so the rat never traverses it fully and "position in field"
            # is not comparable with the other cells; drop those.
            if exclude_edge_fields and (
                pk - track_range[0] < edge_margin or track_range[1] - pk < edge_margin
            ):
                continue
            st = pyr[uid].restrict(ep)
            if len(st) < min_field_spikes:
                continue
            sp_pos = pos_ep.interpolate(st).values
            sp_ph = st.value_from(theta_phase).values
            inside = (sp_pos >= lo) & (sp_pos <= hi) & np.isfinite(sp_pos)
            if inside.sum() < min_field_spikes:
                continue
            x = (sp_pos[inside] - lo) / width  # 0 = field entry, 1 = field exit
            if direction == "left":
                x = 1.0 - x  # travelled distance always increases with x
            phi = sp_ph[inside]
            reg = circ_lin_regress(x, phi)
            fields.append(
                dict(
                    session=session_id, unit=int(uid), direction=direction,
                    lo=lo, hi=hi, peak_pos=pk, width=width, peak_rate=float(peak[k]),
                    si=float(si[k]), n_spikes=int(inside.sum()),
                    x=x, phase=phi, mrl=locking[uid]["mrl"], **reg,
                )
            )

    if verbose:
        n_sig = sum(1 for f in fields if f["p"] < 0.05 and f["slope"] < 0)
        print(
            f"{session_id}: {len(laps)} laps, {len(pyr)} pyramidal cells, "
            f"{sum(maps[d]['is_place'].sum() for d in maps)} directional place fields "
            f"({len(fields)} analysable), {n_sig} with significant negative phase-position slope"
        )

    return dict(
        session=session_id, path=path, h5=h5, spikes=spikes, pyr=pyr, position=pos,
        laps=laps, run_ep=run_ep, dir_ep=dir_ep, speed=speed, maze_ep=maze_ep,
        lfp=lfp, theta_filt=theta_filt, theta_phase=theta_phase, theta_amp=theta_amp,
        best_ch=best_ch, ch_scores=ch_scores, locking=locking, maps=maps, fields=fields,
        track_range=track_range, n_bins=n_bins,
    )


def rayleigh(phases):
    """Rayleigh test for circular non-uniformity. Returns (mean_phase, MRL, p-value)."""
    n = len(phases)
    if n < 3:
        return np.nan, np.nan, np.nan
    c, s = np.cos(phases).mean(), np.sin(phases).mean()
    r = np.hypot(c, s)
    mu = np.arctan2(s, c) % (2 * np.pi)
    z = n * r**2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * r) ** 2)) - (1 + 2 * n))
    return mu, r, p


def circ_lin_regress(x, phi, slope_range=(-2.0, 2.0), n_slopes=2001, min_abs_slope=0.05):
    """Circular-linear regression of phase phi on linear variable x (Kempter et al. 2012).

    Finds the slope a (cycles per unit x) maximising the resultant length of
    phi - 2*pi*a*x, then reports the circular-linear correlation coefficient rho and its
    asymptotic p-value.

    Returns dict with slope (rad per unit x), phase offset, rho, p.
    """
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    n = len(x)
    if n < 8:
        return dict(slope=np.nan, phi0=np.nan, rho=np.nan, p=np.nan, n=n)

    # Slopes are searched in cycles per unit x. Slopes of essentially zero are excluded
    # because the circular-linear correlation is undefined there (the circularised
    # regressor has no spread); the same grid is used for real and shuffled data.
    slopes = np.linspace(slope_range[0], slope_range[1], n_slopes)
    slopes = slopes[np.abs(slopes) >= min_abs_slope]
    # R(a) = |mean(exp(i*(phi - 2*pi*a*x)))|
    ang = phi[None, :] - 2 * np.pi * slopes[:, None] * x[None, :]
    R = np.abs(np.exp(1j * ang).mean(axis=1))
    a = slopes[np.argmax(R)]
    phi0 = np.angle(np.exp(1j * (phi - 2 * np.pi * a * x)).mean()) % (2 * np.pi)

    # circular-linear correlation (Kempter et al. 2012, eq. 3)
    theta_c = (2 * np.pi * np.abs(a) * x) % (2 * np.pi)
    phi_bar = np.angle(np.exp(1j * phi).mean())
    theta_bar = np.angle(np.exp(1j * theta_c).mean())
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta_c - theta_bar))
    den = np.sqrt(
        np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta_c - theta_bar) ** 2)
    )
    rho = num / den if den > 0 else np.nan

    # asymptotic significance test
    lam20 = np.mean(np.sin(phi - phi_bar) ** 2)
    lam02 = np.mean(np.sin(theta_c - theta_bar) ** 2)
    lam22 = np.mean(np.sin(phi - phi_bar) ** 2 * np.sin(theta_c - theta_bar) ** 2)
    z = rho * np.sqrt(n * lam20 * lam02 / lam22) if lam22 > 0 else np.nan
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return dict(slope=2 * np.pi * a, phi0=phi0, rho=rho, p=p, n=n)
