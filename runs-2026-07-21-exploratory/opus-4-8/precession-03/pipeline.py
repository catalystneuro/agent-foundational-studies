"""Reusable phase-precession pipeline for DANDI:000044 (Grosmark & Buzsaki).

Encapsulates the steps validated interactively so they can be run on any
session in the dandiset. All heavy reads are streamed with remfile caching.
"""
import h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import norm

LFP_RATE = 1250.0
THETA_BAND = (6, 12)
SPEED_THRESH = 0.03
MIN_RUN_DUR = 0.5
N_BINS = 50
MAZE_LEN = 1.6
FIELD_FRAC = 0.25
MIN_SPIKES = 30

def _bandpass(x, lo, hi, fs=LFP_RATE):
    b, a = butter(3, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def load_session(s3_url, cache='/tmp/remfile_cache'):
    """Stream one session; return dict with position, theta phase, spikes."""
    h5 = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache(cache)), 'r')
    nwb = NWBHDF5IO(file=h5).read()

    ep = nwb.intervals['epochs'].to_dataframe()
    maze = ep[ep['label'] == 'MazeEpoch'].iloc[0]
    t0, t1 = float(maze['start_time']), float(maze['stop_time'])

    beh = nwb.processing['behavior']
    lin_key = [k for k in beh.data_interfaces if 'Linearized' in k][0]
    lin = list(beh[lin_key].spatial_series.values())[0]
    pos_data = lin.data[:].astype(float).squeeze()
    pos_t = float(lin.starting_time) + np.arange(pos_data.size)/float(lin.rate)
    good = ~np.isnan(pos_data)
    pos_data = np.interp(pos_t, pos_t[good], pos_data[good])
    # normalize to [0, MAZE_LEN] in case a session linearizes to a different scale
    pos_data = (pos_data - pos_data.min())
    if pos_data.max() > 0:
        pos_data = pos_data / pos_data.max() * MAZE_LEN

    u = nwb.units
    cell_type = u['cell_type'][:]
    spikes = {}
    for i in np.where(np.array([c == 'excitatory' for c in cell_type]))[0]:
        st = u['spike_times'][i]
        st = st[(st >= t0) & (st <= t1)]
        if st.size:
            spikes[int(i)] = st

    # best-theta LFP channel from a 120 s window, then full-maze read
    es = nwb.processing['ecephys']['LFP'].electrical_series['LFP']
    conv = float(es.conversion)
    w0 = int((t0+200)*LFP_RATE); w1 = w0 + int(120*LFP_RATE)
    snip = es.data[w0:w1, :].astype(float)*conv
    ratios = [np.sqrt(np.mean(_bandpass(snip[:, c], 6, 12)**2)) /
              np.sqrt(np.mean(_bandpass(snip[:, c], 2, 4)**2)) for c in range(snip.shape[1])]
    ch = int(np.argmax(ratios))
    s0 = int(np.floor(t0*LFP_RATE)); s1 = int(np.ceil(t1*LFP_RATE))
    lfp = es.data[s0:s1, ch].astype(float)*conv
    lfp_t = np.arange(s0, s1)/LFP_RATE
    phase = np.angle(hilbert(_bandpass(lfp, *THETA_BAND)))

    return dict(t0=t0, t1=t1, pos_t=pos_t, pos_data=pos_data,
                lfp_t=lfp_t, theta_phase=phase, spikes=spikes,
                best_ch=ch, theta_ratio=ratios[ch])

def circ_lin_regress(x, phi):
    """Kempter et al. 2012 circular-linear regression. x in [0,1]."""
    x = np.asarray(x); phi = np.asarray(phi)
    a_grid = np.linspace(-4*np.pi, 4*np.pi, 2001)
    R = [np.abs(np.mean(np.exp(1j*(phi - a*x)))) for a in a_grid]
    a = a_grid[int(np.argmax(R))]
    phi0 = np.angle(np.mean(np.exp(1j*(phi - a*x))))
    theta = (a*x) % (2*np.pi)
    pb = np.angle(np.mean(np.exp(1j*phi))); tb = np.angle(np.mean(np.exp(1j*theta)))
    sp = np.sin(phi-pb); st = np.sin(theta-tb)
    rho = np.sum(sp*st)/np.sqrt(np.sum(sp**2)*np.sum(st**2))
    n = x.size
    l20 = np.mean(sp**2); l02 = np.mean(st**2); l22 = np.mean(sp**2*st**2)
    z = rho*np.sqrt(n*l20*l02/l22)
    pval = 2*(1-norm.cdf(abs(z)))
    return a/(2*np.pi), phi0, np.sign(a)*abs(rho), pval

def _field_bounds(rate, bins, peak_pos):
    ipk = int(np.argmin(np.abs(bins-peak_pos)))
    thr = FIELD_FRAC*rate[ipk]; lo = hi = ipk
    while lo > 0 and rate[lo-1] >= thr: lo -= 1
    while hi < len(rate)-1 and rate[hi+1] >= thr: hi += 1
    return bins[lo], bins[hi]

def analyze_session(sess):
    """Full place-field + phase-precession analysis for one loaded session."""
    pos = nap.Tsd(t=sess['pos_t'], d=sess['pos_data'])
    theta_ph = nap.Tsd(t=sess['lfp_t'], d=sess['theta_phase'])
    spikes = {u: nap.Ts(t=st) for u, st in sess['spikes'].items()}
    units = nap.TsGroup(spikes)

    pos_s = pos.smooth(0.1)
    vel = np.gradient(pos_s.values, sess['pos_t'])
    speed = nap.Tsd(t=sess['pos_t'], d=np.abs(vel))
    run_ep = speed.threshold(SPEED_THRESH, 'above').time_support
    run_ep = run_ep.drop_short_intervals(MIN_RUN_DUR).merge_close_intervals(0.2)
    direction = nap.Tsd(t=sess['pos_t'], d=np.sign(vel))
    right = direction.threshold(0.0, 'above').time_support.intersect(run_ep).drop_short_intervals(0.3)
    left = direction.threshold(0.0, 'below').time_support.intersect(run_ep).drop_short_intervals(0.3)

    tc = {'right': nap.compute_1d_tuning_curves(units, pos, N_BINS, ep=right, minmax=(0, MAZE_LEN)),
          'left':  nap.compute_1d_tuning_curves(units, pos, N_BINS, ep=left,  minmax=(0, MAZE_LEN))}
    bins = tc['right'].index.values
    eps = {'right': right, 'left': left}

    results = []
    for uid in units.keys():
        pk = {k: tc[k][uid].max() for k in tc}
        bdir = 'right' if pk['right'] >= pk['left'] else 'left'
        if pk[bdir] < 5:  # not a place cell
            continue
        rate = tc[bdir][uid].values
        ppos = tc[bdir][uid].idxmax()
        f0, f1 = _field_bounds(rate, bins, ppos)
        if f1-f0 < 0.08:
            continue
        sp = spikes[uid].restrict(eps[bdir]).value_from(pos)
        m = (sp.values >= f0) & (sp.values <= f1)
        if m.sum() < MIN_SPIKES:
            continue
        st, spos = sp.index.values[m], sp.values[m]
        ph = np.mod(np.interp(st, theta_ph.index.values, np.unwrap(theta_ph.values)), 2*np.pi)
        xn = (spos-f0)/(f1-f0)
        if bdir == 'left':
            xn = 1-xn
        slope, phi0, rho, pval = circ_lin_regress(xn, ph)
        results.append(dict(uid=uid, bdir=bdir, n=int(m.sum()), slope=slope,
                            phi0=phi0, rho=rho, pval=pval, xn=xn, phase=ph, peak=pk[bdir]))
    return results
