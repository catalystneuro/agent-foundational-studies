"""Load and cache the data needed for phase-precession analysis.

Dataset: DANDI:000044 (Grosmark & Buzsaki), session sub-Achilles ses-10252013.
We stream the ~8.7 GB raw NWB file from S3 with remfile chunk caching and pull
out only what the analysis needs:
  - spike times of CA1 pyramidal cells (TsGroup)
  - linearized position on the 1.6 m linear maze during the MazeEpoch
  - one CA1 LFP channel over the MazeEpoch (chosen for strongest theta)

Everything is cached to loaded_data.npz so downstream scripts run offline.
"""
import h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, hilbert

S3_URL = 'https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028'
LFP_RATE = 1250.0

def theta_power(x, fs=LFP_RATE):
    """Theta (6-12 Hz) / delta (2-4 Hz) power ratio via band-pass RMS."""
    def bp(lo, hi):
        b, a = butter(3, [lo/(fs/2), hi/(fs/2)], btype='band')
        return filtfilt(b, a, x)
    th = np.sqrt(np.mean(bp(6, 12) ** 2))
    de = np.sqrt(np.mean(bp(2, 4) ** 2))
    return th / de

def main():
    print("Opening remote NWB file (streaming)...")
    h5 = h5py.File(remfile.File(S3_URL, disk_cache=remfile.DiskCache('/tmp/remfile_cache')), 'r')
    nwb = NWBHDF5IO(file=h5).read()

    # --- epochs: find the maze (running) epoch ---
    ep = nwb.intervals['epochs'].to_dataframe()
    maze = ep[ep['label'] == 'MazeEpoch'].iloc[0]
    t0, t1 = float(maze['start_time']), float(maze['stop_time'])
    print(f"MazeEpoch: {t0:.1f}-{t1:.1f} s  ({t1-t0:.0f} s)")

    # --- linearized position ---
    lin = nwb.processing['behavior']['1.6mLinearMazeLinearizedPosition'].spatial_series[
        '1.6mLinearMazeLinearizedTimeSeries']
    pos_rate = float(lin.rate)
    pos_t0 = float(lin.starting_time)
    pos_data = lin.data[:].astype(float).squeeze()
    pos_t = pos_t0 + np.arange(pos_data.size) / pos_rate
    print(f"Position: {pos_data.size} samples @ {pos_rate:.2f} Hz, "
          f"t={pos_t[0]:.1f}-{pos_t[-1]:.1f}s, range={np.nanmin(pos_data):.2f}-{np.nanmax(pos_data):.2f} m")

    # --- units: CA1 pyramidal (excitatory) cells ---
    u = nwb.units
    cell_type = u['cell_type'][:]
    location = u['location'][:]
    keep = np.array([ct == 'excitatory' for ct in cell_type])
    spike_times = {}
    for i in np.where(keep)[0]:
        st = u['spike_times'][i]
        st = st[(st >= t0) & (st <= t1)]
        if st.size > 0:
            spike_times[int(i)] = st
    print(f"Kept {len(spike_times)} excitatory CA1 units with spikes in maze epoch")

    # --- select best-theta LFP channel from a 120 s running window ---
    es = nwb.processing['ecephys']['LFP'].electrical_series['LFP']
    conv = float(es.conversion)
    win_s0 = int((t0 + 200) * LFP_RATE)      # 200 s into maze
    win_s1 = win_s0 + int(120 * LFP_RATE)    # 120 s window
    print(f"Scanning theta power across 128 channels on a 120 s window...")
    snippet = es.data[win_s0:win_s1, :].astype(float) * conv  # volts
    ratios = np.array([theta_power(snippet[:, c]) for c in range(snippet.shape[1])])
    best_ch = int(np.argmax(ratios))
    print(f"Best theta channel = {best_ch} (theta/delta ratio {ratios[best_ch]:.2f})")

    # --- read full-maze LFP for the chosen channel ---
    lfp_s0 = int(np.floor(t0 * LFP_RATE))
    lfp_s1 = int(np.ceil(t1 * LFP_RATE))
    print(f"Reading LFP channel {best_ch} over maze epoch ({lfp_s1-lfp_s0} samples)...")
    lfp = es.data[lfp_s0:lfp_s1, best_ch].astype(float) * conv
    lfp_t = np.arange(lfp_s0, lfp_s1) / LFP_RATE

    np.savez_compressed(
        'loaded_data.npz',
        t0=t0, t1=t1,
        pos_data=pos_data, pos_t=pos_t, pos_rate=pos_rate,
        lfp=lfp, lfp_t=lfp_t, lfp_rate=LFP_RATE, best_ch=best_ch,
        theta_ratios=ratios,
        unit_ids=np.array(list(spike_times.keys())),
        # store spikes as an object array of arrays
        spike_times=np.array(list(spike_times.values()), dtype=object),
        unit_location=np.array([location[i] for i in spike_times.keys()]),
    )
    print("Saved loaded_data.npz")

if __name__ == '__main__':
    main()
