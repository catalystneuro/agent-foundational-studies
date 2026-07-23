import h5py
import remfile
import numpy as np
import pynapple as nap
from pynwb import NWBHDF5IO

url = open('s3_url.txt').read().strip()
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")

lin_grp2 = h5f['processing']['behavior']['1.6mLinearMazeLinearizedPosition']
lss2 = lin_grp2['1.6mLinearMazeLinearizedTimeSeries']
lin_grp = h5f['processing']['behavior']['1.6mLinearMazePosition']
lss = lin_grp['1.6mLinearMazeSpatialSeries']
starting_time = lss['starting_time'][()]
rate = lss['starting_time'].attrs['rate'] * 1000
data = lss2['data'][:, 0]
t = starting_time + np.arange(len(data)) / rate
good = ~np.isnan(data)
lin_pos = nap.Tsd(t=t[good], d=data[good])
maze_ep = nap.IntervalSet(start=18079.5, end=20147.0)
lin_pos_maze = lin_pos.restrict(maze_ep)

tt = lin_pos_maze.t; xx = lin_pos_maze.values
dtv = np.diff(tt); dxv = np.diff(xx)
cont_mask = dtv < 0.15
vel = dxv[cont_mask] / dtv[cont_mask]
tmid = (tt[:-1] + dtv/2)[cont_mask]
vel_tsd = nap.Tsd(t=tmid, d=vel)

io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
exc = units[units.cell_type == "excitatory"].getby_threshold("rate", 0.2)

for thr in [0.1, 0.15, 0.2, 0.3]:
    outbound_ep = vel_tsd.threshold(thr, method="above").time_support.drop_short_intervals(0.3)
    spikes = exc.restrict(outbound_ep)
    pos = lin_pos_maze.restrict(outbound_ep)
    tc = nap.compute_tuning_curves(spikes, pos, bins=40, feature_names=["position"])
    occ = tc.attrs["occupancy"]
    rates = tc.values
    mean_rate = np.nansum(rates*occ[None,:],axis=1)/np.nansum(occ)
    p = occ/np.nansum(occ)
    with np.errstate(divide='ignore', invalid='ignore'):
        si_terms = p[None,:]*(rates/mean_rate[:,None])*np.log2(rates/mean_rate[:,None])
    si_terms = np.nan_to_num(si_terms, nan=0.0, posinf=0.0, neginf=0.0)
    si = np.nansum(si_terms, axis=1)
    peak_idx = np.nanargmax(rates, axis=1)
    peak_pos = tc.position.values[peak_idx]
    peak_rate = np.nanmax(rates, axis=1)
    unit_ids = tc.unit.values
    mid_mask = (peak_pos>0.25)&(peak_pos<1.35)
    order = np.argsort(si)[::-1]
    print(f"--- speed thr {thr} m/s: dur={outbound_ep.tot_length():.0f}s n_ep={len(outbound_ep)}, n_mid_field(SI top40)={mid_mask[order[:40]].sum()}")
    for i in order[:8]:
        print("   unit", unit_ids[i], "SI", round(float(si[i]),2), "peak_pos", round(float(peak_pos[i]),2), "peak_rate", round(float(peak_rate[i]),1))
io.close()
