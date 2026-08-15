"""Build direction-specific 1D place fields from Maze running epochs."""
import numpy as np, pynapple as nap, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
C='/tmp/replay_cache'
ep=np.load(f'{C}/epochs.npy', allow_pickle=True)
epd={str(r['label']):(r['start_time'],r['stop_time']) for r in ep}
maze=nap.IntervalSet(start=epd['MazeEpoch'][0], end=epd['MazeEpoch'][1])

sp=np.load(f'{C}/spikes.npy', allow_pickle=True)
loc=np.load(f'{C}/unit_loc.npy')
units=nap.TsGroup({int(uid):nap.Ts(t=np.asarray(times)) for (uid,times),l in zip(sp,loc)})
print('n excitatory units', len(units))

tl=np.load(f'{C}/lin_t.npy'); xl=np.load(f'{C}/lin_x.npy')
valid=~np.isnan(xl)
pos=nap.Tsd(t=tl[valid], d=xl[valid]).restrict(maze)
dt=np.median(np.diff(pos.index))

# smoothed position & velocity
xs=uniform_filter1d(pos.values, int(0.2/dt))
vel=np.gradient(xs, pos.index)              # m/s, signed
speed=np.abs(vel)
SPEED_TH=0.10                               # 10 cm/s
# directional running masks
run_R = (speed>SPEED_TH)&(vel>0)           # rightward (increasing pos)
run_L = (speed>SPEED_TH)&(vel<0)           # leftward

def mask_to_ep(mask):
    e=nap.Tsd(t=pos.index, d=mask.astype(float)).threshold(0.5,'above').time_support
    return e.drop_short_intervals(0.3)
run_R_ep, run_L_ep = mask_to_ep(run_R), mask_to_ep(run_L)
print('run R', run_R_ep.tot_length(), 's | run L', run_L_ep.tot_length(),'s')

# Analyze the track INTERIOR only: the extreme boundary bins are contaminated by
# reward-consumption / turnaround firing (single-bin 100-300 Hz spikes) that swamp
# the real interior place fields. Restrict encoding & decoding to [0.1, 1.5] m.
LO,HI=0.10,1.50; NB=40
tc_R=nap.compute_1d_tuning_curves(units, pos, nb_bins=NB, ep=run_R_ep, minmax=(LO,HI))
tc_L=nap.compute_1d_tuning_curves(units, pos, nb_bins=NB, ep=run_L_ep, minmax=(LO,HI))
np.save(f'{C}/xbins.npy', np.linspace(LO,HI,NB))
# occupancy per direction (seconds per position bin) to mask under-sampled bins
edges=np.linspace(LO,HI,NB+1)
def occ_sec(rep):
    p=pos.restrict(rep); dt_=np.median(np.diff(p.index))
    h,_=np.histogram(p.values, bins=edges); return h*dt_
occR, occL = occ_sec(run_R_ep), occ_sec(run_L_ep)
MIN_OCC=0.3  # s; below this a bin's occupancy-normalized rate is unreliable (inflated ends)
validR, validL = occR>=MIN_OCC, occL>=MIN_OCC
# smooth tuning curves lightly along position, then floor under-sampled bins
for tc,valid in ((tc_R,validR),(tc_L,validL)):
    tc.iloc[:,:]=uniform_filter1d(tc.values, 2, axis=0)
    tc.iloc[~valid,:]=1e-3   # suppress inflated low-occupancy (reward-end) bins
print('valid bins R', validR.sum(), 'L', validL.sum(), 'of', NB)
tc_R.to_pickle(f'{C}/tc_R.pkl'); tc_L.to_pickle(f'{C}/tc_L.pkl')
np.save(f'{C}/valid_bins.npy', np.c_[validR,validL])

# place-cell selection: well-localized INTERIOR fields (exclude reward-end cells that
# pin the ripple posterior to the track ends). Criteria per direction:
#   peak rate > 2 Hz, peak position in track interior, field width < 60% of track.
binc=np.linspace(LO,HI,NB)
def sel(tc):
    r=tc.values; keep=[]
    for j in range(r.shape[1]):
        f=r[:,j]; pk=np.nanmax(f)
        if pk<2.0: continue                        # peak rate > 2 Hz
        width=(f>0.5*pk).sum()/NB*(HI-LO)          # width above half-max (m)
        if width>1.0: continue                     # reject diffuse/multi-peak
        keep.append(j)
    return np.array(keep,dtype=int)
pcR, pcL = sel(tc_R), sel(tc_L)
print(f'interior place cells R={len(pcR)} L={len(pcL)}')
# union of directional interior place cells used for decoding
pc_union=np.union1d(pcR,pcL)
np.save(f'{C}/pc_union.npy', pc_union)
np.save(f'{C}/tc_unit_ids.npy', np.array(tc_R.columns))
np.save(f'{C}/run_R.npy', np.c_[run_R_ep.start, run_R_ep.end])
np.save(f'{C}/run_L.npy', np.c_[run_L_ep.start, run_L_ep.end])

# figure: directional place fields sorted by peak
fig,axes=plt.subplots(1,3,figsize=(15,5))
binc=np.linspace(LO,HI,NB)
for ax,tc,pcs,title in [(axes[0],tc_R,pcR,'Rightward runs'),(axes[1],tc_L,pcL,'Leftward runs')]:
    m=tc.values[:,pcs]
    order=np.argsort(np.argmax(m,0))
    mn=(m[:,order]/(m[:,order].max(0)+1e-9)).T
    im=ax.imshow(mn,aspect='auto',origin='lower',extent=[LO,HI,0,len(order)],cmap='viridis')
    ax.set_xlabel('Position (m)'); ax.set_ylabel('Place cell (sorted)')
    ax.set_title(f'{title} (n={len(pcs)})')
plt.colorbar(im,ax=axes[1],label='Norm. rate')
axes[2].plot(pos.index-maze.start[0], pos.values,'k',lw=0.5)
axes[2].set_xlabel('Time in maze (s)'); axes[2].set_ylabel('Position (m)')
axes[2].set_title('Behavioral trajectory')
plt.tight_layout(); plt.savefig('fig1_place_fields.png',dpi=130); print('saved fig1')
