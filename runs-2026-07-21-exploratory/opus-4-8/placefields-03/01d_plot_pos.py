import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, lindi
from pynwb import NWBHDF5IO
URL="https://lindi.neurosift.org/dandi/dandisets/000044/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"
f=lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=lindi.LocalCache(cache_dir="/tmp/lindi_cache"))
nwbfile=NWBHDF5IO(file=f,mode="r").read()
lin=nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"].spatial_series["1.6mLinearMazeLinearizedTimeSeries"]
xy=nwbfile.processing["behavior"]["1.6mLinearMazePosition"].spatial_series["1.6mLinearMazeSpatialSeries"]
dt=lin.rate; t0=lin.starting_time
d=lin.data[:][:,0]; D=xy.data[:]
t=t0+np.arange(len(d))*dt
print("t span", t[0], t[-1])
nanfrac_lin=np.isnan(d); print("lin nan frac", nanfrac_lin.mean())
print("xy nan frac", np.isnan(D[:,0]).mean(), np.isnan(D[:,1]).mean())
fig,ax=plt.subplots(4,1,figsize=(14,10),sharex=True)
ax[0].plot(t,D[:,0],lw=.4); ax[0].set_ylabel("x (m)")
ax[1].plot(t,D[:,1],lw=.4); ax[1].set_ylabel("y (m)")
ax[2].plot(t,d,lw=.4,color='k'); ax[2].set_ylabel("linearized (m)")
ax[3].plot(t,np.isnan(d).astype(float),lw=.4,color='r'); ax[3].set_ylabel("lin isnan")
ax[3].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("figures/_check_position.png",dpi=110)
# zoom
fig,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
m=(t>18100)&(t<18400)
ax[0].plot(t[m],D[m,0],'.-',ms=2,lw=.5,label='x'); ax[0].plot(t[m],D[m,1],'.-',ms=2,lw=.5,label='y'); ax[0].legend()
ax[1].plot(t[m],d[m],'.-',ms=2,lw=.5,color='k')
plt.tight_layout(); plt.savefig("figures/_check_position_zoom.png",dpi=110)
