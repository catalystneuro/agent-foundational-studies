"""Stage 1: load session, inspect data streams, plot raw data for validation."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
import common

SESSION = "Achilles_10252013"
nwbfile, h = common.open_session(SESSION)
nwb = nap.NWBFile(nwbfile)
print(nwb)

# ---- epochs -------------------------------------------------------------
ep_df = nwbfile.epochs.to_dataframe()
print(ep_df)
epochs = {r.label: nap.IntervalSet(start=r.start_time, end=r.stop_time) for r in ep_df.itertuples()}

# ---- sleep/wake states --------------------------------------------------
st = nwbfile.processing["behavior"]["states"].to_dataframe()
states = {lab: nap.IntervalSet(start=g.start_time.values, end=g.stop_time.values)
          for lab, g in st.groupby("label")}
for k, v in states.items():
    print(f"state {k:8s} n={len(v):3d} total={v.tot_length():.0f} s")

# ---- units --------------------------------------------------------------
units = nwb["units"]
udf = nwbfile.units.to_dataframe()
print("units:", len(units), udf.cell_type.value_counts().to_dict(), udf.location.value_counts().to_dict())

# ---- position -----------------------------------------------------------
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]      # (N,1) linearized, meters
xy  = nwb["1.6mLinearMazeSpatialSeries"]             # (N,2)
lin = nap.Tsd(t=lin.t, d=np.asarray(lin.d).squeeze())
print("position", lin.shape, "t:", lin.t[0], "->", lin.t[-1],
      "range", np.nanmin(lin.d), np.nanmax(lin.d), "n_nan", np.isnan(lin.d).sum())

maze = epochs["MazeEpoch"]
lin_maze = lin.restrict(maze)
print("maze samples", len(lin_maze), "nan", np.isnan(lin_maze.d).sum())

# ---- LFP snippet during POST sleep --------------------------------------
lfp_ds = h["processing/ecephys/LFP/LFP/data"]
FS = nwbfile.processing["ecephys"]["LFP"]["LFP"].rate
CONV = nwbfile.processing["ecephys"]["LFP"]["LFP"].conversion
print("LFP", lfp_ds.shape, "fs", FS, "chunks", lfp_ds.chunks)

# ---- FIGURE: raw data overview -----------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(13, 11))

ax = axs[0]
ax.plot(lin.t / 3600, lin.d, ".", ms=0.5, color="k")
for lab, ep, c in [("PRE", epochs["PREEpoch"], "#8ecae6"),
                   ("MAZE", epochs["MazeEpoch"], "#ffb703"),
                   ("POST", epochs["POSTEpoch"], "#90be6d")]:
    ax.axvspan(ep.start[0] / 3600, ep.end[0] / 3600, alpha=.35, color=c, label=lab)
ax.set(xlabel="time (h)", ylabel="linear position (m)",
       title=f"{SESSION}: session structure (PRE sleep / linear maze / POST sleep)")
ax.legend(loc="upper right", fontsize=8)

ax = axs[1]
ax.plot(lin_maze.t, lin_maze.d, "k-", lw=.7)
ax.set(xlabel="time (s)", ylabel="position (m)", title="Linearized position on the 1.6 m track (maze epoch)")

ax = axs[2]
exc = udf.index[udf.cell_type == "excitatory"].values
tsub = maze.start[0] + np.array([300., 340.])
sub = nap.IntervalSet(start=tsub[0], end=tsub[1])
for i, u in enumerate(exc):
    sp = units[u].restrict(sub)
    ax.plot(sp.t, np.full(len(sp), i), "|", ms=3, color="k", mew=.5)
ax.set(xlabel="time (s)", ylabel="unit #", title=f"Spike raster, {len(exc)} putative pyramidal cells (40 s of running)")
axb = ax.twinx()
axb.plot(lin.restrict(sub).t, lin.restrict(sub).d, color="#e63946", lw=1.2)
axb.set_ylabel("position (m)", color="#e63946")

ax = axs[3]
t0 = epochs["POSTEpoch"].start[0] + 1000
i0 = int(t0 * FS)
seg = lfp_ds[i0:i0 + int(2 * FS), :].astype(np.float32) * CONV * 1e3   # mV
tseg = np.arange(seg.shape[0]) / FS
for k, ch in enumerate(range(0, 128, 16)):
    ax.plot(tseg, seg[:, ch] + k * 1.2, lw=.5, color="k")
ax.set(xlabel="time (s)", ylabel="channel (offset, mV)", title="Raw LFP, 8 channels across shanks (POST sleep)")

plt.tight_layout()
plt.savefig("figures/01_raw_data_overview.png", dpi=140)
print("saved figures/01_raw_data_overview.png")

np.savez("cache/01_basics.npz",
         pos_t=lin.t, pos_d=lin.d,
         epoch_labels=np.array(list(epochs.keys())),
         epoch_start=np.array([e.start[0] for e in epochs.values()]),
         epoch_end=np.array([e.end[0] for e in epochs.values()]))
