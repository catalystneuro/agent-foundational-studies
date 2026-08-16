"""Compute directional place fields and rank place cells.

On a linear track, place fields are direction-specific, so we split running
into rightward and leftward epochs and compute a 1D tuning curve per direction.
Saves place_fields.npz and fig_02_place_fields.png.
"""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

N_BINS = 50
MAZE_LEN = 1.6

def build_objects():
    d = np.load('loaded_data.npz', allow_pickle=True)
    p = np.load('preprocessed.npz', allow_pickle=True)

    pos = nap.Tsd(t=p['pos_t'], d=p['pos_data'])
    run_ep = nap.IntervalSet(start=p['run_starts'], end=p['run_ends'])
    direction = nap.Tsd(t=p['pos_t'], d=p['direction'])

    unit_ids = d['unit_ids']
    spikes = {int(u): nap.Ts(t=st) for u, st in zip(unit_ids, d['spike_times'])}
    units = nap.TsGroup(spikes)
    return d, p, pos, run_ep, direction, units

def direction_epochs(direction, run_ep):
    """Split running epochs by sign of velocity."""
    right = direction.threshold(0.0, 'above').time_support.intersect(run_ep)
    left = direction.threshold(0.0, 'below').time_support.intersect(run_ep)
    right = right.drop_short_intervals(0.3)
    left = left.drop_short_intervals(0.3)
    return right, left

def main():
    d, p, pos, run_ep, direction, units = build_objects()
    right_ep, left_ep = direction_epochs(direction, run_ep)
    print(f"Rightward running: {right_ep.tot_length():.0f} s ({len(right_ep)} runs)")
    print(f"Leftward  running: {left_ep.tot_length():.0f} s ({len(left_ep)} runs)")

    tc = {}
    occ = {}
    for name, ep in [('right', right_ep), ('left', left_ep)]:
        tc[name] = nap.compute_1d_tuning_curves(
            units, pos, nb_bins=N_BINS, ep=ep, minmax=(0, MAZE_LEN))
    bins = tc['right'].index.values

    # rank cells by peak firing rate (best over the two directions)
    peak = np.maximum(tc['right'].max(), tc['left'].max())
    # spatial info per direction
    def spatial_info(rate, pos_ep):
        occ_counts, _ = np.histogram(pos.restrict(pos_ep).values, bins=N_BINS, range=(0, MAZE_LEN))
        P = occ_counts / occ_counts.sum()
        r = rate.values
        rbar = np.nansum(P * r)
        with np.errstate(divide='ignore', invalid='ignore'):
            si = np.nansum(P * (r / rbar) * np.log2(r / rbar))
        return si if rbar > 0 else 0.0

    rows = []
    for uid in units.keys():
        si_r = spatial_info(tc['right'][uid], right_ep)
        si_l = spatial_info(tc['left'][uid], left_ep)
        pk_r, pk_l = tc['right'][uid].max(), tc['left'][uid].max()
        best_dir = 'right' if pk_r >= pk_l else 'left'
        rows.append(dict(uid=uid, peak=max(pk_r, pk_l), best_dir=best_dir,
                         si=max(si_r, si_l),
                         peak_pos=tc[best_dir][uid].idxmax()))
    import pandas as pd
    df = pd.DataFrame(rows).sort_values('peak', ascending=False).reset_index(drop=True)
    print(df.head(20).to_string())

    # place cells: peak rate > 5 Hz and spatial info > 0.5 bits/spike
    place = df[(df['peak'] > 5) & (df['si'] > 0.5)].reset_index(drop=True)
    print(f"\n{len(place)} place cells (peak>5 Hz, SI>0.5 bits/spike)")

    # --- figure: place-field heatmaps sorted by peak location, per direction ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, name in zip(axes, ['right', 'left']):
        sub = place[place['best_dir'] == name].sort_values('peak_pos')
        M = np.stack([tc[name][u].values for u in sub['uid']])
        # normalize each row to its max
        Mn = M / np.nanmax(M, axis=1, keepdims=True)
        im = ax.imshow(Mn, aspect='auto', origin='lower',
                       extent=[0, MAZE_LEN, 0, len(sub)], cmap='viridis',
                       interpolation='nearest')
        ax.set_title(f'{name}ward place fields (n={len(sub)})')
        ax.set_xlabel('Position (m)'); ax.set_ylabel('Cell (sorted by peak)')
        plt.colorbar(im, ax=ax, label='norm. rate')
    plt.tight_layout()
    plt.savefig('fig_02_place_fields.png', dpi=130)
    print("Saved fig_02_place_fields.png")

    np.savez_compressed(
        'place_fields.npz',
        bins=bins,
        tc_right=tc['right'].values, tc_left=tc['left'].values,
        tc_uids=np.array(list(units.keys())),
        right_starts=right_ep.start, right_ends=right_ep.end,
        left_starts=left_ep.start, left_ends=left_ep.end,
        place_uid=place['uid'].values, place_dir=place['best_dir'].values,
        place_peakpos=place['peak_pos'].values, place_peak=place['peak'].values,
        place_si=place['si'].values,
    )
    print("Saved place_fields.npz")

if __name__ == '__main__':
    main()
