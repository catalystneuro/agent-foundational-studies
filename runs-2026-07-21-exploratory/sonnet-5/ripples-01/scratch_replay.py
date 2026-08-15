import numpy as np
import pandas as pd
import pynapple as nap
from scipy.stats import spearmanr
from scratch_load import load_nwb
from scratch_ripples import detect_ripples
from scratch_placefields import get_linearized_position

MIN_ACTIVE_CELLS = 4
SHUFFLES_PER_EVENT = 200


def analyze_window(units_ts, ripples, template_rank, rng):
    rows = []
    all_shuffle_rhos = []
    cell_ids = np.array(list(template_rank.keys()))
    for s, e, p in ripples:
        spike_dict = {}
        for cid in cell_ids:
            ts = units_ts[cid].t
            mask = (ts >= s) & (ts <= e)
            if mask.sum() > 0:
                spike_dict[cid] = ts[mask].min()
        if len(spike_dict) < MIN_ACTIVE_CELLS:
            continue
        active_cells = list(spike_dict.keys())
        event_times = np.array([spike_dict[c] for c in active_cells])
        order = np.argsort(event_times)
        event_rank = np.empty(len(active_cells))
        event_rank[order] = np.arange(len(active_cells))
        base_ranks = np.array([template_rank[c] for c in active_cells])
        rho, _ = spearmanr(event_rank, base_ranks)

        # event-specific shuffle null: permute which template rank is assigned to
        # which active cell (breaks place-field-order <-> spike-order relationship)
        shuf_rhos = np.array([
            spearmanr(event_rank, rng.permutation(base_ranks))[0]
            for _ in range(SHUFFLES_PER_EVENT)
        ])
        pval = (np.sum(np.abs(shuf_rhos) >= np.abs(rho)) + 1) / (SHUFFLES_PER_EVENT + 1)

        rows.append({'start': s, 'end': e, 'rho': rho, 'n_active': len(active_cells), 'pval': pval})
        all_shuffle_rhos.append(shuf_rhos)
    df = pd.DataFrame(rows)
    shuffle_rhos = np.concatenate(all_shuffle_rhos) if all_shuffle_rhos else np.array([])
    return df, shuffle_rhos


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    nwbfile, io = load_nwb()
    nwb = nap.NWBFile(nwbfile)
    units = nwb['units']

    pos, run_ep = get_linearized_position(nwbfile)
    meta = units.metadata
    exc_l = meta[(meta['location'] == 'lCA1') & (meta['cell_type'] == 'excitatory')].index.values
    pyr = units[exc_l]
    tc = nap.compute_tuning_curves(pyr, pos, bins=40, epochs=run_ep, return_pandas=True)

    peak_rate = tc.max(axis=0)
    mean_rate = tc.mean(axis=0)
    selectivity = peak_rate / (mean_rate + 1e-9)
    place_cells = tc.columns[(peak_rate > 1) & (selectivity > 3)]
    peak_pos = tc[place_cells].idxmax(axis=0)
    order_sorted = peak_pos.sort_values()
    template_rank = {cid: rank for rank, cid in enumerate(order_sorted.index)}
    print('place cells', place_cells.tolist())
    print('template_rank', template_rank)

    ripples_pre = np.load('scratch_ripples_PRE.npy')
    ripples_post = np.load('scratch_ripples_POST.npy')

    df_pre, shuf_pre = analyze_window(units, ripples_pre, template_rank, rng)
    df_post, shuf_post = analyze_window(units, ripples_post, template_rank, rng)

    print('PRE: n ripples w/ enough active cells', len(df_pre), 'mean |rho|', df_pre['rho'].abs().mean(),
          'frac sig (p<0.05)', (df_pre['pval'] < 0.05).mean())
    print('POST: n ripples w/ enough active cells', len(df_post), 'mean |rho|', df_post['rho'].abs().mean(),
          'frac sig (p<0.05)', (df_post['pval'] < 0.05).mean())
    print('shuffle mean |rho|', np.abs(np.concatenate([shuf_pre, shuf_post])).mean())

    df_pre.to_pickle('scratch_df_pre.pkl')
    df_post.to_pickle('scratch_df_post.pkl')
    np.save('scratch_shuf_pre.npy', shuf_pre)
    np.save('scratch_shuf_post.npy', shuf_post)
    tc.to_pickle('scratch_tc_final.pkl')
    with open('scratch_template_rank.txt', 'w') as fh:
        fh.write(str(template_rank))
