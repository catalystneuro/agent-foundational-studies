import time
import pickle
import numpy as np
import pynapple as nap
from tqdm import tqdm
from core_analysis import (load_nwb, select_ripple_channel, detect_ripples,
                            compute_place_fields, analyze_replay)

SESSIONS = {
    'Buddy-06272013': 'https://api.dandiarchive.org/api/assets/82714afb-724f-4e2b-b102-c9c47b5cba73/download/',
    'Gatsby-08022013': 'https://api.dandiarchive.org/api/assets/31ea0aab-4777-424e-9a93-9605b2bdcc29/download/',
    'Cicero-09172014': 'https://api.dandiarchive.org/api/assets/e381ebb3-128e-4f3f-9517-11277d7aed9b/download/',
    'Gatsby-08282013': 'https://api.dandiarchive.org/api/assets/f7687af7-3bc9-4d20-8d88-ef293d2a3381/download/',
    'Cicero-09012014': 'https://api.dandiarchive.org/api/assets/3cc5b7b3-02e2-490a-9f19-d20670355084/download/',
    'Achilles-10252013': 'https://api.dandiarchive.org/api/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/download/',
    'Cicero-09102014': 'https://api.dandiarchive.org/api/assets/f61dfe09-3db2-464a-b386-2e828b2e7276/download/',
    'Achilles-11012013': 'https://api.dandiarchive.org/api/assets/8855c8cc-9d8b-4d5b-8ef0-fe87916f839a/download/',
}

PRE_WINDOW_S = 1800.0
POST_WINDOW_S = 2700.0
CHANNEL_SCAN_S = 120.0


def process_session(name, url, rng):
    t0 = time.time()
    nwbfile, io = load_nwb(url)
    nwb = nap.NWBFile(nwbfile)

    epochs = nwbfile.epochs.to_dataframe()
    pre_end = epochs.loc[epochs['label'] == 'PREEpoch', 'stop_time'].values[0]
    post_start = epochs.loc[epochs['label'] == 'POSTEpoch', 'start_time'].values[0]
    post_end = epochs.loc[epochs['label'] == 'POSTEpoch', 'stop_time'].values[0]
    pre_win = (max(0, pre_end - PRE_WINDOW_S), pre_end)
    post_win = (post_start, min(post_end, post_start + POST_WINDOW_S))

    lfp = nwbfile.processing['ecephys']['LFP']['LFP']
    fs = lfp.rate
    ch, power = select_ripple_channel(lfp, fs, pre_win[0] + 300, CHANNEL_SCAN_S)

    i0, i1 = int(pre_win[0] * fs), int(pre_win[1] * fs)
    raw_pre = lfp.data[i0:i1, ch].astype(np.float32)
    ripples_pre = detect_ripples(raw_pre, fs, t_offset=pre_win[0])

    i0, i1 = int(post_win[0] * fs), int(post_win[1] * fs)
    raw_post = lfp.data[i0:i1, ch].astype(np.float32)
    ripples_post = detect_ripples(raw_post, fs, t_offset=post_win[0])

    tc, place_cells, template_rank, pos, run_ep = compute_place_fields(nwbfile, nwb)

    units = nwb['units']
    df_pre, shuf_pre = analyze_replay(units, ripples_pre, template_rank, rng)
    df_post, shuf_post = analyze_replay(units, ripples_post, template_rank, rng)

    result = {
        'name': name,
        'channel': ch,
        'fs': fs,
        'pre_win': pre_win,
        'post_win': post_win,
        'n_ripples_pre': len(ripples_pre),
        'n_ripples_post': len(ripples_post),
        'n_place_cells': len(place_cells),
        'n_units_total': len(nwbfile.units),
        'df_pre': df_pre,
        'df_post': df_post,
        'shuf_pre': shuf_pre,
        'shuf_post': shuf_post,
        'tc': tc,
        'place_cells': place_cells,
        'template_rank': template_rank,
        'ripples_pre_example': ripples_pre[:50],
        'ripples_post_example': ripples_post[:50],
        'raw_pre_snippet': raw_pre[:int(30 * fs)],
        'raw_post_snippet': raw_post[:int(30 * fs)],
        'ripples_pre_snippet': ripples_pre[(ripples_pre[:, 0] < pre_win[0] + 30)] if len(ripples_pre) else ripples_pre,
        'ripples_post_snippet': ripples_post[(ripples_post[:, 0] < post_win[0] + 30)] if len(ripples_post) else ripples_post,
        'elapsed_s': time.time() - t0,
    }
    io.close()
    print(f"{name}: {result['elapsed_s']:.1f}s | ch={ch} | n_place_cells={len(place_cells)} | "
          f"ripples PRE/POST={len(ripples_pre)}/{len(ripples_post)} | "
          f"frac_sig PRE={np.mean(df_pre['pval'] < 0.05):.3f} POST={np.mean(df_post['pval'] < 0.05):.3f}")
    return result


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    results = {}
    for name, url in tqdm(SESSIONS.items()):
        results[name] = process_session(name, url, rng)
    with open('all_sessions_results.pkl', 'wb') as fh:
        pickle.dump(results, fh)
    print('saved all_sessions_results.pkl')
