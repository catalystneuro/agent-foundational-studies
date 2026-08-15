"""Replicate phase precession across sessions/animals in DANDI:000044.

Runs the validated pipeline on three sessions (two animals) and summarizes
the fraction of place cells with negative phase-position slopes.
Saves fig_05_multisession.png and multisession_summary.csv.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pipeline import load_session, analyze_session

SESSIONS = {
    'Achilles-10252013': 'https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028',
    'Achilles-11012013': 'https://dandiarchive.s3.amazonaws.com/blobs/4a9/ef6/4a9ef66e-4d89-44cf-ab50-ad029a665d50',
    'Cicero-09172014':   'https://dandiarchive.s3.amazonaws.com/blobs/aab/623/aab6235a-0144-4072-9a35-c7a3008b9870',
}

def main():
    summary = []
    per_session = {}
    for name, url in tqdm(SESSIONS.items(), desc='sessions'):
        sess = load_session(url)
        res = analyze_session(sess)
        slopes = np.array([r['slope'] for r in res])
        rhos = np.array([r['rho'] for r in res])
        pvals = np.array([r['pval'] for r in res])
        sig = pvals < 0.05
        per_session[name] = dict(slopes=slopes, rhos=rhos, pvals=pvals)
        summary.append(dict(
            session=name, animal=name.split('-')[0], n_place=len(res),
            theta_ratio=round(sess['theta_ratio'], 2),
            frac_neg=round(float((slopes < 0).mean()), 3),
            n_sig=int(sig.sum()),
            frac_sig_neg=round(float((slopes[sig] < 0).mean()) if sig.sum() else np.nan, 3),
            median_slope=round(float(np.median(slopes)), 3),
            median_rho=round(float(np.median(rhos)), 3)))
        print(summary[-1])

    df = pd.DataFrame(summary)
    df.to_csv('multisession_summary.csv', index=False)
    print('\n', df.to_string())

    # --- figure: per-session slope distributions + fraction negative ---
    fig, axes = plt.subplots(1, len(SESSIONS)+1, figsize=(5*len(SESSIONS)+4, 4.5))
    for ax, name in zip(axes[:-1], SESSIONS):
        s = per_session[name]['slopes']
        ax.hist(s, bins=np.arange(-3, 1.6, 0.3), color='steelblue', edgecolor='k')
        ax.axvline(0, color='r', ls='--')
        ax.axvline(np.median(s), color='k')
        ax.set_title(f'{name}\n{100*(s<0).mean():.0f}% neg (n={len(s)})', fontsize=10)
        ax.set_xlabel('slope (cycles/field)'); ax.set_ylabel('# place cells')
    # summary bar of fraction negative
    ax = axes[-1]
    ax.bar(range(len(df)), df['frac_neg']*100, color='seagreen', edgecolor='k')
    ax.axhline(50, color='r', ls='--', label='chance')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([s.split('-')[0]+'\n'+s.split('-')[1][:4] for s in df['session']], fontsize=8)
    ax.set_ylabel('% place cells with negative slope'); ax.set_ylim(0, 100)
    ax.legend(); ax.set_title('Precession is consistent across sessions')
    fig.suptitle('Theta phase precession replicates across sessions and animals (DANDI:000044)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('fig_05_multisession.png', dpi=130)
    print('Saved fig_05_multisession.png')

if __name__ == '__main__':
    main()
