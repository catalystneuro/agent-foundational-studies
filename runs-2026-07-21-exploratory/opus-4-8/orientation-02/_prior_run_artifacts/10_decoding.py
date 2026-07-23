"""Population decoding of grating orientation: V1 vs LGd, static and drifting gratings."""
import warnings, pickle, numpy as np, matplotlib
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_lib import load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
REG_COLORS = {'VISp':'#1f77b4','LGd':'#d62728'}

d, _ = load('cache_715093703.pkl')
res = pickle.load(open('results.pkl','rb')); dg, sg = res['dg'], res['sg']
reg = np.array(d['meta']['region'])

def decode(X, y, seed=0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    yp = cross_val_predict(clf, X, y, cv=cv)
    return (yp == y).mean(), confusion_matrix(y, yp, normalize='true')

sets = {'static gratings (0.25 s)': (sg['R'], sg['ori']),
        'drifting gratings (2 s)': (dg['R'], dg['ori'] % 180)}
n_match = min((reg=='VISp').sum(), (reg=='LGd').sum())
rng = np.random.default_rng(1)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
Rs, ys = sets['static gratings (0.25 s)']
oris = np.unique(ys)
for ax, r in zip(axes[:2], ['VISp','LGd']):
    pop = rng.choice(np.where(reg==r)[0], n_match, replace=False)
    acc, cm = decode(Rs[:, pop], ys)
    im = ax.imshow(cm, vmin=0, vmax=max(0.5, cm.max()), cmap='magma')
    ax.set_xticks(range(len(oris)), [f'{int(x)}' for x in oris])
    ax.set_yticks(range(len(oris)), [f'{int(x)}' for x in oris])
    ax.set_xlabel('decoded orientation (deg)'); ax.set_ylabel('true orientation (deg)')
    ax.set_title(f'{r} — static gratings, n={n_match} units\n'
                 f'accuracy {acc*100:.1f}% (chance {100/len(oris):.1f}%)', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, label='P(decoded | true)')

ax = axes[2]
sizes = [5, 10, 20, 40, n_match]
styles = {'static gratings (0.25 s)': '-', 'drifting gratings (2 s)': '--'}
for name, (R, y) in sets.items():
    for r in ['VISp','LGd']:
        m, s = [], []
        for n in tqdm(sizes, desc=f'{name[:8]}-{r}'):
            a = [decode(R[:, rng.choice(np.where(reg==r)[0], n, replace=False)], y, seed=i)[0]
                 for i in range(3)]
            m.append(np.mean(a)*100); s.append(np.std(a)*100)
        ax.errorbar(sizes, m, yerr=s, marker='o', ms=4, capsize=3, ls=styles[name],
                    color=REG_COLORS[r], label=f'{r}, {name}')
    ax.axhline(100/len(np.unique(y)), color='0.5', ls=':', lw=1)
ax.text(5.5, 100/6+1.5, 'chance (6 orientations)', ha='left', fontsize=7, color='0.45')
ax.text(5.5, 25+1.5, 'chance (4 orientations)', ha='left', fontsize=7, color='0.45')
ax.set_xlabel('number of units in decoding population')
ax.set_ylabel('cross-validated accuracy (%)')
ax.legend(frameon=False, fontsize=7.5, loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.set_title('Population orientation information is present in both\nareas; the V1 advantage is modest', fontsize=9)
fig.tight_layout()
fig.savefig('fig07_decoding.png'); plt.close(fig)
print('fig07 rewritten')
