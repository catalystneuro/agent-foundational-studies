"""Population, cross-stimulus, GLM and decoding figures (prototype session)."""
import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from analysis_lib import load
import nemos as nmo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
REG_COLORS = {'VISp':'#1f77b4','VISl':'#2ca02c','VISrl':'#9467bd',
              'VISam':'#8c564b','VISpm':'#e377c2','LGd':'#d62728'}
ORDER = ['VISp','VISl','VISrl','VISam','VISpm','LGd']

d, tsg = load('cache_715093703.pkl')
res = pickle.load(open('results.pkl','rb'))
dg, sg = res['dg'], res['sg']
reg = np.array(d['meta']['region']); uids = np.array(list(tsg.keys()))
M = dg['metrics']; dirs = dg['dirs']
sig = (M['anova_p'] < 0.01) & (M['perm_p'] < 0.05)

vp_sg = np.where((reg=='VISp') & (sg['anova_p']<0.01))[0]
vp_sg = vp_sg[np.argsort(-sg['gOSI'][vp_sg])][:3]

# ---------------- Fig 6: NeMoS Poisson GLM, direction vs orientation model ----------------
R = dg['R']; o = dg['ori']; f = dg['tf']; tfs = dg['tfs']
dur = 2.0
b_dir = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, order=4, bounds=(0., 360.))
b_ori = nmo.basis.CyclicBSplineEval(n_basis_funcs=6, order=4, bounds=(0., 180.))
grid_dir = np.linspace(0, 359.9, 200)

def cv_pseudo_r2(X, y, n_splits=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_splits)
    ll_m, ll_0 = 0., 0.
    for i in range(n_splits):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        g = nmo.glm.GLM(regularizer='Ridge', regularizer_strength=1e-3,
                        observation_model=nmo.observation_models.PoissonObservations())
        g.fit(X[tr], y[tr])
        mu = np.asarray(g.predict(X[te])); mu = np.clip(mu, 1e-9, None)
        mu0 = np.clip(y[tr].mean(), 1e-9, None)
        ll_m += np.sum(y[te]*np.log(mu) - mu)
        ll_0 += np.sum(y[te]*np.log(mu0) - mu0)
    ll_sat = np.sum(np.where(y > 0, y*np.log(np.clip(y,1e-9,None)) - y, 0.))
    return 1 - (ll_sat - ll_m)/(ll_sat - ll_0)

pr2_dir = np.full(len(uids), np.nan); pr2_ori = np.full(len(uids), np.nan)
glm_curves = {}
glm_units = np.where(np.isin(reg, ['VISp','LGd']))[0]
for u in tqdm(glm_units, desc='GLM'):
    sub = f == M['pref_tf'][u]
    x = o[sub]; y = np.round(R[sub, u]*dur)
    if y.sum() < 20: continue
    Xd = b_dir.compute_features(x); Xo = b_ori.compute_features(x % 180)
    pr2_dir[u] = cv_pseudo_r2(Xd, y); pr2_ori[u] = cv_pseudo_r2(Xo, y)
    if u in list(vp_sg) + list(np.where((reg=='VISp') & sig)[0][:8]):
        g = nmo.glm.GLM(regularizer='Ridge', regularizer_strength=1e-3,
                        observation_model=nmo.observation_models.PoissonObservations())
        g.fit(Xd, y)
        glm_curves[u] = np.asarray(g.predict(b_dir.compute_features(grid_dir)))/dur

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
ex = sorted(glm_curves, key=lambda u: -M['gOSI'][u])[:2]
for ax, u in zip(axes[:2], ex):
    ax.errorbar(dirs, dg['tc'][:, u], yerr=dg['tc_sem'][:, u], fmt='o',
                color='0.25', capsize=2, ms=4, label='measured')
    ax.plot(grid_dir, glm_curves[u], color=REG_COLORS['VISp'], lw=2, label='Poisson GLM')
    ax.set_xticks(dirs); ax.set_xlabel('direction (deg)'); ax.set_ylabel('firing rate (Hz)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"VISp {uids[u]}  gOSI={M['gOSI'][u]:.2f}", fontsize=9)
ax = axes[2]
for r in ['VISp','LGd']:
    k = (reg==r) & np.isfinite(pr2_dir)
    ax.plot(pr2_ori[k], pr2_dir[k], 'o', ms=4, alpha=0.7, color=REG_COLORS[r], mec='none', label=r)
lo = np.nanmin([pr2_ori, pr2_dir]); hi = np.nanmax([pr2_ori, pr2_dir])
ax.plot([lo,hi],[lo,hi],'k--',lw=0.8)
ax.set_xlabel('cross-val. pseudo-$R^2$, orientation model (180° period)')
ax.set_ylabel('pseudo-$R^2$, direction model (360°)')
ax.legend(frameon=False)
ax.set_title('An orientation-only model captures\nmost of the explainable variance', fontsize=9)
fig.suptitle('NeMoS Poisson GLM with cyclic B-spline bases over grating direction', y=1.03)
fig.tight_layout()
fig.savefig('fig06_glm.png'); plt.close(fig)
np.save('glm_pr2.npy', np.vstack([pr2_ori, pr2_dir]))
print('fig06 done')

# ---------------- Fig 7: population decoding of orientation ----------------
sub = np.ones(len(o), bool)
labels_dir = o[sub]; labels_ori = (o[sub] % 180)
def decode(pop_idx, y):
    X = dg['R'][sub][:, pop_idx]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    yp = cross_val_predict(clf, X, y, cv=cv)
    return (yp == y).mean(), confusion_matrix(y, yp, normalize='true')

n_match = min((reg=='VISp').sum(), (reg=='LGd').sum())
rng = np.random.default_rng(1)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
accs = {}
for ax, r in zip(axes[:2], ['VISp','LGd']):
    pop = rng.choice(np.where(reg==r)[0], n_match, replace=False)
    acc, cm = decode(pop, labels_ori)
    accs[r] = acc
    im = ax.imshow(cm, vmin=0, vmax=cm.max(), cmap='magma')
    oris = np.unique(labels_ori)
    ax.set_xticks(range(len(oris)), [f'{int(x)}' for x in oris])
    ax.set_yticks(range(len(oris)), [f'{int(x)}' for x in oris])
    ax.set_xlabel('decoded orientation (deg)'); ax.set_ylabel('true orientation (deg)')
    ax.set_title(f'{r}  (n={n_match} units)\naccuracy {acc*100:.1f}% (chance {100/len(oris):.1f}%)',
                 fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
ax = axes[2]
sizes = [5, 10, 20, 40, n_match]
for r in ['VISp','LGd']:
    ys = []
    for s in sizes:
        a = [decode(rng.choice(np.where(reg==r)[0], s, replace=False), labels_ori)[0]
             for _ in range(5)]
        ys.append(a)
    ys = np.array(ys)
    ax.errorbar(sizes, ys.mean(1)*100, yerr=ys.std(1)*100, marker='o',
                color=REG_COLORS[r], capsize=3, label=r)
ax.axhline(100/len(np.unique(labels_ori)), color='k', ls='--', lw=0.8, label='chance')
ax.set_xlabel('number of units in decoding population')
ax.set_ylabel('decoding accuracy (%)'); ax.legend(frameon=False)
ax.set_title('Orientation is more linearly decodable\nfrom V1 than from LGd', fontsize=9)
fig.suptitle('Cross-validated population decoding of grating orientation (multinomial logistic regression)', y=1.03)
fig.tight_layout()
fig.savefig('fig07_decoding.png'); plt.close(fig)
print('fig07 done', accs)
