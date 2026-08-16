"""Decode which of the five tones was played from single-trial population activity."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

res = pickle.load(open('results_000986.pkl', 'rb'))
sessions = sorted(res)
ufreq = res[sessions[0]]['ufreq']
rng = np.random.default_rng(1)


def decode(X, y, n_splits=5):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, multi_class='multinomial', C=0.1))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    return pred


# --------------------------------------------------- decode in the richest session
main = max(sessions, key=lambda s: res[s]['stats'].tuned.sum())
r = res[main]
X, y = r['counts_ev'].astype(float), r['freqs']
pred = decode(X, y)
acc = (pred == y).mean()
cm = confusion_matrix(y, pred, labels=ufreq, normalize='true')
# shuffled-label control
pred_sh = decode(X, rng.permutation(y))
acc_sh = (pred_sh == rng.permutation(y)).mean()

# ------------------------------------- accuracy vs number of units (same session)
sizes = [1, 2, 5, 10, 20, 40, 80, 160, X.shape[1]]
sizes = sorted(set(s for s in sizes if s <= X.shape[1]))
curve_mean, curve_sd = [], []
for n in tqdm(sizes, desc='units subsampling'):
    accs = []
    for rep in range(8 if n < X.shape[1] else 1):
        idx = rng.choice(X.shape[1], n, replace=False)
        accs.append((decode(X[:, idx], y) == y).mean())
    curve_mean.append(np.mean(accs)); curve_sd.append(np.std(accs))

# ------------------------------------------------------ every session, all units
# The pre-tone window is the key control: if anything other than the tone (slow drift,
# stimulus sequence structure) carried frequency information, it would decode above chance.
per_session, per_session_bl = {}, {}
for s in tqdm(sessions, desc='sessions'):
    rr = res[s]
    per_session[s] = (decode(rr['counts_ev'].astype(float), rr['freqs']) == rr['freqs']).mean()
    per_session_bl[s] = (decode(rr['counts_bl'].astype(float), rr['freqs']) == rr['freqs']).mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.4))
ax = axes[0]
im = ax.imshow(cm, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_yticks(range(5)); ax.set_yticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('decoded frequency (kHz)'); ax.set_ylabel('presented frequency (kHz)')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='w' if cm[i, j] < 0.6 else 'k')
ax.set_title(f'{main.split("/")[-1].replace("_behavior.nwb","")}\n'
             f'{X.shape[1]} units, 5-fold CV, accuracy {acc:.2f}', fontsize=11)
plt.colorbar(im, ax=ax, pad=0.02, label='P(decoded | presented)')

ax = axes[1]
ax.errorbar(sizes, curve_mean, yerr=curve_sd, marker='o', color='crimson', capsize=3)
ax.axhline(0.2, color='0.5', ls='--', label='chance (1/5)')
ax.set_xscale('log'); ax.set_xlabel('number of units in the population')
ax.set_ylabel('decoding accuracy'); ax.set_title('Accuracy grows with population size')
ax.legend(frameon=False)

ax = axes[2]
labels = [s.split('/')[-1].replace('_behavior.nwb', '').replace('sub-', '') for s in sessions]
nun = [len(res[s]['stats']) for s in sessions]
yy = np.arange(len(sessions))
ax.barh(yy + 0.2, [per_session[s] for s in sessions], 0.4, color='steelblue',
        label='tone-evoked window (5-105 ms)')
ax.barh(yy - 0.2, [per_session_bl[s] for s in sessions], 0.4, color='0.75',
        label='pre-tone window (-105 to -5 ms)')
ax.axvline(0.2, color='crimson', ls='--')
ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
ax.set_yticks(yy)
ax.set_yticklabels([f'{l} ({n}u)' for l, n in zip(labels, nun)], fontsize=7.5)
ax.set_xlabel('decoding accuracy'); ax.set_title('Single-trial decoding in every session\n(red dashed = chance, 1/5)')
fig.suptitle('Tone frequency is decodable from single-trial auditory-cortex population activity '
             '(5-105 ms spike counts)', y=1.03, fontsize=12)
fig.savefig('figures/fig05_decoding.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('main session acc %.3f  shuffled %.3f' % (acc, acc_sh))
print('per session:', {k.split("/")[-1]: round(v, 3) for k, v in per_session.items()})
print('baseline-window control:', {k.split('/')[-1]: round(v,3) for k,v in per_session_bl.items()})
pickle.dump(dict(acc=acc, acc_shuffled=acc_sh, cm=cm, per_session=per_session, per_session_bl=per_session_bl,
                 sizes=sizes, curve_mean=curve_mean, main=main),
            open('summary_decoding.pkl', 'wb'))
