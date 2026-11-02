"""Baseline machine learning klasik: fitur buatan tangan + gradient boosting.
Fitur per sadapan: statistik amplitudo, laju jantung dari puncak R (sadapan II), daya spektral."""
import json, numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
d = np.load('data/ptbxl100.npz')
X, Y, fold, age, sex = d['X'], d['Y'], d['fold'], d['age'], d['sex']
FS = 100

def feats(x):  # x: (1000, 12)
    f = [x.mean(0), x.std(0), x.min(0), x.max(0), np.ptp(x, 0), skew(x, 0), kurtosis(x, 0),
         np.percentile(x, 5, 0), np.percentile(x, 95, 0), np.abs(np.diff(x, axis=0)).mean(0)]
    fr, p = welch(x, FS, axis=0, nperseg=256)
    for lo, hi in [(0.5, 3), (3, 8), (8, 15), (15, 40)]:
        f.append(np.log(p[(fr >= lo) & (fr < hi)].sum(0) + 1e-8))
    ii = x[:, 1]  # sadapan II untuk puncak R
    pk, _ = find_peaks(ii, distance=30, prominence=0.3 * np.ptp(ii))
    rr = np.diff(pk) / FS if len(pk) > 2 else np.array([1.0, 1.0])
    hr = [60 / rr.mean(), rr.std(), rr.std() / rr.mean(), len(pk)]
    return np.concatenate([np.concatenate(f), hr])

F = np.stack([feats(x) for x in X])
F = np.column_stack([F, age, sex])
print('jumlah fitur:', F.shape[1])
tr, te = fold <= 9, fold == 10
P = np.zeros((te.sum(), 5))
for k in range(5):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)
    clf.fit(F[tr], Y[tr, k]); P[:, k] = clf.predict_proba(F[te])[:, 1]
np.save('data/pred_ml_test.npy', P)
res = {'macro_auc_test': roc_auc_score(Y[te], P, average='macro'),
       'per_class': {c: roc_auc_score(Y[te][:, k], P[:, k]) for k, c in enumerate(CLASSES)}}
json.dump(res, open('hasil/res_ml.json', 'w'), indent=1); print(json.dumps(res, indent=1))
