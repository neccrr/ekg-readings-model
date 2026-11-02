"""Langkah terakhir model 58 label: ambang minimum 0,10 untuk kode rinci, hitung ulang metrik uji,
lalu tulis model/labels.json (dipakai predict.py, main.py, dan versi web).

Jalankan dari folder utama repo setelah src/train_full.py:
    python src/finalize_full.py
"""
import ast, glob, json, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, 'src')
from predict import ResNet1D

FLOOR = 0.10  # F1 maksimum pada kode jarang memberi ambang terlalu rendah (mis. QT memanjang 0,02)
R = json.load(open('hasil/full_res.json')); labels = R['labels']
d = np.load('data/ptbxl100.npz'); X, fold = d['X'], d['fold']
root = glob.glob('data/raw/*')[0] + '/'
db = pd.read_csv(root + 'ptbxl_database.csv', index_col='ecg_id'); db.scp_codes = db.scp_codes.apply(ast.literal_eval)
cs = [db.loc[i].scp_codes for i in d['ecg_id']]
Y = np.concatenate([d['Y'], np.array([[c in s for c in labels[5:]] for s in cs], dtype=np.float32)], 1)

n = np.load('model/norm.npz')
m = ResNet1D(n_out=len(labels)); m.load_state_dict(torch.load('model/resnet1d_full.pt', map_location='cpu')); m.eval()
te = np.where(fold == 10)[0]
x = X[te] - np.median(X[te], 1, keepdims=True)
x = ((x - n['mu']) / n['sd']).transpose(0, 2, 1).astype(np.float32)
with torch.no_grad():
    pt = np.concatenate([torch.sigmoid(m(torch.from_numpy(x[i:i + 512]))).numpy() for i in range(0, len(x), 512)])

thr, rows = {}, []
for k, lab in enumerate(labels):
    t = R['thr'][lab] if k < 5 else max(R['thr'][lab], FLOOR); thr[lab] = t
    yt = Y[te, k] == 1; yp = pt[:, k] >= t; old = R['metrics'][k]
    rows.append({'label': lab, 'deskripsi': old['deskripsi'], 'n_latih': old['n_latih'], 'n_uji': int(yt.sum()), 'auc': old['auc'],
                 'sens': float((yp & yt).sum() / max(1, yt.sum())), 'spes': float((~yp & ~yt).sum() / max(1, (~yt).sum())),
                 'ppv': float((yp & yt).sum() / max(1, yp.sum())), 'ambang': t})
R.update(thr=thr, metrics=rows, ambang_minimum_kode=FLOOR)
json.dump(R, open('hasil/full_res.json', 'w'), indent=1)
json.dump({'labels': labels, 'thr': thr, 'ambang_minimum_kode': FLOOR, 'auc_super_test': R['auc_super_test'],
           'auc_codes_test': R['auc_codes_test'], 'metrics': rows}, open('model/labels.json', 'w'), indent=1, ensure_ascii=False)
print(f"label {len(labels)} · macro AUC uji superclass {R['auc_super_test']:.3f} · kode {R['auc_codes_test']:.3f} · ditulis model/labels.json")
