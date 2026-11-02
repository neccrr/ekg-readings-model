"""Model multi-tugas: 5 superclass + semua kode SCP PTB-XL dengan >= 50 contoh.
Backbone diinisialisasi dari model 5 kelas; fold 1-8 latih, 9 validasi, 10 uji."""
import ast, glob, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve
import sys; sys.path.insert(0, 'src')
from predict import ResNet1D

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
root = glob.glob('data/raw/*')[0] + '/'
db = pd.read_csv(root + 'ptbxl_database.csv', index_col='ecg_id'); db.scp_codes = db.scp_codes.apply(ast.literal_eval)
scp = pd.read_csv(root + 'scp_statements.csv', index_col=0)
d = np.load('data/ptbxl100.npz'); X, Y5, fold, ids = d['X'], d['Y'], d['fold'], d['ecg_id']
codes_per = [db.loc[i].scp_codes for i in ids]
cnt = pd.Series([c for cs in codes_per for c in cs]).value_counts()
CODES = [c for c in cnt.index if cnt[c] >= 50 and c != 'NORM']  # NORM sudah ada sebagai superclass
YC = np.array([[c in cs for c in CODES] for cs in codes_per], dtype=np.float32)
Y = np.concatenate([Y5, YC], 1); LABELS = ['NORM', 'MI', 'STTC', 'CD', 'HYP'] + CODES
print(len(CODES), 'kode:', CODES, flush=True)

tr, va, te = fold <= 8, fold == 9, fold == 10
norm = np.load('model/norm.npz'); mu, sd = norm['mu'], norm['sd']
Xn = X - np.median(X, 1, keepdims=True)
Xn = ((Xn - mu) / sd).transpose(0, 2, 1).astype(np.float32)

m = ResNet1D(n_out=len(LABELS))
old = torch.load('model/resnet1d_best.pt', map_location='cpu')
m.load_state_dict({k: v for k, v in old.items() if not k.startswith('head')}, strict=False)
with torch.no_grad():  # salin bobot head 5 superclass
    m.head[1].weight[:5] = old['head.1.weight']; m.head[1].bias[:5] = old['head.1.bias']

def predict(x, bs=512):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), bs): out.append(torch.sigmoid(m(torch.from_numpy(x[i:i + bs]))).numpy())
    return np.concatenate(out)

def mauc(y, p):
    ok = (y.sum(0) > 0) & (y.sum(0) < len(y))
    return roc_auc_score(y[:, ok], p[:, ok], average='macro')

EPOCHS, BS = 14, 64
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * int(np.ceil(tr.sum() / BS)))
lossf = nn.BCEWithLogitsLoss()
Xtr, Ytr = Xn[tr], Y[tr]; best = 0; hist = []
for ep in range(EPOCHS):
    m.train(); t0 = time.time(); perm = np.random.permutation(len(Xtr)); tl = 0
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]; xb = torch.from_numpy(Xtr[idx])
        xb = xb * (1 + 0.1 * torch.randn(len(idx), 1, 1)) + 0.02 * torch.randn_like(xb)
        xb = torch.roll(xb, int(np.random.randint(-50, 50)), -1)
        loss = lossf(m(xb), torch.from_numpy(Ytr[idx]))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(idx)
    pv = predict(Xn[va]); a5 = roc_auc_score(Y[va][:, :5], pv[:, :5], average='macro'); ac = mauc(Y[va][:, 5:], pv[:, 5:])
    hist.append({'epoch': ep + 1, 'loss': tl / len(Xtr), 'val_auc5': a5, 'val_auc_codes': ac})
    print(f'ep {ep+1:2d} loss {tl/len(Xtr):.4f} val AUC super {a5:.4f} kode {ac:.4f} ({time.time()-t0:.0f}s)', flush=True)
    if a5 + ac > best: best = a5 + ac; torch.save(m.state_dict(), 'model/resnet1d_full.pt')

m.load_state_dict(torch.load('model/resnet1d_full.pt'))
pv, pt = predict(Xn[va]), predict(Xn[te])
thr, rows = {}, []
for k, lab in enumerate(LABELS):
    yv, yt = Y[va][:, k], Y[te][:, k]
    if k < 5:   # superclass: ambang Youden (sama seperti sebelumnya)
        from sklearn.metrics import roc_curve
        f, t, th = roc_curve(yv, pv[:, k]); thr[lab] = float(th[np.argmax(t - f)])
    else:       # kode rinci: ambang F1 maksimum (lebih sedikit positif palsu untuk kode jarang)
        pr, rc, th = precision_recall_curve(yv, pv[:, k]); f1 = 2 * pr * rc / (pr + rc + 1e-9)
        thr[lab] = float(th[np.argmax(f1[:-1])]) if len(th) else 0.5
    yp = pt[:, k] >= thr[lab]
    rows.append({'label': lab, 'deskripsi': 'superclass' if k < 5 else scp.loc[lab, 'description'], 'n_latih': int(Y[tr][:, k].sum()),
                 'n_uji': int(yt.sum()), 'auc': float(roc_auc_score(yt, pt[:, k])) if 0 < yt.sum() < len(yt) else None,
                 'sens': float((yp & (yt == 1)).sum() / max(1, yt.sum())), 'spes': float((~yp & (yt == 0)).sum() / max(1, (yt == 0).sum())),
                 'ppv': float((yp & (yt == 1)).sum() / max(1, yp.sum())), 'ambang': thr[lab]})
json.dump({'labels': LABELS, 'thr': thr, 'metrics': rows, 'history': hist,
           'auc_super_test': roc_auc_score(Y[te][:, :5], pt[:, :5], average='macro'),
           'auc_codes_test': mauc(Y[te][:, 5:], pt[:, 5:])}, open('hasil/full_res.json', 'w'), indent=1)
print(pd.DataFrame(rows).round(3).to_string())
