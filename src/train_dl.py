"""Deep learning: 1D ResNet CNN untuk klasifikasi 5 superclass PTB-XL (multi-label).
Split standar PTB-XL: fold 1-8 latih, fold 9 validasi, fold 10 uji."""
import time, json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
d = np.load('data/ptbxl100.npz')
X, Y, fold = d['X'], d['Y'], d['fold']
tr, va, te = fold <= 8, fold == 9, fold == 10
mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1))           # normalisasi per sadapan
X = ((X - mu) / sd).transpose(0, 2, 1).astype(np.float32)  # (N, 12, 1000)
np.savez('model/norm.npz', mu=mu, sd=sd)

class Block(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__()
        s.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); s.b1 = nn.BatchNorm1d(cout)
        s.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False); s.b2 = nn.BatchNorm1d(cout)
        s.sk = nn.Sequential() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        s.act = nn.ReLU()
    def forward(s, x):
        return s.act(s.b2(s.c2(s.act(s.b1(s.c1(x))))) + s.sk(x))

class ResNet1D(nn.Module):
    def __init__(s, n_out=5, w=64):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(12, w, 15, 2, 7, bias=False), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        s.layers = nn.Sequential(Block(w, w, 1), Block(w, w, 1), Block(w, 2*w, 2), Block(2*w, 2*w, 1),
                                 Block(2*w, 2*w, 2), Block(2*w, 2*w, 1))
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4*w, n_out))
    def forward(s, x):
        h = s.layers(s.stem(x))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)  # global avg + max pooling
        return s.head(h)

def predict(m, x, bs=512):
    m.eval(); out = []; dv = next(m.parameters()).device
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(torch.sigmoid(m(torch.from_numpy(x[i:i+bs]).to(dv))).cpu().numpy())
    return np.concatenate(out)

if __name__ == '__main__':
    m = ResNet1D().to(dev)
    print('parameter:', sum(p.numel() for p in m.parameters()))
    EPOCHS, BS = 20, 64
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    steps = EPOCHS * int(np.ceil(tr.sum() / BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    Xtr, Ytr = X[tr], Y[tr]
    best, hist = 0, []
    for ep in range(EPOCHS):
        m.train(); t0 = time.time(); perm = np.random.permutation(len(Xtr)); tl = 0
        for i in range(0, len(perm), BS):
            idx = perm[i:i+BS]
            xb = torch.from_numpy(Xtr[idx]).to(dev)
            # augmentasi ringan: skala amplitudo + noise + geser waktu
            xb = xb * (1 + 0.1*torch.randn(len(idx), 1, 1, device=dev)) + 0.02*torch.randn_like(xb)
            xb = torch.roll(xb, int(np.random.randint(-50, 50)), -1)
            loss = lossf(m(xb), torch.from_numpy(Ytr[idx]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item()*len(idx)
        pv = predict(m, X[va]); auc = roc_auc_score(Y[va], pv, average='macro')
        hist.append({'epoch': ep+1, 'train_loss': tl/len(Xtr), 'val_auc': auc})
        print(f'ep {ep+1:2d} loss {tl/len(Xtr):.4f} val macroAUC {auc:.4f} ({time.time()-t0:.0f}s)', flush=True)
        if auc > best: best = auc; torch.save(m.state_dict(), 'model/resnet1d_best.pt')
    m.load_state_dict(torch.load('model/resnet1d_best.pt', map_location=dev))
    pt = predict(m, X[te]); np.save('data/pred_dl_test.npy', pt)
    res = {'macro_auc_test': roc_auc_score(Y[te], pt, average='macro'),
           'per_class': {c: roc_auc_score(Y[te][:, k], pt[:, k]) for k, c in enumerate(CLASSES)},
           'best_val_auc': best, 'history': hist}
    json.dump(res, open('hasil/res_dl.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=1))
