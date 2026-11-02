"""Evaluasi + visualisasi: kurva ROC, perbandingan AUC ML vs DL, contoh EKG + peta saliency."""
import json, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from train_dl import ResNet1D, X, Y, fold, CLASSES

OUT = 'hasil/'
import os; os.makedirs(OUT, exist_ok=True)
te = fold == 10; Yt = Y[te]
pdl = np.load('data/pred_dl_test.npy'); pml = np.load('data/pred_ml_test.npy')
NAMA = {'NORM': 'Normal', 'MI': 'Infark miokard', 'STTC': 'Perubahan ST/T', 'CD': 'Gangguan konduksi', 'HYP': 'Hipertrofi'}
COL = ['#2a7de1', '#e8590c', '#2f9e44', '#9c36b5', '#c2255c']

# 1. Kurva ROC per kelas (DL) + garis putus ML
fig, ax = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
for k, c in enumerate(CLASSES):
    for p, ls, lab in [(pdl, '-', 'CNN'), (pml, '--', 'ML klasik')]:
        f, t, _ = roc_curve(Yt[:, k], p[:, k])
        ax[k].plot(f, t, ls, color=COL[k] if ls == '-' else 'gray', lw=2,
                   label=f'{lab} AUC {roc_auc_score(Yt[:, k], p[:, k]):.3f}')
    ax[k].plot([0, 1], [0, 1], ':', color='lightgray'); ax[k].set_title(f'{c} – {NAMA[c]}')
    ax[k].set_xlabel('1 − spesifisitas'); ax[k].legend(loc='lower right', fontsize=9)
ax[0].set_ylabel('Sensitivitas'); plt.tight_layout(); plt.savefig(OUT + 'roc_curves.png', dpi=130); plt.close()

# 2. Sensitivitas & spesifisitas pada ambang yang dipilih dari set validasi (Youden)
m = ResNet1D(); m.load_state_dict(torch.load('model/resnet1d_best.pt', map_location='cpu')); m.eval()
from train_dl import predict
pv = predict(m, X[fold == 9]); Yv = Y[fold == 9]
rows = []
for k, c in enumerate(CLASSES):
    f, t, th = roc_curve(Yv[:, k], pv[:, k]); thr = th[np.argmax(t - f)]
    yp = pdl[:, k] >= thr; yt = Yt[:, k] == 1
    sens = (yp & yt).sum() / yt.sum(); spec = (~yp & ~yt).sum() / (~yt).sum(); ppv = (yp & yt).sum() / yp.sum()
    rows.append({'kelas': c, 'nama': NAMA[c], 'prevalensi_uji': float(yt.mean()), 'ambang': float(thr),
                 'auc_cnn': roc_auc_score(yt, pdl[:, k]), 'auc_ml': roc_auc_score(yt, pml[:, k]),
                 'sensitivitas': float(sens), 'spesifisitas': float(spec), 'ppv': float(ppv)})
json.dump(rows, open(OUT + 'metrics_test.json', 'w'), indent=1)

# 3. Contoh EKG infark miokard + saliency (gradien input) dari model CNN
idx_te = np.where(te)[0]
cand = [i for j, i in enumerate(idx_te) if Yt[j, 1] == 1 and Yt[j].sum() == 1 and pdl[j, 1] > 0.9]
i = cand[0]
x = torch.from_numpy(X[i:i+1]).requires_grad_(True)
torch.sigmoid(m(x))[0, 1].backward()
sal = x.grad.abs()[0].numpy(); sal = np.convolve(sal.sum(0), np.ones(9)/9, 'same'); sal /= sal.max()
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
raw = np.load('data/ptbxl100.npz')['X'][i]
tt = np.arange(1000) / 100
fig, ax = plt.subplots(12, 1, figsize=(14, 13), sharex=True)
for l in range(12):
    ax[l].imshow(sal[None, :], aspect='auto', cmap='Reds', alpha=0.45, extent=[0, 10, raw[:, l].min(), raw[:, l].max()])
    ax[l].plot(tt, raw[:, l], 'k', lw=0.8); ax[l].set_ylabel(LEADS[l], rotation=0, labelpad=15); ax[l].set_yticks([])
pr = pdl[np.where(idx_te == i)[0][0]]
ax[0].set_title('EKG uji berlabel MI — prediksi CNN: ' + ', '.join(f'{c} {p:.2f}' for c, p in zip(CLASSES, pr)) +
                '\nArsiran merah = bagian sinyal yang paling memengaruhi keputusan model (saliency)')
ax[-1].set_xlabel('Waktu (detik)'); plt.tight_layout(); plt.savefig(OUT + 'contoh_mi_saliency.png', dpi=110); plt.close()

# 4. Kurva pelatihan
h = json.load(open('hasil/res_dl.json'))['history']
fig, a1 = plt.subplots(figsize=(7, 4)); a1.plot([e['epoch'] for e in h], [e['train_loss'] for e in h], 'o-', label='loss latih')
a2 = a1.twinx(); a2.plot([e['epoch'] for e in h], [e['val_auc'] for e in h], 's-', color='#e8590c', label='AUC validasi')
a1.set_xlabel('Epoch'); a1.set_ylabel('BCE loss'); a2.set_ylabel('Macro AUC validasi'); fig.legend(loc='center right')
plt.tight_layout(); plt.savefig(OUT + 'kurva_pelatihan.png', dpi=130); plt.close()
print(json.dumps(rows, indent=1))
