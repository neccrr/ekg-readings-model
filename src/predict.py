"""Inferensi: input rekaman EKG 12 sadapan -> probabilitas diagnosis.

Contoh:
  python src/predict.py contoh/ekg_mi.csv
  python src/predict.py rekaman.csv --fs 500 --plot hasil.png
  python src/predict.py data/raw/.../records100/00000/00001_lr     (format WFDB, tanpa ekstensi)

Format CSV: 12 kolom (satu per sadapan), satuan mV, satu baris per sampel.
Baris header opsional; jika ada nama sadapan (I, II, III, aVR, aVL, aVF, V1..V6), urutan kolom bebas.
Rekaman > 10 detik dipotong per jendela 10 detik (geser 5 detik; geser 10 detik bila > 60 detik) lalu probabilitasnya dirata-rata.
Offset DC (median) tiap sadapan dibuang per jendela, sehingga data Holter dengan baseline bergeser tetap terbaca.

PERINGATAN: model penelitian/pembelajaran, bukan alat diagnosis klinis.
"""
import argparse, json, os, sys
import numpy as np, torch
from scipy.signal import resample_poly
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NAMA = {'NORM': 'EKG normal', 'MI': 'Infark miokard', 'STTC': 'Perubahan segmen ST / gelombang T',
        'CD': 'Gangguan konduksi', 'HYP': 'Hipertrofi'}
FS_MODEL, N_SAMPLES = 100, 1000

# Arsitektur sama dengan train_dl.py (disalin agar predict.py tidak perlu memuat dataset)
import torch.nn as nn
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
        return s.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


def load_signal(path, fs=None):
    """Kembalikan (sinyal [N, 12] dalam mV, urutan LEADS; fs)."""
    if path.endswith('.csv') or path.endswith('.txt'):
        with open(path) as f:
            first = f.readline()
        has_header = any(c.isalpha() for c in first)
        sig = np.loadtxt(path, delimiter=',' if ',' in first else None, skiprows=int(has_header))
        names = [h.strip().strip('"').upper() for h in first.replace(';', ',').split(',')] if has_header else None
    else:  # WFDB
        import wfdb
        rec = wfdb.rdrecord(path.removesuffix('.hea').removesuffix('.dat'))
        sig, names, fs = rec.p_signal, [n.upper() for n in rec.sig_name], fs or rec.fs
    if sig.ndim != 2 or 12 not in sig.shape:
        raise ValueError(f'Butuh 12 sadapan, didapat bentuk {sig.shape}.')
    if sig.shape[1] != 12:
        sig = sig.T
    if names and set(LEADS) <= set(names):
        sig = sig[:, [names.index(l) for l in LEADS]]
    elif names:
        print('Peringatan: nama kolom tidak dikenali; diasumsikan urutan I, II, III, aVR, aVL, aVF, V1-V6.')
    return sig.astype(np.float32), fs or FS_MODEL


def windows(sig, fs):
    if fs != FS_MODEL:
        fr = Fraction(FS_MODEL, int(round(fs))).limit_denominator(10000)
        sig = resample_poly(sig, fr.numerator, fr.denominator, axis=0)
    sig = np.nan_to_num(sig)
    if len(sig) < N_SAMPLES:
        if len(sig) < 5 * FS_MODEL:
            raise ValueError(f'Rekaman terlalu pendek ({len(sig)/FS_MODEL:.1f} s); minimal 5 detik. Periksa frekuensi sampling.')
        sig = np.pad(sig, ((0, N_SAMPLES - len(sig)), (0, 0)))
    step = N_SAMPLES if len(sig) > 6 * N_SAMPLES else N_SAMPLES // 2  # > 60 s: tanpa tumpang tindih
    starts = range(0, len(sig) - N_SAMPLES + 1, step)
    return sig, np.stack([sig[s:s + N_SAMPLES] for s in starts])


def load_model():
    """Muat bobot, normalisasi, dan ambang sekali saja.
    Bila model/resnet1d_full.pt ada, dipakai model lengkap (5 superclass + kode SCP rinci)."""
    full = os.path.join(ROOT, 'model', 'resnet1d_full.pt'), os.path.join(ROOT, 'model', 'labels.json')
    if all(os.path.exists(f) for f in full):
        meta = json.load(open(full[1]))
        m = ResNet1D(n_out=len(meta['labels'])); m.load_state_dict(torch.load(full[0], map_location='cpu'))
        labels, thr = meta['labels'], meta['thr']
        auc = {r['label']: r['auc'] for r in meta.get('metrics', []) if r.get('auc')}
    else:
        m = ResNet1D(); m.load_state_dict(torch.load(os.path.join(ROOT, 'model', 'resnet1d_best.pt'), map_location='cpu'))
        labels, thr = CLASSES, json.load(open(os.path.join(ROOT, 'model', 'thresholds.json')))
        auc = {}
    m.eval()
    return {'model': m, 'norm': np.load(os.path.join(ROOT, 'model', 'norm.npz')), 'labels': labels, 'thr': thr, 'auc': auc}


def predict_signal(sig, fs, M, laporan=True):
    """sig: array [N, 12] dalam mV (urutan I..V6), fs dalam Hz. Kembalikan dict hasil."""
    import time
    from measure import measure
    from report import build, NAMA_KODE
    t0 = time.perf_counter()
    sig100, W = windows(np.asarray(sig, dtype=np.float32), fs)
    W = W - np.median(W, axis=1, keepdims=True)  # buang offset DC per sadapan per jendela (penting untuk data Holter)
    x = ((W - M['norm']['mu']) / M['norm']['sd']).transpose(0, 2, 1).astype(np.float32)
    t1 = time.perf_counter()
    with torch.no_grad():
        p = torch.sigmoid(M['model'](torch.from_numpy(x))).numpy().mean(0)
    t2 = time.perf_counter()
    thr = M['thr']
    sup = M['labels'][:5]  # 5 superclass selalu di depan
    res = [{'kode': c, 'diagnosis': NAMA[c], 'probabilitas': round(float(p[k]), 4),
            'ambang': round(thr[c] if isinstance(thr, dict) and c in thr and not isinstance(thr[c], list) else thr[c], 4),
            'positif': bool(p[k] >= thr[c])} for k, c in enumerate(sup)]
    kode = [{'kode': c, 'nama': NAMA_KODE.get(c, c), 'probabilitas': round(float(p[k]), 4), 'ambang': round(thr[c], 4),
             'positif': bool(p[k] >= thr[c])} for k, c in enumerate(M['labels']) if k >= 5]
    pos = sorted([r for r in res if r['positif'] and r['kode'] != 'NORM'], key=lambda r: -r['probabilitas'])
    if pos:
        kesimpulan = 'Abnormal: ' + ', '.join(f"{r['diagnosis']} ({r['probabilitas']:.0%})" for r in pos)
    elif res[0]['positif']:
        kesimpulan = f"Kemungkinan EKG normal ({res[0]['probabilitas']:.0%})"
    else:
        kesimpulan = 'Tidak meyakinkan: tidak ada kelas yang melewati ambang'
    out = {'kesimpulan': kesimpulan, 'durasi_detik': len(sig100) / FS_MODEL, 'jumlah_jendela': len(W),
           'hasil': res, 'kode': kode,
           'timing': {'preprocess_ms': round((t1 - t0) * 1000, 2), 'inference_ms': round((t2 - t1) * 1000, 2)},
           '_sig100': sig100}
    if laporan:
        t3 = time.perf_counter()
        meas = measure(sig100, FS_MODEL)
        mo = {r['kode']: (r['probabilitas'], r['ambang']) for r in res}
        mo.update({r['kode']: (r['probabilitas'], r['ambang']) for r in kode})
        out['pengukuran'] = meas
        out['laporan'] = build(mo, meas, M.get('auc'))
        # kesimpulan dari laporan (sama dengan tampilan web)
        rep = out['laporan']; g = {b['nama']: b for b in rep[0]['butir']}
        ab = [b['nama'].split(' (')[0] for k in rep[1:] for b in k['butir'] if b['positif'] and b['nama'] != 'Irama sinus']
        if g['Tidak dapat dibaca (artefak/sadapan hilang)']['positif']:
            out['kesimpulan'] = 'Tidak dapat dibaca: ' + ((meas.get('peringatan') or ['kualitas sinyal buruk'])[0])
        elif ab:
            out['kesimpulan'] = 'Abnormal: ' + ', '.join(ab)
        elif g['EKG normal']['positif']:
            out['kesimpulan'] = f"Kemungkinan EKG normal ({res[0]['probabilitas']:.0%})"
        else:
            out['kesimpulan'] = 'Tidak meyakinkan: tidak ada temuan yang melewati ambang'
        out['timing']['measure_ms'] = round((time.perf_counter() - t3) * 1000, 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Prediksi diagnosis dari EKG 12 sadapan.')
    ap.add_argument('input', help='file CSV, atau nama rekaman WFDB')
    ap.add_argument('--fs', type=float, help='frekuensi sampling (Hz) untuk CSV; default 100')
    ap.add_argument('--plot', help='simpan gambar EKG + hasil ke file PNG ini')
    ap.add_argument('--json', action='store_true', help='keluaran JSON')
    a = ap.parse_args(argv)

    if not (os.path.exists(a.input) or os.path.exists(a.input + '.hea')):
        sys.exit(f'File tidak ditemukan: {a.input}')
    try:
        sig, fs = load_signal(a.input, a.fs)
        out = predict_signal(sig, fs, load_model())
    except ValueError as e:
        sys.exit(str(e))
    sig100, W, res, kesimpulan = out['_sig100'], range(out['jumlah_jendela']), out['hasil'], out['kesimpulan']

    if a.json:
        print(json.dumps({k: v for k, v in out.items() if not k.startswith('_')}, indent=1, ensure_ascii=False)); return
    print(f'\nFile    : {a.input}\nDurasi  : {len(sig100)/FS_MODEL:.1f} s ({len(W)} jendela 10 s)\n')
    print(f"{'Kode':<5} {'Diagnosis':<36} {'Prob':>6} {'Ambang':>7}  Hasil")
    for r in res:
        bar = '#' * int(r['probabilitas'] * 20)
        print(f"{r['kode']:<5} {r['diagnosis']:<36} {r['probabilitas']:>6.2f} {r['ambang']:>7.2f}  "
              f"{'POSITIF' if r['positif'] else '-':<8}{bar}")
    print(f'\nKesimpulan: {kesimpulan}')
    m = out.get('pengukuran') or {}
    if m:
        iv = m.get('interval_ms') or {}
        print(f"\nPengukuran: {m.get('irama')} · {m.get('frekuensi_jantung', '—')}/menit · PR {iv.get('PR') or '—'} · QRS {iv.get('QRS') or '—'} · "
              f"QT/QTc {iv.get('QT') or '—'}/{iv.get('QTc_Bazett') or '—'} ms · aksis {m.get('aksis_derajat', '—')}°")
        for w in m.get('peringatan', []):
            print('  ! ' + w)
    for kat in out.get('laporan', []):
        pos = [b for b in kat['butir'] if b['positif']]
        print(f"\n{kat['kategori']}: " + ('—' if not pos else ''))
        for b in pos:
            print(f"  + {b['nama']}  [{'; '.join(b['dasar'])}]")
    print('Catatan: model pembelajaran (PTB-XL), bukan alat diagnosis klinis. Konfirmasi dengan dokter.\n')

    if a.plot:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        seg = sig100[:N_SAMPLES]; t = np.arange(len(seg)) / FS_MODEL
        names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        fig, ax = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
        for l in range(12):
            g = ax[l % 6, l // 6]
            g.plot(t, seg[:, l], 'k', lw=0.8); g.set_ylabel(names[l], rotation=0, labelpad=14)
            g.grid(True, color='#f4b6b6', lw=0.5); g.set_yticks([])
        fig.suptitle(kesimpulan + '\n' + '  ·  '.join(f"{r['kode']} {r['probabilitas']:.2f}" for r in res), fontsize=12)
        ax[-1, 0].set_xlabel('Waktu (s)'); ax[-1, 1].set_xlabel('Waktu (s)')
        plt.tight_layout(); plt.savefig(a.plot, dpi=110); print('Gambar disimpan:', a.plot)


if __name__ == '__main__':
    main()
