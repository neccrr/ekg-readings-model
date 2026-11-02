"""Bangun CSV dari hasil digitize.py.
Pemakaian: python tools/build_csv.py rows.npy mentah.csv siap_model.csv
- mentah: data yang ada di gambar saja (2,5 s per sadapan + strip ritme V1)
- siap_model: 12 sadapan x 10 s; di luar segmen asli diisi median beat pada waktu detak strip ritme
"""
import numpy as np, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from measure import _bp, _detect_qrs
R = np.load(sys.argv[1]); out_raw, out_full = sys.argv[2], sys.argv[3]
T = np.arange(0, 10, 0.01); N = len(T)
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# tata letak cetak 3x4: baris -> [kolom0..3]
LAYOUT = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]
fill = lambda y: np.interp(T, T[np.isfinite(y)], y[np.isfinite(y)])
raw = {l: np.full(N, np.nan) for l in LEADS}
for r, names in enumerate(LAYOUT):
    for c, l in enumerate(names):
        m = (T >= 2.5 * c) & (T < 2.5 * (c + 1)); raw[l][m] = R[r][m]
rhythm = fill(R[3])
# R dari strip ritme V1 (10 s, simultan dengan semua kolom)
x = _bp(np.tile(rhythm[:, None], (1, 12)), 100, 0.5, 40)
rp, _ = _detect_qrs(x, 100)
rr = np.diff(rp) / 100
print('detak di strip V1:', len(rp), 'frekuensi', round(60 / np.median(rr)), '/menit')
A, B = 45, 60   # templat -0,45..+0,6 s
full = {}
for l in LEADS:
    y = raw[l]; ok = np.isfinite(y)
    beats = [y[p - A:p + B] for p in rp if p - A >= 0 and p + B <= N and ok[p - A:p + B].all()]
    if not beats:  # segmen hanya memuat denyut terpotong: pakai bagian yang ada
        beats = [y[p - A:p + B] for p in rp if p - A >= 0 and p + B <= N and ok[p - 10:p + 30].all()]
    tpl = np.nanmedian(np.stack(beats), 0)
    tpl = np.nan_to_num(tpl - np.nanmedian(tpl[A - 25:A - 10]))
    rec = np.zeros(N)
    for i in range(N):                       # tiap sampel ikut denyut terdekat
        k = np.argmin(np.abs(rp - i)); off = i - rp[k] + A
        rec[i] = tpl[off] if 0 <= off < A + B else 0
    full[l] = np.where(ok, fill(y), rec)
    full[l][ok] = y[ok]
hdr = 'waktu_s,' + ','.join(LEADS) + ',V1_ritme'
np.savetxt(out_raw, np.column_stack([T] + [raw[l] for l in LEADS] + [rhythm]), delimiter=',', header=hdr, comments='', fmt='%.4f')
np.savetxt(out_full, np.column_stack([full[l] for l in LEADS]), delimiter=',', header=','.join(LEADS), comments='', fmt='%.4f')
