"""Digitalisasi gambar EKG 12 sadapan format 3x4 + strip ritme (contoh: ecglibrary.com/lvhlah.gif).
Pemakaian: python tools/digitize.py gambar.gif rows.npy
Konstanta (skala, garis dasar, posisi kolom) disetel untuk gambar contoh; sesuaikan untuk gambar lain:
  PX_MV = tinggi pulsa kalibrasi 1 mV (px), PX_S = px per detik, X0 = awal jejak, BASE = garis dasar tiap baris,
  bounds = posisi garis pemisah kolom.
"""
import numpy as np, sys
from PIL import Image
im = np.array(Image.open(sys.argv[1]).convert('L')) < 128
H, W = im.shape
PX_MV, PX_S, X0 = 31.0, 74.4, 26   # 1 mV = 31 px (pulsa kalibrasi); garis pemisah kolom di x=212/398/584 -> 186 px per 2,5 s
BASE = [123, 241, 359, 475]

def runs(col, lo, hi):
    ys = np.where(col[lo:hi])[0] + lo
    if len(ys) == 0: return []
    out, s = [], ys[0]
    for a, b in zip(ys[:-1], ys[1:]):
        if b != a + 1: out.append((s, a)); s = b
    out.append((s, ys[-1])); return out

def trace_seg(row, xa, xb):
    """Telusuri jejak satu segmen [xa, xb) mulai dari garis dasar; pulih otomatis bila hilang."""
    b = BASE[row]; lo, hi = max(0, b - 170), min(H, b + 170)
    y, lost = float(b), 0; xs, vs = [], []
    for x in range(xa, xb):
        rs = runs(im[:, x], lo, hi)
        touch = [r for r in rs if r[0] - 2 <= y <= r[1] + 2]
        if not touch:   # celah antar-kolom pada lonjakan curam: cari run terdekat
            lim = 45 if lost < 3 else 25
            cand = [r for r in rs if min(abs(r[0] - y), abs(r[1] - y)) <= lim]
            if lost >= 3:  # pulihkan dekat garis dasar
                cand = [r for r in rs if r[0] - 12 <= b <= r[1] + 12] or cand
            touch = sorted(cand, key=lambda r: min(abs(r[0] - y), abs(r[1] - y)))[:1]
        if touch:
            r = touch[0]; lost = 0
            if r[1] - r[0] <= 2: y = (r[0] + r[1]) / 2
            else: y = float(r[0] if abs(r[0] - y) > abs(r[1] - y) else r[1])
            xs.append(x); vs.append((b - y) / PX_MV)
        else:
            lost += 1
    return np.array(xs), np.array(vs)

T = np.arange(0, 10, 0.01)
bounds = [X0 + 3, 212, 398, 584, min(W, X0 + 744)]
rows = []
for row in range(4):
    segs = [(bounds[0], bounds[-1])] if row == 3 else [(bounds[k] + (6 if k else 0), bounds[k + 1] - 6) for k in range(4)]
    xs_all, vs_all = [], []
    for xa, xb in segs:
        xs, vs = trace_seg(row, xa, min(xb, W))
        xs_all += list(xs); vs_all += list(vs)
    xs_all, vs_all = np.array(xs_all), np.array(vs_all)
    t = (xs_all - X0) / PX_S
    y = np.interp(T, t, vs_all)
    # tandai celah > 30 ms tanpa data
    gap = np.array([np.min(np.abs(t - tt)) > 0.03 for tt in T]); y[gap] = np.nan
    if row < 3:  # zona garis pemisah antar-kolom dikosongkan
        for k in (1, 2, 3): y[(T > 2.5 * k - 0.08) & (T < 2.5 * k + 0.08)] = np.nan
    rows.append(y)
rows = np.stack(rows); np.save(sys.argv[2], rows)
print('cakupan per baris (s):', [round(np.isfinite(r).sum() / 100, 2) for r in rows])
