"""Pengukuran EKG berbasis aturan (tanpa pembelajaran mesin).

Melengkapi model: model hanya tahu label yang ada di PTB-XL, sedangkan modul ini mengukur
frekuensi jantung, irama, interval (PR, QRS, QT/QTc), aksis, voltase, deviasi ST, denyut prematur,
dan mengenali pola yang tidak ada di PTB-XL (asistol, VF, takikardia kompleks lebar, blok AV derajat 2/3,
flutter, irama junctional/idioventrikular).

Semua angka adalah perkiraan otomatis dari median beat; akurasinya di bawah pembacaan manual.
Input: sig [N, 12] dalam mV, urutan I, II, III, aVR, aVL, aVF, V1..V6; fs dalam Hz.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
GROUPS = {'inferior': ['II', 'III', 'aVF'], 'lateral': ['I', 'aVL', 'V5', 'V6'],
          'septal': ['V1', 'V2'], 'anterior': ['V3', 'V4']}
LI = {l: i for i, l in enumerate(LEADS)}
_r = lambda v: int(np.floor(v + 0.5))  # pembulatan sama dengan Math.round di versi JS


def _bp(x, fs, lo, hi, order=2):
    hi = min(hi, 0.45 * fs)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], 'band')
    return filtfilt(b, a, x, axis=0)


def _quality(sig, fs):
    sd = sig.std(0)
    flat = [LEADS[i] for i in range(12) if sd[i] < 0.01]
    hf = _bp(sig, fs, 35, 0.45 * fs) if fs > 80 else np.zeros_like(sig)
    noise = hf.std(0) / (sd + 1e-6)
    noisy = [LEADS[i] for i in range(12) if noise[i] > 0.5 and LEADS[i] not in flat]
    return flat, noisy


def _detect_qrs(x, fs):
    """Deteksi QRS gabungan 12 sadapan (energi turunan, mirip Pan-Tompkins)."""
    f = _bp(x, fs, 5, 20)
    e = np.abs(np.diff(f, axis=0, prepend=f[:1])).sum(1)
    w = max(1, int(0.08 * fs)); e = np.convolve(e, np.ones(w) / w, 'same')
    if e.max() <= 0:
        return np.array([], int), e
    thr = 0.3 * np.percentile(e, 99)
    pk, _ = find_peaks(e, height=thr, distance=int(0.25 * fs))
    # buang puncak yang jauh lebih kecil dari median (gelombang T besar)
    if len(pk) > 2:
        pk = pk[e[pk] > 0.35 * np.median(e[pk])]
    # perhalus posisi R: puncak |sinyal| di sadapan dengan amplitudo terbesar ±60 ms
    lead = np.argmax(np.ptp(f, 0)); r = []
    h = int(0.06 * fs)
    for p in pk:
        a, b = max(0, p - h), min(len(x), p + h)
        r.append(a + np.argmax(np.abs(f[a:b, lead])))
    return np.unique(np.array(r, int)), e


def _template(x, r, fs, pre=0.35, post=0.65):
    a, b = int(pre * fs), int(post * fs)
    beats = [x[i - a:i + b] for i in r if i - a >= 0 and i + b <= len(x)]
    if not beats:
        return None, a
    return np.median(np.stack(beats), 0), a


def _onset_offset(tpl, a, fs):
    """Awal & akhir QRS dari kecepatan spasial median beat."""
    v = np.abs(np.diff(tpl, axis=0)).sum(1)
    v = np.convolve(v, np.ones(max(1, int(0.012 * fs))) / max(1, int(0.012 * fs)), 'same')
    base = np.percentile(v, 10); pk = v[a - int(0.06 * fs):a + int(0.06 * fs)].max()
    thr = base + 0.07 * (pk - base)
    on = a
    while on > a - int(0.12 * fs) and v[on] > thr:
        on -= 1
    off = a
    while off < a + int(0.16 * fs) and off < len(v) - 1 and v[off] > thr:
        off += 1
    return on, off


def measure(sig, fs):
    sig = np.asarray(sig, dtype=float)
    out = {'temuan': [], 'peringatan': []}
    dur = len(sig) / fs
    flat, noisy = _quality(sig, fs)
    out['kualitas'] = {'sadapan_datar': flat, 'sadapan_bising': noisy}
    x = _bp(sig, fs, 0.5, 40)
    amp = np.ptp(x, 0)

    # ---- Asistol: hampir tidak ada aktivitas listrik di semua sadapan ----
    if np.median(amp) < 0.1:
        out['irama'] = 'Asistol / garis datar'
        out['temuan'].append({'kode': 'ASYST', 'nama': 'Asistol (garis datar)', 'dasar': f'amplitudo median {np.median(amp):.2f} mV'})
        out['dapat_dibaca'] = len(flat) < 12
        return out
    if len(flat) >= 4 or len(noisy) >= 6:
        out['dapat_dibaca'] = False
        out['peringatan'].append(f'Kualitas sinyal buruk: {len(flat)} sadapan datar, {len(noisy)} sadapan bising — hasil tidak dapat diandalkan')
    else:
        out['dapat_dibaca'] = True
        if flat: out['peringatan'].append('Sadapan datar/terlepas: ' + ', '.join(flat))
        if noisy: out['peringatan'].append('Sadapan bising: ' + ', '.join(noisy))

    r, energy = _detect_qrs(x, fs)
    # ---- VF: tidak ada QRS teratur, aktivitas kacau 3-8 Hz ----
    fr, p = welch(x[:, [1, 6, 7, 8]].mean(1), fs, nperseg=min(len(x), int(4 * fs)))
    band = p[(fr >= 3) & (fr <= 8)].sum() / (p[(fr >= 0.5) & (fr <= 40)].sum() + 1e-12)
    rr = np.diff(r) / fs
    out['daya_3_8Hz'] = round(float(band), 2)
    # EKG normal PTB-XL: daya 3-8 Hz maksimal ±0,6; VF (CUDB) 0,8-0,9
    if band > 0.7:
        cvv = np.std(rr) / np.mean(rr) if len(rr) > 1 else 1
        if cvv > 0.12 or len(r) < 3:
            out['irama'] = 'Fibrilasi ventrikel (pola kacau tanpa QRS jelas)'
            out['temuan'].append({'kode': 'VF', 'nama': 'Kemungkinan fibrilasi ventrikel', 'dasar': f'{band*100:.0f}% daya pada 3–8 Hz, interval tidak teratur (CV {cvv:.2f})'})
            return out
        out['temuan'].append({'kode': 'WCT', 'nama': 'Gelombang sinusoidal cepat — curigai VT / flutter ventrikel', 'dasar': f'{band*100:.0f}% daya pada 3–8 Hz, teratur'})
    if len(r) < 3:
        out['irama'] = 'Tidak dapat ditentukan (QRS tidak terdeteksi)'
        out['dapat_dibaca'] = False
        return out

    hr = 60 / np.median(rr); cv = np.std(rr) / np.mean(rr)
    succ = np.abs(np.diff(rr)) / np.mean(rr) if len(rr) > 2 else np.array([0])
    out.update({'frekuensi_jantung': _r(float(hr)), 'jumlah_qrs': int(len(r)), 'durasi_detik': round(dur, 1),
                'rr_cv': round(float(cv), 3)})

    tpl, a = _template(x, r, fs)
    rr_s = float(np.median(rr))
    on, off = _onset_offset(tpl, a, fs)
    qrs_ms = (off - on) / fs * 1000
    base = np.median(tpl[max(0, on - int(0.04 * fs)):on + 1], 0)        # garis dasar = segmen PR
    T = tpl - base

    # ---- Morfologi per denyut: QRS lebar / prematur ----
    seg = [(i, x[i - int(0.08 * fs):i + int(0.12 * fs)]) for i in r if i - int(0.08 * fs) >= 0 and i + int(0.12 * fs) <= len(x)]
    ref = tpl[a - int(0.08 * fs):a + int(0.12 * fs)].ravel()
    corr, idx = [], []
    for i, s in seg:
        c = np.corrcoef(ref, s.ravel())[0, 1]; corr.append(c); idx.append(i)
    corr = np.array(corr); idx = np.array(idx)
    prem = np.zeros(len(r), bool)
    sudden = np.zeros(len(r), bool)
    if len(r) > 3:
        med = np.median(rr)
        for k in range(1, len(r)):
            prem[k] = rr[k - 1] < 0.85 * med
            # mendadak: interval sebelumnya normal (bukan perubahan bertahap sinus aritmia)
            sudden[k] = prem[k] and (k < 2 or rr[k - 2] > 0.92 * med) and rr[k - 1] < 0.8 * med
    pvc = pac = 0; kinds = []
    rpos = {v: k for k, v in enumerate(r)}
    for i, c in zip(idx, corr):
        k = rpos[i]
        if prem[k] and c < 0.7: pvc += 1; kinds.append('V')
        elif c < 0.4 and len(r) > 3: pvc += 1; kinds.append('V')
        elif sudden[k] and c >= 0.9: pac += 1; kinds.append('S')
        else: kinds.append('N')
    out['denyut_prematur'] = {'pvc': pvc, 'pac': pac}
    # run denyut berbeda-morfologi yang cepat (>= 3 berturut-turut, > 100/menit) -> curigai VT
    run, best_run = 0, 0
    for k, (i, c) in enumerate(zip(idx, corr)):
        kk = rpos[i]
        fast = kk > 0 and rr[kk - 1] < 0.6
        run = run + 1 if (c < 0.7 and fast) else 0
        best_run = max(best_run, run)
    out['run_ventrikel_terpanjang'] = int(best_run)

    # ---- Interval ----
    lead_t = max([LI['II'], LI['V5'], LI['V2']], key=lambda l: np.abs(T[off + int(0.08 * fs):, l]).max())
    t_lo, t_hi = off + int(0.06 * fs), min(len(T) - 1, a + int(min(0.6 * rr_s, 0.6) * fs))
    qt_ms = qtc_b = qtc_f = None
    if t_hi > t_lo + 5:
        tl = T[t_lo:t_hi, lead_t]; tp = t_lo + int(np.argmax(np.abs(tl)))
        sgn = np.sign(T[tp, lead_t]); d = np.gradient(T[:, lead_t] * sgn)
        after = d[tp:t_hi + int(0.15 * fs)]
        if len(after) > 2 and T[tp, lead_t] * sgn > 0.05:
            j = tp + int(np.argmin(after)); slope = d[j]
            if slope < 0:
                tend = j + (T[j, lead_t] * sgn) / (-slope)  # metode tangen
                qt_ms = (tend - on) / fs * 1000
                if 240 < qt_ms < 700:
                    qtc_b = qt_ms / np.sqrt(rr_s); qtc_f = qt_ms / np.cbrt(rr_s)
                else:
                    qt_ms = None
    # gelombang P di sadapan II
    pr_ms = None; p_amp = 0.0; p_present = False
    # jendela P: 250 ms sebelum QRS, tidak boleh melewati akhir T denyut sebelumnya (~60% RR setelah R)
    p_lo = max(0, on - int(0.25 * fs), a - int(0.4 * rr_s * fs))
    pw = T[p_lo:on - int(0.03 * fs), LI['II']]
    if len(pw) > 3:
        k = int(np.argmax(np.abs(pw))); p_amp = float(pw[k])
        alt = max(np.abs(T[p_lo:on - int(0.03 * fs), LI[l]]).max() for l in ('V1', 'aVF', 'I'))
        p_present = bool((abs(p_amp) >= 0.05 or alt >= 0.07) and corr.size and np.median(corr) > 0.8)
        if p_present:
            ppos = p_lo + k; po = ppos
            while po > max(p_lo, ppos - int(0.08 * fs)) and abs(T[po, LI['II']]) > 0.15 * abs(p_amp):
                po -= 1
            pr_ms = (on - po) / fs * 1000
            if not 60 < pr_ms < 400: pr_ms = None
    out['interval_ms'] = {'PR': None if pr_ms is None else _r(pr_ms), 'QRS': _r(qrs_ms),
                          'QT': None if qt_ms is None else _r(qt_ms),
                          'QTc_Bazett': None if qtc_b is None else _r(qtc_b), 'QTc_Fridericia': None if qtc_f is None else _r(qtc_f)}
    out['gelombang_P'] = {'ada': bool(p_present), 'amplitudo_II_mV': round(p_amp, 2)}

    # ---- Aksis & voltase ----
    area = T[on:off + 1].sum(0) / fs
    ax = float(np.degrees(np.arctan2(area[LI['aVF']], area[LI['I']])))
    out['aksis_derajat'] = _r(ax)
    qamp = np.ptp(T[on:off + 1], 0)
    sok = abs(min(0, T[on:off + 1, LI['V1']].min())) + max(T[on:off + 1, LI['V5']].max(), T[on:off + 1, LI['V6']].max())
    out['voltase'] = {'sokolow_lyon_mV': round(float(sok), 2), 'qrs_ekstremitas_maks_mV': round(float(qamp[:6].max()), 2),
                      'qrs_prekordial_maks_mV': round(float(qamp[6:].max()), 2)}
    # ---- ST & T ----
    jp = min(len(T) - 1, off + int(0.04 * fs))
    st = T[jp]; out['ST_J40_mV'] = {LEADS[i]: round(float(st[i]), 2) for i in range(12)}
    tpeak = {}
    if t_hi > t_lo:
        for l in range(12):
            w = T[t_lo:t_hi, l]; tpeak[LEADS[l]] = float(w[np.argmax(np.abs(w))])

    F = out['temuan']
    def add(kode, nama, dasar): F.append({'kode': kode, 'nama': nama, 'dasar': dasar})
    regular = cv < 0.08 and (succ.max() if len(succ) else 0) < 0.25
    wide = qrs_ms >= 120
    # ---- Irama ----
    if not p_present and not regular and cv > 0.12 and not wide:
        irama = 'Irama tidak teratur tanpa gelombang P (curiga fibrilasi atrium)'
        add('AF?', 'Kemungkinan fibrilasi atrium', f'RR tidak teratur (CV {cv:.2f}), gelombang P tidak jelas')
    elif wide and hr > 100 and regular and not p_present:
        irama = 'Takikardia kompleks lebar'
        add('WCT', 'Takikardia kompleks lebar — curigai takikardia ventrikel', f'QRS {qrs_ms:.0f} ms, {hr:.0f}/menit, teratur, P tidak jelas')
    elif not p_present and regular and not wide and hr > 100:
        irama = 'Takikardia QRS sempit teratur tanpa P (SVT/AVNRT/flutter?)'
        add('SVT?', 'Takikardia supraventrikular (kemungkinan AVNRT atau flutter)', f'{hr:.0f}/menit, teratur, QRS {qrs_ms:.0f} ms, P tidak jelas')
    elif not p_present and regular and not wide and 40 <= hr <= 60:
        irama = 'Kemungkinan irama junctional'
        add('JR?', 'Kemungkinan irama junctional (escape)', f'{hr:.0f}/menit, teratur, QRS sempit, P tidak jelas')
    elif not p_present and wide and 50 <= hr <= 110 and regular:
        irama = 'Kemungkinan irama idioventrikular dipercepat (AIVR)'
        add('AIVR?', 'Kemungkinan irama idioventrikular dipercepat', f'{hr:.0f}/menit, QRS lebar {qrs_ms:.0f} ms, P tidak jelas')
    elif not p_present and wide and hr < 50:
        irama = 'Kemungkinan irama escape ventrikel'
        add('VESC?', 'Kemungkinan irama escape ventrikel', f'{hr:.0f}/menit, QRS lebar')
    elif p_present:
        irama = 'Irama sinus' if p_amp > 0 else 'Irama dengan gelombang P terbalik di II (ektopik atrium/junctional?)'
        if p_amp > 0:
            if hr < 60: add('SB', 'Sinus bradikardia', f'{hr:.0f}/menit')
            elif hr > 100: add('ST', 'Sinus takikardia', f'{hr:.0f}/menit')
            if not regular and cv > 0.08 and pvc + pac <= 1: add('SA', 'Sinus aritmia', f'variasi RR {cv:.2f}')
    else:
        irama = 'Irama tidak dapat dipastikan (gelombang P tidak jelas)'
    out['irama'] = irama

    # ---- Aktivitas atrium terpisah: blok AV derajat 2/3, flutter ----
    xa = x[:, LI['II']].copy(); blank = np.ones(len(xa), bool)
    for i in r:
        blank[max(0, i - int(0.1 * fs)):min(len(xa), i + int(0.45 * fs))] = False
    if blank.sum() > 2 * fs:
        fa, pa = welch(np.where(blank, xa, 0), fs, nperseg=min(len(xa), int(4 * fs)))
        m = (fa >= 3.8) & (fa <= 6.7)
        if pa[m].sum() / (pa[(fa > 0.5) & (fa < 20)].sum() + 1e-12) > 0.6 and not p_present:
            add('AFL?', 'Kemungkinan flutter atrium', f'aktivitas atrium {fa[m][np.argmax(pa[m])]*60:.0f}/menit')
    if True:  # hitung gelombang P di luar kompleks QRS-T
        pk, _ = find_peaks(np.where(blank, _bp(xa[:, None], fs, 1, 15)[:, 0], 0), height=0.05, distance=int(0.2 * fs))
        ppi = np.diff(pk) / fs
        p_reg = len(ppi) > 2 and np.std(ppi) / np.mean(ppi) < 0.2
        p_rate = 60 / np.median(ppi) if len(ppi) > 1 else 0
        if hr < 60 and p_reg and len(pk) >= 1.4 * len(r) and p_rate > 1.4 * hr:
            # P lebih banyak dari QRS
            if regular and cv < 0.05:
                add('3AVB?', 'Kemungkinan blok AV derajat 3 (total)', f'P ±{p_rate:.0f}/menit vs QRS {hr:.0f}/menit, QRS teratur')
            else:
                add('2AVB?', 'Kemungkinan blok AV derajat 2', f'P ±{p_rate:.0f}/menit vs QRS {hr:.0f}/menit')

    # ---- Konduksi, aksis, interval ----
    if pr_ms and pr_ms > 220: add('1AVB', 'Blok AV derajat 1', f'PR {pr_ms:.0f} ms (> 200; ambang otomatis 220)')
    if pr_ms and pr_ms < 120 and wide: add('WPW?', 'PR pendek + QRS lebar (curigai preeksitasi/WPW)', f'PR {pr_ms:.0f} ms, QRS {qrs_ms:.0f} ms')
    if wide:
        v1 = T[on:off + 1, LI['V1']]; v6 = T[on:off + 1, LI['V6']]
        if v1.max() > abs(v1.min()) * 0.8 and v6.min() < -0.1:
            add('RBBB?', 'Pola blok cabang berkas kanan', f'QRS {qrs_ms:.0f} ms, V1 dominan positif (rSR\')')
        elif v1.min() < -abs(v1.max()) and v6.max() > abs(v6.min()):
            add('LBBB?', 'Pola blok cabang berkas kiri', f'QRS {qrs_ms:.0f} ms, V1 dominan negatif, V6 lebar positif')
        else:
            add('IVCD', 'QRS lebar (gangguan konduksi intraventrikular)', f'QRS {qrs_ms:.0f} ms')
    if -90 <= ax < -30: add('LAD', 'Deviasi aksis kiri', f'aksis {ax:.0f}°')
    elif 90 < ax <= 180: add('RAD', 'Deviasi aksis kanan', f'aksis {ax:.0f}°')
    elif ax < -90: add('EAD', 'Aksis ekstrem (northwest)', f'aksis {ax:.0f}°')
    if qtc_b:
        if qtc_b > 490 and qtc_f > 470: add('LQT', 'QT memanjang', f'QTc {qtc_b:.0f} ms (Bazett), {qtc_f:.0f} ms (Fridericia)')
        elif qtc_b < 340: add('SQT', 'QT memendek', f'QTc {qtc_b:.0f} ms (Bazett)')
    if out['voltase']['qrs_ekstremitas_maks_mV'] < 0.5:
        add('LVOLT', 'Voltase QRS rendah', 'QRS < 0,5 mV di semua sadapan ekstremitas')
    cornell = max(0, T[on:off + 1, LI['aVL']].max()) + abs(min(0, T[on:off + 1, LI['V3']].min()))
    out['voltase']['cornell_mV'] = round(float(cornell), 2)
    if sok >= 3.5 or cornell > 2.8:
        add('LVH-V', 'Kriteria voltase LVH', f'Sokolow-Lyon {sok:.1f} mV (≥ 3,5) / Cornell {cornell:.1f} mV (> 2,8)')
    # ---- Denyut prematur ----
    if pvc:
        pat = ''.join(kinds)
        big = 'NV' * 3 in pat or 'VN' * 3 in pat
        add('PVC', f'Kontraksi ventrikel prematur ({pvc}× dalam {dur:.0f} s)' + (' — pola bigemini' if big else ''), 'denyut prematur dengan morfologi berbeda')

    if best_run >= 3:
        add('NSVT?', f'{best_run} denyut berbeda-morfologi berturut-turut > 100/menit (curigai VT non-sustained)', 'run denyut ektopik cepat')
    if pac: add('PAC', f'Kontraksi atrium prematur ({pac}× dalam {dur:.0f} s)', 'denyut prematur dengan morfologi QRS sama')
    # ---- ST/T ----
    if not wide:
        for g, ls in GROUPS.items():
            lim = 0.25 if g in ('septal', 'anterior') else 0.1
            el = [l for l in ls if st[LI[l]] >= lim]
            if len(el) >= 2:
                add('STE', f'Elevasi ST di sadapan {g} ({", ".join(el)})', 'ST J+40 ms ≥ ' + ('0,25' if lim == 0.25 else '0,1') + ' mV di ≥ 2 sadapan bersebelahan')
            dep = [l for l in ls if st[LI[l]] <= -0.1]
            if len(dep) >= 2:
                add('STD', f'Depresi ST di sadapan {g} ({", ".join(dep)})', 'ST J+40 ms ≤ −0,1 mV di ≥ 2 sadapan bersebelahan')
        inv = [l for g, ls in GROUPS.items() for l in ls if tpeak.get(l, 0) < -0.15 and l not in ('aVR', 'V1', 'III', 'aVL')]
        if len(inv) >= 2: add('TINV', 'Inversi gelombang T (' + ', '.join(inv) + ')', 'puncak T < −0,15 mV')
    return out


if __name__ == '__main__':
    import sys, json, os
    sys.path.insert(0, os.path.dirname(__file__))
    from predict import load_signal
    s, fs = load_signal(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(json.dumps(measure(s, fs), indent=1, ensure_ascii=False))
