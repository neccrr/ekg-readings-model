"""Laporan EKG lengkap: menggabungkan model (label PTB-XL) dan pengukuran berbasis aturan
ke dalam 8 kategori diagnosis (Nurseslabs / LITFL)."""

# Nama Indonesia untuk kode SCP-ECG PTB-XL yang dipelajari model
NAMA_KODE = {
    'NORM': 'EKG normal', 'MI': 'Infark miokard (superclass)', 'STTC': 'Perubahan ST/T (superclass)',
    'CD': 'Gangguan konduksi (superclass)', 'HYP': 'Hipertrofi (superclass)',
    'SR': 'Irama sinus', 'SBRAD': 'Sinus bradikardia', 'STACH': 'Sinus takikardia', 'SARRH': 'Sinus aritmia',
    'AFIB': 'Fibrilasi atrium', 'AFLT': 'Flutter atrium', 'PAC': 'Kontraksi atrium prematur (PAC)', 'SVARR': 'Aritmia supraventrikular',
    'PVC': 'Kontraksi ventrikel prematur (PVC)', 'BIGU': 'Pola bigemini', '1AVB': 'Blok AV derajat 1', 'LPR': 'Interval PR memanjang',
    'CLBBB': 'Blok cabang berkas kiri komplit (LBBB)', 'CRBBB': 'Blok cabang berkas kanan komplit (RBBB)',
    'ILBBB': 'LBBB inkomplit', 'IRBBB': 'RBBB inkomplit', 'IVCD': 'Gangguan konduksi intraventrikular non-spesifik',
    'LAFB': 'Hemiblok anterior kiri (LAFB)', 'LPFB': 'Hemiblok posterior kiri (LPFB)', 'WPW': 'Sindrom Wolff-Parkinson-White',
    'IMI': 'Infark miokard inferior', 'ASMI': 'Infark miokard anteroseptal', 'ILMI': 'Infark miokard inferolateral',
    'AMI': 'Infark miokard anterior', 'ALMI': 'Infark miokard anterolateral', 'LMI': 'Infark miokard lateral',
    'IPLMI': 'Infark miokard inferoposterolateral', 'INJAS': 'Cedera subendokardial anteroseptal', 'INJAL': 'Cedera subendokardial anterolateral',
    'QWAVE': 'Gelombang Q patologis', 'ISC_': 'Iskemia non-spesifik', 'ISCAL': 'Iskemia anterolateral', 'ISCIN': 'Iskemia inferior',
    'ISCIL': 'Iskemia inferolateral', 'ISCAS': 'Iskemia anteroseptal', 'ISCLA': 'Iskemia lateral', 'NDT': 'Kelainan gelombang T non-diagnostik',
    'NST_': 'Perubahan ST non-spesifik', 'STD_': 'Depresi segmen ST', 'INVT': 'Inversi gelombang T', 'LOWT': 'Gelombang T amplitudo rendah',
    'NT_': 'Perubahan gelombang T non-spesifik', 'ANEUR': 'Perubahan ST-T sesuai aneurisma ventrikel', 'DIG': 'Efek digitalis',
    'EL': 'Gangguan elektrolit atau efek obat', 'LNGQT': 'QT memanjang', 'LVH': 'Hipertrofi ventrikel kiri (LVH)',
    'RVH': 'Hipertrofi ventrikel kanan (RVH)', 'LAO/LAE': 'Pembesaran atrium kiri', 'RAO/RAE': 'Pembesaran atrium kanan',
    'VCLVH': 'Kriteria voltase LVH', 'HVOLT': 'Voltase QRS tinggi', 'LVOLT': 'Voltase QRS rendah', 'ABQRS': 'QRS abnormal',
}

# Katalog 8 kategori. Tiap butir: (nama, kode model, kode aturan, catatan cakupan)
# cakupan: 'model' = dipelajari dari PTB-XL; 'aturan' = pengukuran/heuristik; 'tidak' = belum didukung
KATALOG = [
    ('1. Interpretasi umum', [
        ('EKG normal', ['NORM*'], [], None),
        ('EKG abnormal', ['MI*', 'STTC*', 'CD*', 'HYP*'], [], 'positif bila ada temuan abnormal'),
        ('Tidak dapat dibaca (artefak/sadapan hilang)', [], ['UNREAD'], None),
    ]),
    ('2. Irama sinus', [
        ('Irama sinus', ['SR'], ['SR'], None),
        ('Sinus bradikardia (< 60/menit)', ['SBRAD'], ['SB'], None),
        ('Sinus takikardia (> 100/menit)', ['STACH'], ['ST'], None),
        ('Sinus aritmia', ['SARRH'], [], None),
    ]),
    ('3. Irama atrium', [
        ('Kontraksi atrium prematur (PAC)', ['PAC'], [], None),
        ('Fibrilasi atrium', ['AFIB'], ['AF?'], None),
        ('Flutter atrium', [], ['AFL?'], 'heuristik frekuensi atrium 250–400/menit; data latih terlalu sedikit'),
        ('Takikardia atrium / MAT', ['SVARR'], [], 'model hanya mengenali "aritmia supraventrikular" umum; sensitivitas rendah'),
    ]),
    ('4. Irama junctional', [
        ('Kontraksi junctional prematur (PJC)', [], [], 'tidak didukung: tidak ada data latih, sulit dibedakan dari PAC'),
        ('Irama junctional / escape (40–60/menit)', [], ['JR?'], 'heuristik: teratur, QRS sempit, tanpa gelombang P'),
        ('Takikardia junctional / AVNRT', [], ['SVT?'], 'heuristik: takikardia QRS sempit teratur tanpa P'),
    ]),
    ('5. Blok AV', [
        ('Blok AV derajat 1 (PR > 200 ms)', ['1AVB', 'LPR'], ['1AVB'], None),
        ('Blok AV derajat 2 (Mobitz I / II)', [], ['2AVB?'], 'heuristik: P lebih banyak dari QRS; tipe Mobitz belum dibedakan'),
        ('Blok AV derajat 3 (total)', [], ['3AVB?'], 'heuristik: P lebih cepat dari QRS yang teratur dan lambat'),
    ]),
    ('6. Irama ventrikel', [
        ('Kontraksi ventrikel prematur (PVC)', ['PVC'], [], None),
        ('Bigemini / trigemini', ['BIGU'], [], None),
        ('Irama idioventrikular dipercepat (AIVR)', [], ['AIVR?', 'VESC?'], 'heuristik: QRS lebar 50–110/menit tanpa P'),
        ('Takikardia ventrikel (VT, termasuk NSVT)', [], ['WCT', 'NSVT?'], 'heuristik takikardia kompleks lebar; tidak membedakan Torsades'),
        ('Fibrilasi ventrikel', [], ['VF'], 'heuristik: aktivitas kacau 3–8 Hz tanpa QRS'),
        ('Asistol', [], ['ASYST'], 'heuristik: amplitudo hampir nol'),
    ]),
    ('7. Konduksi & aksis', [
        ('Blok cabang berkas kanan (RBBB)', ['CRBBB', 'IRBBB'], [], None),
        ('Blok cabang berkas kiri (LBBB)', ['CLBBB', 'ILBBB'], [], None),
        ('Gangguan konduksi intraventrikular', ['IVCD'], [], 'model lemah (AUC uji 0,73)'),
        ('Hemiblok anterior kiri (LAFB)', ['LAFB'], [], None),
        ('Hemiblok posterior kiri (LPFB)', ['LPFB'], [], None),
        ('Deviasi aksis kiri (< −30°)', [], ['LAD', 'EAD'], None),
        ('Deviasi aksis kanan (> +90°)', [], ['RAD'], None),
        ('Preeksitasi / WPW', ['WPW'], [], None),
    ]),
    ('8. Iskemia, infark, hipertrofi & QT', [
        ('Infark miokard (lokasi)', ['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'LMI', 'IPLMI', 'QWAVE', 'MI*'], [], None),
        ('Elevasi ST (curiga STEMI)', ['INJAS', 'INJAL'], ['STE'], 'model hanya mengenali cedera subendokardial; elevasi ST dari pengukuran'),
        ('Depresi ST / iskemia', ['STD_', 'ISC_', 'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA', 'NST_', 'STTC*'], [], None),
        ('Inversi / kelainan gelombang T', ['INVT', 'NDT', 'NT_', 'LOWT'], [], None),
        ('Hipertrofi ventrikel', ['LVH', 'RVH', 'VCLVH', 'HVOLT', 'HYP*'], [], None),
        ('Pembesaran atrium', ['LAO/LAE', 'RAO/RAE'], [], None),
        ('QT memanjang', ['LNGQT'], ['LQT'], 'terutama dari pengukuran QTc; model jarang positif (11 contoh uji)'),
        ('QT memendek', [], ['SQT'], 'pengukuran QTc < 340 ms; tidak divalidasi'),
        ('Voltase QRS rendah', ['LVOLT'], [], None),
        ('Efek obat / elektrolit', ['DIG', 'EL'], [], 'data latih sedikit; sensitivitas rendah'),
    ]),
]

# Temuan aturan yang ditampilkan meski tidak ada di katalog model, plus aturan yang diredam
ATURAN_DIREDAM = {'PVC', 'PAC', 'STD', 'TINV', 'LVH-V', 'LVOLT', 'RBBB?', 'LBBB?', 'WPW?', 'SA'}


def build(model_out, meas, auc=None):
    auc = auc or {}
    """model_out: dict kode -> (p, ambang); meas: hasil measure(). Kembalikan list kategori."""
    rk = {t['kode']: t for t in meas.get('temuan', [])}
    sinus_kuat = model_out.get('SR', (0, 1))[0] >= model_out.get('SR', (0, 1))[1]
    # aturan irama non-sinus diredam bila model yakin irama sinus
    if sinus_kuat:
        for k in ('JR?', 'AIVR?', 'VESC?', 'AF?', 'SVT?', 'AFL?'):
            rk.pop(k, None)
    if meas.get('irama') == 'Irama sinus':
        rk['SR'] = {'kode': 'SR', 'nama': 'Irama sinus', 'dasar': 'gelombang P positif di II sebelum tiap QRS'}
    if not meas.get('dapat_dibaca', True):
        rk['UNREAD'] = {'kode': 'UNREAD', 'nama': 'Tidak dapat dibaca', 'dasar': '; '.join(meas.get('peringatan', []))}
    out = []
    for kat, items in KATALOG:
        rows = []
        for nama, mk, ak, cat in items:
            hits, basis = [], []
            spesifik = any(model_out.get(c, (0, 1))[0] >= model_out.get(c, (0, 1))[1] for c in mk if not c.endswith('*'))
            for c in mk:
                key = c.rstrip('*')
                if c.endswith('*') and spesifik and nama != 'EKG normal':
                    continue
                if key in model_out:
                    p, thr = model_out[key]
                    if p >= thr:
                        hits.append(key); a = auc.get(key)
                        basis.append(f'model: {NAMA_KODE.get(key, key)} p={p:.2f}' + (f' (AUC uji {a:.2f})' if a else ''))
            for c in ak:
                if c in rk:
                    hits.append(c); basis.append(f"pengukuran: {rk[c]['dasar']}")
            cak = 'tidak' if not mk and not ak else ('model+aturan' if mk and ak else 'model' if mk else 'aturan')
            rows.append({'nama': nama, 'positif': bool(hits), 'dasar': basis, 'cakupan': cak, 'catatan': cat})
        out.append({'kategori': kat, 'butir': rows})
    # Interpretasi umum: abnormal = ada temuan positif selain "Irama sinus"
    ab = [b['nama'] for k in out[1:] for b in k['butir'] if b['positif'] and b['nama'] != 'Irama sinus']
    g = {b['nama']: b for b in out[0]['butir']}
    g['EKG abnormal'].update(positif=bool(ab), dasar=[', '.join(ab)] if ab else [])
    if ab and g['EKG normal']['positif']:
        g['EKG normal'].update(positif=False, dasar=g['EKG normal']['dasar'] + ['diabaikan karena ada temuan abnormal'])
    return out
