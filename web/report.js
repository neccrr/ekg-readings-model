// ===== Laporan 8 kategori (port dari src/report.py; katalog dibaca dari KATALOG) =====
function buildReport(modelOut, meas, auc = {}) {
  const rk = Object.fromEntries((meas.temuan || []).map(t => [t.kode, t]));
  const on = k => modelOut[k] && modelOut[k][0] >= modelOut[k][1];
  if (on('SR')) for (const k of ['JR?', 'AIVR?', 'VESC?', 'AF?', 'SVT?', 'AFL?']) delete rk[k];
  if (meas.irama === 'Irama sinus') rk.SR = { kode: 'SR', nama: 'Irama sinus', dasar: 'gelombang P positif di II sebelum tiap QRS' };
  if (meas.dapat_dibaca === false) rk.UNREAD = { kode: 'UNREAD', nama: 'Tidak dapat dibaca', dasar: (meas.peringatan || []).join('; ') };
  const out = KATALOG.katalog.map(([kat, items]) => ({ kategori: kat, butir: items.map(([nama, mk, ak, cat]) => {
    const basis = []; let hit = false;
    const spes = mk.some(c => !c.endsWith('*') && on(c));
    for (const c of mk) { const key = c.replace('*', '');
      if (c.endsWith('*') && spes && nama !== 'EKG normal') continue;
      if (on(key)) { hit = true; basis.push(`model: ${KATALOG.nama[key] || key} p=${modelOut[key][0].toFixed(2)}` + (auc[key] ? ` (AUC uji ${auc[key].toFixed(2)})` : '')); } }
    for (const c of ak) if (rk[c]) { hit = true; basis.push(`pengukuran: ${rk[c].dasar}`); }
    const cak = !mk.length && !ak.length ? 'tidak' : mk.length && ak.length ? 'model+aturan' : mk.length ? 'model' : 'aturan';
    return { nama, positif: hit, dasar: basis, cakupan: cak, catatan: cat };
  }) }));
  const ab = out.slice(1).flatMap(k => k.butir.filter(b => b.positif && b.nama !== 'Irama sinus').map(b => b.nama));
  const g = Object.fromEntries(out[0].butir.map(b => [b.nama, b]));
  Object.assign(g['EKG abnormal'], { positif: ab.length > 0, dasar: ab.length ? [ab.join(', ')] : [] });
  if (ab.length && g['EKG normal'].positif) Object.assign(g['EKG normal'], { positif: false, dasar: [...g['EKG normal'].dasar, 'diabaikan karena ada temuan abnormal'] });
  return out;
}
if (typeof module !== 'undefined') module.exports = { buildReport };
