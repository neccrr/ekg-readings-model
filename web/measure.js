// ===== Pengukuran EKG berbasis aturan (port dari src/measure.py, fs tetap 100 Hz) =====
const M_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
const M_LI = Object.fromEntries(M_LEADS.map((l, i) => [l, i]));
const M_GROUPS = { inferior: ['II', 'III', 'aVF'], lateral: ['I', 'aVL', 'V5', 'V6'], septal: ['V1', 'V2'], anterior: ['V3', 'V4'] };

function m_lfilter(b, a, x, zi0) {
  const n = b.length, z = zi0.slice(), y = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i], yi = b[0] * xi + z[0]; y[i] = yi;
    for (let k = 1; k < n - 1; k++) z[k - 1] = b[k] * xi + z[k] - a[k] * yi;
    z[n - 2] = b[n - 1] * xi - a[n - 1] * yi;
  }
  return y;
}
function m_filtfilt(F, x) {        // sama dengan scipy.signal.filtfilt (padtype odd, padlen 15)
  const pl = 15, N = x.length;
  if (N <= pl) return Float64Array.from(x);
  const ext = new Float64Array(N + 2 * pl);
  for (let i = 0; i < pl; i++) ext[i] = 2 * x[0] - x[pl - i];
  ext.set(x, pl);
  for (let i = 0; i < pl; i++) ext[N + pl + i] = 2 * x[N - 1] - x[N - 2 - i];
  let y = m_lfilter(F.b, F.a, ext, F.zi.map(v => v * ext[0]));
  y.reverse(); y = m_lfilter(F.b, F.a, y, F.zi.map(v => v * y[0])); y.reverse();
  return y.subarray(pl, pl + N);
}
const m_bp = (sig, key) => sig.map(ch => m_filtfilt(FILTERS[key], ch));
function m_pct(arr, q) { const b = Float64Array.from(arr).sort(); const p = (b.length - 1) * q / 100, lo = Math.floor(p), hi = Math.ceil(p); return b[lo] + (b[hi] - b[lo]) * (p - lo); }
const m_median = a => m_pct(a, 50);
const m_mean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
const m_std = a => { const m = m_mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / a.length); };
const m_ptp = a => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return hi - lo; };
const m_argmax = a => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };
const m_argmin = a => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[k]) k = i; return k; };
const m_diff = a => { const o = new Float64Array(Math.max(0, a.length - 1)); for (let i = 1; i < a.length; i++) o[i - 1] = a[i] - a[i - 1]; return o; };
function m_boxsame(a, w) {          // np.convolve(a, ones(w)/w, 'same')
  if (w <= 1) return Float64Array.from(a);
  const o = new Float64Array(a.length), sh = Math.floor((w - 1) / 2);
  for (let i = 0; i < a.length; i++) { let s = 0; const hi = i + sh; for (let j = hi - w + 1; j <= hi; j++) if (j >= 0 && j < a.length) s += a[j]; o[i] = s / w; }
  return o;
}
function m_findpeaks(x, height, distance) {   // scipy.signal.find_peaks (height, distance)
  const pk = []; let i = 1;
  while (i < x.length - 1) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1;
      while (ahead < x.length - 1 && x[ahead] === x[i]) ahead++;
      if (x[ahead] < x[i]) { pk.push(Math.floor((i + ahead - 1) / 2)); i = ahead; }
    }
    i++;
  }
  let p = pk.filter(k => x[k] >= height);
  if (distance > 1 && p.length > 1) {
    const keep = new Array(p.length).fill(true), order = p.map((_, k) => k).sort((u, v) => x[p[u]] - x[p[v]] || u - v).reverse();
    for (const k of order) {
      if (!keep[k]) continue;
      for (let j = k - 1; j >= 0 && p[k] - p[j] < distance; j--) keep[j] = false;
      for (let j = k + 1; j < p.length && p[j] - p[k] < distance; j++) keep[j] = false;
    }
    p = p.filter((_, k) => keep[k]);
  }
  return p;
}
function m_welch(x, fs, nper) {      // scipy.signal.welch (hann, 50% overlap, detrend konstan)
  const step = nper - Math.floor(nper / 2), nseg = Math.floor((x.length - Math.floor(nper / 2)) / step);
  const w = Float64Array.from({ length: nper }, (_, n) => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / nper));
  const nf = Math.floor(nper / 2) + 1, P = new Float64Array(nf), cs = new Float64Array(nper), sn = new Float64Array(nper);
  for (let k = 0; k < nf; k++) {
    for (let n = 0; n < nper; n++) { cs[n] = Math.cos(2 * Math.PI * k * n / nper); sn[n] = Math.sin(2 * Math.PI * k * n / nper); }
    for (let s = 0; s < nseg; s++) {
      const o = s * step; let m = 0; for (let n = 0; n < nper; n++) m += x[o + n]; m /= nper;
      let re = 0, im = 0; for (let n = 0; n < nper; n++) { const v = (x[o + n] - m) * w[n]; re += v * cs[n]; im -= v * sn[n]; }
      P[k] += re * re + im * im;
    }
  }
  return { f: Float64Array.from({ length: nf }, (_, k) => k * fs / nper), p: P };
}
const m_bandsum = (W, lo, hi, incl = true) => { let s = 0; for (let k = 0; k < W.f.length; k++) if (incl ? (W.f[k] >= lo && W.f[k] <= hi) : (W.f[k] > lo && W.f[k] < hi)) s += W.p[k]; return s; };
function m_corr(a, b) {
  const ma = m_mean(a), mb = m_mean(b); let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { const u = a[i] - ma, v = b[i] - mb; sab += u * v; saa += u * u; sbb += v * v; }
  return sab / Math.sqrt(saa * sbb);
}
const m_r0 = v => Math.round(v), m_r2 = v => Math.round(v * 100) / 100;

function ecgMeasure(sig) {           // sig: 12 array (mV, 100 Hz)
  const fs = 100, N = sig[0].length, out = { temuan: [], peringatan: [] }, dur = N / fs;
  const add = (kode, nama, dasar) => out.temuan.push({ kode, nama, dasar });
  // kualitas
  const sd = sig.map(m_std), flat = M_LEADS.filter((_, i) => sd[i] < 0.01);
  const hf = m_bp(sig, '35-45'), noisy = M_LEADS.filter((l, i) => m_std(hf[i]) / (sd[i] + 1e-6) > 0.5 && !flat.includes(l));
  out.kualitas = { sadapan_datar: flat, sadapan_bising: noisy };
  const x = m_bp(sig, '0.5-40'), amp = x.map(m_ptp);
  if (m_median(amp) < 0.1) {
    out.irama = 'Asistol / garis datar'; add('ASYST', 'Asistol (garis datar)', `amplitudo median ${m_median(amp).toFixed(2)} mV`);
    out.dapat_dibaca = flat.length < 12; return out;
  }
  if (flat.length >= 4 || noisy.length >= 6) { out.dapat_dibaca = false; out.peringatan.push(`Kualitas sinyal buruk: ${flat.length} sadapan datar, ${noisy.length} sadapan bising — hasil tidak dapat diandalkan`); }
  else { out.dapat_dibaca = true; if (flat.length) out.peringatan.push('Sadapan datar/terlepas: ' + flat.join(', ')); if (noisy.length) out.peringatan.push('Sadapan bising: ' + noisy.join(', ')); }
  // deteksi QRS
  const f = m_bp(x, '5-20'), e0 = new Float64Array(N);
  for (const ch of f) for (let t = 1; t < N; t++) e0[t] += Math.abs(ch[t] - ch[t - 1]);
  const e = m_boxsame(e0, Math.floor(0.08 * fs));
  let r = [];
  let emax = 0; for (const u of e) if (u > emax) emax = u;
  if (emax > 0) {
    let pk = m_findpeaks(e, 0.3 * m_pct(e, 99), Math.floor(0.25 * fs));
    if (pk.length > 2) { const md = m_median(pk.map(k => e[k])); pk = pk.filter(k => e[k] > 0.35 * md); }
    const lead = m_argmax(f.map(m_ptp)), h = Math.floor(0.06 * fs);
    r = [...new Set(pk.map(p => { const a = Math.max(0, p - h), b = Math.min(N, p + h); let k = a; for (let i = a; i < b; i++) if (Math.abs(f[lead][i]) > Math.abs(f[lead][k])) k = i; return k; }))].sort((u, v) => u - v);
  }
  // VF
  const mix = new Float64Array(N); for (let t = 0; t < N; t++) mix[t] = (x[1][t] + x[6][t] + x[7][t] + x[8][t]) / 4;
  const W = m_welch(mix, fs, Math.min(N, 4 * fs));
  const band = m_bandsum(W, 3, 8) / (m_bandsum(W, 0.5, 40) + 1e-12);
  const rr = m_diff(r).map(v => v / fs);
  out.daya_3_8Hz = m_r2(band);
  if (band > 0.7) {
    const cvv = rr.length > 1 ? m_std(rr) / m_mean(rr) : 1;
    if (cvv > 0.12 || r.length < 3) { out.irama = 'Fibrilasi ventrikel (pola kacau tanpa QRS jelas)'; add('VF', 'Kemungkinan fibrilasi ventrikel', `${Math.round(band * 100)}% daya pada 3–8 Hz, interval tidak teratur (CV ${cvv.toFixed(2)})`); return out; }
    add('WCT', 'Gelombang sinusoidal cepat — curigai VT / flutter ventrikel', `${Math.round(band * 100)}% daya pada 3–8 Hz, teratur`);
  }
  if (r.length < 3) { out.irama = 'Tidak dapat ditentukan (QRS tidak terdeteksi)'; out.dapat_dibaca = false; return out; }
  const hr = 60 / m_median(rr), cv = m_std(rr) / m_mean(rr), mrr = m_mean(rr);
  const succ = rr.length > 2 ? m_diff(rr).map(v => Math.abs(v) / mrr) : [0];
  Object.assign(out, { frekuensi_jantung: m_r0(hr), jumlah_qrs: r.length, durasi_detik: Math.round(dur * 10) / 10, rr_cv: Math.round(cv * 1000) / 1000 });
  // median beat
  const A = Math.floor(0.35 * fs), B = Math.floor(0.65 * fs), L = A + B;
  const rb = r.filter(i => i - A >= 0 && i + B <= N);
  const tpl = x.map(ch => { const o = new Float64Array(L); const tmp = new Float64Array(rb.length); for (let t = 0; t < L; t++) { rb.forEach((i, k) => tmp[k] = ch[i - A + t]); o[t] = m_median(tmp); } return o; });
  const a = A, rr_s = m_median(rr);
  // onset/offset QRS
  const v0 = new Float64Array(L - 1); for (const ch of tpl) for (let t = 0; t < L - 1; t++) v0[t] += Math.abs(ch[t + 1] - ch[t]);
  const v = m_boxsame(v0, Math.max(1, Math.floor(0.012 * fs)));
  const vb = m_pct(v, 10); let vpk = -Infinity; for (let t = a - Math.floor(0.06 * fs); t < a + Math.floor(0.06 * fs); t++) vpk = Math.max(vpk, v[t]);
  const thr = vb + 0.07 * (vpk - vb);
  let on = a; while (on > a - Math.floor(0.12 * fs) && v[on] > thr) on--;
  let off = a; while (off < a + Math.floor(0.16 * fs) && off < v.length - 1 && v[off] > thr) off++;
  const qrs_ms = (off - on) / fs * 1000;
  const T = tpl.map(ch => { const b0 = m_median(ch.subarray(Math.max(0, on - Math.floor(0.04 * fs)), on + 1)); return ch.map(u => u - b0); });
  // morfologi per denyut
  const s0 = Math.floor(0.08 * fs), s1 = Math.floor(0.12 * fs);
  const ref = []; for (let t = a - s0; t < a + s1; t++) for (let l = 0; l < 12; l++) ref.push(tpl[l][t]);
  const idx = [], corr = [];
  for (const i of r) { if (i - s0 < 0 || i + s1 > N) continue; const sg = []; for (let t = i - s0; t < i + s1; t++) for (let l = 0; l < 12; l++) sg.push(x[l][t]); idx.push(i); corr.push(m_corr(ref, sg)); }
  const prem = new Array(r.length).fill(false), sudden = new Array(r.length).fill(false);
  if (r.length > 3) { const med = m_median(rr); for (let k = 1; k < r.length; k++) { prem[k] = rr[k - 1] < 0.85 * med; sudden[k] = prem[k] && (k < 2 || rr[k - 2] > 0.92 * med) && rr[k - 1] < 0.8 * med; } }
  const rpos = new Map(r.map((v2, k) => [v2, k]));
  let pvc = 0, pac = 0; const kinds = [];
  idx.forEach((i, j) => { const k = rpos.get(i), c = corr[j];
    if (prem[k] && c < 0.7) { pvc++; kinds.push('V'); } else if (c < 0.4 && r.length > 3) { pvc++; kinds.push('V'); }
    else if (sudden[k] && c >= 0.9) { pac++; kinds.push('S'); } else kinds.push('N'); });
  out.denyut_prematur = { pvc, pac };
  let run = 0, best_run = 0;
  idx.forEach((i, j) => { const kk = rpos.get(i), fast = kk > 0 && rr[kk - 1] < 0.6; run = (corr[j] < 0.7 && fast) ? run + 1 : 0; best_run = Math.max(best_run, run); });
  out.run_ventrikel_terpanjang = best_run;
  // QT
  const after80 = l => { let m = 0; for (let t = off + Math.floor(0.08 * fs); t < L; t++) m = Math.max(m, Math.abs(T[l][t])); return m; };
  const lead_t = [M_LI.II, M_LI.V5, M_LI.V2].reduce((p, c) => after80(c) > after80(p) ? c : p);
  const t_lo = off + Math.floor(0.06 * fs), t_hi = Math.min(L - 1, a + Math.floor(Math.min(0.6 * rr_s, 0.6) * fs));
  let qt_ms = null, qtc_b = null, qtc_f = null;
  if (t_hi > t_lo + 5) {
    const TL = T[lead_t]; let tp = t_lo; for (let t = t_lo; t < t_hi; t++) if (Math.abs(TL[t]) > Math.abs(TL[tp])) tp = t;
    const sgn = Math.sign(TL[tp]), g = TL.map(u => u * sgn), d = new Float64Array(L);
    d[0] = g[1] - g[0]; d[L - 1] = g[L - 1] - g[L - 2]; for (let t = 1; t < L - 1; t++) d[t] = (g[t + 1] - g[t - 1]) / 2;
    const af = d.subarray(tp, Math.min(L, t_hi + Math.floor(0.15 * fs)));
    if (af.length > 2 && TL[tp] * sgn > 0.05) {
      const j = tp + m_argmin(af), slope = d[j];
      if (slope < 0) { const tend = j + (TL[j] * sgn) / (-slope); qt_ms = (tend - on) / fs * 1000;
        if (qt_ms > 240 && qt_ms < 700) { qtc_b = qt_ms / Math.sqrt(rr_s); qtc_f = qt_ms / Math.cbrt(rr_s); } else qt_ms = null; }
    }
  }
  // gelombang P
  let pr_ms = null, p_amp = 0, p_present = false;
  const p_lo = Math.max(0, on - Math.floor(0.25 * fs), a - Math.floor(0.4 * rr_s * fs)), p_hi = on - Math.floor(0.03 * fs);
  if (p_hi - p_lo > 3) {
    const pw = T[M_LI.II].subarray(p_lo, p_hi); let k = 0; for (let i = 1; i < pw.length; i++) if (Math.abs(pw[i]) > Math.abs(pw[k])) k = i; p_amp = pw[k];
    const alt = Math.max(...['V1', 'aVF', 'I'].map(l => Math.max(...Array.from(T[M_LI[l]].subarray(p_lo, p_hi), Math.abs))));
    p_present = (Math.abs(p_amp) >= 0.05 || alt >= 0.07) && corr.length > 0 && m_median(corr) > 0.8;
    if (p_present) { const ppos = p_lo + k; let po = ppos; while (po > Math.max(p_lo, ppos - Math.floor(0.08 * fs)) && Math.abs(T[M_LI.II][po]) > 0.15 * Math.abs(p_amp)) po--;
      pr_ms = (on - po) / fs * 1000; if (!(pr_ms > 60 && pr_ms < 400)) pr_ms = null; }
  }
  out.interval_ms = { PR: pr_ms == null ? null : m_r0(pr_ms), QRS: m_r0(qrs_ms), QT: qt_ms == null ? null : m_r0(qt_ms), QTc_Bazett: qtc_b == null ? null : m_r0(qtc_b), QTc_Fridericia: qtc_f == null ? null : m_r0(qtc_f) };
  out.gelombang_P = { ada: p_present, amplitudo_II_mV: m_r2(p_amp) };
  // aksis & voltase
  const qseg = l => T[l].subarray(on, off + 1), area = l => m_mean(qseg(l)) * (off - on + 1) / fs;
  const ax = Math.atan2(area(M_LI.aVF), area(M_LI.I)) * 180 / Math.PI; out.aksis_derajat = m_r0(ax);
  const qamp = T.map((_, l) => m_ptp(qseg(l)));
  const mn = l => Math.min(...qseg(l)), mx = l => Math.max(...qseg(l));
  const sok = Math.abs(Math.min(0, mn(M_LI.V1))) + Math.max(mx(M_LI.V5), mx(M_LI.V6));
  out.voltase = { sokolow_lyon_mV: m_r2(sok), qrs_ekstremitas_maks_mV: m_r2(Math.max(...qamp.slice(0, 6))), qrs_prekordial_maks_mV: m_r2(Math.max(...qamp.slice(6))) };
  const jp = Math.min(L - 1, off + Math.floor(0.04 * fs)), st = T.map(ch => ch[jp]);
  out.ST_J40_mV = Object.fromEntries(M_LEADS.map((l, i) => [l, m_r2(st[i])]));
  const tpeak = {};
  if (t_hi > t_lo) T.forEach((ch, l) => { let k = t_lo; for (let t = t_lo; t < t_hi; t++) if (Math.abs(ch[t]) > Math.abs(ch[k])) k = t; tpeak[M_LEADS[l]] = ch[k]; });
  const regular = cv < 0.08 && Math.max(...succ) < 0.25, wide = qrs_ms >= 120;
  let irama;
  if (!p_present && !regular && cv > 0.12 && !wide) { irama = 'Irama tidak teratur tanpa gelombang P (curiga fibrilasi atrium)'; add('AF?', 'Kemungkinan fibrilasi atrium', `RR tidak teratur (CV ${cv.toFixed(2)}), gelombang P tidak jelas`); }
  else if (wide && hr > 100 && regular && !p_present) { irama = 'Takikardia kompleks lebar'; add('WCT', 'Takikardia kompleks lebar — curigai takikardia ventrikel', `QRS ${m_r0(qrs_ms)} ms, ${m_r0(hr)}/menit, teratur, P tidak jelas`); }
  else if (!p_present && regular && !wide && hr > 100) { irama = 'Takikardia QRS sempit teratur tanpa P (SVT/AVNRT/flutter?)'; add('SVT?', 'Takikardia supraventrikular (kemungkinan AVNRT atau flutter)', `${m_r0(hr)}/menit, teratur, QRS ${m_r0(qrs_ms)} ms, P tidak jelas`); }
  else if (!p_present && regular && !wide && hr >= 40 && hr <= 60) { irama = 'Kemungkinan irama junctional'; add('JR?', 'Kemungkinan irama junctional (escape)', `${m_r0(hr)}/menit, teratur, QRS sempit, P tidak jelas`); }
  else if (!p_present && wide && hr >= 50 && hr <= 110 && regular) { irama = 'Kemungkinan irama idioventrikular dipercepat (AIVR)'; add('AIVR?', 'Kemungkinan irama idioventrikular dipercepat', `${m_r0(hr)}/menit, QRS lebar ${m_r0(qrs_ms)} ms, P tidak jelas`); }
  else if (!p_present && wide && hr < 50) { irama = 'Kemungkinan irama escape ventrikel'; add('VESC?', 'Kemungkinan irama escape ventrikel', `${m_r0(hr)}/menit, QRS lebar`); }
  else if (p_present) {
    irama = p_amp > 0 ? 'Irama sinus' : 'Irama dengan gelombang P terbalik di II (ektopik atrium/junctional?)';
    if (p_amp > 0) { if (hr < 60) add('SB', 'Sinus bradikardia', `${m_r0(hr)}/menit`); else if (hr > 100) add('ST', 'Sinus takikardia', `${m_r0(hr)}/menit`);
      if (!regular && cv > 0.08 && pvc + pac <= 1) add('SA', 'Sinus aritmia', `variasi RR ${cv.toFixed(2)}`); }
  } else irama = 'Irama tidak dapat dipastikan (gelombang P tidak jelas)';
  out.irama = irama;
  // aktivitas atrium terpisah
  const xa = x[M_LI.II], blank = new Uint8Array(N).fill(1);
  for (const i of r) for (let t = Math.max(0, i - Math.floor(0.1 * fs)); t < Math.min(N, i + Math.floor(0.45 * fs)); t++) blank[t] = 0;
  const nb = blank.reduce((s2, u) => s2 + u, 0);
  if (nb > 2 * fs) {
    const Wa = m_welch(xa.map((u, t) => blank[t] ? u : 0), fs, Math.min(N, 4 * fs));
    if (m_bandsum(Wa, 3.8, 6.7) / (m_bandsum(Wa, 0.5, 20, false) + 1e-12) > 0.6 && !p_present) {
      let kb = -1; for (let k = 0; k < Wa.f.length; k++) if (Wa.f[k] >= 3.8 && Wa.f[k] <= 6.7 && (kb < 0 || Wa.p[k] > Wa.p[kb])) kb = k;
      add('AFL?', 'Kemungkinan flutter atrium', `aktivitas atrium ${m_r0(Wa.f[kb] * 60)}/menit`);
    }
  }
  const xa15 = m_filtfilt(FILTERS['1-15'], xa).map((u, t) => blank[t] ? u : 0);
  const ppk = m_findpeaks(xa15, 0.05, Math.floor(0.2 * fs)), ppi = m_diff(ppk).map(u => u / fs);
  const p_reg = ppi.length > 2 && m_std(ppi) / m_mean(ppi) < 0.2, p_rate = ppi.length > 1 ? 60 / m_median(ppi) : 0;
  if (hr < 60 && p_reg && ppk.length >= 1.4 * r.length && p_rate > 1.4 * hr) {
    if (regular && cv < 0.05) add('3AVB?', 'Kemungkinan blok AV derajat 3 (total)', `P ±${m_r0(p_rate)}/menit vs QRS ${m_r0(hr)}/menit, QRS teratur`);
    else add('2AVB?', 'Kemungkinan blok AV derajat 2', `P ±${m_r0(p_rate)}/menit vs QRS ${m_r0(hr)}/menit`);
  }
  if (pr_ms && pr_ms > 220) add('1AVB', 'Blok AV derajat 1', `PR ${m_r0(pr_ms)} ms (> 200; ambang otomatis 220)`);
  if (pr_ms && pr_ms < 120 && wide) add('WPW?', 'PR pendek + QRS lebar (curigai preeksitasi/WPW)', `PR ${m_r0(pr_ms)} ms, QRS ${m_r0(qrs_ms)} ms`);
  if (wide) {
    const v1 = qseg(M_LI.V1), v6 = qseg(M_LI.V6), v1x = Math.max(...v1), v1n = Math.min(...v1), v6x = Math.max(...v6), v6n = Math.min(...v6);
    if (v1x > Math.abs(v1n) * 0.8 && v6n < -0.1) add('RBBB?', 'Pola blok cabang berkas kanan', `QRS ${m_r0(qrs_ms)} ms, V1 dominan positif (rSR')`);
    else if (v1n < -Math.abs(v1x) && v6x > Math.abs(v6n)) add('LBBB?', 'Pola blok cabang berkas kiri', `QRS ${m_r0(qrs_ms)} ms, V1 dominan negatif, V6 lebar positif`);
    else add('IVCD', 'QRS lebar (gangguan konduksi intraventrikular)', `QRS ${m_r0(qrs_ms)} ms`);
  }
  if (ax >= -90 && ax < -30) add('LAD', 'Deviasi aksis kiri', `aksis ${m_r0(ax)}°`);
  else if (ax > 90 && ax <= 180) add('RAD', 'Deviasi aksis kanan', `aksis ${m_r0(ax)}°`);
  else if (ax < -90) add('EAD', 'Aksis ekstrem (northwest)', `aksis ${m_r0(ax)}°`);
  if (qtc_b) { if (qtc_b > 490 && qtc_f > 470) add('LQT', 'QT memanjang', `QTc ${m_r0(qtc_b)} ms (Bazett), ${m_r0(qtc_f)} ms (Fridericia)`); else if (qtc_b < 340) add('SQT', 'QT memendek', `QTc ${m_r0(qtc_b)} ms (Bazett)`); }
  if (out.voltase.qrs_ekstremitas_maks_mV < 0.5) add('LVOLT', 'Voltase QRS rendah', 'QRS < 0,5 mV di semua sadapan ekstremitas');
  const cornell = Math.max(0, mx(M_LI.aVL)) + Math.abs(Math.min(0, mn(M_LI.V3))); out.voltase.cornell_mV = m_r2(cornell);
  if (sok >= 3.5 || cornell > 2.8) add('LVH-V', 'Kriteria voltase LVH', `Sokolow-Lyon ${sok.toFixed(1)} mV (≥ 3,5) / Cornell ${cornell.toFixed(1)} mV (> 2,8)`);
  if (pvc) { const pat = kinds.join(''), big = pat.includes('NVNVNV') || pat.includes('VNVNVN');
    add('PVC', `Kontraksi ventrikel prematur (${pvc}× dalam ${m_r0(dur)} s)` + (big ? ' — pola bigemini' : ''), 'denyut prematur dengan morfologi berbeda'); }
  if (best_run >= 3) add('NSVT?', `${best_run} denyut berbeda-morfologi berturut-turut > 100/menit (curigai VT non-sustained)`, 'run denyut ektopik cepat');
  if (pac) add('PAC', `Kontraksi atrium prematur (${pac}× dalam ${m_r0(dur)} s)`, 'denyut prematur dengan morfologi QRS sama');
  if (!wide) {
    for (const [g, ls] of Object.entries(M_GROUPS)) {
      const lim = (g === 'septal' || g === 'anterior') ? 0.25 : 0.1;
      const el = ls.filter(l => st[M_LI[l]] >= lim);
      if (el.length >= 2) add('STE', `Elevasi ST di sadapan ${g} (${el.join(', ')})`, `ST J+40 ms ≥ ${lim === 0.25 ? '0,25' : '0,1'} mV di ≥ 2 sadapan bersebelahan`);
      const dep = ls.filter(l => st[M_LI[l]] <= -0.1);
      if (dep.length >= 2) add('STD', `Depresi ST di sadapan ${g} (${dep.join(', ')})`, 'ST J+40 ms ≤ −0,1 mV di ≥ 2 sadapan bersebelahan');
    }
    const inv = Object.values(M_GROUPS).flat().filter(l => (tpeak[l] ?? 0) < -0.15 && !['aVR', 'V1', 'III', 'aVL'].includes(l));
    if (inv.length >= 2) add('TINV', 'Inversi gelombang T (' + inv.join(', ') + ')', 'puncak T < −0,15 mV');
  }
  return out;
}
if (typeof module !== 'undefined') module.exports = { ecgMeasure, setFilters: F => { globalThis.FILTERS = F; } };
