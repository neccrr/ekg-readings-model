// ===== Mesin inferensi 1D ResNet (BatchNorm sudah dilebur ke konvolusi) =====
function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
           : e === 31 ? (f ? NaN : s * Infinity)
           : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}
function b64ToBytes(b64) {
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
function conv(x, L, P, W, relu) {
  const { co, ci, k, s, p } = P, Lo = Math.floor((L + 2 * p - k) / s) + 1;
  const y = new Float32Array(co * Lo);
  for (let o = 0; o < co; o++) {
    const yo = o * Lo, bias = W[P.b + o];
    for (let t = 0; t < Lo; t++) y[yo + t] = bias;
    for (let c = 0; c < ci; c++) {
      const xc = c * L, wb = P.w + (o * ci + c) * k;
      for (let kk = 0; kk < k; kk++) {
        const w = W[wb + kk], off = kk - p;
        const t0 = Math.max(0, Math.ceil(-off / s)), t1 = Math.min(Lo, Math.floor((L - 1 - off) / s) + 1);
        for (let t = t0, idx = xc + t0 * s + off; t < t1; t++, idx += s) y[yo + t] += w * x[idx];
      }
    }
    if (relu) for (let t = 0; t < Lo; t++) if (y[yo + t] < 0) y[yo + t] = 0;
  }
  return [y, Lo];
}
function maxpool(x, C, L) { // k3 s2 p1
  const Lo = Math.floor((L + 2 - 3) / 2) + 1, y = new Float32Array(C * Lo);
  for (let c = 0; c < C; c++) for (let t = 0; t < Lo; t++) {
    let m = -Infinity;
    for (let kk = -1; kk <= 1; kk++) { const i = 2 * t + kk; if (i >= 0 && i < L && x[c * L + i] > m) m = x[c * L + i]; }
    y[c * Lo + t] = m;
  }
  return [y, Lo];
}
// x: Float32Array 12*1000 (channel-major, sudah dinormalisasi) -> 5 probabilitas
function forward(M, W, x) {
  let [h, L] = conv(x, 1000, M.stem, W, true);
  [h, L] = maxpool(h, M.stem.co, L);
  for (const B of M.blocks) {
    const [a, La] = conv(h, L, B.c1, W, true);
    const [b, Lb] = conv(a, La, B.c2, W, false);
    const sk = B.sk ? conv(h, L, B.sk, W, false)[0] : h;
    for (let i = 0; i < b.length; i++) { const v = b[i] + sk[i]; b[i] = v > 0 ? v : 0; }
    h = b; L = Lb;
  }
  const C = M.head.i / 2, feat = new Float32Array(M.head.i);
  for (let c = 0; c < C; c++) {
    let s = 0, m = -Infinity;
    for (let t = 0; t < L; t++) { const v = h[c * L + t]; s += v; if (v > m) m = v; }
    feat[c] = s / L; feat[C + c] = m;
  }
  const out = [];
  for (let o = 0; o < M.head.o; o++) {
    let z = W[M.head.b + o];
    for (let i = 0; i < M.head.i; i++) z += W[M.head.w + o * M.head.i + i] * feat[i];
    out.push(1 / (1 + Math.exp(-z)));
  }
  return out;
}
// sig: array of 12 Float32Array (mV, 100 Hz). Jendela 10 s, geser 5 s, rata-rata.
function predictSignal(M, W, sig) {
  const N = sig[0].length, starts = [];
  for (let s = 0; s + 1000 <= N; s += 500) starts.push(s);
  const acc = [0, 0, 0, 0, 0];
  for (const s0 of starts) {
    const x = new Float32Array(12000);
    for (let l = 0; l < 12; l++) for (let t = 0; t < 1000; t++) x[l * 1000 + t] = (sig[l][s0 + t] - M.mu[l]) / M.sd[l];
    forward(M, W, x).forEach((p, i) => acc[i] += p / starts.length);
  }
  return { probs: acc, windows: starts.length };
}
// Ubah frekuensi sampling ke 100 Hz: FIR low-pass windowed-sinc (cutoff 45 Hz) lalu interpolasi linear
function resampleTo100(ch, fs) {
  if (Math.abs(fs - 100) < 1e-6) return Float32Array.from(ch);
  let src = ch;
  if (fs > 100) {
    const fc = 45 / fs, half = Math.round(4 * fs / 100) * 4, taps = new Float32Array(2 * half + 1);
    let sum = 0;
    for (let i = -half; i <= half; i++) {
      const sinc = i === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * i) / (Math.PI * i);
      const win = 0.54 + 0.46 * Math.cos(Math.PI * i / half);
      taps[i + half] = sinc * win; sum += sinc * win;
    }
    src = new Float32Array(ch.length);
    for (let t = 0; t < ch.length; t++) {
      let acc = 0;
      for (let i = -half; i <= half; i++) {
        const j = Math.min(ch.length - 1, Math.max(0, t + i));
        acc += taps[i + half] * ch[j];
      }
      src[t] = acc / sum;
    }
  }
  const n = Math.floor((src.length - 1) * 100 / fs) + 1, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const pos = i * fs / 100, a = Math.floor(pos), f = pos - a;
    out[i] = a + 1 < src.length ? src[a] * (1 - f) + src[a + 1] * f : src[a];
  }
  return out;
}
if (typeof module !== 'undefined') module.exports = { f16ToF32, b64ToBytes, forward, predictSignal, resampleTo100 };
