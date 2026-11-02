"""Pembaca EKG 12 sadapan — satu perintah untuk semuanya.

  python3 main.py                      # siapkan lingkungan, uji model, buka web di http://localhost:8000
  python3 main.py file.csv --fs 257    # prediksi satu file langsung di terminal
  python3 main.py --port 9000 --no-browser --log-file logs/server.log -v

Saat pertama dijalankan, main.py membuat virtualenv .venv dan memasang dependensi (±1-3 menit).
Web lokal memakai model PyTorch lewat API /api/predict; statistik performa di /api/stats.
PERINGATAN: model pembelajaran, bukan alat diagnosis klinis.
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, '.venv')
DEPS = ['torch', 'numpy', 'scipy', 'wfdb', 'matplotlib']


# ---------- 1. Siapkan lingkungan Python otomatis ----------
def ensure_env():
    try:
        import torch, numpy, scipy, wfdb  # noqa: F401
        return
    except ImportError:
        pass
    vpy = os.path.join(VENV, 'Scripts' if os.name == 'nt' else 'bin', 'python')
    if os.path.abspath(sys.executable) == os.path.abspath(vpy):
        print('Memasang dependensi ke .venv (sekali saja, ±1-3 menit)…', flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *DEPS])
        return
    if not os.path.exists(vpy):
        print('Membuat virtualenv .venv …', flush=True)
        subprocess.check_call([sys.executable, '-m', 'venv', VENV])
    os.execv(vpy, [vpy, os.path.abspath(__file__), *sys.argv[1:]])  # jalankan ulang di dalam .venv


ensure_env()

import argparse, json, logging, platform, socket, statistics, threading, time, webbrowser  # noqa: E402
from collections import deque  # noqa: E402
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler  # noqa: E402
from urllib.parse import unquote, urlparse, parse_qs  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'src'))
import predict as P  # noqa: E402

log = logging.getLogger('ekg')


# ---------- 2. Logging terminal ----------
class Fmt(logging.Formatter):
    COL = {'DEBUG': '\033[2m', 'INFO': '\033[36m', 'WARNING': '\033[33m', 'ERROR': '\033[31m'}

    def __init__(self, color):
        super().__init__('%(asctime)s %(levelname)-5s %(message)s', '%H:%M:%S')
        self.color = color

    def format(self, r):
        s = super().format(r)
        if self.color:
            c = self.COL.get(r.levelname, '')
            s = s.replace(r.levelname, f'{c}{r.levelname}\033[0m', 1)
        return s


def setup_logging(verbose, log_file):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(Fmt(sys.stdout.isatty() and os.environ.get('NO_COLOR') is None))
    log.addHandler(h)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)-5s %(message)s'))
        log.addHandler(fh)


def rss_mb():
    """Memori puncak proses (MB)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1e6 if sys.platform == 'darwin' else r / 1e3, 1)  # macOS: byte, Linux: KB
    except Exception:  # noqa: BLE001
        return None


# ---------- 3. Statistik server ----------
class Stats:
    def __init__(self, M, load_ms):
        self.t_start = time.time()
        self.lock = threading.Lock()
        self.recent = deque(maxlen=50)
        self.lat = deque(maxlen=500)
        self.n = self.err = 0
        self.ekg_s = 0.0
        wpath = os.path.join(ROOT, 'model', 'resnet1d_best.pt')
        self.info = {
            'python': platform.python_version(), 'torch': P.torch.__version__,
            'platform': f'{platform.system()} {platform.release()} ({platform.machine()})',
            'cpu_count': os.cpu_count(), 'torch_threads': P.torch.get_num_threads(),
            'params': sum(p.numel() for p in M['model'].parameters()),
            'model_mb': round(os.path.getsize(wpath) / 1e6, 2), 'load_ms': round(load_ms, 1),
        }

    def add(self, rec):
        with self.lock:
            self.n += 1
            if rec.get('error'):
                self.err += 1
            else:
                self.lat.append(rec['total_ms']); self.ekg_s += rec['durasi_detik']
            self.recent.appendleft(rec)

    def snapshot(self):
        with self.lock:
            lat = sorted(self.lat)
            pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)) if lat else (lambda q: None)
            return {'uptime_s': round(time.time() - self.t_start), 'requests': self.n, 'errors': self.err,
                    'latency_ms': {'mean': round(statistics.fmean(lat), 1) if lat else None, 'p50': pct(.5), 'p95': pct(.95),
                                   'max': round(lat[-1], 1) if lat else None},
                    'ekg_detik_total': round(self.ekg_s, 1), 'rss_mb': rss_mb(), 'info': self.info, 'recent': list(self.recent)[:20]}


# ---------- 4. Server web lokal ----------
def make_handler(M, S):
    web = os.path.join(ROOT, 'web')

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=web, **k)

        def log_message(self, fmt, *args):  # permintaan file statis -> DEBUG saja
            log.debug('%s %s', self.address_string(), fmt % args)

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith('/api/health'):
                return self._json(200, {'ok': True, 'engine': f'PyTorch {P.torch.__version__}', 'kelas': P.CLASSES})
            if self.path.startswith('/api/stats'):
                return self._json(200, S.snapshot())
            if self.path in ('/', '/index.html'):
                log.info('Halaman dibuka dari %s', self.client_address[0])
            return super().do_GET()

        def do_POST(self):
            if not self.path.startswith('/api/predict'):
                return self._json(404, {'error': 'tidak ada'})
            name = unquote(self.headers.get('X-File-Name') or 'tanpa-nama')
            src_fs = self.headers.get('X-Source-Fs')
            t0 = time.perf_counter(); n = 0
            try:
                n = int(self.headers.get('Content-Length', 0))
                if n > 60_000_000:
                    raise ValueError('Data terlalu besar (> 60 MB).')
                raw = self.rfile.read(n)
                t1 = time.perf_counter()
                if 'json' in (self.headers.get('Content-Type') or ''):
                    req = json.loads(raw)                       # {"sig": [12 x N], "fs": 100}
                    sig, fs = np.asarray(req['sig'], dtype=np.float32).T, float(req.get('fs', 100))
                else:                                           # CSV mentah: curl --data-binary @file.csv "…?fs=257"
                    fs = float(parse_qs(urlparse(self.path).query).get('fs', ['100'])[0])
                    import tempfile
                    with tempfile.NamedTemporaryFile('wb', suffix='.csv', delete=False) as f:
                        f.write(raw); tmp = f.name
                    try:
                        sig, fs = P.load_signal(tmp, fs)
                    finally:
                        os.unlink(tmp)
                t2 = time.perf_counter()
                out = P.predict_signal(sig, fs, M)
                out = {k: v for k, v in out.items() if not k.startswith('_')}
                t3 = time.perf_counter()
                tm = {'read_ms': round((t1 - t0) * 1000, 2), 'parse_ms': round((t2 - t1) * 1000, 2),
                      **out['timing'], 'total_ms': round((t3 - t0) * 1000, 2)}
                out['timing'] = tm
                out['payload_bytes'] = n
                out['realtime_x'] = round(out['durasi_detik'] / max(tm['total_ms'] / 1000, 1e-6))
                out['engine'] = 'PyTorch (server lokal)'
                self._json(200, out)
                pos = [h for h in out['hasil'] if h['positif']]
                fs_txt = f'{src_fs}→{fs:g} Hz (diubah di browser)' if src_fs and float(src_fs) != fs else f'{fs:g} Hz'
                log.info('PREDIKSI %s | %s, %.1f s, %d jendela, %.0f KB | baca %.0f · parse %.0f · pra-proses %.0f · '
                         'inferensi %.0f · total %.0f ms (%d× realtime) | %s',
                         name, fs_txt, out['durasi_detik'], out['jumlah_jendela'], n / 1024, tm['read_ms'], tm['parse_ms'],
                         tm['preprocess_ms'], tm['inference_ms'], tm['total_ms'], out['realtime_x'], out['kesimpulan'])
                log.debug('  probabilitas: %s', '  '.join(f"{h['kode']} {h['probabilitas']:.3f}{'*' if h['positif'] else ''}"
                                                          for h in out['hasil']))
                S.add({'waktu': time.strftime('%H:%M:%S'), 'file': name, 'durasi_detik': out['durasi_detik'],
                       'jendela': out['jumlah_jendela'], 'kb': round(n / 1024, 1), **tm, 'realtime_x': out['realtime_x'],
                       'kesimpulan': out['kesimpulan'], 'positif': [h['kode'] for h in pos]})
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                log.warning('DITOLAK %s | %s', name, e)
                S.add({'waktu': time.strftime('%H:%M:%S'), 'file': name, 'error': str(e)})
                self._json(400, {'error': str(e)})
            except Exception as e:  # noqa: BLE001
                log.exception('GALAT %s', name)
                S.add({'waktu': time.strftime('%H:%M:%S'), 'file': name, 'error': f'{type(e).__name__}: {e}'})
                self._json(500, {'error': f'{type(e).__name__}: {e}'})
    return H


def free_port(start):
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    raise SystemExit('Tidak ada port kosong.')


def self_test(M):
    log.info('Uji cepat pada contoh/ :')
    for f in sorted(os.listdir(os.path.join(ROOT, 'contoh'))):
        if f.endswith('.csv'):
            sig, fs = P.load_signal(os.path.join(ROOT, 'contoh', f), 100)
            out = P.predict_signal(sig, fs, M)
            log.info('  %-16s %6.1f ms  -> %s', f, out['timing']['preprocess_ms'] + out['timing']['inference_ms'], out['kesimpulan'])


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        return P.main(sys.argv[1:])                 # mode terminal: python3 main.py file.csv --fs 257
    ap = argparse.ArgumentParser(description='Jalankan Pembaca EKG di localhost.')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--log-file', help='simpan log juga ke file ini, mis. logs/server.log')
    ap.add_argument('-v', '--verbose', action='store_true', help='log rinci (probabilitas, file statis)')
    a = ap.parse_args()
    setup_logging(a.verbose, a.log_file)

    log.info('Python %s · PyTorch %s · %s · %d CPU (%d thread PyTorch)', platform.python_version(),
             P.torch.__version__, platform.platform(terse=True), os.cpu_count(), P.torch.get_num_threads())
    t0 = time.perf_counter(); M = P.load_model(); load_ms = (time.perf_counter() - t0) * 1000
    S = Stats(M, load_ms)
    log.info('Model dimuat: %d parameter, %.1f MB, %.0f ms', S.info['params'], S.info['model_mb'], load_ms)
    with P.torch.no_grad():                         # pemanasan agar permintaan pertama tidak lambat
        M['model'](P.torch.zeros(1, 12, 1000))
    self_test(M)
    port = free_port(a.port)
    if port != a.port:
        log.warning('Port %d terpakai, memakai %d', a.port, port)
    srv = ThreadingHTTPServer(('127.0.0.1', port), make_handler(M, S))
    url = f'http://localhost:{port}'
    log.info('Web aktif: %s   (Ctrl+C untuk berhenti)', url)
    log.info('API: POST %s/api/predict · GET %s/api/stats', url, url)
    if a.log_file:
        log.info('Log juga disimpan ke %s', a.log_file)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        snap = S.snapshot()
        log.info('Berhenti. %d permintaan (%d gagal), median %s ms, %.1f menit EKG dianalisis, memori puncak %s MB.',
                 snap['requests'], snap['errors'], snap['latency_ms']['p50'] or '-', snap['ekg_detik_total'] / 60, snap['rss_mb'])
        srv.server_close()


if __name__ == '__main__':
    main()
