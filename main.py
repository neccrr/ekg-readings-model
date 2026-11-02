"""Pembaca EKG 12 sadapan — satu perintah untuk semuanya.

  python3 main.py                      # siapkan lingkungan, uji model, buka web di http://localhost:8000
  python3 main.py file.csv --fs 257    # prediksi satu file langsung di terminal
  python3 main.py --port 9000 --no-browser

Saat pertama dijalankan, main.py membuat virtualenv .venv dan memasang dependensi (±1-3 menit).
Web lokal memakai model PyTorch lewat API /api/predict; halaman yang sama juga bisa jalan tanpa server.
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

import argparse, json, socket, threading, time, webbrowser  # noqa: E402
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'src'))
import predict as P  # noqa: E402


# ---------- 2. Server web lokal ----------
def make_handler(M):
    web = os.path.join(ROOT, 'web')

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=web, **k)

        def log_message(self, fmt, *args):
            if '/api/' in (args[0] if args else ''):
                sys.stderr.write(f"  {time.strftime('%H:%M:%S')} {args[0]} -> {args[1]}\n")

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith('/api/health'):
                return self._json(200, {'ok': True, 'engine': f'PyTorch {P.torch.__version__}', 'kelas': P.CLASSES})
            return super().do_GET()

        def do_POST(self):
            if not self.path.startswith('/api/predict'):
                return self._json(404, {'error': 'tidak ada'})
            try:
                n = int(self.headers.get('Content-Length', 0))
                if n > 60_000_000:
                    raise ValueError('Data terlalu besar (> 60 MB).')
                raw = self.rfile.read(n)
                t0 = time.perf_counter()
                if 'json' in (self.headers.get('Content-Type') or ''):
                    req = json.loads(raw)                       # {"sig": [12 x N], "fs": 100}
                    sig, fs = np.asarray(req['sig'], dtype=np.float32).T, float(req.get('fs', 100))
                else:                                           # CSV mentah: curl --data-binary @file.csv "…?fs=257"
                    from urllib.parse import urlparse, parse_qs
                    fs = float(parse_qs(urlparse(self.path).query).get('fs', ['100'])[0])
                    import tempfile
                    with tempfile.NamedTemporaryFile('wb', suffix='.csv', delete=False) as f:
                        f.write(raw); tmp = f.name
                    try:
                        sig, fs = P.load_signal(tmp, fs)
                    finally:
                        os.unlink(tmp)
                out = P.predict_signal(sig, fs, M)
                out = {k: v for k, v in out.items() if not k.startswith('_')}
                out['ms'] = round((time.perf_counter() - t0) * 1000)
                out['engine'] = 'PyTorch (server lokal)'
                self._json(200, out)
            except (ValueError, KeyError) as e:
                self._json(400, {'error': str(e)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {'error': f'{type(e).__name__}: {e}'})
    return H


def free_port(start):
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    raise SystemExit('Tidak ada port kosong.')


def self_test(M):
    print('\nUji cepat model pada contoh/ :')
    for f in sorted(os.listdir(os.path.join(ROOT, 'contoh'))):
        if f.endswith('.csv'):
            sig, fs = P.load_signal(os.path.join(ROOT, 'contoh', f), 100)
            out = P.predict_signal(sig, fs, M)
            print(f'  {f:<18} -> {out["kesimpulan"]}')


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        return P.main(sys.argv[1:])                 # mode terminal: python3 main.py file.csv --fs 257
    ap = argparse.ArgumentParser(description='Jalankan Pembaca EKG di localhost.')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()

    t0 = time.time(); M = P.load_model()
    print(f'Model dimuat ({sum(p.numel() for p in M["model"].parameters()):,} parameter, {time.time()-t0:.1f} s).')
    self_test(M)
    port = free_port(a.port)
    srv = ThreadingHTTPServer(('127.0.0.1', port), make_handler(M))
    url = f'http://localhost:{port}'
    print(f'\nWeb aktif: {url}   (Ctrl+C untuk berhenti)')
    print(f'API      : curl --data-binary @file.csv "{url}/api/predict?fs=257"\n')
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nBerhenti.')
        srv.server_close()


if __name__ == '__main__':
    main()
