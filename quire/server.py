"""Local HTTP server: serves the UI and exposes the evaluator as JSON."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine.worker import EvalWorker

UI_DIR = Path(__file__).parent / "ui"
FILE_RE = re.compile(r"^[\w][\w \-.]{0,80}$")
EXT = ".quire.json"


class App:
    def __init__(self, worksheets: Path, module_dirs: list[Path], examples: Path, password: str | None = None):
        self.worksheets = worksheets
        self.module_dirs = module_dirs
        self.examples = examples
        self.password = password or None  # when set, every request needs HTTP Basic auth with this password
        self.lock = threading.Lock()
        self.reload()

    def authorized(self, header: str | None) -> bool:
        if not self.password:
            return True
        if not header or not header.startswith("Basic "):
            return False
        try:
            _, _, given = base64.b64decode(header[6:]).decode().partition(":")
        except Exception:  # noqa: BLE001
            return False
        return given == self.password

    def reload(self):
        with self.lock:
            if hasattr(self, "worker"):
                self.worker.restart()
            else:
                self.worker = EvalWorker(self.module_dirs)
            self._catalog = self.worker.catalog()

    def catalog(self):
        return self._catalog

    def evaluate(self, cells):
        with self.lock:
            t0 = time.perf_counter()
            results = self.worker.evaluate(cells)
            return {"results": results, "ms": round((time.perf_counter() - t0) * 1000)}

    def _path(self, name: str, example=False) -> Path:
        if not FILE_RE.match(name):
            raise ValueError("Use letters, digits, spaces, '-' or '_' in the name.")
        base = self.examples if example else self.worksheets
        return base / (name if name.endswith(EXT) else name + EXT)

    def list_files(self):
        def names(d: Path):
            return sorted(p.name[: -len(EXT)] for p in d.glob("*" + EXT)) if d.is_dir() else []

        return {"files": names(self.worksheets), "examples": names(self.examples),
                "dir": str(self.worksheets)}

    def open(self, name, example=False):
        return json.loads(self._path(name, example).read_text())

    def save(self, name, doc):
        self.worksheets.mkdir(parents=True, exist_ok=True)
        p = self._path(name)
        p.write_text(json.dumps(doc, indent=1))
        return {"saved": p.name[: -len(EXT)], "path": str(p)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, status, body: bytes, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=200):
            self._send(status, json.dumps(obj).encode())

        def _guard(self) -> bool:
            if app.authorized(self.headers.get("Authorization")):
                return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Quire"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self._json({"ok": True})
            if not self._guard():
                return
            if path == "/api/catalog":
                return self._json(app.catalog())
            if path == "/api/files":
                return self._json(app.list_files())
            if path == "/":
                path = "/index.html"
            f = (UI_DIR / path.lstrip("/")).resolve()
            if UI_DIR.resolve() in f.parents and f.is_file():
                ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._guard():
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            try:
                if self.path == "/api/eval":
                    return self._json(app.evaluate(body.get("cells", [])))
                if self.path == "/api/open":
                    return self._json({"doc": app.open(body["name"], bool(body.get("example")))})
                if self.path == "/api/save":
                    return self._json(app.save(body["name"], body["doc"]))
                if self.path == "/api/reload":
                    app.reload()
                    return self._json(app.catalog())
            except FileNotFoundError:
                return self._json({"error": "No such worksheet."}, 404)
            except (ValueError, KeyError) as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            self._json({"error": "not found"}, 404)

    return Handler


def serve(app: App, host="127.0.0.1", port=8765, open_browser=True):
    server = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print(f"Quire running at {url}  (worksheets: {app.worksheets})")
    if open_browser:
        import webbrowser

        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
