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
    def __init__(self, worksheets: Path, module_dirs: list[Path], examples: Path, password: str | None = None,
                 mirror=None):
        self.worksheets = worksheets
        self.module_dirs = module_dirs
        self.examples = examples
        self.password = password or None  # when set, every request needs HTTP Basic auth with this password
        self.mirror = mirror  # optional quire.storage.Mirror: every write is copied to a bucket
        self.mirror_error: str | None = None
        self.lock = threading.Lock()
        if mirror is not None:
            try:
                self.worksheets.mkdir(parents=True, exist_ok=True)
                down, up = mirror.sync_down(), mirror.sync_up()
                print(f"Object storage: {down} file(s) downloaded, {up} uploaded ({mirror.client.bucket}/{mirror.prefix})")
            except Exception as exc:  # noqa: BLE001 - keep serving from the local folder
                self.mirror_error = f"object storage unreachable at startup: {exc}"
                print("warning:", self.mirror_error)
        self.reload()

    def _mirror(self, action: str, path: Path) -> None:
        if getattr(self, "mirror", None) is None:
            return
        try:
            getattr(self.mirror, action)(path)
            self.mirror_error = None
        except Exception as exc:  # noqa: BLE001
            self.mirror_error = f"not copied to object storage: {exc}"

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
                self.worker = EvalWorker(self.module_dirs, doc_dirs=[self.worksheets, self.examples])
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
        saved_at = time.strftime("%Y-%m-%d %H:%M")
        doc = dict(doc, saved_at=saved_at)
        if p.is_file():
            try:
                old = json.loads(p.read_text())
            except json.JSONDecodeError:
                old = None
            if old != doc and {k: v for k, v in (old or {}).items() if k != "saved_at"} != {k: v for k, v in doc.items() if k != "saved_at"}:
                self._keep_version(p)
        p.write_text(json.dumps(doc, indent=1))
        self._mirror("put", p)
        out = {"saved": p.name[: -len(EXT)], "path": str(p), "saved_at": saved_at}
        if self.mirror_error:
            out["warning"] = self.mirror_error
        return out

    # -- version history: previous saves live in <worksheets>/.history/<name>/ ---------------
    def _history_dir(self, name: str) -> Path:
        return self.worksheets / ".history" / self._path(name).name[: -len(EXT)]

    def _keep_version(self, p: Path, keep: int = 40):
        d = self._history_dir(p.name[: -len(EXT)])
        d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(p.stat().st_mtime))
        target = d / (stamp + EXT)
        k = 1
        while target.exists():
            target = d / (f"{stamp}-{k}" + EXT)
            k += 1
        target.write_text(p.read_text())
        self._mirror("put", target)
        old = sorted(d.glob("*" + EXT))
        for extra in old[: max(0, len(old) - keep)]:
            extra.unlink()
            self._mirror("delete", extra)

    def history(self, name: str) -> dict:
        d = self._history_dir(name)
        versions = []
        for f in sorted(d.glob("*" + EXT), reverse=True) if d.is_dir() else []:
            try:
                doc = json.loads(f.read_text())
                cells = len(doc.get("cells", []))
            except (json.JSONDecodeError, AttributeError):
                cells = None
            stamp = f.name[: -len(EXT)]
            t = time.strptime(stamp[:15], "%Y%m%d-%H%M%S")
            versions.append({"stamp": stamp, "time": time.strftime("%Y-%m-%d %H:%M:%S", t), "cells": cells})
        return {"versions": versions}

    def version(self, name: str, stamp: str) -> dict:
        if not re.match(r"^\d{8}-\d{6}(-\d+)?$", stamp):
            raise ValueError("Bad version.")
        f = self._history_dir(name) / (stamp + EXT)
        return json.loads(f.read_text())

    # -- read-only share links: token -> worksheet name, in <worksheets>/.shares.json ----------
    def _shares(self) -> dict:
        f = self.worksheets / ".shares.json"
        try:
            return json.loads(f.read_text()) if f.is_file() else {}
        except json.JSONDecodeError:
            return {}

    def _write_shares(self, shares: dict) -> None:
        self.worksheets.mkdir(parents=True, exist_ok=True)
        f = self.worksheets / ".shares.json"
        f.write_text(json.dumps(shares, indent=1))
        self._mirror("put", f)

    def share(self, name: str) -> dict:
        """Create (or return) the read-only link token of a saved worksheet."""
        import secrets

        p = self._path(name)
        if not p.is_file():
            raise FileNotFoundError(name)
        saved = p.name[: -len(EXT)]
        shares = self._shares()
        token = next((t for t, n in shares.items() if n == saved), None)
        if token is None:
            token = secrets.token_urlsafe(12)
            shares[token] = saved
            self._write_shares(shares)
        return {"token": token, "url": f"/s/{token}"}

    def unshare(self, name: str) -> dict:
        saved = self._path(name).name[: -len(EXT)]
        shares = {t: n for t, n in self._shares().items() if n != saved}
        self._write_shares(shares)
        return {"ok": True}

    def share_token(self, name: str) -> str | None:
        saved = self._path(name).name[: -len(EXT)]
        return next((t for t, n in self._shares().items() if n == saved), None)

    def shared_doc(self, token: str) -> dict | None:
        name = self._shares().get(token) if re.match(r"^[\w-]{8,64}$", token or "") else None
        if name is None:
            return None
        p = self._path(name)
        if not p.is_file():
            return None
        doc = json.loads(p.read_text())
        cells = []
        for cell in doc.get("cells", []):
            if cell.get("type") == "check":  # the reference and the hint stay on the server
                cell = dict(cell, reference="", hint="", locked=True, has_hint=bool((cell.get("hint") or "").strip()))
            cells.append(cell)
        return dict(doc, cells=cells)

    def _stored_doc(self, token: str) -> dict | None:
        name = self._shares().get(token) if re.match(r"^[\w-]{8,64}$", token or "") else None
        p = self._path(name) if name else None
        return json.loads(p.read_text()) if p and p.is_file() else None

    def shared_evaluate(self, token: str, body: dict) -> dict:
        """Evaluate a shared worksheet with the viewer's slider values and plot ranges applied, nothing else."""
        import copy

        doc = self._stored_doc(token)
        if doc is None:
            raise FileNotFoundError(token)
        cells = copy.deepcopy(doc.get("cells", []))
        sliders, plots = body.get("sliders") or {}, body.get("plots") or {}
        answers = body.get("answers") or {}
        for cell in cells:
            if cell.get("type") == "check" and isinstance(answers.get(str(cell.get("id"))), str):
                cell["answer"] = answers[str(cell.get("id"))][:2000]
        number = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")
        for cell in cells:
            cid = str(cell.get("id"))
            if cell.get("type", "math") == "math" and isinstance(sliders.get(cid), dict):
                lines = cell.get("source", "").split("\n")
                for line_s, value in sliders[cid].items():
                    try:
                        i, v = int(line_s), float(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= i < len(lines) and "slider(" in lines[i]:
                        lines[i] = re.sub(r"slider\(\s*[-+0-9.eE]+", f"slider({v:.6g}", lines[i], count=1)
                cell["source"] = "\n".join(lines)
            if cell.get("type") == "plot" and isinstance(plots.get(cid), dict):
                for k in ("xmin", "xmax", "ymin", "ymax"):
                    v = plots[cid].get(k)
                    if isinstance(v, str) and (v == "" or number.match(v)):
                        cell[k] = v.strip()
        out = self.evaluate(cells)
        for r in out.get("results", []):  # never echo the reference back
            r.pop("reference", None)
        return out

    def file_path(self, name: str) -> Path | None:
        """An uploaded data or image file, for /files/<name>."""
        if not re.match(r"^[\w][\w\-.]{0,80}$", name) or ".." in name:
            return None
        f = self.worksheets / "data" / name
        return f if f.is_file() else None

    def upload(self, filename: str, content_b64: str):
        """Store a data file (csv, tsv, txt, xlsx) or an image (png, jpg, gif, svg, webp) under <worksheets>/data."""
        stem, dot, ext = filename.rpartition(".")
        ext = ("." + ext.lower()) if dot else ""
        if ext == ".jpeg":
            ext = ".jpg"
        if ext not in (".csv", ".tsv", ".txt", ".xlsx", ".png", ".jpg", ".gif", ".svg", ".webp"):
            raise ValueError("Only .csv, .tsv, .txt, .xlsx and image files (.png, .jpg, .gif, .svg, .webp) can be uploaded.")
        safe = re.sub(r"\W+", "_", stem or "data").strip("_") or "data"
        if safe[0].isdigit():
            safe = "d_" + safe
        raw = base64.b64decode(content_b64)
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError("Files are limited to 20 MB.")
        d = self.worksheets / "data"
        d.mkdir(parents=True, exist_ok=True)
        (d / (safe + ext)).write_bytes(raw)
        self._mirror("put", d / (safe + ext))
        return {"name": safe, "file": safe + ext, "path": str(d / (safe + ext)),
                "image": ext in (".png", ".jpg", ".gif", ".svg", ".webp")}


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

        def _static(self, path: str):
            f = (UI_DIR / path.lstrip("/")).resolve()
            if UI_DIR.resolve() in f.parents and f.is_file():
                ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
            self._json({"error": "not found"}, 404)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/health":
                return self._json({"ok": True})
            # Read-only share links and the app's own files need no password; the API and uploads do.
            if path.startswith("/s/"):
                return self._static("/index.html")
            if path.startswith("/api/shared/"):
                doc = app.shared_doc(path.split("/")[3] if len(path.split("/")) > 3 else "")
                return self._json({"doc": doc}) if doc is not None else self._json({"error": "This link is no longer shared."}, 404)
            if path in ("/", "/index.html", "/app.js", "/style.css") or path.startswith("/vendor/"):
                return self._static("/index.html" if path == "/" else path)
            if path.startswith("/files/"):
                from urllib.parse import parse_qs, unquote

                token = parse_qs(query).get("share", [""])[0]
                if not (app.shared_doc(token) is not None or app.authorized(self.headers.get("Authorization"))):
                    return self._guard()
                f = app.file_path(unquote(path[len("/files/"):]))
                if f is None:
                    return self._json({"error": "not found"}, 404)
                ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype)
            if not self._guard():
                return
            if path == "/api/catalog":
                return self._json(app.catalog())
            if path == "/api/files":
                return self._json(app.list_files())
            if path.startswith("/api/history"):
                from urllib.parse import parse_qs, urlparse

                q = parse_qs(urlparse(self.path).query)
                try:
                    name = q.get("name", [""])[0]
                    return self._json(dict(app.history(name), share=app.share_token(name)))
                except ValueError as exc:
                    return self._json({"error": str(exc)}, 400)
            self._static(path)

        def do_POST(self):
            shared = re.match(r"^/api/shared/([\w-]+)/eval$", self.path)
            if not shared and not self._guard():
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            try:
                if shared:
                    return self._json(app.shared_evaluate(shared.group(1), body))
                if self.path == "/api/share":
                    return self._json(app.share(body["name"]))
                if self.path == "/api/unshare":
                    return self._json(app.unshare(body["name"]))
                if self.path == "/api/eval":
                    return self._json(app.evaluate(body.get("cells", [])))
                if self.path == "/api/open":
                    return self._json({"doc": app.open(body["name"], bool(body.get("example")))})
                if self.path == "/api/save":
                    return self._json(app.save(body["name"], body["doc"]))
                if self.path == "/api/reload":
                    app.reload()
                    return self._json(app.catalog())
                if self.path == "/api/upload":
                    return self._json(app.upload(body["filename"], body["content"]))
                if self.path == "/api/version":
                    return self._json({"doc": app.version(body["name"], body["stamp"])})
            except FileNotFoundError:
                return self._json({"error": "No such worksheet (save it first)."}, 404)
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
