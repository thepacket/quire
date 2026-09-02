"""Quire as a desktop application: the local server in a thread, shown in a native window (pywebview).

    pip install 'quire[desktop]'
    quire-desktop [--dir FOLDER]
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(worksheets: Path, module_dirs: list[Path]) -> tuple[ThreadingHTTPServer, str]:
    """Start the Quire server on a free local port in a daemon thread; returns (server, url)."""
    from .server import App, make_handler

    os.environ["QUIRE_WORKSHEETS"] = str(worksheets)
    app = App(worksheets, module_dirs, ROOT / "examples")
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}/"


def main(argv=None, webview=None) -> int:
    ap = argparse.ArgumentParser(prog="quire-desktop", description="Quire in its own window.")
    ap.add_argument("--dir", type=Path, default=Path(os.environ.get("QUIRE_WORKSHEETS", Path.home() / "Quire")),
                    help="where worksheets are saved (default ~/Quire)")
    ap.add_argument("--modules", type=Path, action="append", help="extra module directory (repeatable)")
    args = ap.parse_args(argv)
    if webview is None:
        try:
            import webview  # type: ignore
        except ImportError:
            print("The desktop window needs pywebview: pip install 'quire[desktop]'", file=sys.stderr)
            return 1
    args.dir.mkdir(parents=True, exist_ok=True)
    server, url = start_server(args.dir, [ROOT / "modules"] + (args.modules or []))
    try:
        webview.create_window("Quire", url, width=1220, height=840, min_size=(720, 520))
        webview.start()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
