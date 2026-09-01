import argparse
from pathlib import Path

from .server import App, serve

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(prog="quire", description="Reactive, unit-aware math worksheets.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dir", type=Path, default=ROOT / "worksheets", help="where worksheets are saved")
    ap.add_argument("--modules", type=Path, action="append", help="extra module directory (repeatable)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    module_dirs = [ROOT / "modules"] + (args.modules or [])
    app = App(args.dir, module_dirs, ROOT / "examples")
    serve(app, args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
