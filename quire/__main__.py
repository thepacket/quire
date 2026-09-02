import argparse
from pathlib import Path

from .server import App, serve

ROOT = Path(__file__).resolve().parent.parent


def main():
    import os

    ap = argparse.ArgumentParser(prog="quire", description="Reactive, unit-aware math worksheets.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    ap.add_argument("--host", default=os.environ.get("QUIRE_HOST", "127.0.0.1"))
    ap.add_argument("--dir", type=Path, default=Path(os.environ.get("QUIRE_WORKSHEETS", ROOT / "worksheets")),
                    help="where worksheets are saved (env QUIRE_WORKSHEETS)")
    ap.add_argument("--modules", type=Path, action="append", help="extra module directory (repeatable)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--password", default=os.environ.get("QUIRE_PASSWORD"),
                    help="require this password (HTTP Basic auth); env QUIRE_PASSWORD")
    args = ap.parse_args()
    module_dirs = [ROOT / "modules"] + (args.modules or [])
    app = App(args.dir, module_dirs, ROOT / "examples", password=args.password)
    serve(app, args.host, args.port, not args.no_browser and args.host in ("127.0.0.1", "localhost"))


if __name__ == "__main__":
    main()
