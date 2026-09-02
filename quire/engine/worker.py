"""Run evaluation in a separate process so a runaway computation cannot freeze the app.

Evaluation is stateless (the whole document is re-evaluated each time), so the
worker holds nothing but the loaded modules. On timeout the process is killed
and a fresh one started; the cell that was running gets a clear error.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

TIMEOUT_MESSAGE = ("This took longer than {s} s and was stopped. Try nsolve or nintegrate for a numeric "
                   "answer, or simplify the expression.")


def make_loader(dirs: list[Path]):
    """name -> (doc, stamp) from the first directory holding <name>.quire.json, else None."""
    import json
    import re

    def loader(name: str):
        if not re.match(r"^[\w][\w \-.]{0,80}$", name):
            return None
        for d in dirs:
            p = Path(d) / (name if name.endswith(".quire.json") else name + ".quire.json")
            if p.is_file():
                return json.loads(p.read_text()), p.stat().st_mtime_ns
        return None
    return loader


def _loop(conn, module_dirs, doc_dirs=()):
    from .evaluator import Evaluator
    from ..modules.registry import load_registry

    registry = load_registry([Path(d) for d in module_dirs])
    ev = Evaluator(registry, loader=make_loader([Path(d) for d in doc_dirs]) if doc_dirs else None)
    cache: dict = {}  # per-cell results, replayed when a cell and everything it depends on are unchanged
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        kind = msg[0]
        if kind == "catalog":
            conn.send(registry.catalog())
        elif kind == "eval":
            cells = msg[1]
            # Stream results one cell at a time so a timeout can name the culprit.
            for res in ev.iter_evaluate(cells, cache):
                conn.send(("cell", res))
            conn.send(("done",))
        elif kind == "quit":
            return


class EvalWorker:
    def __init__(self, module_dirs: list[Path], timeout: float = 20.0, doc_dirs: list[Path] = ()):
        self.module_dirs = [str(d) for d in module_dirs]
        self.doc_dirs = [str(d) for d in doc_dirs]
        self.timeout = timeout
        self._ctx = mp.get_context("spawn")
        self._start()

    def _start(self):
        self._booting = True
        self.conn, child = self._ctx.Pipe()
        self.proc = self._ctx.Process(target=_loop, args=(child, self.module_dirs, self.doc_dirs), daemon=True)
        self.proc.start()
        child.close()

    def restart(self):
        try:
            self.proc.kill()
            self.proc.join(2)
        except Exception:  # noqa: BLE001
            pass
        self._start()

    def _recv(self, timeout: float | None = None):
        """Receive one message, or None on timeout / worker death (worker is restarted).

        The first answer after a (re)start waits for the modules to load, which can take
        longer than the evaluation timeout on a slow machine.
        """
        if timeout is None:
            timeout = max(self.timeout, 60.0) if self._booting else self.timeout
        try:
            if self.conn.poll(timeout):
                self._booting = False
                return self.conn.recv()
            self.restart()
        except (EOFError, OSError):
            self.restart()
        return None

    def catalog(self) -> dict:
        self.conn.send(("catalog",))
        cat = self._recv()
        return cat if cat is not None else {"modules": [], "entries": [], "error": "The evaluation worker did not start."}

    def evaluate(self, cells: list[dict]) -> list[dict]:
        try:
            self.conn.send(("eval", cells))
        except (BrokenPipeError, OSError):
            self.restart()
            self.conn.send(("eval", cells))
        results = []
        for cell in cells:
            msg = self._recv()
            if msg is None:
                msg = TIMEOUT_MESSAGE.format(s=int(self.timeout))
                results.append({"id": cell.get("id"), "ok": False, "outputs": [], "defines": [], "uses": [],
                                "error": msg, "warning": None})
                for later in cells[len(results):]:
                    results.append({"id": later.get("id"), "ok": False, "outputs": [], "defines": [], "uses": [],
                                    "error": "Not evaluated: a cell above timed out.", "warning": None})
                return results
            results.append(msg[1])
        self._recv()  # ("done",)
        return results
