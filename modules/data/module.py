"""Data files: CSV and TSV (and Excel when openpyxl is installed) read from the worksheet folder's data/ directory.

Files are referred to by name without extension: read_csv(measurements) loads data/measurements.csv.
Upload files with the Data button in the app, or copy them into <worksheets>/data/.
"""
import csv
import os
from pathlib import Path

import sympy as sp

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "data"
DESCRIPTION = "Read CSV/TSV/Excel files from the data folder into tables, columns and lists."

ROOT = Path(__file__).resolve().parent.parent.parent


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _dirs():
    ws = Path(os.environ.get("QUIRE_WORKSHEETS", ROOT / "worksheets"))
    return [ws / "data", ROOT / "examples" / "data"]


def _find(name):
    name = str(name)
    for d in _dirs():
        for ext in (".csv", ".tsv", ".txt", ".xlsx"):
            p = d / f"{name}{ext}"
            if p.is_file():
                return p
    raise EvalError(f"No data file named '{name}' in {', '.join(str(d) for d in _dirs())}. Upload one with the Data button.")


def _cell(text):
    t = text.strip()
    if t == "":
        return sp.nan
    try:
        return sp.Integer(int(t))
    except ValueError:
        pass
    try:
        float(t)
        digits = len(t.lstrip("+-").replace(".", "").split("e")[0].split("E")[0].lstrip("0")) or 1
        return sp.Float(t, max(digits, 3))  # keeps the digits as written
    except ValueError:
        return sp.Symbol(t.replace(" ", "_"))


def _load(name):
    p = _find(name)
    if p.suffix == ".xlsx":
        try:
            import openpyxl
        except ImportError:
            raise EvalError("Reading .xlsx needs the openpyxl package.") from None
        ws = openpyxl.load_workbook(p, read_only=True, data_only=True).active
        rows = [["" if v is None else str(v) for v in r] for r in ws.iter_rows(values_only=True)]
    else:
        with open(p, newline="", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            rows = [r for r in csv.reader(f, dialect) if any(c.strip() for c in r)]
    if not rows:
        raise EvalError(f"'{p.name}' is empty.")
    header = None
    first = rows[0]
    if any(not _is_number(c) for c in first):
        header = [c.strip() for c in first]
        rows = rows[1:]
    width = max(len(r) for r in rows) if rows else len(header or [])
    data = [[_cell(c) for c in r] + [sp.nan] * (width - len(r)) for r in rows]
    return p, header, data


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def read_csv(name):
    """The file as a matrix (numbers; text cells become symbols). The header goes in a note."""
    p, header, data = _load(name)
    _note(f"{p.name}: {len(data)} rows × {len(data[0]) if data else 0} columns" + (f"; columns: {', '.join(header)}" if header else ""))
    return sp.ImmutableMatrix(data)


def headers(name):
    _, header, _ = _load(name)
    return [sp.Symbol(h.replace(" ", "_")) for h in (header or [])]


def column(name, which):
    """A column as a list, by header name (symbol) or 1-based index."""
    p, header, data = _load(name)
    if isinstance(which, sp.Symbol) or isinstance(which, str):
        key = str(which)
        if not header:
            raise EvalError(f"'{p.name}' has no header row; use a column number.")
        names = [h.replace(" ", "_") for h in header]
        if key not in names:
            raise EvalError(f"No column '{key}'; columns are {', '.join(names)}.")
        k = names.index(key)
    else:
        k = int(which) - 1
        if not 0 <= k < len(data[0]):
            raise EvalError(f"Column number out of range (1..{len(data[0])}).")
    return [row[k] for row in data]


def row_of(name, k):
    _, _, data = _load(name)
    k = int(k) - 1
    if not 0 <= k < len(data):
        raise EvalError(f"Row number out of range (1..{len(data)}).")
    return list(data[k])


def table_size(name):
    _, header, data = _load(name)
    return [sp.Integer(len(data)), sp.Integer(len(data[0]) if data else 0)]


def data_files():
    names = sorted({p.stem for d in _dirs() if d.is_dir() for p in d.iterdir() if p.suffix in (".csv", ".tsv", ".txt", ".xlsx")})
    _note("files in the data folders")
    return [sp.Symbol(n) for n in names]


def register(api):
    D = "Data files"
    api.function("read_csv", read_csv, signature="read_csv(name)", doc="the file as a matrix", category=D, example="read_csv(measurements)")
    api.function("column", column, signature="column(name, header_or_number)", doc="one column as a list", category=D,
                 example="column(measurements, temperature)")
    api.function("row_of", row_of, signature="row_of(name, k)", doc="one row as a list", category=D)
    api.function("headers", headers, signature="headers(name)", doc="column names", category=D)
    api.function("table_size", table_size, signature="table_size(name)", doc="[rows, columns]", category=D)
    api.function("data_files", data_files, signature="data_files()", doc="names of the available files", category=D)
