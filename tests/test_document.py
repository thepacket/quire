"""Worksheet as a document: live values in text, imports, version history, uploads."""
import base64
import json
from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator, placeholders
from quire.modules.registry import load_registry
from quire.server import App

ROOT = Path(__file__).resolve().parent.parent

DOCS = {
    "beam": ({"cells": [{"type": "math", "source": "L = 2 m\nE = 200 GPa"}, {"type": "text", "source": "notes"},
                        {"type": "math", "source": "assume w > 0\nI(b, h) = b h^3/12"}]}, 1),
    "loop": ({"cells": [{"type": "math", "source": "import loop"}]}, 1),
    "uses_beam": ({"cells": [{"type": "math", "source": "import beam\nL2 = 2*L"}]}, 1),
    "broken": ({"cells": [{"type": "math", "source": "x = 1/0)"}]}, 1),
}


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]), loader=lambda name: DOCS.get(name))


def test_placeholders_skip_math_spans():
    assert placeholders("beam $\\frac{1}{2}$ deflects {{delta -> mm}} and {{ x^2 }} $$a{{b}}$$ `{{code}}` {{") == ["delta -> mm", "x^2"]


def test_live_values_in_text(ev):
    rs = ev.evaluate([{"id": 1, "type": "math", "source": "L = 2 m\ndigits 3"},
                      {"id": 2, "type": "text", "source": "Length {{L -> cm}}, third {{L/3}}, bad {{nothing(}}, and {{y = 3}} defines nothing"},
                      {"id": 3, "type": "math", "source": "y"}])
    v = rs[1]["values"]
    assert rs[1]["ok"] and len(v) == 4
    assert v[0]["plain"] == "200*centimeter" and v[0]["latex"] == "200\\,\\text{cm}"
    assert v[1]["plain"] == "2*meter/3" and v[1]["approx_plain"] == "0.667 meter"  # digits 3 applies in text too
    assert "error" in v[2] and "')'" in v[2]["error"]
    assert rs[2]["outputs"][0]["plain"] == "y"                   # the placeholder's definition did not leak
    assert "values" not in ev.evaluate([{"id": 1, "type": "text", "source": "plain $x^{2}$ text"}])[0]


def test_import_brings_definitions(ev):
    rs = ev.evaluate([{"id": 1, "type": "math", "source": "import beam\nF = 10 kN"},
                      {"id": 2, "type": "math", "source": "I(2, 3)"},
                      {"id": 3, "type": "math", "source": "import uses_beam"},
                      {"id": 4, "type": "math", "source": "L2 -> cm"}])
    assert rs[0]["ok"] and rs[0]["defines"] == ["L", "E", "I", "F"]
    assert rs[0]["outputs"][0]["kind"] == "import" and rs[0]["outputs"][0]["imported"] == ["L", "E", "I"]
    assert rs[1]["outputs"][0]["plain"] == "9/2"
    assert rs[2]["defines"] == ["L", "E", "I", "L2"] and rs[3]["outputs"][0]["plain"] == "400*centimeter"


def test_import_errors(ev):
    r = ev.evaluate([{"id": 1, "type": "math", "source": "import loop"}])[0]
    assert not r["ok"] and "imports itself" in r["error"]
    r = ev.evaluate([{"id": 1, "type": "math", "source": "import nope"}])[0]
    assert not r["ok"] and "No worksheet named 'nope'" in r["error"]
    r = ev.evaluate([{"id": 1, "type": "math", "source": "import broken"}])[0]
    assert not r["ok"] and r["error"].startswith("In 'broken'")
    plain = Evaluator(load_registry([ROOT / "modules"]))
    r = plain.evaluate([{"id": 1, "type": "math", "source": "import beam"}])[0]
    assert not r["ok"] and "not available" in r["error"]


def test_import_cache_follows_the_file(ev):
    docs = {"k": ({"cells": [{"type": "math", "source": "k = 1"}]}, 1)}
    e = Evaluator(load_registry([ROOT / "modules"]), loader=lambda n: docs.get(n))
    assert e.evaluate([{"id": 1, "type": "math", "source": "import k\nk"}])[0]["outputs"][1]["plain"] == "1"
    docs["k"] = ({"cells": [{"type": "math", "source": "k = 2"}]}, 2)  # a new stamp: re-read
    assert e.evaluate([{"id": 1, "type": "math", "source": "import k\nk"}])[0]["outputs"][1]["plain"] == "2"


def _app(tmp_path) -> App:
    app = App.__new__(App)  # no worker process: only the file side is exercised
    app.worksheets = tmp_path / "ws"
    app.examples = ROOT / "examples"
    return app


def test_history_keeps_previous_saves(tmp_path):
    app = _app(tmp_path)
    doc = {"quire": 1, "title": "T", "cells": [{"id": "a", "type": "math", "source": "x = 1"}]}
    r = app.save("beam calc", doc)
    assert r["saved"] == "beam calc" and r["saved_at"]
    assert app.history("beam calc")["versions"] == []           # the first save has nothing before it
    app.save("beam calc", doc)                                   # identical content: no version
    assert app.history("beam calc")["versions"] == []
    doc2 = {"quire": 1, "title": "T", "cells": [{"id": "a", "type": "math", "source": "x = 2"}]}
    app.save("beam calc", doc2)
    versions = app.history("beam calc")["versions"]
    assert len(versions) == 1 and versions[0]["cells"] == 1
    old = app.version("beam calc", versions[0]["stamp"])
    assert old["cells"][0]["source"] == "x = 1"
    assert json.loads(app._path("beam calc").read_text())["cells"][0]["source"] == "x = 2"
    with pytest.raises(ValueError):
        app.version("beam calc", "../../etc")


def test_image_upload_is_served_from_files(tmp_path):
    app = _app(tmp_path)
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
    r = app.upload("beam photo.PNG", png)
    assert r["image"] and r["file"] == "beam_photo.png"
    assert app.file_path("beam_photo.png").read_bytes().startswith(b"\x89PNG")
    assert app.file_path("../beam_photo.png") is None and app.file_path("missing.png") is None
    r = app.upload("readings.csv", base64.b64encode(b"a,b\n1,2\n").decode())
    assert not r["image"] and r["name"] == "readings"
    with pytest.raises(ValueError):
        app.upload("virus.exe", png)
