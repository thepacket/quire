"""Read-only share links, the storage mirror wired into the app, and the server's public/private routes."""
import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from quire.server import App, make_handler

ROOT = Path(__file__).resolve().parent.parent


class FakeMirror:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
        self.client = type("C", (), {"bucket": "b"})()
        self.prefix = "worksheets/"

    def sync_down(self):
        return 0

    def sync_up(self):
        return 0

    def put(self, path):
        if self.fail:
            raise OSError("bucket down")
        self.calls.append(("put", Path(path).name))

    def delete(self, path):
        self.calls.append(("delete", Path(path).name))


def _app(tmp_path, mirror=None) -> App:
    app = App.__new__(App)
    app.worksheets = tmp_path / "ws"
    app.examples = ROOT / "examples"
    app.mirror, app.mirror_error = mirror, None
    app.evaluate = lambda cells: {"results": cells}  # the file side only: hand back what would be evaluated
    return app


DOC = {"quire": 1, "title": "Beam", "cells": [
    {"id": "s", "type": "math", "source": "a = slider(1, 0, 5, 0.5)\nb = 2 a"},
    {"id": "p", "type": "plot", "kind": "function", "exprs": "a x", "xmin": "0", "xmax": "3"},
    {"id": "t", "type": "text", "source": "hello {{b}}"}]}


def test_share_links(tmp_path):
    app = _app(tmp_path)
    with pytest.raises(FileNotFoundError):
        app.share("beam")                                         # nothing saved under that name yet
    app.save("beam", DOC)
    r = app.share("beam")
    token = r["token"]
    assert r["url"] == f"/s/{token}" and len(token) >= 12
    assert app.share("beam")["token"] == token                    # sharing again keeps the link
    assert app.share_token("beam") == token and app.share_token("other") is None
    assert app.shared_doc(token)["title"] == "Beam"
    assert app.shared_doc("nope") is None and app.shared_doc("../..") is None
    app.unshare("beam")
    assert app.shared_doc(token) is None and app.share_token("beam") is None


def test_shared_evaluation_only_moves_sliders_and_ranges(tmp_path):
    app = _app(tmp_path)
    app.save("beam", DOC)
    token = app.share("beam")["token"]
    cells = app.shared_evaluate(token, {
        "sliders": {"s": {"0": 3.5, "1": 9, "9": 1}, "p": {"0": 2}},
        "plots": {"p": {"xmin": "-1.5", "xmax": "x^2", "ymin": "", "kind": "surface", "exprs": "rm -rf"}},
        "cells": [{"type": "math", "source": "evil"}]})["results"]
    assert cells[0]["source"] == "a = slider(3.5, 0, 5, 0.5)\nb = 2 a"      # only the slider line changed
    assert cells[1]["xmin"] == "-1.5" and cells[1]["xmax"] == "3"            # a number yes, an expression no
    assert cells[1]["kind"] == "function" and cells[1]["exprs"] == "a x" and cells[1]["ymin"] == ""
    stored = json.loads(app._path("beam").read_text())
    assert len(cells) == 3 and stored["saved_at"] and stored == dict(DOC, saved_at=stored["saved_at"])  # the saved copy is untouched
    with pytest.raises(FileNotFoundError):
        app.shared_evaluate("nope", {})


def test_mirror_receives_every_write(tmp_path):
    m = FakeMirror()
    app = _app(tmp_path, m)
    app.save("beam", DOC)
    app.save("beam", dict(DOC, title="Beam 2"))
    app.share("beam")
    app.upload("photo.png", base64.b64encode(b"\x89PNG").decode())
    kinds = [c for c in m.calls]
    assert ("put", "beam.quire.json") in kinds and ("put", ".shares.json") in kinds and ("put", "photo.png") in kinds
    assert any(k == "put" and n.endswith(".quire.json") and n[0].isdigit() for k, n in kinds)  # the history version
    failing = _app(tmp_path / "2", FakeMirror(fail=True))
    r = failing.save("beam", DOC)
    assert "object storage" in r["warning"] and failing._path("beam").is_file()   # the local save still happened


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("srv")
    app = App(tmp / "ws", [ROOT / "modules"], ROOT / "examples", password="pw")
    app.save("beam", DOC)
    app.upload("photo.png", base64.b64encode(b"\x89PNG").decode())
    token = app.share("beam")["token"]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", token
    srv.shutdown()
    app.worker.proc.kill()


def _get(url, auth=None, data=None):
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(f"x:{auth}".encode()).decode()
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_routes_public_and_private(server):
    base, token = server
    assert _get(base + "/health")[0] == 200
    assert _get(base + "/app.js")[0] == 200 and _get(base + "/vendor/plotly.min.js")[0] == 200
    assert _get(base + "/api/catalog")[0] == 401 and _get(base + "/api/catalog", auth="pw")[0] == 200
    assert _get(base + "/api/eval", data={"cells": []})[0] == 401
    assert _get(base + "/files/photo.png")[0] == 401
    assert _get(base + "/files/photo.png?share=" + token)[0] == 200
    assert _get(base + "/files/photo.png?share=bogus")[0] == 401
    status, body = _get(base + "/s/" + token)
    assert status == 200 and b"<title>Quire</title>" in body
    status, body = _get(base + "/api/shared/" + token)
    assert status == 200 and json.loads(body)["doc"]["title"] == "Beam"
    assert _get(base + "/api/shared/bogus")[0] == 404
    status, body = _get(base + f"/api/shared/{token}/eval", data={"sliders": {"s": {"0": 4}}})
    results = json.loads(body)["results"]
    assert status == 200 and results[0]["outputs"][1]["plain"] == "8"       # b = 2 a with the moved slider
    assert results[2]["values"][0]["plain"] == "8"
    assert _get(base + "/api/shared/bogus/eval", data={})[0] == 404
