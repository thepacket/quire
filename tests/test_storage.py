"""Object-storage mirror: request signing, the S3 client against a fake bucket, and the mirror sync."""
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from xml.sax.saxutils import escape

import pytest

from quire.storage import Mirror, S3Client, StorageError, mirror_from_env, sign_request


def test_signature_matches_the_aws_test_vector():
    # "get-vanilla" from the AWS Signature Version 4 test suite
    auth = sign_request("GET", "example.amazonaws.com", "/", {}, {"x-amz-date": "20150830T123600Z"},
                        hashlib.sha256(b"").hexdigest(), "20150830T123600Z", "AKIDEXAMPLE",
                        "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "us-east-1", "service")
    assert auth == ("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
                    "SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31")


class FakeS3(BaseHTTPRequestHandler):
    objects: dict = {}
    requests: list = []

    def log_message(self, *a):
        pass

    def _key(self):
        u = urlparse(self.path)
        parts = u.path.split("/", 2)
        return (unquote(parts[2]) if len(parts) > 2 else ""), parse_qs(u.query)

    def _check_auth(self):
        auth = self.headers.get("Authorization", "")
        FakeS3.requests.append((self.command, self.path))
        assert auth.startswith("AWS4-HMAC-SHA256 Credential=TESTKEY/") and "x-amz-content-sha256" in auth

    def do_PUT(self):
        self._check_auth()
        key, _ = self._key()
        FakeS3.objects[key] = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

    def do_DELETE(self):
        self._check_auth()
        key, _ = self._key()
        FakeS3.objects.pop(key, None)
        self.send_response(204); self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        self._check_auth()
        key, q = self._key()
        if "list-type" in q:
            prefix = q.get("prefix", [""])[0]
            items = "".join(f"<Contents><Key>{escape(k)}</Key><Size>{len(v)}</Size></Contents>"
                            for k, v in sorted(FakeS3.objects.items()) if k.startswith(prefix))
            body = (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    f"<IsTruncated>false</IsTruncated>{items}</ListBucketResult>").encode()
            self.send_response(200)
        elif key in FakeS3.objects:
            body = FakeS3.objects[key]
            self.send_response(200)
        else:
            body = b"<Error><Code>NoSuchKey</Code></Error>"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


@pytest.fixture
def bucket():
    FakeS3.objects = {}
    FakeS3.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeS3)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = S3Client("quire-test", f"http://127.0.0.1:{server.server_address[1]}", "TESTKEY", "TESTSECRET")
    yield client
    server.shutdown()


def test_client_round_trip(bucket):
    assert bucket.get("missing") is None
    bucket.put("worksheets/beam calc.quire.json", b'{"a": 1}', "application/json")
    bucket.put("worksheets/.history/beam calc/20260101-000000.quire.json", b"{}")
    assert bucket.get("worksheets/beam calc.quire.json") == b'{"a": 1}'
    assert bucket.list("worksheets/") == [("worksheets/.history/beam calc/20260101-000000.quire.json", 2),
                                          ("worksheets/beam calc.quire.json", 8)]
    bucket.delete("worksheets/beam calc.quire.json")
    assert bucket.get("worksheets/beam calc.quire.json") is None
    assert any("beam%20calc" in path for _, path in FakeS3.requests)  # keys are URI-encoded in the request


def test_mirror_syncs_both_ways(bucket, tmp_path):
    remote_only = tmp_path / "other"
    other = Mirror(bucket, remote_only)
    (remote_only / "data").mkdir(parents=True)
    (remote_only / "old.quire.json").write_text("{}")
    (remote_only / "data" / "readings.csv").write_text("a,b\n")
    assert other.sync_up() == 2 and other.sync_up() == 0

    root = tmp_path / "ws"
    root.mkdir()
    (root / "local.quire.json").write_text('{"local": true}')
    m = Mirror(bucket, root)
    assert m.sync_down() == 2                                   # the two objects arrive
    assert (root / "data" / "readings.csv").read_text() == "a,b\n"
    assert m.sync_up() == 1                                     # and the local file goes up
    assert bucket.get("worksheets/local.quire.json") == b'{"local": true}'
    (root / "local.quire.json").write_text('{"local": 2}')
    m.put(root / "local.quire.json")
    assert bucket.get("worksheets/local.quire.json") == b'{"local": 2}'
    m.delete(root / "local.quire.json")
    assert bucket.get("worksheets/local.quire.json") is None
    assert m.sync_down() == 0                                   # nothing missing now
    FakeS3.objects["worksheets/../evil.json"] = b"x"            # a hostile key never leaves the folder
    assert m.sync_down() == 0 and not (tmp_path / "evil.json").exists()


def test_mirror_from_env(tmp_path):
    assert mirror_from_env(tmp_path, {}) is None
    with pytest.raises(StorageError):
        mirror_from_env(tmp_path, {"BUCKET_NAME": "b"})
    m = mirror_from_env(tmp_path, {"BUCKET_NAME": "b", "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s",
                                   "AWS_ENDPOINT_URL_S3": "https://fly.storage.tigris.dev", "AWS_REGION": "auto"})
    assert m.client.host == "fly.storage.tigris.dev" and m.client.region == "auto" and m.prefix == "worksheets/"
    assert m.key(tmp_path / "a" / "b.json") == "worksheets/a/b.json"


def test_desktop_launcher_uses_a_stub_window(tmp_path):
    from quire.desktop import main

    calls = []

    class Stub:
        @staticmethod
        def create_window(title, url, **kw):
            calls.append((title, url, kw))

        @staticmethod
        def start():
            calls.append("start")

    assert main(["--dir", str(tmp_path / "Quire")], webview=Stub) == 0
    assert calls[0][0] == "Quire" and calls[0][1].startswith("http://127.0.0.1:") and calls[1] == "start"
    assert (tmp_path / "Quire").is_dir()
