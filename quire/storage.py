"""Object-storage mirror for the worksheet folder (any S3-compatible bucket, e.g. Tigris on Fly.io).

The local folder stays the working copy: modules and the evaluation worker read
files from disk. Every write is copied to the bucket, and at startup the bucket is
synced down, so a fresh machine with an empty disk rebuilds the folder and the
deployment no longer depends on a volume pinned to one host. Only the standard
library is used (AWS Signature Version 4 over urllib).

Configuration comes from the environment, matching what ``fly storage create``
sets: BUCKET_NAME (or QUIRE_S3_BUCKET), AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_ENDPOINT_URL_S3 (or AWS_ENDPOINT_URL), AWS_REGION.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


class StorageError(Exception):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _q(value) -> str:
    return urllib.parse.quote(str(value), safe="-_.~")


def sign_request(method: str, host: str, path: str, query: dict, headers: dict, payload_hash: str, amz_date: str,
                 access_key: str, secret_key: str, region: str, service: str = "s3") -> str:
    """The Authorization header value for a request (AWS Signature Version 4)."""
    canonical_query = "&".join(f"{_q(k)}={_q(v)}" for k, v in sorted(query.items()))
    hdrs = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    hdrs["host"] = host
    signed = ";".join(sorted(hdrs))
    canonical_headers = "".join(f"{k}:{hdrs[k]}\n" for k in sorted(hdrs))
    canonical = "\n".join([method, path, canonical_query, canonical_headers, signed, payload_hash])
    date = amz_date[:8]
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, _sha256(canonical.encode())])
    k = _hmac(("AWS4" + secret_key).encode(), date)
    k = _hmac(k, region)
    k = _hmac(k, service)
    k = _hmac(k, "aws4_request")
    signature = hmac.new(k, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed}, Signature={signature}"


class S3Client:
    """A minimal S3 client: get, put, delete, list (path-style addressing)."""

    def __init__(self, bucket: str, endpoint: str, access_key: str, secret_key: str, region: str = "auto",
                 timeout: float = 30.0):
        u = urllib.parse.urlparse(endpoint if "://" in endpoint else "https://" + endpoint)
        self.scheme, self.host = u.scheme, u.netloc
        self.bucket, self.access_key, self.secret_key, self.region, self.timeout = bucket, access_key, secret_key, region, timeout

    def _request(self, method: str, key: str = "", query: dict | None = None, data: bytes = b"",
                 content_type: str | None = None) -> tuple[int, bytes]:
        query = query or {}
        path = "/" + _q(self.bucket) + ("/" + urllib.parse.quote(key, safe="/-_.~") if key else "")
        amz_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload_hash = _sha256(data)
        headers = {"x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
        if content_type:
            headers["content-type"] = content_type
        auth = sign_request(method, self.host, path, query, headers, payload_hash, amz_date, self.access_key,
                            self.secret_key, self.region)
        url = f"{self.scheme}://{self.host}{path}"
        if query:
            url += "?" + "&".join(f"{_q(k)}={_q(v)}" for k, v in sorted(query.items()))
        req = urllib.request.Request(url, data=data if method == "PUT" else None, method=method,
                                     headers={**headers, "Authorization": auth})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, b""
            raise StorageError(f"{method} {key or '/'}: HTTP {exc.code} {exc.read()[:200]!r}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise StorageError(f"{method} {key or '/'}: {exc}") from None

    def get(self, key: str) -> bytes | None:
        status, body = self._request("GET", key)
        return None if status == 404 else body

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._request("PUT", key, data=data, content_type=content_type)

    def delete(self, key: str) -> None:
        self._request("DELETE", key)

    def list(self, prefix: str = "") -> list[tuple[str, int]]:
        """(key, size) for every object under prefix."""
        out, token = [], None
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            status, body = self._request("GET", query=query)
            if status == 404:
                raise StorageError(f"bucket '{self.bucket}' not found")
            root = ET.fromstring(body)
            for el in root.iter():
                if el.tag.endswith("Contents"):
                    fields = {c.tag.split("}")[-1]: c.text for c in el}
                    out.append((fields.get("Key", ""), int(fields.get("Size") or 0)))
            truncated = any(el.tag.endswith("IsTruncated") and (el.text or "").lower() == "true" for el in root.iter())
            token = next((el.text for el in root.iter() if el.tag.endswith("NextContinuationToken")), None)
            if not truncated or not token:
                return out


class Mirror:
    """Keeps a local folder and a bucket prefix in step: writes go up, startup syncs down."""

    def __init__(self, client: S3Client, root: Path, prefix: str = "worksheets/"):
        self.client, self.root, self.prefix = client, Path(root), prefix.rstrip("/") + "/" if prefix else ""

    def key(self, path: Path) -> str:
        return self.prefix + Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def put(self, path: Path) -> None:
        path = Path(path)
        if path.is_file():
            self.client.put(self.key(path), path.read_bytes())

    def delete(self, path: Path) -> None:
        self.client.delete(self.key(path))

    def _local(self, key: str) -> Path | None:
        rel = key[len(self.prefix):]
        if not rel or rel.endswith("/") or any(part in ("..", "") for part in rel.split("/")):
            return None
        return self.root / rel

    def sync_down(self) -> int:
        """Download every object that is missing locally or differs in size. Returns the count."""
        n = 0
        for key, size in self.client.list(self.prefix):
            local = self._local(key)
            if local is None or (local.is_file() and local.stat().st_size == size):
                continue
            data = self.client.get(key)
            if data is None:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            n += 1
        return n

    def sync_up(self) -> int:
        """Upload local files the bucket does not have (a folder that predates the bucket). Returns the count."""
        if not self.root.is_dir():
            return 0
        remote = {key for key, _ in self.client.list(self.prefix)}
        n = 0
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and self.key(p) not in remote:
                self.put(p)
                n += 1
        return n


def mirror_from_env(root: Path, env: dict | None = None) -> Mirror | None:
    """A Mirror from the environment (see the module docstring), or None when no bucket is configured."""
    env = os.environ if env is None else env
    bucket = env.get("QUIRE_S3_BUCKET") or env.get("BUCKET_NAME")
    if not bucket:
        return None
    key, secret = env.get("AWS_ACCESS_KEY_ID"), env.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise StorageError("Object storage: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
    region = env.get("AWS_REGION") or "auto"
    endpoint = env.get("AWS_ENDPOINT_URL_S3") or env.get("AWS_ENDPOINT_URL") or f"https://s3.{region}.amazonaws.com"
    return Mirror(S3Client(bucket, endpoint, key, secret, region), root, env.get("QUIRE_S3_PREFIX", "worksheets/"))
