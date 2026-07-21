from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", normalized).strip()


def canonicalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("source URL must be absolute HTTP(S)")
    scheme = parsed.scheme.lower()
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    port = parsed.port
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, val)
        for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMETERS and not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, host, path, query, ""))


def content_fingerprint(value: str) -> str:
    return hashlib.sha256(normalize_text(value).casefold().encode("utf-8")).hexdigest()
