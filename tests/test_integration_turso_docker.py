"""Real-Docker integration test against a libsql-server (sqld) container.

This complements the in-process emulator tests (``test_turso_bulk_ingest.py``,
which drive ``dev/turso_emulator.py``) by exercising the library against an
actual ``ghcr.io/tursodatabase/libsql-server:latest`` server in Docker: real
network, real Hrana protocol, real durable SQLite on the server side.

Run it explicitly (it is excluded from the default suite via ``addopts``)::

    uv run pytest -m integration
    uv run pytest -m integration -s   # to see the printed timings/round-trips

It skips cleanly (never errors) when the docker CLI/daemon is missing or the
image cannot be pulled, so machines without Docker stay green.

What "real round trip" means here, and the one honest caveat
-------------------------------------------------------------
The library's transparent-URL feature backs a remote URL with a libsql
*embedded replica* (local file + ``sync_url``). The libsql client bootstraps
that replica via Turso Cloud's sync API: ``GET /info`` then ``GET /export`` then
WAL frames. Stock self-hosted ``sqld`` does **not** serve those endpoints -- it
speaks only Hrana (``POST /v3/pipeline``) -- so on this image ``GET /info`` and
``GET /export`` return 404 and the embedded-replica ``sync()`` fails on its
first call. (This is exactly why the repo ships its own emulator for the
replica-bootstrap tests.) ``test_embedded_replica_export_unsupported_on_sqld``
pins that behaviour so a future ``sqld`` that adds ``/export`` flips this test
to a failure and tells us to enable the real embedded-replica path.

So the faithful round trip we *can* run against real ``sqld`` uses the same
libsql client the library wraps, in pure-remote (Hrana) mode: every statement
the index issues -- schema creation, batched ``bulk_insert``, vector-index rows,
search probes -- is forwarded to and answered by the container. We prove the
writes are durable on the server by opening a SECOND, fresh client (no shared
local state) and reading back all N rows plus correct nearest neighbours --
which is the real-server equivalent of "a fresh replica bootstraps all N".
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("libsql")

import libsql  # noqa: E402

from sqlitesearch import TextSearchIndex, VectorSearchIndex  # noqa: E402
from sqlitesearch.connection import _ConnectionWrapper  # noqa: E402

pytestmark = pytest.mark.integration

N = 120
DIM = 32

DOCS = [
    {
        "id": i,
        "question": f"q{i} how do I install python and a venv",
        "answer": f"answer {i} use uv or venv",
        "section": "setup" if i % 2 == 0 else "account",
    }
    for i in range(N)
]


def _remote_connect_factory(url: str):
    """Build a drop-in ``connect`` that routes every statement to real ``sqld``.

    The library's ``connect()`` always wraps a remote URL as an embedded replica
    (``sync_url``), which stock ``sqld`` can't bootstrap (see module docstring).
    Here we instead open a pure-remote Hrana connection to the same container and
    wrap it with the library's own ``_ConnectionWrapper`` so ``row["col"]``
    access keeps working. Tests monkeypatch the index modules' ``connect`` with
    this so the FULL public API (fit/count/search) runs against the real server.
    """

    def connect(db_path, *, backend="sqlite3", auth_token=None, replica_path=None):
        return _ConnectionWrapper(libsql.connect(url))

    return connect


def test_embedded_replica_export_unsupported_on_sqld(sqld_container):
    """Document + pin: stock sqld lacks the /export sync API embedded replicas need.

    If this ever starts succeeding, ``sqld`` has gained the replica-bootstrap
    endpoints and the real embedded-replica path below can be enabled.
    """
    url = sqld_container.url
    # The endpoints the libsql embedded-replica client probes during sync().
    import urllib.error
    import urllib.request

    for path in ("/info", "/export/0"):
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(url + path, timeout=5).read()
        assert ei.value.code == 404, f"{path} unexpectedly returned {ei.value.code}"

    # And the high-level symptom: the transparent embedded-replica setup fails on
    # its first sync() against this server.
    import os
    import tempfile

    fd, replica = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(replica)
    with pytest.raises(Exception) as ex:
        libsql.connect(replica, sync_url=url, auth_token="x").sync()
    assert "export" in str(ex.value).lower() or "404" in str(ex.value)


def test_vector_round_trip_against_real_sqld(sqld_container, monkeypatch):
    """Full vector round trip through the public API against the real container."""
    import sqlitesearch.vector.index as vmod

    url = sqld_container.url
    monkeypatch.setattr(vmod, "connect", _remote_connect_factory(url))

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)

    t0 = time.time()
    idx = VectorSearchIndex(keyword_fields=["section"], db_path=url, mode="lsh", seed=1)
    idx.fit(vectors, DOCS)
    ingest_s = time.time() - t0

    # The writes reached the server: a fresh, independent remote client (no shared
    # local file) counts all N rows back out of the real database.
    fresh = libsql.connect(url)
    server_count = fresh.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    assert server_count == N, f"server has {server_count} rows, expected {N}"

    # A second, fresh index built against the same server bootstraps all N and
    # returns the known nearest neighbour for several query vectors.
    t1 = time.time()
    idx2 = VectorSearchIndex(keyword_fields=["section"], db_path=url, mode="lsh", seed=1)
    bootstrap_count = idx2.count()
    assert bootstrap_count == N, f"fresh index sees {bootstrap_count}, expected {N}"
    for probe in (0, 7, 42, N - 1):
        top = [r["id"] for r in idx2.search(vectors[probe], num_results=3)]
        assert top[0] == probe, f"probe {probe}: got {top}, expected {probe} first"
    bootstrap_s = time.time() - t1

    print(
        f"\n[real sqld] vector: ingest {N} vecs in {ingest_s:.2f}s; "
        f"fresh-client count={bootstrap_count} (server={server_count}); "
        f"fresh index bootstrap+search {bootstrap_s:.2f}s"
    )


def test_text_round_trip_against_real_sqld(sqld_container, monkeypatch):
    """Briefly exercise TextSearchIndex (FTS5) against the real container too."""
    import sqlitesearch.text.fts as tmod

    url = sqld_container.url
    monkeypatch.setattr(tmod, "connect", _remote_connect_factory(url))

    t0 = time.time()
    idx = TextSearchIndex(
        text_fields=["question", "answer", "section"],
        keyword_fields=["section"],
        db_path=url,
    )
    idx.fit(DOCS)
    ingest_s = time.time() - t0

    assert idx.count() == N

    # Fresh index against the same server sees everything and search works.
    idx2 = TextSearchIndex(
        text_fields=["question", "answer", "section"],
        keyword_fields=["section"],
        db_path=url,
    )
    assert idx2.count() == N
    results = idx2.search("install python venv", num_results=5)
    assert results, "expected text-search hits from the real server"
    filtered = idx2.search("python", filter_dict={"section": "setup"}, num_results=10)
    assert {r["section"] for r in filtered} == {"setup"}

    print(f"\n[real sqld] text: ingest {N} docs in {ingest_s:.2f}s; search OK")
