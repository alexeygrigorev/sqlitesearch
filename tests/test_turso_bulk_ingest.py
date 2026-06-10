"""Regression tests for bulk ingest over the libsql/Turso embedded replica.

Over an embedded replica (``sync_url`` set), every SQL statement is forwarded
to the remote primary as a network round-trip. A naive per-row ``executemany``
turns an N-row ``fit()`` into O(N) round-trips, which is unusably slow over a
real network (issue #3). These tests pin the round-trip count low by driving a
local Turso emulator (no network, no Turso account) that counts pushes.

The emulator lives in ``dev/turso_emulator.py`` and runs as a subprocess (the
Rust libsql client holds the GIL while syncing, which would deadlock a
same-process server thread).
"""

import os
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("libsql")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev"))
from turso_emulator import EmulatorProcess  # noqa: E402

from sqlitesearch import TextSearchIndex, VectorSearchIndex  # noqa: E402

N = 200
DOCS = [
    {"id": i, "question": f"q{i} install python venv", "answer": f"ans {i}", "section": "setup"}
    for i in range(N)
]


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


@pytest.fixture
def emulator():
    # Distinct port per test to avoid collisions across the parametrized runs.
    import random

    emu = EmulatorProcess(port=8200 + random.randint(0, 300)).start()
    try:
        yield emu
    finally:
        emu.stop()


def test_text_ingest_round_trips_bounded(emulator, monkeypatch):
    # The emulator can't replicate frames back to the writing replica, so the
    # pre-write _is_empty() read would see no table; skip it (we're measuring
    # the write path, which is what issue #3 is about).
    monkeypatch.setattr(TextSearchIndex, "_is_empty", lambda self: True)
    idx = TextSearchIndex(
        text_fields=["question", "answer", "section"],
        keyword_fields=["section"],
        db_path=_fresh_db(),
        backend="libsql",
        sync_url=emulator.url,
        auth_token="x",
    )
    idx.fit(DOCS)
    round_trips = emulator.stats()["push_count"]
    # Per-row executemany would be ~4*N (≈800); batching keeps it tiny.
    assert round_trips < N // 2, f"too many round-trips: {round_trips} for {N} docs"


@pytest.mark.parametrize("mode", ["lsh", "ivf", "hnsw"])
def test_vector_ingest_round_trips_bounded(emulator, monkeypatch, mode):
    monkeypatch.setattr(VectorSearchIndex, "_is_empty", lambda self: True)
    vectors = np.random.default_rng(0).standard_normal((N, 32)).astype(np.float32)
    idx = VectorSearchIndex(
        keyword_fields=["section"],
        db_path=_fresh_db(),
        mode=mode,
        seed=1,
        backend="libsql",
        sync_url=emulator.url,
        auth_token="x",
    )
    idx.fit(vectors, DOCS)
    round_trips = emulator.stats()["push_count"]
    # Naive ingest (per-row docs + per-row index rows) is hundreds-to-thousands
    # of round-trips; batching keeps it well under the doc count.
    assert round_trips < N, f"too many round-trips for {mode}: {round_trips} for {N} docs"


def test_emulator_cold_reload_persists():
    """A fresh replica bootstraps the written data from the emulator's /export."""
    import libsql

    with EmulatorProcess(port=8190) as emu:
        c1 = libsql.connect(_fresh_db(), sync_url=emu.url, auth_token="x")
        c1.sync()
        c1.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")
        c1.executemany("INSERT INTO t(v) VALUES (?)", [(f"r{i}",) for i in range(10)])
        c1.commit()
        c1.sync()

        c2 = libsql.connect(_fresh_db(), sync_url=emu.url, auth_token="x")
        c2.sync()
        count = c2.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert count == 10
