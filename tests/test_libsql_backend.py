"""Tests for the libsql / Turso backend.

These exercise the libsql client against a *local* libsql file (no network or
Turso account needed), which runs the exact same code path as a Turso embedded
replica. They assert the libsql backend returns the same results as the default
sqlite3 backend for both FTS5 text search and every vector mode.
"""

import os
import tempfile

import numpy as np
import pytest

from sqlitesearch import TextSearchIndex, VectorSearchIndex

# The libsql backend is optional; skip the whole module if it isn't installed.
pytest.importorskip("libsql")


DOCS = [
    {"id": 1, "question": "How do I install Python?", "answer": "Use the official installer.", "section": "setup"},
    {"id": 2, "question": "How do I reset my password?", "answer": "Click forgot password.", "section": "account"},
    {"id": 3, "question": "Where are the course videos?", "answer": "On the course page.", "section": "content"},
    {"id": 4, "question": "How to submit homework?", "answer": "Use the submission form.", "section": "homework"},
    {"id": 5, "question": "Python virtual environments?", "answer": "Use venv or uv.", "section": "setup"},
]


@pytest.fixture
def two_db_paths():
    paths = []
    for _ in range(2):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)  # let the backend create it fresh
        paths.append(path)
    yield paths
    for path in paths:
        for p in (path, path + "-wal", path + "-shm", path + "-info"):
            try:
                os.unlink(p)
            except OSError:
                pass


def test_text_search_libsql_matches_sqlite3(two_db_paths):
    sqlite_db, libsql_db = two_db_paths

    def build(db_path, backend):
        idx = TextSearchIndex(
            text_fields=["question", "answer", "section"],
            keyword_fields=["section"],
            db_path=db_path,
            backend=backend,
        )
        idx.fit(DOCS)
        return idx

    a = build(sqlite_db, "sqlite3")
    b = build(libsql_db, "libsql")

    ids_a = [r["id"] for r in a.search("install python", num_results=3)]
    ids_b = [r["id"] for r in b.search("install python", num_results=3)]
    assert ids_a == ids_b
    assert 1 in ids_b  # named row access works through the libsql adapter

    # keyword filtering works on the libsql backend
    filtered = b.search("python", filter_dict={"section": "setup"}, num_results=5)
    assert {r["section"] for r in filtered} == {"setup"}


@pytest.mark.parametrize("mode", ["lsh", "ivf", "hnsw"])
def test_vector_search_libsql_matches_sqlite3(two_db_paths, mode):
    sqlite_db, libsql_db = two_db_paths
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal(size=(len(DOCS), 32)).astype(np.float32)

    def build(db_path, backend):
        idx = VectorSearchIndex(
            keyword_fields=["section"],
            db_path=db_path,
            mode=mode,
            backend=backend,
        )
        idx.fit(vectors, DOCS)
        return idx

    a = build(sqlite_db, "sqlite3")
    b = build(libsql_db, "libsql")

    # querying with doc 3's own vector should rank doc 3 first on both backends
    ids_a = [r["id"] for r in a.search(vectors[2], num_results=3)]
    ids_b = [r["id"] for r in b.search(vectors[2], num_results=3)]
    assert ids_b[0] == 3
    assert ids_a == ids_b
