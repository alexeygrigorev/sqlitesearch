"""Tests for sharing one SQLite file between a text and a vector index (#2).

The two index types use the same `docs` table. With an `id_field` they
deduplicate by that id (one row per document, vector_hash filled by the vector
index, FTS built by the text index), enabling hybrid search over a single file.
"""

import os
import tempfile

import numpy as np
import pytest

from sqlitesearch import TextSearchIndex, VectorSearchIndex

DOCS = [
    {"doc_id": i, "question": f"install python {i}", "answer": f"use venv {i}", "section": "setup"}
    for i in range(8)
]


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    yield path
    for p in (path, path + "-wal", path + "-shm"):
        try:
            os.unlink(p)
        except OSError:
            pass


def _vectors():
    return np.random.default_rng(0).standard_normal((len(DOCS), 16)).astype(np.float32)


def _build_text(db_path):
    return TextSearchIndex(
        text_fields=["question", "answer"], keyword_fields=["section"],
        id_field="doc_id", db_path=db_path,
    )


def _build_vector(db_path):
    return VectorSearchIndex(
        keyword_fields=["section"], id_field="doc_id", db_path=db_path, mode="lsh", seed=1,
    )


@pytest.mark.parametrize("vector_first", [True, False])
def test_shared_file_dedups_and_searches(db_path, vector_first):
    vectors = _vectors()
    if vector_first:
        _build_vector(db_path).fit(vectors, DOCS)
        _build_text(db_path).fit(DOCS)
    else:
        _build_text(db_path).fit(DOCS)
        _build_vector(db_path).fit(vectors, DOCS)

    # Documents are stored once, not duplicated across the two indexes.
    import sqlite3

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == len(DOCS)
    assert conn.execute(
        "SELECT COUNT(*) FROM docs WHERE vector_hash IS NOT NULL"
    ).fetchone()[0] == len(DOCS)
    conn.close()

    # Both modalities work from a cold reopen of the shared file.
    vidx = _build_vector(db_path)
    tidx = _build_text(db_path)
    assert vidx.search(vectors[3], num_results=3)  # vector search returns hits
    text_hits = tidx.search("install python", num_results=3)
    assert text_hits


def test_cold_load_with_text_only_rows_does_not_crash(db_path):
    """Regression for the original #2 crash: a vector cold-load over a file that
    also has text-only rows (NULL vector_hash) must not dereference NULL."""
    vectors = _vectors()
    # Vector index covers only the first 4 docs; text covers all 8 -> docs 4..7
    # exist with NULL vector_hash.
    _build_vector(db_path).fit(vectors[:4], DOCS[:4])
    _build_text(db_path).fit(DOCS)

    # Cold reopen + search must not raise (was: TypeError on np.frombuffer(None)).
    vidx = _build_vector(db_path)
    results = vidx.search(vectors[1], num_results=3)
    assert isinstance(results, list)
