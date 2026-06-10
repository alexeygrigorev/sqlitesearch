"""Tests for the local in-process ``pyturso`` backend (``backend="turso"``).

pyturso (tursodatabase/turso) runs fully locally with no cloud. It has no FTS5,
so it supports vector search only; TextSearchIndex must reject it. The vector
results must match the stdlib sqlite3 backend.
"""

import os
import tempfile

import numpy as np
import pytest

from sqlitesearch import TextSearchIndex, VectorSearchIndex

pytest.importorskip("turso")


DOCS = [
    {"doc_id": i, "title": f"doc {i}", "section": "a" if i % 2 else "b"} for i in range(20)
]


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def test_text_index_rejects_turso_backend():
    with pytest.raises(ValueError, match="FTS5|turso"):
        TextSearchIndex(text_fields=["title"], db_path=_fresh_db(), backend="turso")


@pytest.mark.parametrize("mode", ["lsh", "ivf", "hnsw"])
def test_vector_turso_matches_sqlite3(mode):
    vectors = np.random.default_rng(3).standard_normal((len(DOCS), 24)).astype(np.float32)

    def build(backend):
        idx = VectorSearchIndex(
            keyword_fields=["section"], id_field="doc_id",
            db_path=_fresh_db(), mode=mode, seed=7, backend=backend,
        )
        idx.fit(vectors, DOCS)
        return idx

    a = build("sqlite3")
    b = build("turso")
    ids_a = [d["doc_id"] for d in a.search(vectors[5], num_results=5)]
    ids_b = [d["doc_id"] for d in b.search(vectors[5], num_results=5)]
    assert ids_b[0] == 5
    assert ids_a == ids_b


def test_vector_turso_cold_reload():
    vectors = np.random.default_rng(3).standard_normal((len(DOCS), 24)).astype(np.float32)
    path = _fresh_db()
    VectorSearchIndex(
        keyword_fields=["section"], id_field="doc_id", db_path=path, mode="lsh",
        seed=7, backend="turso",
    ).fit(vectors, DOCS)
    # Fresh instance reads from the same local file.
    reopened = VectorSearchIndex(
        keyword_fields=["section"], id_field="doc_id", db_path=path, mode="lsh",
        seed=7, backend="turso",
    )
    assert reopened.search(vectors[5], num_results=3)[0]["doc_id"] == 5
