"""Tests for the pure-numpy int8-cache LSH mode (``mode="lsh_int8"``).

``lsh_int8`` keeps an int8 quantized shortlist cache instead of the full
float32 vector cache, but preserves exact results by reranking the shortlist's
SQLite BLOBs, so its search output matches plain ``mode="lsh"``.
"""

import os
import tempfile

import numpy as np
import pytest

from sqlitesearch import VectorSearchIndex


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    yield path
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


def test_lsh_int8_preserves_lsh_search_semantics(temp_db):
    int8_db = temp_db
    py_db = temp_db + ".py"
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal(size=(200, 64)).astype(np.float32)
    payload = [
        {"id": i, "category": "even" if i % 2 == 0 else "odd"}
        for i in range(len(vectors))
    ]

    py_index = VectorSearchIndex(
        mode="lsh",
        keyword_fields=["category"],
        db_path=py_db,
        seed=7,
    )
    int8_index = VectorSearchIndex(
        mode="lsh_int8",
        keyword_fields=["category"],
        db_path=int8_db,
        seed=7,
    )
    py_index.fit(vectors, payload)
    int8_index.fit(vectors, payload)

    # The int8 cache replaces the float32 vector cache.
    assert int8_index._cached_vectors is None

    for query in (vectors[0], vectors[17], vectors[42]):
        int8_results = int8_index.search(query, num_results=10)
        py_results = py_index.search(query, num_results=10)
        assert int8_results
        assert py_results
        # Top result must match the exact-float32 LSH result: the int8 cache
        # only narrows the shortlist, then reranks it exactly.
        assert int8_results[0] == py_results[0]
        assert int8_index.search(
            query,
            filter_dict={"category": "even"},
            num_results=10,
        ) == py_index.search(
            query,
            filter_dict={"category": "even"},
            num_results=10,
        )

    py_index.close()
    int8_index.close()
    try:
        os.unlink(py_db)
    except OSError:
        pass


def test_lsh_int8_search_without_index_vector_cache(temp_db):
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal(size=(80, 32)).astype(np.float32)
    payload = [{"id": i} for i in range(len(vectors))]

    index = VectorSearchIndex(mode="lsh_int8", db_path=temp_db, seed=1)
    index.fit(vectors, payload)

    assert index._cached_vectors is None

    results = index.search(vectors[0], num_results=5)

    assert results
    assert results[0]["id"] == 0
    assert index._cached_vectors is None
    index.close()


def test_lsh_int8_cold_search_does_not_load_index_vector_cache(temp_db):
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal(size=(50, 32)).astype(np.float32)
    payload = [{"id": i} for i in range(len(vectors))]
    index = VectorSearchIndex(mode="lsh_int8", db_path=temp_db, seed=1)
    index.fit(vectors, payload)
    index.close()

    reloaded = VectorSearchIndex(mode="lsh_int8", db_path=temp_db, seed=1)
    assert reloaded._cached_vectors is None

    results = reloaded.search(vectors[0], num_results=5)

    assert results
    assert results[0]["id"] == 0
    # The int8 cache is rebuilt from the docs table on first search; the float32
    # cache must never be populated.
    assert reloaded._cached_vectors is None
    reloaded.close()


def test_lsh_int8_add_appends_to_cache(temp_db):
    """add() must extend the int8 cache, not rebuild it from the new chunk only
    (regression guard for the cache append lifecycle)."""
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal(size=(40, 16)).astype(np.float32)
    payload = [{"id": i} for i in range(40)]

    index = VectorSearchIndex(mode="lsh_int8", db_path=temp_db, seed=1)
    index.fit(vectors[:20], payload[:20])

    strategy = index._strategy
    assert strategy._q_vectors is not None
    fitted_doc_ids = list(strategy._q_doc_ids)
    assert len(fitted_doc_ids) == 20

    index.add(vectors[20], payload[20])

    # 20 fitted + 1 added = 21 quantized rows; the fitted rows must still be
    # present (the old add() path rebuilt the cache from the new chunk only).
    assert len(strategy._q_doc_ids) == 21
    assert set(fitted_doc_ids).issubset(set(strategy._q_doc_ids))
    assert len(strategy._q_id_to_idx) == 21

    # A query for an early vector must still find it (cache wasn't truncated).
    results = index.search(vectors[0], num_results=5)
    assert results
    assert results[0]["id"] == 0
    index.close()
