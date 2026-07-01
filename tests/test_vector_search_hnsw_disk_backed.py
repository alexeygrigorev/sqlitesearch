"""Tests for the HNSW ``disk_backed`` profile and its memmap lifecycle.

``disk_backed=True`` drops the index's float32 vector cache and serves the
graph-walk nav vectors from a file-backed ``np.memmap`` instead of a resident
array, so steady-state search RSS stays flat as the corpus grows. Exact results
are unchanged (``index.py._rerank`` reranks the small candidate set from SQLite
BLOBs). These tests pin the two profiles apart and guard the memmap tempfile
cleanup (np.memmap flushes on collection but never unlinks its own file).
"""

import gc
import glob
import os
import tempfile

import numpy as np
import pytest

from sqlitesearch import VectorSearchIndex


def _nvec_tempfiles():
    return glob.glob(os.path.join(tempfile.gettempdir(), "sqlitesearch-hnsw-*.nvec"))


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


def _make_corpus(n=300, dim=32, seed=7):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    payload = [{"id": i} for i in range(n)]
    return vectors, payload


def test_default_keeps_cache_and_never_opens_memmap(temp_db):
    vectors, payload = _make_corpus()
    idx = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1)
    idx.fit(vectors, payload)

    assert idx._strategy._disk_backed is False
    assert idx._strategy.use_sqlite_rerank is False
    assert idx._cached_vectors is not None

    idx.search(vectors[0], num_results=5)
    assert idx._strategy._vec_mmap_path is None
    idx.close()


def test_disk_backed_drops_cache_and_opens_memmap_on_search(temp_db):
    vectors, payload = _make_corpus()
    idx = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1, disk_backed=True)
    idx.fit(vectors, payload)

    assert idx._strategy._disk_backed is True
    assert idx._strategy.use_sqlite_rerank is True
    assert idx._cached_vectors is None

    # The memmap is opened lazily on the first search, not at fit time.
    assert idx._strategy._vec_mmap_path is None
    idx.search(vectors[0], num_results=5)
    assert idx._strategy._vec_mmap_path is not None
    idx.close()


def test_disk_backed_recall_matches_default(temp_db):
    """Same seed + same vectors -> identical graph; only the nav-vector backing
    store differs, so the two profiles must return byte-identical result sets."""
    vectors, payload = _make_corpus(n=500, dim=32)

    default_idx = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1)
    default_idx.fit(vectors, payload)

    disk_db = temp_db + ".disk"
    disk_idx = VectorSearchIndex(mode="hnsw", db_path=disk_db, seed=1, disk_backed=True)
    disk_idx.fit(vectors, payload)

    k = 10
    for q in range(0, 500, 37):
        default_ids = [r["id"] for r in default_idx.search(vectors[q], num_results=k)]
        disk_ids = [r["id"] for r in disk_idx.search(vectors[q], num_results=k)]
        assert default_ids == disk_ids, f"top-{k} differs at q={q}"

    default_idx.close()
    disk_idx.close()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(disk_db + suffix)
        except OSError:
            pass


def test_memmap_tempfile_cleaned_after_close(temp_db):
    vectors, payload = _make_corpus()
    idx = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1, disk_backed=True)
    idx.fit(vectors, payload)
    idx.search(vectors[0], num_results=5)  # open the memmap

    path = idx._strategy._vec_mmap_path
    assert path and os.path.exists(path)
    idx.close()
    assert not os.path.exists(path), "close() must unlink the nav-vector tempfile"
    assert _nvec_tempfiles() == []


def test_memmap_tempfile_cleaned_after_gc_without_close(temp_db):
    """An index abandoned without close() must still not leak its tempfile:
    ``__del__`` best-effort unlinks it (np.memmap itself never does)."""
    vectors, payload = _make_corpus()
    idx = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1, disk_backed=True)
    idx.fit(vectors, payload)
    idx.search(vectors[0], num_results=5)

    path = idx._strategy._vec_mmap_path
    assert path and os.path.exists(path)
    del idx
    gc.collect()
    assert not os.path.exists(path), "__del__ must unlink the nav-vector tempfile"


def test_disk_backed_cold_reload_then_add(temp_db):
    """The cold-reopen + add() fix must also hold for the disk_backed profile,
    where nav vectors are not held by the index cache and must be rebuilt from
    the docs table before appending."""
    vectors, payload = _make_corpus(n=300)
    idx = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1, disk_backed=True)
    idx.fit(vectors, payload)
    idx.close()

    idx2 = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1, disk_backed=True)
    idx2.add(vectors[0], {"id": 300})
    assert idx2._strategy._n_nodes == 301
    early = {r["id"] for r in idx2.search(vectors[5], num_results=5)}
    assert 5 in early, f"early doc lost after disk_backed reopen+add: {early}"
    idx2.close()
    assert _nvec_tempfiles() == []
