"""
Tests for HNSW vector search mode.
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
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class TestHNSWBasics:
    def test_fit_and_search(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 64)).astype(np.float32)
        payload = [{"id": i} for i in range(100)]

        index = VectorSearchIndex(mode="hnsw", m=16, ef_construction=100, ef_search=50, db_path=temp_db)
        index.fit(vectors, payload)

        results = index.search(vectors[0], num_results=10)
        assert len(results) > 0
        assert results[0]["id"] == 0

    def test_num_results(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 64)).astype(np.float32)
        payload = [{"id": i} for i in range(100)]

        index = VectorSearchIndex(mode="hnsw", db_path=temp_db)
        index.fit(vectors, payload)

        for n in [1, 5, 10]:
            results = index.search(vectors[0], num_results=n)
            assert len(results) <= n

    def test_empty_search(self, temp_db):
        index = VectorSearchIndex(mode="hnsw", db_path=temp_db)
        query = np.random.randn(64).astype(np.float32)
        results = index.search(query)
        assert len(results) == 0

    def test_small_dataset(self, temp_db):
        """Test with very small dataset (fewer nodes than m)."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((5, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(5)]

        index = VectorSearchIndex(mode="hnsw", m=16, db_path=temp_db)
        index.fit(vectors, payload)

        results = index.search(vectors[0], num_results=5)
        assert len(results) > 0
        assert results[0]["id"] == 0


class TestHNSWRecall:
    def test_recall_at_1000(self, temp_db):
        """HNSW should achieve high recall@10 on 1000 random vectors."""
        rng = np.random.default_rng(42)
        n, dim = 1000, 64
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i} for i in range(n)]

        index = VectorSearchIndex(
            mode="hnsw", m=16, ef_construction=200, ef_search=100, db_path=temp_db
        )
        index.fit(vectors, payload)

        # Compute ground truth
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = vectors / norms

        hits = 0
        total = 0
        k = 10
        n_queries = 50
        query_indices = rng.choice(n, n_queries, replace=False)

        for qi in query_indices:
            query = vectors[qi]
            q_norm = query / (np.linalg.norm(query) + 1e-10)
            sims = normed @ q_norm
            true_top_k = set(np.argsort(sims)[-k:][::-1])

            results = index.search(query, num_results=k)
            result_ids = {r["id"] for r in results}

            hits += len(true_top_k & result_ids)
            total += k

        recall = hits / total
        assert recall >= 0.5, f"HNSW recall@{k} = {recall:.3f}, expected >= 0.5"


class TestHNSWFiltering:
    def test_keyword_filter(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [
            {"id": i, "category": "a" if i % 2 == 0 else "b"}
            for i in range(20)
        ]

        index = VectorSearchIndex(
            mode="hnsw", keyword_fields=["category"], db_path=temp_db
        )
        index.fit(vectors, payload)

        results = index.search(vectors[0], filter_dict={"category": "a"})
        assert all(r["category"] == "a" for r in results)

    def test_numeric_filter(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [{"id": i, "price": float(i * 10)} for i in range(20)]

        index = VectorSearchIndex(
            mode="hnsw", numeric_fields=["price"], db_path=temp_db
        )
        index.fit(vectors, payload)

        results = index.search(vectors[0], filter_dict={"price": [(">=", 100)]})
        assert all(r["price"] >= 100 for r in results)


class TestHNSWPersistence:
    def test_persist_and_reload(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(50)]

        index1 = VectorSearchIndex(mode="hnsw", db_path=temp_db)
        index1.fit(vectors, payload)
        index1.close()

        index2 = VectorSearchIndex(mode="hnsw", db_path=temp_db)
        results = index2.search(vectors[0], num_results=5)
        assert len(results) > 0
        assert results[0]["id"] == 0
        index2.close()


class TestHNSWAdd:
    def test_add_after_fit(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(20)]

        index = VectorSearchIndex(mode="hnsw", db_path=temp_db)
        index.fit(vectors, payload)

        new_vec = rng.standard_normal(32).astype(np.float32)
        index.add(new_vec, {"id": 20})

        results = index.search(new_vec, num_results=5)
        assert any(r["id"] == 20 for r in results)

    def test_add_without_fit(self, temp_db):
        rng = np.random.default_rng(42)
        index = VectorSearchIndex(mode="hnsw", db_path=temp_db)

        vec = rng.standard_normal(32).astype(np.float32)
        index.add(vec, {"id": 1})

        results = index.search(vec)
        assert len(results) == 1
        assert results[0]["id"] == 1


class TestHNSWClear:
    def test_clear(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(20)]

        index = VectorSearchIndex(mode="hnsw", db_path=temp_db)
        index.fit(vectors, payload)
        assert len(index.search(vectors[0])) > 0

        index.clear()
        assert len(index.search(vectors[0])) == 0

        # Can fit again
        index.fit(vectors, payload)
        assert len(index.search(vectors[0])) > 0


class TestHNSWNumbaFallback:
    """The layer-0 beam search has a numba-compiled kernel and a pure-numpy
    fallback. numba is an optional extra, so the fallback must stay correct on
    its own (an earlier bug surfaced the wrong result ids and read as ~0 recall).
    """

    @staticmethod
    def _brute_force_top_k(vectors, query, k):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = vectors / norms
        q = query / (np.linalg.norm(query) + 1e-10)
        sims = normed @ q
        return set(np.argsort(sims)[-k:][::-1])

    def test_numba_absent_fallback_recall(self, temp_db, monkeypatch):
        # Force the numpy fallback path exactly as if numba were not installed.
        import sqlitesearch.vector.strategy_hnsw as hnsw_mod

        monkeypatch.setattr(hnsw_mod, "_beam_search_0_njit", None)

        rng = np.random.default_rng(42)
        n, dim = 800, 48
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i} for i in range(n)]

        index = VectorSearchIndex(
            mode="hnsw", m=16, ef_construction=200, ef_search=100, db_path=temp_db, seed=1
        )
        index.fit(vectors, payload)

        # The numba kernel must not have been wired in.
        assert index._strategy._cand_sim is None

        k = 10
        hits = 0
        total = 0
        for qi in range(0, n, 17):
            gt = self._brute_force_top_k(vectors, vectors[qi], k)
            got = {r["id"] for r in index.search(vectors[qi], num_results=k)}
            hits += len(gt & got)
            total += k
        recall = hits / total
        assert recall >= 0.5, f"numpy-fallback recall@{k} = {recall:.3f}, expected >= 0.5"
        index.close()

    def test_numba_and_numpy_fallback_agree(self, temp_db, monkeypatch):
        """When numba is installed, the compiled kernel and the numpy fallback
        build the same graph (the two are documented as algorithmically identical)."""
        import sqlitesearch.vector.strategy_hnsw as hnsw_mod
        from sqlitesearch.vector._hnsw_numba import _NUMBA_OK

        if not _NUMBA_OK:
            pytest.skip("numba not installed; parity check needs both paths")

        rng = np.random.default_rng(99)
        n, dim = 500, 40
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i} for i in range(n)]

        # Build with the numba kernel, then rebuild with it disabled.
        numba_index = VectorSearchIndex(
            mode="hnsw", m=16, ef_construction=200, ef_search=100, db_path=temp_db, seed=1
        )
        numba_index.fit(vectors, payload)
        assert numba_index._strategy._cand_sim is not None
        numba_results = [
            {r["id"] for r in numba_index.search(vectors[q], num_results=10)}
            for q in range(0, n, 23)
        ]
        numba_index.close()
        # clear() so the second fit is a fresh build on the same db.
        monkeypatch.setattr(hnsw_mod, "_beam_search_0_njit", None)
        fallback_index = VectorSearchIndex(
            mode="hnsw", m=16, ef_construction=200, ef_search=100, db_path=temp_db, seed=1
        )
        fallback_index.clear()
        fallback_index.fit(vectors, payload)
        assert fallback_index._strategy._cand_sim is None

        overlaps = []
        for q, numba_set in zip(range(0, n, 23), numba_results):
            fb_set = {r["id"] for r in fallback_index.search(vectors[q], num_results=10)}
            overlaps.append(len(numba_set & fb_set) / 10)
        fallback_index.close()

        mean_overlap = float(np.mean(overlaps))
        assert mean_overlap >= 0.9, (
            f"numba vs numpy-fallback top-10 overlap = {mean_overlap:.3f}, expected >= 0.9"
        )


class TestHNSWColdReload:
    """Guards the cold-reopen + add() path: a reopened index must load its
    persisted graph/params before appending, instead of rebuilding from just
    the new chunk (which previously clobbered everything fitted before)."""

    def test_cold_reload_then_add_preserves_prior_graph(self, temp_db):
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((300, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(300)]

        idx = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1)
        idx.fit(vectors, payload)
        idx.close()

        idx2 = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1)
        idx2.add(vectors[0], {"id": 300})

        # All 300 originally-fitted nodes must still be indexed (not just the 1
        # added node), and an early vector must still be retrievable.
        assert idx2._strategy._n_nodes == 301
        early = {r["id"] for r in idx2.search(vectors[5], num_results=5)}
        assert 5 in early, f"early doc lost after reopen+add: {early}"
        idx2.close()

    def test_cold_reload_search_matches_in_process_build(self, temp_db):
        rng = np.random.default_rng(11)
        n, dim = 500, 32
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i} for i in range(n)]
        queries = list(range(0, n, 25))

        fresh = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1)
        fresh.fit(vectors, payload)
        fresh_sets = [
            {r["id"] for r in fresh.search(vectors[q], num_results=10)} for q in queries
        ]
        fresh.close()

        reopened = VectorSearchIndex(mode="hnsw", db_path=temp_db, seed=1)
        for q, expected in zip(queries, fresh_sets):
            got = {r["id"] for r in reopened.search(vectors[q], num_results=10)}
            assert got == expected, f"cold-reload search diverged at q={q}"
        reopened.close()
