"""
Tests for IVF vector search mode.
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


class TestIVFBasics:
    def test_fit_and_search(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 64)).astype(np.float32)
        payload = [{"id": i} for i in range(100)]

        index = VectorSearchIndex(mode="ivf", db_path=temp_db)
        index.fit(vectors, payload)

        results = index.search(vectors[0], num_results=10)
        assert len(results) > 0
        assert results[0]["id"] == 0

    def test_num_results(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 64)).astype(np.float32)
        payload = [{"id": i} for i in range(100)]

        index = VectorSearchIndex(mode="ivf", db_path=temp_db)
        index.fit(vectors, payload)

        for n in [1, 5, 10]:
            results = index.search(vectors[0], num_results=n)
            assert len(results) <= n

    def test_empty_search(self, temp_db):
        index = VectorSearchIndex(mode="ivf", db_path=temp_db)
        query = np.random.randn(64).astype(np.float32)
        results = index.search(query)
        assert len(results) == 0

    def test_custom_clusters(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(50)]

        index = VectorSearchIndex(mode="ivf", n_clusters=5, n_probe_clusters=3, db_path=temp_db)
        index.fit(vectors, payload)

        results = index.search(vectors[0], num_results=5)
        assert len(results) > 0
        assert results[0]["id"] == 0


class TestIVFRecall:
    def test_recall_at_1000(self, temp_db):
        """IVF should achieve at least 0.5 recall@10 on 1000 random vectors."""
        rng = np.random.default_rng(42)
        n, dim = 1000, 64
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i} for i in range(n)]

        index = VectorSearchIndex(mode="ivf", n_probe_clusters=8, db_path=temp_db)
        index.fit(vectors, payload)

        # Compute ground truth for 50 queries
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
        assert recall >= 0.5, f"IVF recall@{k} = {recall:.3f}, expected >= 0.5"


class TestIVFFiltering:
    def test_keyword_filter(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [
            {"id": i, "category": "a" if i % 2 == 0 else "b"}
            for i in range(20)
        ]

        index = VectorSearchIndex(
            mode="ivf", keyword_fields=["category"], db_path=temp_db
        )
        index.fit(vectors, payload)

        results = index.search(vectors[0], filter_dict={"category": "a"})
        assert all(r["category"] == "a" for r in results)

    def test_numeric_filter(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [{"id": i, "price": float(i * 10)} for i in range(20)]

        index = VectorSearchIndex(
            mode="ivf", numeric_fields=["price"], db_path=temp_db
        )
        index.fit(vectors, payload)

        results = index.search(vectors[0], filter_dict={"price": [(">=", 100)]})
        assert all(r["price"] >= 100 for r in results)


class TestIVFPersistence:
    def test_persist_and_reload(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(50)]

        index1 = VectorSearchIndex(mode="ivf", db_path=temp_db)
        index1.fit(vectors, payload)
        index1.close()

        index2 = VectorSearchIndex(mode="ivf", db_path=temp_db)
        results = index2.search(vectors[0], num_results=5)
        assert len(results) > 0
        assert results[0]["id"] == 0
        index2.close()


class TestIVFAdd:
    def test_add_after_fit(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(20)]

        index = VectorSearchIndex(mode="ivf", db_path=temp_db)
        index.fit(vectors, payload)

        new_vec = rng.standard_normal(32).astype(np.float32)
        index.add(new_vec, {"id": 20})

        results = index.search(new_vec, num_results=5)
        assert any(r["id"] == 20 for r in results)

    def test_add_without_fit(self, temp_db):
        rng = np.random.default_rng(42)
        index = VectorSearchIndex(mode="ivf", db_path=temp_db)

        vec = rng.standard_normal(32).astype(np.float32)
        index.add(vec, {"id": 1})

        results = index.search(vec)
        assert len(results) == 1
        assert results[0]["id"] == 1


class TestIVFClear:
    def test_clear(self, temp_db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 32)).astype(np.float32)
        payload = [{"id": i} for i in range(20)]

        index = VectorSearchIndex(mode="ivf", db_path=temp_db)
        index.fit(vectors, payload)
        assert len(index.search(vectors[0])) > 0

        index.clear()
        assert len(index.search(vectors[0])) == 0

        # Can fit again
        index.fit(vectors, payload)
        assert len(index.search(vectors[0])) > 0
