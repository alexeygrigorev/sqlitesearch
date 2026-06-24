"""
Recall tests for cardinality-aware filtered search.

These guard the fix for the post-filtering recall collapse: before the planner,
``search(filter_dict=...)`` gathered ANN candidates over the whole index and
intersected them with the filter afterwards, so a selective filter (e.g.
``course == X``) starved the result set and returned the wrong / too-few docs.
With the planner, selective filters take an exact-scan branch that returns the
true top-k within the filter.
"""

import os
import tempfile
from datetime import date, timedelta

import numpy as np
import pytest

from sqlitesearch import VectorSearchIndex
from sqlitesearch.operators import OPERATORS, is_range_filter


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# --- ground-truth helpers -------------------------------------------------


def _doc_matches(doc, field, value):
    v = doc.get(field)
    if value is None:
        return v is None
    if is_range_filter(value):
        if v is None:
            return False
        for op, opv in value:
            if opv is None:
                continue
            if not OPERATORS[op](v, opv):
                return False
        return True
    if isinstance(value, (list, tuple, set)):
        return v in value
    return v == value


def _matches_filter(doc, filter_dict):
    return all(_doc_matches(doc, f, v) for f, v in filter_dict.items())


def _ground_truth_topk(vectors, payload, filter_dict, query, k):
    idxs = [i for i, doc in enumerate(payload) if _matches_filter(doc, filter_dict)]
    if not idxs:
        return set()
    sub = vectors[idxs]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = sub / norms
    q = query / (np.linalg.norm(query) + 1e-10)
    sims = normed @ q
    k_eff = min(k, len(idxs))
    order = np.argsort(sims)[::-1][:k_eff]
    return {payload[idxs[i]]["id"] for i in order}


def _recall(index, query, filter_dict, vectors, payload, k):
    gt = _ground_truth_topk(vectors, payload, filter_dict, query, k)
    if not gt:
        return 1.0
    results = index.search(query, filter_dict=filter_dict, num_results=k)
    result_ids = {r["id"] for r in results}
    return len(gt & result_ids) / len(gt)


# --- headline: selective keyword filter, all three modes ------------------


def _build_course_index(mode, temp_db, n=2000, dim=64, n_courses=10):
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    payload = [{"id": i, "course": f"c{i % n_courses}"} for i in range(n)]
    kwargs = {"mode": mode, "keyword_fields": ["course"], "db_path": temp_db}
    if mode == "hnsw":
        kwargs.update(m=16, ef_construction=100, ef_search=50)
    elif mode == "ivf":
        kwargs.update(n_probe_clusters=8)
    index = VectorSearchIndex(**kwargs)
    index.fit(vectors, payload)
    return index, vectors, payload


@pytest.mark.parametrize("mode", ["lsh", "ivf", "hnsw"])
class TestSelectiveFilterRecall:
    def test_selective_keyword_filter_recall(self, mode, temp_db):
        """A selective course filter must return the true top-k within that
        course, not the starved post-filter set (fails before the planner)."""
        index, vectors, payload = _build_course_index(mode, temp_db)

        rng = np.random.default_rng(7)
        k = 10
        query_indices = rng.choice(len(vectors), 40, replace=False)

        recalls = []
        for qi in query_indices:
            course = payload[qi]["course"]
            recalls.append(
                _recall(
                    index,
                    vectors[qi],
                    {"course": course},
                    vectors,
                    payload,
                    k,
                )
            )
        mean_recall = float(np.mean(recalls))
        assert mean_recall >= 0.9, f"{mode} selective-filter recall@{k} = {mean_recall:.3f}"


# --- planner dispatch -----------------------------------------------------


class TestPlannerDispatch:
    def test_no_filter_skips_planner(self, temp_db):
        """The unfiltered hot path must not invoke the cardinality planner."""
        index, vectors, payload = _build_course_index("lsh", temp_db, n=200, dim=32)

        def boom(*args, **kwargs):
            raise AssertionError("planner must not run without a filter")

        index._count_filtered = boom  # would explode if the planner ran
        results = index.search(vectors[0], num_results=5)
        assert len(results) > 0

    def test_empty_filter_result_returns_empty(self, temp_db):
        index, vectors, payload = _build_course_index("lsh", temp_db, n=200, dim=32)
        results = index.search(vectors[0], filter_dict={"course": "does-not-exist"})
        assert results == []

    def test_count_with_filter(self, temp_db):
        index, vectors, payload = _build_course_index("lsh", temp_db, n=200, dim=32)
        expected = sum(1 for d in payload if d["course"] == "c0")
        assert index.count({"course": "c0"}) == expected
        assert index.count() == len(payload)

    def test_forced_ann_branch_still_respects_filter(self, temp_db):
        """exact_filter_threshold=0 forces the ANN branch; results must still
        honour the filter."""
        index, vectors, payload = _build_course_index("hnsw", temp_db, n=200, dim=32)
        index._exact_filter_threshold = 0  # always take the filtered-ANN branch
        results = index.search(vectors[0], filter_dict={"course": "c0"}, num_results=10)
        assert all(r["course"] == "c0" for r in results)


# --- exact branch across filter shapes ------------------------------------


class TestExactBranchRecall:
    def _build(self, temp_db, n=600, dim=48):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [
            {
                "id": i,
                "course": f"c{i % 6}",
                "price": float(i % 100),
                "created_at": date(2024, 1, 1) + timedelta(days=i),
            }
            for i in range(n)
        ]
        index = VectorSearchIndex(
            keyword_fields=["course"],
            numeric_fields=["price"],
            date_fields=["created_at"],
            db_path=temp_db,
        )
        index.fit(vectors, payload)
        return index, vectors, payload

    def test_multivalue_in_recall(self, temp_db):
        index, vectors, payload = self._build(temp_db)
        recalls = []
        for qi in range(0, 200, 7):
            recalls.append(
                _recall(index, vectors[qi], {"course": ["c0", "c1"]}, vectors, payload, 10)
            )
        assert float(np.mean(recalls)) >= 0.9

    def test_numeric_range_recall(self, temp_db):
        index, vectors, payload = self._build(temp_db)
        flt = {"price": [(">=", 20), ("<", 60)]}
        recalls = []
        for qi in range(0, 200, 7):
            recalls.append(_recall(index, vectors[qi], flt, vectors, payload, 10))
        assert float(np.mean(recalls)) >= 0.9

    def test_date_range_recall(self, temp_db):
        index, vectors, payload = self._build(temp_db)
        flt = {"created_at": [(">=", date(2024, 1, 10)), ("<=", date(2024, 3, 1))]}
        recalls = []
        for qi in range(0, 200, 7):
            recalls.append(_recall(index, vectors[qi], flt, vectors, payload, 10))
        assert float(np.mean(recalls)) >= 0.9

    def test_empty_list_matches_nothing(self, temp_db):
        index, vectors, payload = self._build(temp_db, n=50, dim=16)
        assert index.search(vectors[0], filter_dict={"course": []}) == []


# --- adaptive over-fetch (HNSW) -------------------------------------------


class TestAdaptiveOverFetch:
    def test_ann_branch_fills_results(self, temp_db):
        """With a small initial budget and a forced ANN branch, the over-fetch
        loop must widen the budget until enough post-filter results remain."""
        rng = np.random.default_rng(42)
        n, dim = 800, 48
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i, "course": f"c{i % 4}"} for i in range(n)]

        index = VectorSearchIndex(
            mode="hnsw",
            m=16,
            ef_construction=100,
            ef_search=8,  # deliberately tiny
            exact_filter_threshold=0,  # force the filtered-ANN branch
            keyword_fields=["course"],
            db_path=temp_db,
        )
        index.fit(vectors, payload)

        results = index.search(vectors[0], filter_dict={"course": "c0"}, num_results=10)
        assert len(results) == 10
        assert all(r["course"] == "c0" for r in results)

        # The transient ef_search widening must not leak to later queries.
        assert index._strategy.ef_search == 8


# --- Phase 2: HNSW node-skipping traversal ---------------------------------


class TestFilteredHNSWNodeSkipping:
    """The HNSW filter-aware (node-skipping) traversal walks the graph through
    non-matching nodes but collects only matching ones. With the exact branch
    disabled this path alone must still return the true top-k within the
    filter."""

    def test_nodeskipping_recall_forced_ann(self, temp_db):
        rng = np.random.default_rng(42)
        n, dim = 2000, 64
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i, "course": f"c{i % 10}"} for i in range(n)]

        index = VectorSearchIndex(
            mode="hnsw",
            m=16,
            ef_construction=100,
            ef_search=64,
            exact_filter_threshold=0,  # force filtered-ANN -> node-skipping
            keyword_fields=["course"],
            db_path=temp_db,
        )
        index.fit(vectors, payload)

        recalls = []
        for qi in range(0, n, 53):
            course = payload[qi]["course"]
            recalls.append(_recall(index, vectors[qi], {"course": course}, vectors, payload, 10))
        mean_recall = float(np.mean(recalls))
        assert mean_recall >= 0.9, f"node-skipping recall@10 = {mean_recall:.3f}"

    def test_non_matching_entry_point_still_finds_matches(self, temp_db):
        """If the entry point itself fails the filter, the walk must still
        expand its neighbors and reach the allowed region (results must be
        non-empty and all-matching)."""
        rng = np.random.default_rng(1)
        n, dim = 500, 48
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = [{"id": i, "course": f"c{i % 5}"} for i in range(n)]

        index = VectorSearchIndex(
            mode="hnsw",
            m=16,
            ef_construction=100,
            ef_search=32,
            exact_filter_threshold=0,
            keyword_fields=["course"],
            db_path=temp_db,
        )
        index.fit(vectors, payload)

        # Query for a course that the nearest vector (vectors[0]) does NOT have,
        # so the entry neighbourhood is dominated by non-matching nodes.
        target_course = "c2"
        results = index.search(vectors[0], filter_dict={"course": target_course}, num_results=10)
        assert len(results) == 10
        assert all(r["course"] == target_course for r in results)


class TestNoStrategyMutation:
    """The ``override`` kwarg widens the budget per-call only. The shared
    strategy object must be untouched after a filtered search that runs the
    over-fetch loop (strategies are shared across threads; connections are
    thread-local)."""

    @pytest.mark.parametrize(
        "mode,knob",
        [("hnsw", "ef_search"), ("ivf", "n_probe_clusters"), ("lsh", "n_probe")],
    )
    def test_budget_attr_unchanged_after_filtered_search(self, mode, knob, temp_db):
        index, vectors, payload = _build_course_index(mode, temp_db, n=600, dim=32)
        original = getattr(index._strategy, knob)
        index._exact_filter_threshold = 0  # force the over-fetch loop
        results = index.search(
            vectors[0],
            filter_dict={"course": payload[0]["course"]},
            num_results=10,
        )
        assert all(r["course"] == payload[0]["course"] for r in results)
        assert getattr(index._strategy, knob) == original
