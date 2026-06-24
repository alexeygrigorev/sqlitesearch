"""
VectorSearchIndex - Persistent vector search with pluggable strategies.

Supports LSH, IVF, and HNSW modes for approximate nearest neighbor search,
followed by exact cosine similarity reranking.
"""

import json
import pickle
import sqlite3
import threading
from datetime import date, datetime
from typing import Any

import numpy as np

from sqlitesearch.connection import (
    bulk_insert_returning_ids,
    bulk_upsert,
    connect,
    fetch_ids_by_key,
    is_remote_url,
    max_sql_vars,
)
from sqlitesearch.operators import OPERATORS, is_range_filter
from sqlitesearch.vector.base import VectorMode
from sqlitesearch.vector.strategy_lsh import LSHStrategy

# When a filter matches at most this many vectors, skip the ANN index and run
# an exact cosine rerank over the filtered subset (cheap: vectors are cached in
# memory). Beyond it, fall back to the ANN path with adaptive over-fetch.
# Cardinality-aware filtering mirrors what Qdrant/Milvus/pgvector do: exact scan
# for selective filters, filtered ANN otherwise.
_DEFAULT_EXACT_FILTER_THRESHOLD = 20_000


class VectorSearchIndex:
    """
    A persistent vector search index with pluggable search strategies.

    Supports mode="lsh" (default), mode="ivf", and mode="hnsw".

    API:
    - __init__(mode="lsh", keyword_fields=None, numeric_fields=None, date_fields=None, id_field=None, db_path=..., **kwargs)
    - fit(vectors, payload) - Index vectors (only if index is empty)
    - add(vector, doc) - Add a single vector with document
    - search(query_vector, filter_dict=None, num_results=10, output_ids=False)
    """

    def __init__(
        self,
        mode: str = "lsh",
        keyword_fields: list[str] | None = None,
        numeric_fields: list[str] | None = None,
        date_fields: list[str] | None = None,
        id_field: str | None = None,
        db_path: str = "sqlitesearch_vectors.db",
        backend: str = "sqlite3",
        auth_token: str | None = None,
        replica_path: str | None = None,
        # LSH params (kept as explicit kwargs for backward compatibility)
        n_tables: int = 8,
        hash_size: int = 16,
        n_probe: int = 2,
        seed: int | None = None,
        # IVF params
        n_clusters: int | None = None,
        n_probe_clusters: int = 4,
        # HNSW params
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        # Filtered search: max filtered-subset size for the exact-scan branch.
        # None -> _DEFAULT_EXACT_FILTER_THRESHOLD. Set 0 to always use the ANN
        # path for filtered queries.
        exact_filter_threshold: int | None = None,
    ):
        self.keyword_fields = list(keyword_fields) if keyword_fields is not None else []
        self.numeric_fields = list(numeric_fields) if numeric_fields is not None else []
        self.date_fields = list(date_fields) if date_fields is not None else []
        self.id_field = id_field
        self.db_path = db_path
        # A remote URL as db_path means a Turso embedded replica (set up in
        # connect()). Treat it as the libsql backend here so bulk inserts use
        # the network-sized batches.
        if is_remote_url(db_path) and backend == "sqlite3":
            backend = "libsql"
        self.backend = backend
        self.auth_token = auth_token
        self.replica_path = replica_path
        self._max_vars = max_sql_vars(backend)
        self._local = threading.local()

        # Expose LSH params as attributes for backward compatibility with tests
        self.n_tables = n_tables
        self.hash_size = hash_size
        self.n_probe = n_probe

        # In-memory vector cache for fast reranking
        self._dimension: int | None = None
        self._cached_vectors: np.ndarray | None = None
        self._cached_doc_ids: list[int] | None = None
        self._cached_docs: list[dict] | None = None
        self._id_to_cache_idx: dict[int, int] | None = None

        # Add id_field to keyword_fields if provided and not already there
        if self.id_field and self.id_field not in self.keyword_fields:
            self.keyword_fields.append(self.id_field)

        # Create strategy
        mode_enum = VectorMode(mode)
        self._mode = mode_enum
        self._exact_filter_threshold = exact_filter_threshold
        if mode_enum == VectorMode.LSH:
            self._strategy = LSHStrategy(
                n_tables=n_tables, hash_size=hash_size, n_probe=n_probe, seed=seed
            )
        elif mode_enum == VectorMode.IVF:
            from sqlitesearch.vector.strategy_ivf import IVFStrategy

            self._strategy = IVFStrategy(
                n_clusters=n_clusters, n_probe_clusters=n_probe_clusters, seed=seed
            )
        elif mode_enum == VectorMode.HNSW:
            from sqlitesearch.vector.strategy_hnsw import HNSWStrategy

            self._strategy = HNSWStrategy(
                m=m, ef_construction=ef_construction, ef_search=ef_search, seed=seed
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Let the strategy size its bulk inserts for the backend too (#13).
        self._strategy._max_vars = self._max_vars

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = connect(
                self.db_path,
                backend=self.backend,
                auth_token=self.auth_token,
                replica_path=self.replica_path,
            )
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Build column definitions
        keyword_cols = [f', "{field}" TEXT' for field in self.keyword_fields]
        numeric_cols = [f', "{field}" REAL' for field in self.numeric_fields]
        date_cols = [f', "{field}" TEXT' for field in self.date_fields]
        extra_sql = "\n".join(keyword_cols + numeric_cols + date_cols)

        # Main documents table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_json TEXT NOT NULL,
                vector_hash BLOB{extra_sql}
            )
        """)

        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)

        # Field indexes
        for field in self.keyword_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_vec_{field} ON docs ("{field}")')
        for field in self.numeric_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_vec_num_{field} ON docs ("{field}")')
        for field in self.date_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_vec_date_{field} ON docs ("{field}")')

        # Unique index on the user id field enables upsert-by-id so a shared
        # docs table is deduplicated across the text and vector index (#2).
        if self.id_field and self.id_field != "id":
            cursor.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS uidx_docs_id ON docs ("{self.id_field}")'
            )

        # Strategy-specific tables
        self._strategy.init_tables(cursor)

        conn.commit()

    def count(self, filter_dict: dict[str, Any] | None = None) -> int:
        """Return the number of documents in the index.

        With a ``filter_dict``, count only the vector rows matching it (the same
        semantics used by the filtered-search planner).
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        if filter_dict:
            return self._count_filtered(cursor, filter_dict)
        cursor.execute("SELECT COUNT(*) as count FROM docs")
        row = cursor.fetchone()
        return row["count"]

    def _is_empty(self) -> bool:
        """Check whether this index has any *vector* rows.

        Counts rows carrying a vector rather than all of ``docs`` so that a
        VectorSearchIndex can be fitted into a file whose ``docs`` table was
        already populated (with NULL vector_hash) by a TextSearchIndex
        (shared/hybrid file, issue #2).
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM docs WHERE vector_hash IS NOT NULL")
        return cursor.fetchone()["count"] == 0

    def fit(
        self,
        vectors: np.ndarray,
        payload: list[dict[str, Any]],
    ) -> "VectorSearchIndex":
        """
        Index the provided vectors with payload documents.

        Only works if the index is empty. Use add() to append documents.
        """
        if not self._is_empty():
            raise ValueError(
                "Index already contains documents. "
                "Use clear() to reset the index or add() to append documents."
            )
        return self._add_vectors(vectors, payload, is_fit=True)

    def add(
        self,
        vector: np.ndarray,
        doc: dict[str, Any],
    ) -> "VectorSearchIndex":
        """Add a single vector with document to the index."""
        vectors = np.asarray(vector, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return self._add_vectors(vectors, [doc], is_fit=False)

    def _add_vectors(
        self,
        vectors: np.ndarray,
        payload: list[dict[str, Any]],
        is_fit: bool = False,
    ) -> "VectorSearchIndex":
        """Internal method to add vectors to the index."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array, got shape {vectors.shape}")

        if len(vectors) != len(payload):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) must match "
                f"number of payload documents ({len(payload)})"
            )

        # Initialize dimension + strategy params if first time
        if self._dimension is None:
            self._dimension = vectors.shape[1]
            if hasattr(self._strategy, "set_dimension"):
                self._strategy.set_dimension(self._dimension)

            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("dimension", pickle.dumps(self._dimension)),
            )
            self._strategy.save_params(cursor)
            conn.commit()

        conn = self._get_conn()
        cursor = conn.cursor()

        # Build column lists
        filter_cols = (
            [f'"{field}"' for field in self.keyword_fields]
            + [f'"{field}"' for field in self.numeric_fields]
            + [f'"{field}"' for field in self.date_fields]
        )
        all_cols = ["doc_json", "vector_hash"] + filter_cols

        # Prepare doc rows
        doc_rows = []
        for vector, doc in zip(vectors, payload):
            doc_for_json = {}
            for key, value in doc.items():
                if isinstance(value, (date, datetime)):
                    doc_for_json[key] = value.isoformat()
                else:
                    doc_for_json[key] = value
            doc_json = json.dumps(doc_for_json)
            vector_bytes = vector.tobytes()

            keyword_vals = [doc.get(field) for field in self.keyword_fields]
            numeric_vals = [doc.get(field) for field in self.numeric_fields]
            date_vals = []
            for field in self.date_fields:
                value = doc.get(field)
                if isinstance(value, (date, datetime)):
                    date_vals.append(value.isoformat())
                else:
                    date_vals.append(value)

            doc_rows.append([doc_json, vector_bytes] + keyword_vals + numeric_vals + date_vals)

        # Insert docs and collect IDs. bulk_insert_returning_ids collapses the
        # inserts into chunked multi-row statements, which matters for the
        # libsql/Turso backend where each statement is a network round-trip
        # (issue #3) -- the old per-row loop was one round-trip per document.
        if self.id_field and self.id_field != "id":
            # Shared/hybrid file (#2): upsert by the user id so we reuse rows a
            # TextSearchIndex may already have written, filling vector_hash,
            # rather than duplicating them.
            id_col = f'"{self.id_field}"'
            bulk_upsert(
                cursor, "docs", all_cols, doc_rows, id_col, all_cols, max_vars=self._max_vars
            )
            key_vals = [doc.get(self.id_field) for doc in payload]
            id_map = fetch_ids_by_key(cursor, "docs", id_col, key_vals, max_vars=self._max_vars)
            doc_ids = [id_map[str(v)] for v in key_vals]
        else:
            doc_ids = bulk_insert_returning_ids(
                cursor, "docs", all_cols, doc_rows, max_vars=self._max_vars
            )

        # Build strategy index
        if is_fit:
            self._strategy.build_index(cursor, vectors, doc_ids)
        else:
            self._strategy.add_to_index(cursor, vectors, doc_ids)

        conn.commit()

        # Update in-memory vector cache
        new_docs = []
        for doc_row in doc_rows:
            doc = json.loads(doc_row[0])
            doc = self._convert_dates(doc)
            new_docs.append(doc)

        if self._cached_vectors is None:
            self._cached_vectors = vectors.copy()
            self._cached_doc_ids = list(doc_ids)
            self._cached_docs = new_docs
            self._id_to_cache_idx = {did: i for i, did in enumerate(doc_ids)}
        else:
            offset = len(self._cached_doc_ids)
            self._cached_vectors = np.vstack([self._cached_vectors, vectors])
            self._cached_doc_ids.extend(doc_ids)
            self._cached_docs.extend(new_docs)
            for i, did in enumerate(doc_ids):
                self._id_to_cache_idx[did] = offset + i

        return self

    def clear(self) -> "VectorSearchIndex":
        """Clear all documents from the index."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM docs")
        cursor.execute("DELETE FROM metadata")
        self._strategy.clear_index(cursor)

        self._dimension = None
        self._cached_vectors = None
        self._cached_doc_ids = None
        self._cached_docs = None
        self._id_to_cache_idx = None

        conn.commit()
        return self

    def search(
        self,
        query_vector: np.ndarray,
        filter_dict: dict[str, Any] | None = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the index with the given query vector.

        When a ``filter_dict`` is supplied, a cardinality-aware planner picks
        the strategy: for selective filters (matching at most
        ``exact_filter_threshold`` vectors) it skips the ANN index and runs an
        exact cosine rerank over the filtered subset (always correct); for
        non-selective filters it uses the ANN path with adaptive over-fetch so
        the post-filter result set isn't starved. Unfiltered search is
        unchanged.
        """
        if filter_dict is None:
            filter_dict = {}

        query_vector = np.asarray(query_vector, dtype=np.float32).flatten()

        if self._dimension is None:
            self._load_metadata()

        if self._dimension is None:
            return []

        if self._cached_vectors is None:
            self._load_vector_cache()

        if query_vector.shape[0] != self._dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[0]} "
                f"does not match index dimension {self._dimension}"
            )

        conn = self._get_conn()
        cursor = conn.cursor()

        # No filter: standard ANN path (unchanged, no planner overhead).
        if not filter_dict:
            candidate_ids = self._strategy.find_candidates(cursor, query_vector)
            if not candidate_ids:
                return []
            return self._rerank(cursor, query_vector, candidate_ids, num_results, output_ids)

        # Filtered path: cardinality-aware planner.
        filtered_count = self._count_filtered(cursor, filter_dict)
        if filtered_count == 0:
            return []

        threshold = (
            _DEFAULT_EXACT_FILTER_THRESHOLD
            if self._exact_filter_threshold is None
            else self._exact_filter_threshold
        )

        if filtered_count <= threshold:
            # Selective filter: exact cosine over the filtered subset. Cheap
            # because vectors are cached in memory and _rerank already does
            # exact cosine over an arbitrary id set.
            filtered_ids = self._enumerate_filtered_ids(cursor, filter_dict)
            if not filtered_ids:
                return []
            return self._rerank(cursor, query_vector, filtered_ids, num_results, output_ids)

        # Non-selective filter: ANN candidates -> filter -> adaptive over-fetch.
        return self._filtered_ann_search(cursor, query_vector, filter_dict, num_results, output_ids)

    def _filtered_ann_search(
        self,
        cursor: sqlite3.Cursor,
        query_vector: np.ndarray,
        filter_dict: dict[str, Any],
        num_results: int,
        output_ids: bool,
    ) -> list[dict[str, Any]]:
        """Filtered ANN search widened until the result set is large enough.

        Mirrors Milvus "iterative filter" / pgvector "iterative index scans": if
        the survivors are fewer than ``num_results``, bump the candidate budget
        and retry. The widened budget is passed via the ``override`` kwarg rather
        than mutating the shared strategy object, so it never leaks across
        concurrent searches (the strategy is shared across threads; connections
        are thread-local).

        HNSW uses node-skipping traversal (Phase 2): the allowed id set is
        enumerated once and pushed into the graph walk, so disqualified subtrees
        are skipped in a single pass instead of post-filtering many passes.
        LSH/IVF have no graph, so they gather candidates and post-filter.
        """
        knob, cap = self._ann_budget_knob_and_cap()

        if self._mode == VectorMode.HNSW:
            allowed = self._enumerate_filtered_ids(cursor, filter_dict)
            if not allowed:
                return []
            current = self._strategy.ef_search
            candidates: set[int] = set()
            while True:
                candidates = self._strategy.find_candidates(
                    cursor,
                    query_vector,
                    override={knob: current},
                    filter_ids=allowed,
                )
                if len(candidates) >= num_results or current >= cap:
                    break
                nxt = min(current * 2, cap)
                if nxt <= current:
                    break  # can't widen further
                current = nxt
            if not candidates:
                return []
            return self._rerank(cursor, query_vector, candidates, num_results, output_ids)

        # LSH / IVF: widen the budget via override, then post-filter.
        current = getattr(self._strategy, knob)
        survivors: set[int] = set()
        while True:
            candidate_ids = self._strategy.find_candidates(
                cursor, query_vector, override={knob: current}
            )
            survivors = self._apply_filters(cursor, candidate_ids, filter_dict)
            if len(survivors) >= num_results or current >= cap:
                break
            nxt = min(current * 2, cap)
            if nxt <= current:
                break  # can't widen further
            current = nxt
        if not survivors:
            return []
        return self._rerank(cursor, query_vector, survivors, num_results, output_ids)

    def _ann_budget_knob_and_cap(self) -> tuple[str, int]:
        """Return (strategy attribute to widen, max useful value) by mode."""
        if self._mode == VectorMode.HNSW:
            n_nodes = getattr(self._strategy, "_n_nodes", 0) or 0
            return ("ef_search", min(max(n_nodes, 1), 10_000))
        if self._mode == VectorMode.IVF:
            centroids = getattr(self._strategy, "_centroids", None)
            cap = len(centroids) if centroids is not None else 4
            return ("n_probe_clusters", max(cap, 1))
        # LSH: n_probe beyond ~2 adds no probe keys (1-/2-bit flips), so the
        # retry converges fast; selective filters are handled by the exact
        # branch anyway.
        return ("n_probe", 2)

    # --- Shared internal methods ---

    def _chunked_in_query(self, cursor, sql_template, ids_list, extra_params=None, chunk_size=None):
        if extra_params is None:
            extra_params = []
        # Respect the backend's variable limit (900 for sqlite3, 30000 for the
        # libsql/Turso network backend) instead of a hardcoded 900.
        if chunk_size is None:
            chunk_size = self._max_vars
        all_rows = []
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            sql = sql_template.format(placeholders=placeholders)
            cursor.execute(sql, chunk + extra_params)
            all_rows.extend(cursor.fetchall())
        return all_rows

    def _build_field_predicate(self, field: str, value: Any) -> tuple[str | None, list]:
        """Build a WHERE fragment for one field, WITHOUT the ``id IN (...)`` anchor.

        Returns ``(fragment, params)`` where ``fragment`` is one of:
          - ``(None, [])``   -> matches nothing (e.g. an empty list value); the
            caller should short-circuit to an empty result.
          - ``("", [])``     -> no constraint on this field (skip it).
          - ``("<sql>", params)`` -> e.g. ``('"category" = ?', ['x'])``.
        """
        if field in self.keyword_fields:
            if value is None:
                return (f'"{field}" IS NULL', [])
            if isinstance(value, (list, tuple, set)):
                # Multi-value membership: field matches ANY of these values.
                values = list(value)
                if not values:
                    # Empty list matches nothing.
                    return (None, [])
                ph = ", ".join("?" for _ in values)
                return (f'"{field}" IN ({ph})', values)
            return (f'"{field}" = ?', [value])

        if field in self.numeric_fields:
            return self._build_typed_predicate(field, value, convert_date=False)

        if field in self.date_fields:
            return self._build_typed_predicate(field, value, convert_date=True)

        # Unknown field: ignore (same as the legacy per-field loop).
        return ("", [])

    def _build_typed_predicate(
        self, field: str, value: Any, *, convert_date: bool
    ) -> tuple[str | None, list]:
        """Build a WHERE fragment for a numeric/date field.

        ``convert_date`` converts ``date``/``datetime`` values to ISO strings
        (date columns are stored as TEXT). Mirrors the legacy numeric/date
        filter behaviour exactly.
        """
        if value is None:
            return (f'"{field}" IS NULL', [])
        if is_range_filter(value):
            conds: list[str] = []
            params: list = []
            for op, op_value in value:
                if op in OPERATORS and op_value is not None:
                    if convert_date and isinstance(op_value, (date, datetime)):
                        op_value = op_value.isoformat()
                    conds.append(f'"{field}" {op} ?')
                    params.append(op_value)
            if not conds:
                return ("", [])  # no usable condition -> no constraint
            return (" AND ".join(conds), params)
        # scalar equality
        if convert_date and isinstance(value, (date, datetime)):
            value = value.isoformat()
        return (f'"{field}" = ?', [value])

    def _build_filter_where(self, filter_dict: dict[str, Any]) -> tuple[str | None, list]:
        """Compose the full WHERE clause for a ``filter_dict`` (AND-combined).

        Like :meth:`_build_field_predicate`, returns ``(None, [])`` when the
        filter provably matches nothing, ``("", [])`` for no constraint, or a
        ready ``("<sql>", params)`` fragment. No ``id IN (...)`` anchor.
        """
        fragments: list[str] = []
        params: list = []
        for field, value in filter_dict.items():
            frag, field_params = self._build_field_predicate(field, value)
            if frag is None:
                return (None, [])  # matches nothing
            if frag:
                fragments.append(frag)
                params.extend(field_params)
        if not fragments:
            return ("", [])
        return (" AND ".join(fragments), params)

    def _filtered_vector_where(self, fragment: str | None, params: list) -> tuple[str | None, list]:
        """Scope a filter fragment to vector-bearing rows.

        Adds ``vector_hash IS NOT NULL`` so text-only rows from a shared
        (hybrid) ``docs`` table are excluded (#2). ``fragment=None`` (matches
        nothing) is passed through as a sentinel.
        """
        if fragment is None:
            return (None, [])
        if not fragment:
            return ("vector_hash IS NOT NULL", params)
        return (f"{fragment} AND vector_hash IS NOT NULL", params)

    def _count_filtered(self, cursor: sqlite3.Cursor, filter_dict: dict[str, Any]) -> int:
        """Count vector rows matching ``filter_dict`` (no candidate anchor)."""
        where, params = self._filtered_vector_where(*self._build_filter_where(filter_dict))
        if where is None:
            return 0
        cursor.execute(f"SELECT COUNT(*) AS c FROM docs WHERE {where}", params)
        return cursor.fetchone()["c"]

    def _enumerate_filtered_ids(
        self, cursor: sqlite3.Cursor, filter_dict: dict[str, Any]
    ) -> set[int]:
        """Return all vector-row ids matching ``filter_dict`` (no candidate anchor)."""
        where, params = self._filtered_vector_where(*self._build_filter_where(filter_dict))
        if where is None:
            return set()
        cursor.execute(f"SELECT id FROM docs WHERE {where}", params)
        return {row["id"] for row in cursor.fetchall()}

    def _apply_filters(
        self,
        cursor: sqlite3.Cursor,
        candidate_ids: set[int],
        filter_dict: dict[str, Any],
    ) -> set[int]:
        """Intersect ``candidate_ids`` with the docs matching ``filter_dict``."""
        if not filter_dict or not candidate_ids:
            return candidate_ids

        fragment, params = self._build_filter_where(filter_dict)
        if fragment is None:
            return set()  # empty-list value matches nothing

        ids_list = list(candidate_ids)
        if fragment:
            sql = f"SELECT id FROM docs WHERE id IN ({{placeholders}}) AND {fragment}"
            extra_params = params
        else:
            sql = "SELECT id FROM docs WHERE id IN ({placeholders})"
            extra_params = []
        rows = self._chunked_in_query(cursor, sql, ids_list, extra_params=extra_params)
        return {row["id"] for row in rows}

    def _rerank(self, cursor, query_vector, candidate_ids, num_results, output_ids):
        if not candidate_ids:
            return []

        if self._cached_vectors is not None and self._id_to_cache_idx is not None:
            cache_indices = [
                self._id_to_cache_idx[did] for did in candidate_ids if did in self._id_to_cache_idx
            ]
            if not cache_indices:
                return []

            cache_indices = np.array(cache_indices)
            candidate_matrix = self._cached_vectors[cache_indices]
        else:
            ids_list = list(candidate_ids)
            rows = self._chunked_in_query(
                cursor,
                "SELECT id, doc_json, vector_hash FROM docs "
                "WHERE id IN ({placeholders}) AND vector_hash IS NOT NULL",
                ids_list,
            )
            if not rows:
                return []
            cache_indices = None
            doc_ids_fb = [row["id"] for row in rows]
            docs_fb = [self._convert_dates(json.loads(row["doc_json"])) for row in rows]
            candidate_matrix = np.stack(
                [np.frombuffer(row["vector_hash"], dtype=np.float32) for row in rows]
            )

        # Vectorized cosine similarity
        query_norm = np.linalg.norm(query_vector)
        query_normalized = query_vector if query_norm == 0 else query_vector / query_norm

        norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_matrix = candidate_matrix / norms

        similarities = normalized_matrix @ query_normalized

        if num_results < len(similarities):
            top_indices = np.argpartition(similarities, -num_results)[-num_results:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                if cache_indices is not None:
                    ci = cache_indices[idx]
                    doc = self._cached_docs[ci].copy()
                    doc_id = self._cached_doc_ids[ci]
                else:
                    doc = docs_fb[idx]
                    doc_id = doc_ids_fb[idx]

                if output_ids:
                    result_id = doc.get(self.id_field, doc_id) if self.id_field else doc_id
                    doc = {**doc, "_id": result_id}
                results.append(doc)

        return results

    def _load_vector_cache(self) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        # Only rows that actually carry a vector. In a shared (hybrid) file the
        # text index may have written rows with a NULL vector_hash; skip those
        # so a cold load doesn't try to np.frombuffer(None) (issue #2).
        cursor.execute(
            "SELECT id, doc_json, vector_hash FROM docs WHERE vector_hash IS NOT NULL ORDER BY id"
        )
        rows = cursor.fetchall()
        if not rows:
            return

        doc_ids = []
        docs = []
        vectors_list = []
        for row in rows:
            doc_ids.append(row["id"])
            doc = json.loads(row["doc_json"])
            doc = self._convert_dates(doc)
            docs.append(doc)
            vectors_list.append(np.frombuffer(row["vector_hash"], dtype=np.float32))

        self._cached_vectors = np.stack(vectors_list)
        self._cached_doc_ids = doc_ids
        self._cached_docs = docs
        self._id_to_cache_idx = {did: i for i, did in enumerate(doc_ids)}

        # Sync vectors to strategy if it needs them (e.g., HNSW)
        self._sync_vectors_to_strategy()

    def _sync_vectors_to_strategy(self) -> None:
        """Give the strategy a reference to the vector cache if it needs one."""
        if hasattr(self._strategy, "_vectors") and self._cached_vectors is not None:
            from sqlitesearch.vector.strategy_hnsw import HNSWStrategy

            if isinstance(self._strategy, HNSWStrategy):
                self._strategy._vectors = HNSWStrategy._normalize(self._cached_vectors)
                self._strategy._doc_ids = self._cached_doc_ids
                self._strategy._id_to_idx = self._id_to_cache_idx

    def _load_metadata(self) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM metadata WHERE key = 'dimension'")
        row = cursor.fetchone()
        if row:
            self._dimension = pickle.loads(row["value"])

        self._strategy.load_params(cursor)

    def _convert_dates(self, doc: dict[str, Any]) -> dict[str, Any]:
        if not self.date_fields:
            return doc

        for field in self.date_fields:
            if field in doc and doc[field] is not None:
                value = doc[field]
                if isinstance(value, str):
                    has_time = "T" in value or " " in value
                    if has_time:
                        try:
                            doc[field] = datetime.fromisoformat(value)
                        except ValueError:
                            pass
                    else:
                        try:
                            doc[field] = date.fromisoformat(value)
                        except ValueError:
                            pass
        return doc

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")

    def __enter__(self) -> "VectorSearchIndex":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
