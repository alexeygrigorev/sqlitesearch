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
from typing import Any, Optional

import numpy as np

from sqlitesearch.operators import OPERATORS, is_range_filter
from sqlitesearch.vector.base import VectorMode
from sqlitesearch.vector.strategy_lsh import LSHStrategy


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
        keyword_fields: Optional[list[str]] = None,
        numeric_fields: Optional[list[str]] = None,
        date_fields: Optional[list[str]] = None,
        id_field: Optional[str] = None,
        db_path: str = "sqlitesearch_vectors.db",
        # LSH params (kept as explicit kwargs for backward compatibility)
        n_tables: int = 8,
        hash_size: int = 16,
        n_probe: int = 0,
        seed: Optional[int] = None,
        # IVF params
        n_clusters: Optional[int] = None,
        n_probe_clusters: int = 4,
        # HNSW params
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        self.keyword_fields = list(keyword_fields) if keyword_fields is not None else []
        self.numeric_fields = list(numeric_fields) if numeric_fields is not None else []
        self.date_fields = list(date_fields) if date_fields is not None else []
        self.id_field = id_field
        self.db_path = db_path
        self._local = threading.local()

        # Expose LSH params as attributes for backward compatibility with tests
        self.n_tables = n_tables
        self.hash_size = hash_size

        # In-memory vector cache for fast reranking
        self._dimension: Optional[int] = None
        self._cached_vectors: Optional[np.ndarray] = None
        self._cached_doc_ids: Optional[list[int]] = None
        self._cached_docs: Optional[list[dict]] = None
        self._id_to_cache_idx: Optional[dict[int, int]] = None

        # Add id_field to keyword_fields if provided and not already there
        if self.id_field and self.id_field not in self.keyword_fields:
            self.keyword_fields.append(self.id_field)

        # Create strategy
        mode_enum = VectorMode(mode)
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

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")
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

        # Strategy-specific tables
        self._strategy.init_tables(cursor)

        conn.commit()

    def _is_empty(self) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM docs")
        row = cursor.fetchone()
        return row["count"] == 0

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
        col_names = ", ".join(all_cols)
        placeholders = ", ".join(["?"] * len(all_cols))

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

            doc_rows.append(
                [doc_json, vector_bytes] + keyword_vals + numeric_vals + date_vals
            )

        # Insert docs and collect IDs
        insert_sql = f"INSERT INTO docs ({col_names}) VALUES ({placeholders})"
        doc_ids = []
        for row in doc_rows:
            cursor.execute(insert_sql, row)
            doc_ids.append(cursor.lastrowid)

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
        filter_dict: Optional[dict[str, Any]] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the index with the given query vector."""
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

        # Step 1: Find candidates via strategy
        candidate_ids = self._strategy.find_candidates(cursor, query_vector)

        if not candidate_ids:
            return []

        # Step 2: Apply filters
        candidate_ids = self._apply_filters(cursor, candidate_ids, filter_dict)

        if not candidate_ids:
            return []

        # Step 3: Exact reranking
        results = self._rerank(cursor, query_vector, candidate_ids, num_results, output_ids)

        return results

    # --- Shared internal methods ---

    @staticmethod
    def _chunked_in_query(cursor, sql_template, ids_list, extra_params=None, chunk_size=900):
        if extra_params is None:
            extra_params = []
        all_rows = []
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            sql = sql_template.format(placeholders=placeholders)
            cursor.execute(sql, chunk + extra_params)
            all_rows.extend(cursor.fetchall())
        return all_rows

    def _apply_filters(
        self,
        cursor: sqlite3.Cursor,
        candidate_ids: set[int],
        filter_dict: dict[str, Any],
    ) -> set[int]:
        if not filter_dict:
            return candidate_ids

        filtered_ids = candidate_ids.copy()
        ids_list = list(candidate_ids)

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                if value is None:
                    rows = self._chunked_in_query(
                        cursor,
                        f'SELECT id FROM docs WHERE id IN ({{placeholders}}) '
                        f'AND "{field}" IS NULL',
                        ids_list,
                    )
                else:
                    rows = self._chunked_in_query(
                        cursor,
                        f'SELECT id FROM docs WHERE id IN ({{placeholders}}) '
                        f'AND "{field}" = ?',
                        ids_list,
                        [value],
                    )
                valid_ids = set(row["id"] for row in rows)
                filtered_ids &= valid_ids
            elif field in self.numeric_fields:
                filtered_ids = self._apply_numeric_filter(cursor, field, value, filtered_ids)
            elif field in self.date_fields:
                filtered_ids = self._apply_date_filter(cursor, field, value, filtered_ids)

        return filtered_ids

    def _apply_numeric_filter(self, cursor, field, value, candidate_ids):
        if not candidate_ids:
            return candidate_ids
        ids_list = list(candidate_ids)

        if value is None:
            rows = self._chunked_in_query(
                cursor,
                f'SELECT id FROM docs WHERE id IN ({{placeholders}}) AND "{field}" IS NULL',
                ids_list,
            )
        elif is_range_filter(value):
            where_conditions = []
            extra_params = []
            for op, op_value in value:
                if op in OPERATORS and op_value is not None:
                    where_conditions.append(f'"{field}" {op} ?')
                    extra_params.append(op_value)
            if where_conditions:
                where_sql = " AND " + " AND ".join(where_conditions)
                rows = self._chunked_in_query(
                    cursor,
                    f'SELECT id FROM docs WHERE id IN ({{placeholders}}){where_sql}',
                    ids_list,
                    extra_params,
                )
            else:
                rows = self._chunked_in_query(
                    cursor,
                    f"SELECT id FROM docs WHERE id IN ({{placeholders}})",
                    ids_list,
                )
        else:
            rows = self._chunked_in_query(
                cursor,
                f'SELECT id FROM docs WHERE id IN ({{placeholders}}) AND "{field}" = ?',
                ids_list,
                [value],
            )
        return set(row["id"] for row in rows) & candidate_ids

    def _apply_date_filter(self, cursor, field, value, candidate_ids):
        if not candidate_ids:
            return candidate_ids
        ids_list = list(candidate_ids)

        if value is None:
            rows = self._chunked_in_query(
                cursor,
                f'SELECT id FROM docs WHERE id IN ({{placeholders}}) AND "{field}" IS NULL',
                ids_list,
            )
        elif is_range_filter(value):
            where_conditions = []
            extra_params = []
            for op, op_value in value:
                if op in OPERATORS and op_value is not None:
                    if isinstance(op_value, (date, datetime)):
                        op_value = op_value.isoformat()
                    where_conditions.append(f'"{field}" {op} ?')
                    extra_params.append(op_value)
            if where_conditions:
                where_sql = " AND " + " AND ".join(where_conditions)
                rows = self._chunked_in_query(
                    cursor,
                    f'SELECT id FROM docs WHERE id IN ({{placeholders}}){where_sql}',
                    ids_list,
                    extra_params,
                )
            else:
                rows = self._chunked_in_query(
                    cursor,
                    f"SELECT id FROM docs WHERE id IN ({{placeholders}})",
                    ids_list,
                )
        else:
            if isinstance(value, (date, datetime)):
                value = value.isoformat()
            rows = self._chunked_in_query(
                cursor,
                f'SELECT id FROM docs WHERE id IN ({{placeholders}}) AND "{field}" = ?',
                ids_list,
                [value],
            )
        return set(row["id"] for row in rows) & candidate_ids

    def _rerank(self, cursor, query_vector, candidate_ids, num_results, output_ids):
        if not candidate_ids:
            return []

        if self._cached_vectors is not None and self._id_to_cache_idx is not None:
            cache_indices = [
                self._id_to_cache_idx[did]
                for did in candidate_ids
                if did in self._id_to_cache_idx
            ]
            if not cache_indices:
                return []

            cache_indices = np.array(cache_indices)
            candidate_matrix = self._cached_vectors[cache_indices]
        else:
            ids_list = list(candidate_ids)
            rows = self._chunked_in_query(
                cursor,
                "SELECT id, doc_json, vector_hash FROM docs WHERE id IN ({placeholders})",
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
        cursor.execute("SELECT id, doc_json, vector_hash FROM docs ORDER BY id")
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
