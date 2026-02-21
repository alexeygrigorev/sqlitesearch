"""
VectorSearchIndex - Persistent vector search using LSH with exact reranking.

This module provides approximate nearest neighbor search using Locality-Sensitive
Hashing (LSH) with random projections, followed by exact cosine similarity reranking.
"""

import json
import pickle
import sqlite3
import threading
from datetime import date, datetime
from typing import Any, Optional

import numpy as np

from sqlitesearch.operators import OPERATORS, is_range_filter


class VectorSearchIndex:
    """
    A persistent vector search index using LSH with exact reranking.

    Uses random projections for LSH to find candidate matches, then reranks
    using exact cosine similarity. This provides a good balance of speed and
    accuracy for small to medium datasets.

    API:
    - __init__(keyword_fields=None, numeric_fields=None, date_fields=None, id_field=None, n_tables=8, hash_size=16)
    - fit(vectors, payload) - Index vectors (only if index is empty)
    - add(vector, doc) - Add a single vector with document
    - search(query_vector, filter_dict=None, num_results=10, output_ids=False)

    Example:
        >>> import numpy as np
        >>> index = VectorSearchIndex(
        ...     keyword_fields=["category"],
        ...     numeric_fields=["price"],
        ...     date_fields=["created_at"],
        ...     id_field="doc_id",
        ...     db_path="vectors.db"
        ... )
        >>> vectors = np.random.rand(100, 384)
        >>> payload = [{"doc_id": i, "category": "test", "price": 100} for i in range(100)]
        >>> index.fit(vectors, payload)
        >>> query = np.random.rand(384)
        >>> results = index.search(query, filter_dict={"price": [('>=', 50)]})
    """

    def __init__(
        self,
        keyword_fields: Optional[list[str]] = None,
        numeric_fields: Optional[list[str]] = None,
        date_fields: Optional[list[str]] = None,
        id_field: Optional[str] = None,
        n_tables: int = 8,
        hash_size: int = 16,
        db_path: str = "sqlitesearch_vectors.db",
    ):
        """
        Initialize the VectorSearchIndex.

        Args:
            keyword_fields: List of field names for exact filtering.
            numeric_fields: List of field names for numeric range filtering.
            date_fields: List of field names for date range filtering.
            id_field: Field name to use as document ID. If None, auto-generates IDs.
            n_tables: Number of LSH hash tables (more = better recall, slower).
            hash_size: Number of bits per hash (more = better precision, slower).
            db_path: Path to the SQLite database file.
        """
        self.keyword_fields = list(keyword_fields) if keyword_fields is not None else []
        self.numeric_fields = list(numeric_fields) if numeric_fields is not None else []
        self.date_fields = list(date_fields) if date_fields is not None else []
        self.id_field = id_field
        self.n_tables = n_tables
        self.hash_size = hash_size
        self.db_path = db_path
        self._local = threading.local()

        # LSH parameters (will be initialized during fit)
        self._dimension = None
        self._random_vectors = None  # Shape: (n_tables, hash_size, dimension)

        # Add id_field to keyword_fields if provided and not already there
        if self.id_field and self.id_field not in self.keyword_fields:
            self.keyword_fields.append(self.id_field)

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

        # Build keyword column definitions
        keyword_cols = []
        for field in self.keyword_fields:
            keyword_cols.append(f', "{field}" TEXT')
        keyword_sql = "\n".join(keyword_cols)

        # Build numeric column definitions
        numeric_cols = []
        for field in self.numeric_fields:
            numeric_cols.append(f', "{field}" REAL')
        numeric_sql = "\n".join(numeric_cols)

        # Build date column definitions (store as ISO 8601 strings for comparison)
        date_cols = []
        for field in self.date_fields:
            date_cols.append(f', "{field}" TEXT')
        date_sql = "\n".join(date_cols)

        # Main documents table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_json TEXT NOT NULL,
                vector_hash BLOB{keyword_sql}{numeric_sql}{date_sql}
            )
        """)

        # LSH hash buckets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lsh_buckets (
                table_id INTEGER NOT NULL,
                hash_key TEXT NOT NULL,
                doc_id INTEGER NOT NULL,
                PRIMARY KEY (table_id, hash_key, doc_id)
            )
        """)

        # Metadata table for LSH parameters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lsh_lookup ON lsh_buckets (table_id, hash_key)")
        for field in self.keyword_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_vec_{field} ON docs ("{field}")')
        for field in self.numeric_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_vec_num_{field} ON docs ("{field}")')
        for field in self.date_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_vec_date_{field} ON docs ("{field}")')

        conn.commit()

    def _is_empty(self) -> bool:
        """Check if the index is empty."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM docs")
        row = cursor.fetchone()
        return row["count"] == 0

    def _generate_random_vectors(self, dimension: int) -> None:
        """
        Generate random projection vectors for LSH.

        Uses random projections from a Gaussian distribution.
        For cosine similarity, we can use random projections and hash based on sign.
        """
        rng = np.random.default_rng()
        self._random_vectors = rng.standard_normal(
            size=(self.n_tables, self.hash_size, dimension)
        ).astype(np.float32)

    def _hash_vector(self, vector: np.ndarray, table_id: int) -> str:
        """
        Compute LSH hash for a vector.

        Uses random projection + sign hashing.
        For cosine similarity: hash = sign(random_projection @ vector)

        Args:
            vector: 1D numpy array of shape (dimension,).
            table_id: Which hash table to use.

        Returns:
            Hash string (binary as hex).
        """
        projection = self._random_vectors[table_id] @ vector
        # Convert to binary hash based on sign
        binary_hash = (projection > 0).astype(np.uint8)
        # Convert to hex string for storage
        return "".join(str(b) for b in binary_hash)

    def fit(
        self,
        vectors: np.ndarray,
        payload: list[dict[str, Any]],
    ) -> "VectorSearchIndex":
        """
        Index the provided vectors with payload documents.

        Only works if the index is empty. Use add() to append documents.

        Args:
            vectors: 2D numpy array of shape (n_docs, dimension).
            payload: List of documents as payload (same length as vectors).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If the index already contains documents.
        """
        if not self._is_empty():
            raise ValueError(
                "Index already contains documents. "
                "Use clear() to reset the index or add() to append documents."
            )

        return self._add_vectors(vectors, payload)

    def add(
        self,
        vector: np.ndarray,
        doc: dict[str, Any],
    ) -> "VectorSearchIndex":
        """
        Add a single vector with document to the index.

        Args:
            vector: 1D numpy array.
            doc: Document to associate with this vector.

        Returns:
            self for method chaining.
        """
        vectors = np.asarray(vector, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        return self._add_vectors(vectors, [doc])

    def _add_vectors(
        self,
        vectors: np.ndarray,
        payload: list[dict[str, Any]],
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

        # Initialize LSH parameters if first time
        if self._dimension is None:
            self._dimension = vectors.shape[1]
            self._generate_random_vectors(self._dimension)

            # Store LSH parameters
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("dimension", pickle.dumps(self._dimension))
            )
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("random_vectors", pickle.dumps(self._random_vectors))
            )
            conn.commit()

        conn = self._get_conn()
        cursor = conn.cursor()

        # Build column lists including keyword, numeric, and date fields
        filter_cols = (
            [f'"{field}"' for field in self.keyword_fields] +
            [f'"{field}"' for field in self.numeric_fields] +
            [f'"{field}"' for field in self.date_fields]
        )
        all_cols = ["doc_json", "vector_hash"] + filter_cols
        col_names = ", ".join(all_cols)
        placeholders = ", ".join(["?"] * len(all_cols))

        # Insert documents and LSH buckets
        for i, (vector, doc) in enumerate(zip(vectors, payload)):
            # Convert date/datetime objects to ISO format for JSON serialization
            doc_for_json = {}
            for key, value in doc.items():
                if isinstance(value, (date, datetime)):
                    doc_for_json[key] = value.isoformat()
                else:
                    doc_for_json[key] = value
            doc_json = json.dumps(doc_for_json)
            vector_bytes = pickle.dumps(vector)

            keyword_vals = [doc.get(field) for field in self.keyword_fields]

            # Extract numeric values
            numeric_vals = [doc.get(field) for field in self.numeric_fields]

            # Extract date values and convert to ISO format
            date_vals = []
            for field in self.date_fields:
                value = doc.get(field)
                if isinstance(value, (date, datetime)):
                    date_vals.append(value.isoformat())
                else:
                    date_vals.append(value)

            # Insert into docs table
            cursor.execute(
                f"INSERT INTO docs ({col_names}) VALUES ({placeholders})",
                [doc_json, vector_bytes] + keyword_vals + numeric_vals + date_vals
            )
            doc_id = cursor.lastrowid

            # Insert into LSH buckets for each table
            for table_id in range(self.n_tables):
                hash_key = self._hash_vector(vector, table_id)
                cursor.execute(
                    "INSERT INTO lsh_buckets (table_id, hash_key, doc_id) VALUES (?, ?, ?)",
                    (table_id, hash_key, doc_id)
                )

        conn.commit()
        return self

    def clear(self) -> "VectorSearchIndex":
        """
        Clear all documents from the index.

        Returns:
            self for method chaining.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM docs")
        cursor.execute("DELETE FROM lsh_buckets")
        cursor.execute("DELETE FROM metadata")

        self._dimension = None
        self._random_vectors = None

        conn.commit()
        return self

    def search(
        self,
        query_vector: np.ndarray,
        filter_dict: Optional[dict[str, Any]] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search the index with the given query vector.

        Args:
            query_vector: 1D numpy array of shape (dimension,).
            filter_dict: Dictionary of keyword fields to filter by.
            num_results: Maximum number of results to return.
            output_ids: If True, adds an 'id' field with the document ID.

        Returns:
            List of documents matching the search criteria, ranked by cosine similarity.
        """
        if filter_dict is None:
            filter_dict = {}

        query_vector = np.asarray(query_vector, dtype=np.float32).flatten()

        if self._dimension is None:
            # Try to load from metadata
            self._load_metadata()

        # If still None, index is empty
        if self._dimension is None:
            return []

        if query_vector.shape[0] != self._dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[0]} "
                f"does not match index dimension {self._dimension}"
            )

        conn = self._get_conn()
        cursor = conn.cursor()

        # Step 1: Find candidates using LSH
        candidate_ids = self._find_candidates(cursor, query_vector)

        if not candidate_ids:
            return []

        # Step 2: Apply keyword filters
        candidate_ids = self._apply_filters(cursor, candidate_ids, filter_dict)

        if not candidate_ids:
            return []

        # Step 3: Exact reranking with cosine similarity
        results = self._rerank(cursor, query_vector, candidate_ids, num_results, output_ids)

        return results

    def _find_candidates(self, cursor: sqlite3.Cursor, query_vector: np.ndarray) -> set[int]:
        """Find candidate document IDs using LSH."""
        candidate_ids = set()

        for table_id in range(self.n_tables):
            hash_key = self._hash_vector(query_vector, table_id)
            cursor.execute(
                "SELECT DISTINCT doc_id FROM lsh_buckets WHERE table_id = ? AND hash_key = ?",
                (table_id, hash_key)
            )
            candidate_ids.update(row["doc_id"] for row in cursor.fetchall())

        return candidate_ids

    def _apply_filters(
        self,
        cursor: sqlite3.Cursor,
        candidate_ids: set[int],
        filter_dict: dict[str, Any],
    ) -> set[int]:
        """Apply keyword, numeric, and date filters to candidate IDs."""
        if not filter_dict:
            return candidate_ids

        filtered_ids = candidate_ids.copy()
        ids_list = list(candidate_ids)

        for field, value in filter_dict.items():
            # Keyword field filters
            if field in self.keyword_fields:
                placeholders = ",".join("?" * len(ids_list))
                if value is None:
                    cursor.execute(
                        f'SELECT id FROM docs WHERE id IN ({placeholders}) '
                        f'AND "{field}" IS NULL',
                        ids_list
                    )
                else:
                    cursor.execute(
                        f'SELECT id FROM docs WHERE id IN ({placeholders}) '
                        f'AND "{field}" = ?',
                        ids_list + [value]
                    )
                valid_ids = set(row["id"] for row in cursor.fetchall())
                filtered_ids &= valid_ids

            # Numeric field filters
            elif field in self.numeric_fields:
                filtered_ids = self._apply_numeric_filter(
                    cursor, field, value, filtered_ids
                )

            # Date field filters
            elif field in self.date_fields:
                filtered_ids = self._apply_date_filter(
                    cursor, field, value, filtered_ids
                )

        return filtered_ids

    def _apply_numeric_filter(
        self,
        cursor: sqlite3.Cursor,
        field: str,
        value: Any,
        candidate_ids: set[int],
    ) -> set[int]:
        """Apply a numeric filter to candidate IDs."""
        if not candidate_ids:
            return candidate_ids

        ids_list = list(candidate_ids)
        placeholders = ",".join("?" * len(ids_list))

        if value is None:
            cursor.execute(
                f'SELECT id FROM docs WHERE id IN ({placeholders}) '
                f'AND "{field}" IS NULL',
                ids_list
            )
        elif is_range_filter(value):
            # Range filter: [('>=', 100), ('<', 200)]
            where_conditions = []
            params = ids_list.copy()
            for op, op_value in value:
                if op in OPERATORS and op_value is not None:
                    where_conditions.append(f'"{field}" {op} ?')
                    params.append(op_value)
            if where_conditions:
                where_sql = " AND " + " AND ".join(where_conditions)
                cursor.execute(
                    f'SELECT id FROM docs WHERE id IN ({placeholders}){where_sql}',
                    params
                )
            else:
                # No valid conditions, return all candidates
                cursor.execute(
                    f'SELECT id FROM docs WHERE id IN ({placeholders})',
                    ids_list
                )
        else:
            # Exact match
            cursor.execute(
                f'SELECT id FROM docs WHERE id IN ({placeholders}) '
                f'AND "{field}" = ?',
                ids_list + [value]
            )

        return set(row["id"] for row in cursor.fetchall()) & candidate_ids

    def _apply_date_filter(
        self,
        cursor: sqlite3.Cursor,
        field: str,
        value: Any,
        candidate_ids: set[int],
    ) -> set[int]:
        """Apply a date filter to candidate IDs."""
        if not candidate_ids:
            return candidate_ids

        ids_list = list(candidate_ids)
        placeholders = ",".join("?" * len(ids_list))

        if value is None:
            cursor.execute(
                f'SELECT id FROM docs WHERE id IN ({placeholders}) '
                f'AND "{field}" IS NULL',
                ids_list
            )
        elif is_range_filter(value):
            # Range filter: [('>=', date(...)), ('<', date(...))]
            where_conditions = []
            params = ids_list.copy()
            for op, op_value in value:
                if op in OPERATORS and op_value is not None:
                    # Convert date/datetime to ISO format for comparison
                    if isinstance(op_value, (date, datetime)):
                        op_value = op_value.isoformat()
                    where_conditions.append(f'"{field}" {op} ?')
                    params.append(op_value)
            if where_conditions:
                where_sql = " AND " + " AND ".join(where_conditions)
                cursor.execute(
                    f'SELECT id FROM docs WHERE id IN ({placeholders}){where_sql}',
                    params
                )
            else:
                # No valid conditions, return all candidates
                cursor.execute(
                    f'SELECT id FROM docs WHERE id IN ({placeholders})',
                    ids_list
                )
        else:
            # Exact match - convert date/datetime to ISO format
            if isinstance(value, (date, datetime)):
                value = value.isoformat()
            cursor.execute(
                f'SELECT id FROM docs WHERE id IN ({placeholders}) '
                f'AND "{field}" = ?',
                ids_list + [value]
            )

        return set(row["id"] for row in cursor.fetchall()) & candidate_ids

    def _rerank(
        self,
        cursor: sqlite3.Cursor,
        query_vector: np.ndarray,
        candidate_ids: set[int],
        num_results: int,
        output_ids: bool,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using exact cosine similarity."""
        if not candidate_ids:
            return []

        # Fetch all candidate vectors and documents
        ids_list = list(candidate_ids)
        placeholders = ",".join("?" * len(ids_list))
        cursor.execute(
            f'SELECT id, doc_json, vector_hash FROM docs '
            f'WHERE id IN ({placeholders})',
            ids_list
        )

        candidates = []
        for row in cursor.fetchall():
            vector = pickle.loads(row["vector_hash"])
            doc = json.loads(row["doc_json"])
            # Convert ISO date strings back to date/datetime objects
            doc = self._convert_dates(doc)
            candidates.append((row["id"], vector, doc))

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_normalized = query_vector
        else:
            query_normalized = query_vector / query_norm

        scored_candidates = []
        for doc_id, vector, doc in candidates:
            vec_norm = np.linalg.norm(vector)
            if vec_norm == 0:
                similarity = 0.0
            else:
                vec_normalized = vector / vec_norm
                similarity = float(np.dot(query_normalized, vec_normalized))

            scored_candidates.append((doc_id, doc, similarity))

        # Sort by similarity (descending)
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        # Take top results
        results = []
        for doc_id, doc, score in scored_candidates[:num_results]:
            if score > 0:  # Only return results with positive similarity
                if output_ids:
                    # Use id_field value if available, otherwise use database id
                    result_id = doc.get(self.id_field, doc_id) if self.id_field else doc_id
                    doc = {**doc, "_id": result_id}
                results.append(doc)

        return results

    def _load_metadata(self) -> None:
        """Load LSH parameters from database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM metadata WHERE key = 'dimension'")
        row = cursor.fetchone()
        if row:
            self._dimension = pickle.loads(row["value"])

        cursor.execute("SELECT value FROM metadata WHERE key = 'random_vectors'")
        row = cursor.fetchone()
        if row:
            self._random_vectors = pickle.loads(row["value"])

    def _convert_dates(self, doc: dict[str, Any]) -> dict[str, Any]:
        """
        Convert ISO date strings back to date/datetime objects for date_fields.

        Args:
            doc: Document with potentially ISO formatted date strings.

        Returns:
            Document with date fields converted back to date/datetime objects.
        """
        if not self.date_fields:
            return doc

        for field in self.date_fields:
            if field in doc and doc[field] is not None:
                value = doc[field]
                if isinstance(value, str):
                    # Check if string contains time component (has 'T' or ' ')
                    has_time = 'T' in value or ' ' in value

                    if has_time:
                        # Parse as datetime
                        try:
                            doc[field] = datetime.fromisoformat(value)
                        except ValueError:
                            pass
                    else:
                        # Parse as date only
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
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
