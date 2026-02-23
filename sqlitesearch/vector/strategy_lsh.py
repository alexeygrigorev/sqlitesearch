"""
LSH (Locality-Sensitive Hashing) search strategy.

Uses random projections for approximate nearest neighbor search.
"""

import pickle
import sqlite3
from typing import Optional

import numpy as np


class LSHStrategy:
    """LSH search strategy using random projections."""

    def __init__(
        self,
        n_tables: int = 8,
        hash_size: int = 16,
        n_probe: int = 0,
        seed: Optional[int] = None,
    ):
        self.n_tables = n_tables
        self.hash_size = hash_size
        self.n_probe = n_probe
        self._seed = seed

        self._dimension: Optional[int] = None
        self._random_vectors: Optional[np.ndarray] = None  # (n_tables, hash_size, dim)
        self._random_vectors_flat: Optional[np.ndarray] = None  # (n_tables*hash_size, dim)

    def init_tables(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lsh_buckets (
                table_id INTEGER NOT NULL,
                hash_key TEXT NOT NULL,
                doc_id INTEGER NOT NULL,
                PRIMARY KEY (table_id, hash_key, doc_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_lsh_lookup "
            "ON lsh_buckets (table_id, hash_key)"
        )

    def save_params(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("random_vectors", pickle.dumps(self._random_vectors)),
        )

    def load_params(self, cursor: sqlite3.Cursor) -> bool:
        cursor.execute("SELECT value FROM metadata WHERE key = 'random_vectors'")
        row = cursor.fetchone()
        if row:
            self._random_vectors = pickle.loads(row["value"])
            self._dimension = self._random_vectors.shape[2]
            self._random_vectors_flat = self._random_vectors.reshape(
                self.n_tables * self.hash_size, self._dimension
            )
            return True
        return False

    def set_dimension(self, dimension: int) -> None:
        """Initialize random projections for the given dimension."""
        self._dimension = dimension
        self._generate_random_vectors(dimension)

    def build_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        all_signs = self._hash_vectors_batch(vectors)
        lsh_rows = []
        for i, doc_id in enumerate(doc_ids):
            for table_id in range(self.n_tables):
                hash_key = self._signs_to_hash_str(all_signs[i, table_id])
                lsh_rows.append((table_id, hash_key, doc_id))
        cursor.executemany(
            "INSERT INTO lsh_buckets (table_id, hash_key, doc_id) VALUES (?, ?, ?)",
            lsh_rows,
        )

    def add_to_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        # Same as build_index for LSH
        self.build_index(cursor, vectors, doc_ids)

    def find_candidates(self, cursor: sqlite3.Cursor, query_vector: np.ndarray) -> set[int]:
        hash_keys = self._hash_vector_all_tables(query_vector)

        hit_counts: dict[int, int] = {}
        for table_id, exact_key in enumerate(hash_keys):
            probe_keys = self._generate_probe_keys(exact_key, self.n_probe)
            placeholders = ",".join("?" * len(probe_keys))
            cursor.execute(
                f"SELECT doc_id FROM lsh_buckets "
                f"WHERE table_id = ? AND hash_key IN ({placeholders})",
                [table_id] + probe_keys,
            )
            for row in cursor.fetchall():
                hit_counts[row["doc_id"]] = hit_counts.get(row["doc_id"], 0) + 1

        if len(hit_counts) <= 50000:
            return set(hit_counts.keys())

        sorted_candidates = sorted(hit_counts.items(), key=lambda x: x[1], reverse=True)
        return {doc_id for doc_id, _ in sorted_candidates[:50000]}

    def clear_index(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("DELETE FROM lsh_buckets")
        self._dimension = None
        self._random_vectors = None
        self._random_vectors_flat = None

    # --- Internal LSH methods ---

    def _generate_random_vectors(self, dimension: int) -> None:
        rng = np.random.default_rng(self._seed)
        self._random_vectors = rng.standard_normal(
            size=(self.n_tables, self.hash_size, dimension)
        ).astype(np.float32)
        self._random_vectors_flat = self._random_vectors.reshape(
            self.n_tables * self.hash_size, dimension
        )

    def _hash_vector_all_tables(self, vector: np.ndarray) -> list[str]:
        projections = self._random_vectors_flat @ vector
        projections = projections.reshape(self.n_tables, self.hash_size)
        signs = (projections > 0).astype(np.uint8)
        return ["".join(str(b) for b in row) for row in signs]

    def _hash_vectors_batch(self, vectors: np.ndarray) -> np.ndarray:
        projections = self._random_vectors_flat @ vectors.T
        projections = projections.reshape(self.n_tables, self.hash_size, len(vectors))
        return (projections > 0).transpose(2, 0, 1)

    @staticmethod
    def _signs_to_hash_str(signs_row: np.ndarray) -> str:
        return "".join(str(int(b)) for b in signs_row)

    @staticmethod
    def _generate_probe_keys(exact_hash: str, n_probe: int) -> list[str]:
        keys = [exact_hash]
        if n_probe <= 0:
            return keys
        n = len(exact_hash)
        chars = list(exact_hash)
        flip = {"0": "1", "1": "0"}
        for i in range(n):
            flipped = chars.copy()
            flipped[i] = flip[flipped[i]]
            keys.append("".join(flipped))
        if n_probe >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    flipped = chars.copy()
                    flipped[i] = flip[flipped[i]]
                    flipped[j] = flip[flipped[j]]
                    keys.append("".join(flipped))
        return keys
