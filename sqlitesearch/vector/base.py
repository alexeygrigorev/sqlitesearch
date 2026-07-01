"""
Base protocol for vector search strategies.
"""

import sqlite3
from enum import Enum
from typing import Protocol

import numpy as np


class VectorMode(str, Enum):
    LSH = "lsh"
    LSH_INT8 = "lsh_int8"
    IVF = "ivf"
    HNSW = "hnsw"


class SearchStrategy(Protocol):
    """Protocol that all vector search strategies must implement."""

    def init_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create strategy-specific database tables."""
        ...

    def save_params(self, cursor: sqlite3.Cursor) -> None:
        """Save strategy parameters to the metadata table."""
        ...

    def load_params(self, cursor: sqlite3.Cursor) -> bool:
        """Load strategy parameters from the metadata table. Returns True if loaded."""
        ...

    def build_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Build the index from scratch (called during fit)."""
        ...

    def add_to_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Add vectors to the existing index (called during add)."""
        ...

    def find_candidates(
        self,
        cursor: sqlite3.Cursor,
        query_vector: np.ndarray,
        *,
        override: dict[str, int] | None = None,
        filter_ids: set[int] | None = None,
    ) -> set[int]:
        """Find candidate document IDs for a query vector.

        ``override`` temporarily replaces strategy parameters (``ef_search``,
        ``n_probe_clusters``, ``n_probe``) for this call only — used by the
        filtered-ANN branch to widen the candidate budget without mutating the
        shared strategy object (strategies are shared across threads).
        ``filter_ids`` restricts results to an allowed id set; the HNSW strategy
        honors it graph-aware (skipping non-matching subtrees in a single
        walk), while LSH/IVF have no graph and ignore it (they post-filter).
        """
        ...

    def clear_index(self, cursor: sqlite3.Cursor) -> None:
        """Clear all strategy-specific data."""
        ...
