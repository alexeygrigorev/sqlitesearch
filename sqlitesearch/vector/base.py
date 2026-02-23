"""
Base protocol for vector search strategies.
"""

import sqlite3
from enum import Enum
from typing import Protocol

import numpy as np


class VectorMode(str, Enum):
    LSH = "lsh"
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

    def find_candidates(self, cursor: sqlite3.Cursor, query_vector: np.ndarray) -> set[int]:
        """Find candidate document IDs for a query vector."""
        ...

    def clear_index(self, cursor: sqlite3.Cursor) -> None:
        """Clear all strategy-specific data."""
        ...
