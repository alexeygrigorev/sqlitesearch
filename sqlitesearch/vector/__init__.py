"""
Vector search module with pluggable strategies (LSH, IVF, HNSW).

This module provides persistent vector search with approximate nearest
neighbor search, followed by exact cosine similarity reranking.
"""

from sqlitesearch.vector.index import VectorSearchIndex

__all__ = ["VectorSearchIndex"]
