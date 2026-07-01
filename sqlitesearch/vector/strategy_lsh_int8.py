"""
int8-quantized shortlist cache for LSH strategies.

Instead of holding the full float32 vector matrix in RAM (the default
``mode="lsh"`` cache), we keep an int8-quantized, disk-backed (``np.memmap``)
copy of the *normalized* vectors. LSH bucket candidates are narrowed to a
shortlist by a cheap int8 dot product; the final top-k is an exact float32
cosine rerank over the shortlist's SQLite BLOBs (handled by
``VectorSearchIndex._rerank`` when ``use_sqlite_rerank`` is set).

This is pure numpy with no compiled extension: at 100K x 768d the float32 cache
is ~293 MB while the int8 cache is ~73 MB, and because it is file-backed the OS
page cache (evictable under memory pressure) holds it rather than resident RSS.

The cache machinery lives in ``Int8ShortlistMixin`` so the int8 shortlist
scoring is a single self-contained mix-in layered on ``LSHStrategy``.
"""

import os
import sqlite3
import tempfile

import numpy as np

from sqlitesearch.vector.strategy_lsh import LSHStrategy


class Int8ShortlistMixin:
    """int8 quantized shortlist cache layered on ``LSHStrategy``.

    Subclasses must call ``_init_int8_cache()`` from ``__init__`` and arrange for
    LSH buckets to be inserted (e.g. via ``LSHStrategy._insert_lsh_buckets``)
    before/alongside ``_set_quantized_vectors`` / ``_append_quantized_vectors``.
    """

    # Drop the float32 in-memory vector cache: the int8 cache narrows
    # candidates and VectorSearchIndex._rerank does the exact float32 rerank
    # over the shortlist's SQLite BLOBs.
    use_sqlite_rerank = True

    def _init_int8_cache(self) -> None:
        self._q_vectors: np.ndarray | None = None
        self._q_doc_ids: list[int] | None = None
        self._q_id_to_idx: dict[int, int] | None = None
        self._q_mmap_path: str | None = None

    def rank_candidate_ids(
        self,
        cursor: sqlite3.Cursor,
        query_vector: np.ndarray,
        candidate_ids: set[int],
        num_results: int,
    ) -> list[int]:
        if self._q_vectors is None or self._q_id_to_idx is None:
            self._load_quantized_vectors(cursor)
        if self._q_vectors is None or self._q_id_to_idx is None:
            return list(candidate_ids)

        indexed = [
            (doc_id, self._q_id_to_idx[doc_id])
            for doc_id in candidate_ids
            if doc_id in self._q_id_to_idx
        ]
        if not indexed:
            return []

        doc_ids = [doc_id for doc_id, _ in indexed]
        indices = np.asarray([idx for _, idx in indexed], dtype=np.int64)
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        # Int8 normalized-vector cache gives a cheap shortlist without holding
        # the full float32 vector matrix in memory. Exact reranking still uses
        # SQLite vector BLOBs for the returned shortlist.
        sims = self._q_vectors[indices].astype(np.float32) @ q
        shortlist_size = min(len(doc_ids), max(num_results * 4, 32))
        if shortlist_size < len(doc_ids):
            top = np.argpartition(sims, -shortlist_size)[-shortlist_size:]
            top = top[np.argsort(sims[top])[::-1]]
        else:
            top = np.argsort(sims)[::-1]
        return [doc_ids[int(i)] for i in top]

    def clear_index(self, cursor: sqlite3.Cursor) -> None:
        super().clear_index(cursor)
        self._q_vectors = None
        self._q_doc_ids = None
        self._q_id_to_idx = None
        self._cleanup_quantized_mmap()

    def close(self) -> None:
        self._q_vectors = None
        self._cleanup_quantized_mmap()
        parent_close = getattr(super(), "close", None)
        if callable(parent_close):
            parent_close()

    def _set_quantized_vectors(self, vectors: np.ndarray, doc_ids: list[int]) -> None:
        self._cleanup_quantized_mmap()
        self._q_vectors = self._quantize_to_mmap(vectors)
        self._q_doc_ids = list(doc_ids)
        self._q_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    def _append_quantized_vectors(self, vectors: np.ndarray, doc_ids: list[int]) -> None:
        q_new = self._quantize(vectors)
        if self._q_vectors is None or self._q_doc_ids is None or self._q_id_to_idx is None:
            self._set_quantized_vectors(vectors, doc_ids)
            return
        offset = len(self._q_doc_ids)
        self._q_vectors = np.vstack([self._q_vectors, q_new])
        self._q_doc_ids.extend(doc_ids)
        for i, doc_id in enumerate(doc_ids):
            self._q_id_to_idx[doc_id] = offset + i

    def _load_quantized_vectors(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "SELECT id, vector_hash FROM docs WHERE vector_hash IS NOT NULL ORDER BY id"
        )
        rows = cursor.fetchall()
        if not rows:
            return
        doc_ids = [row["id"] for row in rows]
        vectors = np.stack([np.frombuffer(row["vector_hash"], dtype=np.float32) for row in rows])
        self._set_quantized_vectors(vectors, doc_ids)

    @staticmethod
    def _quantize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = vectors / norms
        return np.clip(np.rint(normalized * 127), -127, 127).astype(np.int8)

    def _quantize_to_mmap(self, vectors: np.ndarray) -> np.memmap:
        vectors = np.asarray(vectors, dtype=np.float32)
        fd, path = tempfile.mkstemp(prefix="sqlitesearch-lsh-int8-", suffix=".qvec")
        os.close(fd)
        out = np.memmap(path, dtype=np.int8, mode="w+", shape=vectors.shape)
        chunk_size = 8192
        for start in range(0, len(vectors), chunk_size):
            chunk = vectors[start : start + chunk_size]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            out[start : start + len(chunk)] = np.clip(
                np.rint((chunk / norms) * 127), -127, 127
            ).astype(np.int8)
        out.flush()
        self._q_mmap_path = path
        return out

    def _cleanup_quantized_mmap(self) -> None:
        path = self._q_mmap_path
        self._q_mmap_path = None
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


class LSHInt8Strategy(Int8ShortlistMixin, LSHStrategy):
    """Pure-numpy LSH with an int8 quantized shortlist cache.

    Hashing uses ``LSHStrategy``'s numpy random projections; the int8 cache and
    exact SQLite-BLOB rerank come from ``Int8ShortlistMixin``. Search flows
    through ``VectorSearchIndex._rerank`` (no ``search_unfiltered`` override),
    which narrows candidates via ``rank_candidate_ids`` then exact-reranks the
    shortlist from the docs table.
    """

    def __init__(
        self,
        n_tables: int = 8,
        hash_size: int = 16,
        n_probe: int = 2,
        seed: int | None = None,
    ):
        super().__init__(n_tables=n_tables, hash_size=hash_size, n_probe=n_probe, seed=seed)
        self._init_int8_cache()

    def build_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        # LSHStrategy._insert_lsh_buckets hashes (numpy) and bulk-inserts buckets.
        self._insert_lsh_buckets(cursor, vectors, doc_ids)
        self._set_quantized_vectors(vectors, doc_ids)

    def add_to_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        # Insert only the new buckets, then append to the int8 cache (don't
        # rebuild it, which would discard the previously quantized vectors).
        self._insert_lsh_buckets(cursor, vectors, doc_ids)
        self._append_quantized_vectors(vectors, doc_ids)
