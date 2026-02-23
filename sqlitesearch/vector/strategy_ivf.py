"""
IVF (Inverted File Index) search strategy.

Clusters vectors using k-means, then at query time searches only the nearest clusters.
"""

import math
import pickle
import sqlite3
from typing import Optional

import numpy as np


class IVFStrategy:
    """IVF search strategy using k-means clustering."""

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        n_probe_clusters: int = 4,
        seed: Optional[int] = None,
    ):
        self._n_clusters_param = n_clusters  # None = auto-scale
        self.n_clusters: Optional[int] = n_clusters
        self.n_probe_clusters = n_probe_clusters
        self._seed = seed

        self._dimension: Optional[int] = None
        self._centroids: Optional[np.ndarray] = None  # (n_clusters, dim), normalized

    def init_tables(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ivf_centroids (
                cluster_id INTEGER PRIMARY KEY,
                centroid BLOB NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ivf_assignments (
                doc_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                PRIMARY KEY (doc_id, cluster_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ivf_cluster "
            "ON ivf_assignments (cluster_id)"
        )

    def save_params(self, cursor: sqlite3.Cursor) -> None:
        if self._centroids is not None:
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("ivf_centroids", pickle.dumps(self._centroids)),
            )
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("ivf_n_clusters", pickle.dumps(self.n_clusters)),
            )

    def load_params(self, cursor: sqlite3.Cursor) -> bool:
        cursor.execute("SELECT value FROM metadata WHERE key = 'ivf_centroids'")
        row = cursor.fetchone()
        if row:
            self._centroids = pickle.loads(row["value"])
            self._dimension = self._centroids.shape[1]
            cursor.execute("SELECT value FROM metadata WHERE key = 'ivf_n_clusters'")
            row2 = cursor.fetchone()
            if row2:
                self.n_clusters = pickle.loads(row2["value"])
            return True
        return False

    def set_dimension(self, dimension: int) -> None:
        self._dimension = dimension

    def build_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Run k-means and build the inverted file index."""
        n = len(vectors)

        # Auto-scale clusters
        if self._n_clusters_param is None:
            self.n_clusters = min(int(math.sqrt(n)), 256)
            self.n_clusters = max(self.n_clusters, 1)
        else:
            self.n_clusters = min(self._n_clusters_param, n)

        # Normalize vectors for cosine similarity
        normed = self._normalize(vectors)

        # Run k-means
        self._centroids = self._kmeans(normed, self.n_clusters)

        # Store centroids in DB
        for cid in range(len(self._centroids)):
            cursor.execute(
                "INSERT INTO ivf_centroids (cluster_id, centroid) VALUES (?, ?)",
                (cid, self._centroids[cid].tobytes()),
            )

        # Assign vectors to nearest centroid
        assignments = self._assign(normed)  # (n,) array of cluster IDs
        rows = [(int(doc_ids[i]), int(assignments[i])) for i in range(n)]
        cursor.executemany(
            "INSERT INTO ivf_assignments (doc_id, cluster_id) VALUES (?, ?)",
            rows,
        )

        # Save params
        self.save_params(cursor)

    def add_to_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Assign new vectors to nearest existing centroid."""
        if self._centroids is None:
            # First add — treat as build
            self.build_index(cursor, vectors, doc_ids)
            return

        normed = self._normalize(vectors)
        assignments = self._assign(normed)
        rows = [(int(doc_ids[i]), int(assignments[i])) for i in range(len(vectors))]
        cursor.executemany(
            "INSERT INTO ivf_assignments (doc_id, cluster_id) VALUES (?, ?)",
            rows,
        )

    def find_candidates(self, cursor: sqlite3.Cursor, query_vector: np.ndarray) -> set[int]:
        """Find candidates by probing nearest clusters."""
        if self._centroids is None:
            return set()

        query_normed = query_vector / (np.linalg.norm(query_vector) + 1e-10)

        # Cosine similarity to all centroids
        sims = self._centroids @ query_normed
        n_probe = min(self.n_probe_clusters, len(self._centroids))
        top_clusters = np.argsort(sims)[-n_probe:][::-1]

        # Fetch doc_ids from those clusters
        candidate_ids: set[int] = set()
        for cid in top_clusters:
            cursor.execute(
                "SELECT doc_id FROM ivf_assignments WHERE cluster_id = ?",
                (int(cid),),
            )
            for row in cursor.fetchall():
                candidate_ids.add(row["doc_id"])

        return candidate_ids

    def clear_index(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("DELETE FROM ivf_centroids")
        cursor.execute("DELETE FROM ivf_assignments")
        self._dimension = None
        self._centroids = None

    # --- K-means implementation ---

    def _kmeans(self, vectors: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
        """K-means with k-means++ init on normalized vectors (cosine similarity)."""
        n = len(vectors)
        rng = np.random.default_rng(self._seed)

        if k >= n:
            # More clusters than vectors — each vector is its own centroid
            return vectors.copy()

        # K-means++ initialization
        centroids = np.empty((k, vectors.shape[1]), dtype=np.float32)
        idx = rng.integers(n)
        centroids[0] = vectors[idx]

        for c in range(1, k):
            # Squared distance from nearest centroid (using 1 - cosine as distance)
            sims = vectors @ centroids[:c].T  # (n, c)
            max_sim = sims.max(axis=1)  # (n,)
            dists = 1.0 - max_sim
            dists = np.maximum(dists, 0.0)
            probs = dists / (dists.sum() + 1e-10)
            idx = rng.choice(n, p=probs)
            centroids[c] = vectors[idx]

        # Iterate
        for _ in range(max_iter):
            # Assign
            sims = vectors @ centroids.T  # (n, k)
            labels = sims.argmax(axis=1)  # (n,)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    cluster_mean = vectors[mask].mean(axis=0)
                    norm = np.linalg.norm(cluster_mean)
                    if norm > 0:
                        new_centroids[c] = cluster_mean / norm
                    else:
                        new_centroids[c] = centroids[c]
                else:
                    new_centroids[c] = centroids[c]

            # Check convergence
            movement = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if movement < tol:
                break

        return centroids

    def _assign(self, normed_vectors: np.ndarray) -> np.ndarray:
        """Assign normalized vectors to nearest centroid. Returns array of cluster IDs."""
        sims = normed_vectors @ self._centroids.T  # (n, k)
        return sims.argmax(axis=1)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
