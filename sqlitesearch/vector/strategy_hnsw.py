"""
HNSW (Hierarchical Navigable Small World) search strategy.

Builds a multi-layer proximity graph for fast approximate nearest neighbor search.
Graph is stored using vector-array indices internally (not doc_ids) to avoid
dict lookups in the hot path.
"""

import heapq
import math
import pickle
import sqlite3
from typing import Optional

import numpy as np


class HNSWStrategy:
    """HNSW search strategy using a hierarchical proximity graph."""

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: Optional[int] = None,
    ):
        self.m = m
        self.m_max0 = m * 2  # max connections at layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._seed = seed

        self._dimension: Optional[int] = None
        # Entry point and graph stored as vector-array indices (not doc_ids)
        self._entry_point: Optional[int] = None  # index into _vectors
        self._max_layer: int = 0
        self._ml = 1.0 / math.log(m) if m > 1 else 1.0

        # Graph: layer -> idx -> list[neighbor_idx]
        # All values are vector-array indices (0..N-1)
        self._graph: dict[int, dict[int, list[int]]] = {}
        self._node_layers: dict[int, int] = {}  # idx -> max layer
        self._graph_loaded = False

        # Normalized vectors (contiguous numpy array)
        self._vectors: Optional[np.ndarray] = None
        # Mapping between doc_ids and vector-array indices
        self._doc_ids: Optional[list[int]] = None
        self._id_to_idx: Optional[dict[int, int]] = None
        self._n_nodes: int = 0

    def init_tables(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hnsw_edges (
                layer INTEGER NOT NULL,
                src_id INTEGER NOT NULL,
                dst_id INTEGER NOT NULL,
                PRIMARY KEY (layer, src_id, dst_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_hnsw_src "
            "ON hnsw_edges (layer, src_id)"
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hnsw_meta (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)

    def save_params(self, cursor: sqlite3.Cursor) -> None:
        for key, value in [
            ("hnsw_entry_point", self._entry_point),
            ("hnsw_max_layer", self._max_layer),
            ("hnsw_m", self.m),
            ("hnsw_ef_construction", self.ef_construction),
            ("hnsw_node_layers", self._node_layers),
            ("hnsw_n_nodes", self._n_nodes),
        ]:
            cursor.execute(
                "INSERT OR REPLACE INTO hnsw_meta (key, value) VALUES (?, ?)",
                (key, pickle.dumps(value)),
            )

    def load_params(self, cursor: sqlite3.Cursor) -> bool:
        cursor.execute("SELECT key, value FROM hnsw_meta")
        rows = cursor.fetchall()
        if not rows:
            return False

        meta = {row["key"]: pickle.loads(row["value"]) for row in rows}
        self._entry_point = meta.get("hnsw_entry_point")
        self._max_layer = meta.get("hnsw_max_layer", 0)
        self.m = meta.get("hnsw_m", self.m)
        self.ef_construction = meta.get("hnsw_ef_construction", self.ef_construction)
        self._node_layers = meta.get("hnsw_node_layers", {})
        self._n_nodes = meta.get("hnsw_n_nodes", 0)
        self._ml = 1.0 / math.log(self.m) if self.m > 1 else 1.0
        self.m_max0 = self.m * 2
        return True

    def set_dimension(self, dimension: int) -> None:
        self._dimension = dimension

    def build_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Build HNSW graph from scratch."""
        self._dimension = vectors.shape[1]

        self._vectors = self._normalize(vectors)
        self._doc_ids = list(doc_ids)
        self._id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        self._n_nodes = len(doc_ids)

        self._graph = {}
        self._node_layers = {}
        self._entry_point = None
        self._max_layer = 0

        rng = np.random.default_rng(self._seed)
        vecs = self._vectors  # local ref for speed

        for idx in range(self._n_nodes):
            self._insert_node_idx(idx, rng, vecs)

        self._save_graph_with_docids(cursor, doc_ids)
        self.save_params(cursor)

    def add_to_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Incrementally add nodes to HNSW graph."""
        if self._entry_point is None:
            self.build_index(cursor, vectors, doc_ids)
            return

        if not self._graph_loaded:
            self._load_graph(cursor)

        normed_new = self._normalize(vectors)
        if self._vectors is not None:
            self._vectors = np.vstack([self._vectors, normed_new])
        else:
            self._vectors = normed_new
            self._doc_ids = []
            self._id_to_idx = {}

        offset = self._n_nodes
        for i, did in enumerate(doc_ids):
            self._doc_ids.append(did)
            self._id_to_idx[did] = offset + i

        rng = np.random.default_rng(self._seed)
        vecs = self._vectors

        for i in range(len(doc_ids)):
            idx = offset + i
            self._n_nodes += 1
            self._insert_node_idx(idx, rng, vecs)

        self._save_graph_with_docids(cursor, self._doc_ids)
        self.save_params(cursor)

    def find_candidates(self, cursor: sqlite3.Cursor, query_vector: np.ndarray) -> set[int]:
        """Search HNSW graph for nearest neighbors."""
        if self._entry_point is None:
            return set()

        if not self._graph_loaded:
            self._load_graph(cursor)

        query_normed = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        vecs = self._vectors

        current = self._entry_point
        for layer in range(self._max_layer, 0, -1):
            current = self._greedy_search(query_normed, current, layer, vecs)

        results = self._beam_search(query_normed, current, 0, self.ef_search, vecs)

        # Convert indices back to doc_ids
        return {self._doc_ids[idx] for _, idx in results}

    def clear_index(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("DELETE FROM hnsw_edges")
        cursor.execute("DELETE FROM hnsw_meta")
        self._graph = {}
        self._node_layers = {}
        self._entry_point = None
        self._max_layer = 0
        self._dimension = None
        self._vectors = None
        self._doc_ids = None
        self._id_to_idx = None
        self._graph_loaded = False
        self._n_nodes = 0

    # --- Core HNSW operations (index-based, no dict lookups in hot path) ---

    def _insert_node_idx(self, idx: int, rng: np.random.Generator, vecs: np.ndarray) -> None:
        """Insert node by vector-array index."""
        level = int(-math.log(rng.random() + 1e-10) * self._ml)
        self._node_layers[idx] = level

        if self._entry_point is None:
            self._entry_point = idx
            self._max_layer = level
            for l in range(level + 1):
                if l not in self._graph:
                    self._graph[l] = {}
                self._graph[l][idx] = []
            return

        q_vec = vecs[idx]
        current = self._entry_point

        # Greedy descent
        for layer in range(self._max_layer, level, -1):
            current = self._greedy_search(q_vec, current, layer, vecs)

        # Insert at layers level..0
        for layer in range(min(level, self._max_layer), -1, -1):
            candidates = self._beam_search(q_vec, current, layer, self.ef_construction, vecs)

            m_max = self.m_max0 if layer == 0 else self.m
            neighbors = [nid for _, nid in candidates[:m_max]]

            # Set forward edges
            if layer not in self._graph:
                self._graph[layer] = {}
            self._graph[layer][idx] = neighbors

            # Set reverse edges + prune
            layer_graph = self._graph[layer]
            for neighbor in neighbors:
                nbr_list = layer_graph.get(neighbor)
                if nbr_list is None:
                    layer_graph[neighbor] = [idx]
                    continue
                nbr_list.append(idx)
                if len(nbr_list) > m_max:
                    # Batch prune
                    nbr_arr = np.array(nbr_list, dtype=np.int64)
                    sims = vecs[nbr_arr] @ vecs[neighbor]
                    top_idx = np.argpartition(sims, -m_max)[-m_max:]
                    layer_graph[neighbor] = nbr_arr[top_idx].tolist()

            if candidates:
                current = candidates[0][1]

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = idx

    def _greedy_search(self, q_vec: np.ndarray, entry: int, layer: int, vecs: np.ndarray) -> int:
        """Greedy search — all operations use array indices, no dict lookups."""
        current = entry
        current_sim = float(q_vec @ vecs[current])
        layer_graph = self._graph.get(layer, {})

        while True:
            neighbors = layer_graph.get(current)
            if not neighbors:
                break

            nbr_arr = np.array(neighbors, dtype=np.int64)
            sims = vecs[nbr_arr] @ q_vec
            best_i = int(np.argmax(sims))

            if sims[best_i] > current_sim:
                current = neighbors[best_i]
                current_sim = float(sims[best_i])
            else:
                break

        return current

    def _beam_search(
        self, q_vec: np.ndarray, entry: int, layer: int, ef: int, vecs: np.ndarray
    ) -> list[tuple[float, int]]:
        """Beam search with heaps. Returns [(sim, idx), ...] sorted desc."""
        entry_sim = float(q_vec @ vecs[entry])
        visited: set[int] = {entry}

        # Max-heap for candidates (negate sim), min-heap for results
        candidates = [(-entry_sim, entry)]
        results = [(entry_sim, entry)]
        worst_sim = entry_sim
        layer_graph = self._graph.get(layer, {})

        while candidates:
            neg_sim, current = heapq.heappop(candidates)
            current_sim = -neg_sim

            if current_sim < worst_sim and len(results) >= ef:
                break

            neighbors = layer_graph.get(current)
            if not neighbors:
                continue

            # Filter visited
            new_nbrs = [n for n in neighbors if n not in visited]
            if not new_nbrs:
                continue
            visited.update(new_nbrs)

            # Batch similarity
            nbr_arr = np.array(new_nbrs, dtype=np.int64)
            sims = vecs[nbr_arr] @ q_vec

            # Vectorized filter: only process sims above worst (or all if results < ef)
            if len(results) >= ef:
                mask = sims > worst_sim
                good_indices = np.where(mask)[0]
            else:
                good_indices = np.arange(len(new_nbrs))

            for i in good_indices:
                sim_f = float(sims[i])
                nbr = new_nbrs[i]
                heapq.heappush(candidates, (-sim_f, nbr))
                heapq.heappush(results, (sim_f, nbr))
                if len(results) > ef:
                    heapq.heappop(results)
                worst_sim = results[0][0]

        results.sort(reverse=True)
        return results

    def _save_graph_with_docids(self, cursor: sqlite3.Cursor, doc_ids: list[int]) -> None:
        """Save graph to DB, converting indices to doc_ids for persistence."""
        cursor.execute("DELETE FROM hnsw_edges")
        rows = []
        for layer, nodes in self._graph.items():
            for src_idx, neighbor_indices in nodes.items():
                src_id = doc_ids[src_idx]
                for dst_idx in neighbor_indices:
                    rows.append((layer, src_id, doc_ids[dst_idx]))
        if rows:
            cursor.executemany(
                "INSERT INTO hnsw_edges (layer, src_id, dst_id) VALUES (?, ?, ?)",
                rows,
            )

    def _load_graph(self, cursor: sqlite3.Cursor) -> None:
        """Load graph from DB, converting doc_ids back to indices."""
        self._graph = {}
        cursor.execute("SELECT layer, src_id, dst_id FROM hnsw_edges")
        for row in cursor.fetchall():
            layer = row["layer"]
            src_idx = self._id_to_idx[row["src_id"]]
            dst_idx = self._id_to_idx[row["dst_id"]]
            if layer not in self._graph:
                self._graph[layer] = {}
            if src_idx not in self._graph[layer]:
                self._graph[layer][src_idx] = []
            self._graph[layer][src_idx].append(dst_idx)

        # Also rebuild node_layers from graph structure
        for layer in self._graph:
            for idx in self._graph[layer]:
                if idx not in self._node_layers or layer > self._node_layers[idx]:
                    self._node_layers[idx] = layer

        self._graph_loaded = True

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
