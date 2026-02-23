"""
HNSW (Hierarchical Navigable Small World) search strategy.

Builds a multi-layer proximity graph for fast approximate nearest neighbor search.
"""

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
        self._entry_point: Optional[int] = None
        self._max_layer: int = 0
        self._ml = 1.0 / math.log(m) if m > 1 else 1.0

        # In-memory graph: layer -> node_id -> list[neighbor_ids]
        self._graph: dict[int, dict[int, list[int]]] = {}
        self._node_layers: dict[int, int] = {}  # node_id -> max layer for that node
        self._graph_loaded = False

        # Reference to orchestrator's vector cache (set by orchestrator)
        self._vectors: Optional[np.ndarray] = None
        self._doc_ids: Optional[list[int]] = None
        self._id_to_idx: Optional[dict[int, int]] = None

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
        self._ml = 1.0 / math.log(self.m) if self.m > 1 else 1.0
        self.m_max0 = self.m * 2
        return True

    def set_dimension(self, dimension: int) -> None:
        self._dimension = dimension

    def build_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Build HNSW graph from scratch."""
        self._dimension = vectors.shape[1]

        # Set up vector lookup
        self._vectors = self._normalize(vectors)
        self._doc_ids = list(doc_ids)
        self._id_to_idx = {did: i for i, did in enumerate(doc_ids)}

        # Reset graph
        self._graph = {}
        self._node_layers = {}
        self._entry_point = None
        self._max_layer = 0

        rng = np.random.default_rng(self._seed)

        # Insert all nodes
        for i, doc_id in enumerate(doc_ids):
            self._insert_node(doc_id, rng)

        # Save graph to DB
        self._save_graph(cursor)
        self.save_params(cursor)

    def add_to_index(self, cursor: sqlite3.Cursor, vectors: np.ndarray, doc_ids: list[int]) -> None:
        """Incrementally add nodes to HNSW graph."""
        if self._entry_point is None:
            self.build_index(cursor, vectors, doc_ids)
            return

        # Ensure graph is loaded
        if not self._graph_loaded:
            self._load_graph(cursor)

        # Update vector lookup (orchestrator updates cache, we reference it)
        # We need to extend our local references
        normed_new = self._normalize(vectors)
        if self._vectors is not None:
            self._vectors = np.vstack([self._vectors, normed_new])
        else:
            self._vectors = normed_new
            self._doc_ids = []
            self._id_to_idx = {}

        offset = len(self._doc_ids)
        for i, did in enumerate(doc_ids):
            self._doc_ids.append(did)
            self._id_to_idx[did] = offset + i

        rng = np.random.default_rng(self._seed)

        for doc_id in doc_ids:
            self._insert_node(doc_id, rng)

        # Save new edges
        self._save_graph(cursor)
        self.save_params(cursor)

    def find_candidates(self, cursor: sqlite3.Cursor, query_vector: np.ndarray) -> set[int]:
        """Search HNSW graph for nearest neighbors."""
        if self._entry_point is None:
            return set()

        # Ensure graph is loaded into memory
        if not self._graph_loaded:
            self._load_graph(cursor)

        query_normed = query_vector / (np.linalg.norm(query_vector) + 1e-10)

        # Greedy search from top layer to layer 1
        current = self._entry_point
        for layer in range(self._max_layer, 0, -1):
            current = self._greedy_search(query_normed, current, layer)

        # Beam search at layer 0
        candidates = self._beam_search(query_normed, current, 0, self.ef_search)

        return {node_id for node_id, _ in candidates}

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

    # --- Internal HNSW methods ---

    def _similarity(self, id_a: int, id_b: int) -> float:
        """Cosine similarity between two nodes using cached normalized vectors."""
        idx_a = self._id_to_idx[id_a]
        idx_b = self._id_to_idx[id_b]
        return float(self._vectors[idx_a] @ self._vectors[idx_b])

    def _query_similarity(self, query_normed: np.ndarray, node_id: int) -> float:
        """Cosine similarity between query and a node."""
        idx = self._id_to_idx[node_id]
        return float(query_normed @ self._vectors[idx])

    def _random_level(self, rng: np.random.Generator) -> int:
        """Generate random level for a new node."""
        return int(-math.log(rng.random() + 1e-10) * self._ml)

    def _get_neighbors(self, node_id: int, layer: int) -> list[int]:
        """Get neighbors of a node at a given layer."""
        if layer in self._graph and node_id in self._graph[layer]:
            return self._graph[layer][node_id]
        return []

    def _set_neighbors(self, node_id: int, layer: int, neighbors: list[int]) -> None:
        """Set neighbors of a node at a given layer."""
        if layer not in self._graph:
            self._graph[layer] = {}
        self._graph[layer][node_id] = neighbors

    def _insert_node(self, node_id: int, rng: np.random.Generator) -> None:
        """Insert a single node into the HNSW graph."""
        level = self._random_level(rng)
        self._node_layers[node_id] = level

        if self._entry_point is None:
            # First node
            self._entry_point = node_id
            self._max_layer = level
            for l in range(level + 1):
                self._set_neighbors(node_id, l, [])
            return

        current = self._entry_point
        q_normed = self._vectors[self._id_to_idx[node_id]]

        # Greedy descent from top layer to level+1
        for layer in range(self._max_layer, level, -1):
            current = self._greedy_search(q_normed, current, layer)

        # Insert at layers level down to 0
        for layer in range(min(level, self._max_layer), -1, -1):
            # Find ef_construction nearest neighbors at this layer
            candidates = self._beam_search(q_normed, current, layer, self.ef_construction)

            # Select M best neighbors
            m_max = self.m_max0 if layer == 0 else self.m
            neighbors = self._select_neighbors(node_id, candidates, m_max)

            # Set bidirectional connections
            self._set_neighbors(node_id, layer, neighbors)
            for neighbor in neighbors:
                nbr_neighbors = self._get_neighbors(neighbor, layer)
                nbr_neighbors.append(node_id)
                # Prune if over capacity
                if len(nbr_neighbors) > m_max:
                    nbr_neighbors = self._select_neighbors(
                        neighbor,
                        [(n, self._similarity(neighbor, n)) for n in nbr_neighbors],
                        m_max,
                    )
                self._set_neighbors(neighbor, layer, nbr_neighbors)

            if candidates:
                current = candidates[0][0]  # best candidate for next layer

        # Update entry point if new node has higher layer
        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = node_id

    def _select_neighbors(
        self, node_id: int, candidates: list, m_max: int
    ) -> list[int]:
        """Select best neighbors from candidates. Candidates can be (id, sim) tuples or just ids."""
        if not candidates:
            return []

        if isinstance(candidates[0], tuple):
            scored = candidates
        else:
            scored = [
                (c, self._similarity(node_id, c)) for c in candidates
            ]

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:m_max]]

    def _greedy_search(self, query_normed: np.ndarray, entry: int, layer: int) -> int:
        """Greedy search at a single layer — returns closest node."""
        current = entry
        current_sim = self._query_similarity(query_normed, current)

        while True:
            changed = False
            for neighbor in self._get_neighbors(current, layer):
                sim = self._query_similarity(query_normed, neighbor)
                if sim > current_sim:
                    current = neighbor
                    current_sim = sim
                    changed = True
            if not changed:
                break

        return current

    def _beam_search(
        self, query_normed: np.ndarray, entry: int, layer: int, ef: int
    ) -> list[tuple[int, float]]:
        """Beam search at a single layer — returns list of (node_id, similarity)."""
        visited: set[int] = {entry}
        entry_sim = self._query_similarity(query_normed, entry)

        # candidates: max-heap by similarity (we want to expand best first)
        # results: min-heap by similarity (we want to evict worst)
        candidates = [(entry, entry_sim)]
        results = [(entry, entry_sim)]

        while candidates:
            # Pop best candidate (highest similarity)
            candidates.sort(key=lambda x: x[1], reverse=True)
            current, current_sim = candidates.pop(0)

            # Worst in results
            results.sort(key=lambda x: x[1])
            worst_sim = results[0][1]

            if current_sim < worst_sim and len(results) >= ef:
                break

            for neighbor in self._get_neighbors(current, layer):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                sim = self._query_similarity(query_normed, neighbor)

                results.sort(key=lambda x: x[1])
                if len(results) < ef or sim > results[0][1]:
                    candidates.append((neighbor, sim))
                    results.append((neighbor, sim))
                    if len(results) > ef:
                        results.sort(key=lambda x: x[1])
                        results.pop(0)

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _save_graph(self, cursor: sqlite3.Cursor) -> None:
        """Save in-memory graph to database."""
        cursor.execute("DELETE FROM hnsw_edges")
        rows = []
        for layer, nodes in self._graph.items():
            for src_id, neighbors in nodes.items():
                for dst_id in neighbors:
                    rows.append((layer, src_id, dst_id))
        if rows:
            cursor.executemany(
                "INSERT INTO hnsw_edges (layer, src_id, dst_id) VALUES (?, ?, ?)",
                rows,
            )

    def _load_graph(self, cursor: sqlite3.Cursor) -> None:
        """Load graph from database into memory."""
        self._graph = {}
        cursor.execute("SELECT layer, src_id, dst_id FROM hnsw_edges")
        for row in cursor.fetchall():
            layer = row["layer"]
            src_id = row["src_id"]
            dst_id = row["dst_id"]
            if layer not in self._graph:
                self._graph[layer] = {}
            if src_id not in self._graph[layer]:
                self._graph[layer][src_id] = []
            self._graph[layer][src_id].append(dst_id)
        self._graph_loaded = True

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
