"""
HNSW (Hierarchical Navigable Small World) search strategy.

Builds a multi-layer proximity graph for fast approximate nearest neighbor search.
Layer 0 graph uses pre-allocated numpy arrays for fast neighbor lookups.
Upper layers use dicts (sparse, few nodes). After initial construction, an
NN-descent refinement pass improves graph quality by checking 2-hop neighbors.
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
        self._entry_point: Optional[int] = None
        self._max_layer: int = 0
        self._ml = 1.0 / math.log(m) if m > 1 else 1.0

        # Layer 0: numpy arrays (hot path)
        self._adj: Optional[np.ndarray] = None  # (capacity, m_max0) int32
        self._adj_count: Optional[np.ndarray] = None  # (capacity,) int32
        self._capacity: int = 0

        # Upper layers: dict (sparse, rarely accessed)
        self._upper: dict[int, dict[int, list[int]]] = {}
        self._node_layers: dict[int, int] = {}
        self._graph_loaded = False

        # Normalized vectors (contiguous numpy array)
        self._vectors: Optional[np.ndarray] = None
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

    _PARAM_KEYS = frozenset({
        "hnsw_entry_point", "hnsw_max_layer", "hnsw_m",
        "hnsw_ef_construction", "hnsw_node_layers", "hnsw_n_nodes",
    })

    def load_params(self, cursor: sqlite3.Cursor) -> bool:
        cursor.execute("SELECT key, value FROM hnsw_meta")
        rows = cursor.fetchall()
        if not rows:
            return False

        meta = {
            row["key"]: pickle.loads(row["value"])
            for row in rows if row["key"] in self._PARAM_KEYS
        }
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
        """Build HNSW graph from scratch using sequential insertion.

        Reverse edges use an overflow buffer that is periodically pruned
        when full, maintaining graph quality during construction.
        Pre-allocated visited array with generation counter minimizes
        per-insertion overhead.
        """
        self._dimension = vectors.shape[1]

        self._vectors = self._normalize(vectors)
        self._doc_ids = list(doc_ids)
        self._id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        self._n_nodes = len(doc_ids)

        n = self._n_nodes
        # Overflow buffer for reverse edges — pruned when full during insertion
        overflow = 16
        adj_width = self.m_max0 + overflow
        self._adj = np.full((n, adj_width), -1, dtype=np.int32)
        self._adj_count = np.zeros(n, dtype=np.int32)
        self._capacity = n
        self._upper = {}
        self._node_layers = {}
        self._entry_point = None
        self._max_layer = 0

        rng = np.random.default_rng(self._seed)
        vecs = self._vectors

        # Pre-allocate visited generation counter (avoids per-search allocation)
        visited_gen = np.zeros(n, dtype=np.int32)
        gen = 0

        # Sequential HNSW insertion with periodic overflow pruning
        import time as _time
        _t0 = _time.perf_counter()
        for idx in range(n):
            gen = self._insert_node_fast(idx, rng, vecs, adj_width,
                                         visited_gen, gen)
        _t1 = _time.perf_counter()

        # Batch prune any remaining overflow neighbors
        self._batch_prune(vecs)
        _t2 = _time.perf_counter()

        # NN-descent refinement to improve graph quality
        # Skip for large datasets — periodic overflow pruning maintains quality
        if 100 < n <= 500_000:
            self._refine_graph(vecs, n_iters=1)
        _t3 = _time.perf_counter()

        self._save_graph(cursor, doc_ids)
        _t4 = _time.perf_counter()
        self.save_params(cursor)

        if n >= 10000:
            print(f"  HNSW build phases: insert={_t1-_t0:.1f}s  prune={_t2-_t1:.1f}s  "
                  f"refine={_t3-_t2:.1f}s  save={_t4-_t3:.1f}s  total={_t4-_t0:.1f}s")

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

        # Ensure capacity for new nodes
        new_total = offset + len(doc_ids)
        self._ensure_capacity(new_total)

        rng = np.random.default_rng(self._seed)
        vecs = self._vectors

        for i in range(len(doc_ids)):
            idx = offset + i
            self._n_nodes += 1
            self._insert_node(idx, rng, vecs)

        self._save_graph(cursor, self._doc_ids)
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
            current = self._greedy_search_upper(query_normed, current, layer, vecs)

        results = self._beam_search_0(query_normed, current, self.ef_search, vecs)

        return {self._doc_ids[idx] for _, idx in results}

    def clear_index(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("DELETE FROM hnsw_edges")
        cursor.execute("DELETE FROM hnsw_meta")
        self._adj = None
        self._adj_count = None
        self._capacity = 0
        self._upper = {}
        self._node_layers = {}
        self._entry_point = None
        self._max_layer = 0
        self._dimension = None
        self._vectors = None
        self._doc_ids = None
        self._id_to_idx = None
        self._graph_loaded = False
        self._n_nodes = 0

    # --- Array management ---

    def _ensure_capacity(self, n: int) -> None:
        """Ensure layer 0 arrays can hold at least n nodes."""
        if self._adj is not None and self._capacity >= n:
            return
        new_cap = max(n, int(self._capacity * 1.5), 1024)
        if self._adj is None:
            self._adj = np.full((new_cap, self.m_max0), -1, dtype=np.int32)
            self._adj_count = np.zeros(new_cap, dtype=np.int32)
        else:
            new_adj = np.full((new_cap, self.m_max0), -1, dtype=np.int32)
            new_adj[:self._capacity] = self._adj
            self._adj = new_adj
            new_count = np.zeros(new_cap, dtype=np.int32)
            new_count[:self._capacity] = self._adj_count
            self._adj_count = new_count
        self._capacity = new_cap

    # --- Fast batch build ---

    def _build_layer0_fast(self, vecs: np.ndarray, rng: np.random.Generator) -> None:
        """Build layer 0 using cluster-based batch construction.

        1. K-means clustering to partition vectors into groups
        2. Within each cluster, brute-force find nearest neighbors (batch matmul)
        3. Similarity-based cross-cluster edges (batch matmul per cluster pair)
        4. NN-descent refinement spreads good edges across cluster boundaries
        """
        n = self._n_nodes
        adj = self._adj
        adj_count = self._adj_count
        m_max0 = self.m_max0

        # Step 1: K-means clustering
        n_clusters = min(int(math.sqrt(n)), 256)
        assignments, centers = self._kmeans(vecs, n_clusters, rng)

        # Build cluster membership lists
        cluster_nodes = [[] for _ in range(n_clusters)]
        for i in range(n):
            cluster_nodes[assignments[i]].append(i)
        cluster_nodes = [np.array(c, dtype=np.int32) for c in cluster_nodes]

        # Step 2: Within-cluster brute-force nearest neighbors (vectorized)
        for c_idx in range(n_clusters):
            nodes = cluster_nodes[c_idx]
            if len(nodes) <= 1:
                continue

            c_vecs = vecs[nodes]
            sims = c_vecs @ c_vecs.T
            np.fill_diagonal(sims, -np.inf)

            k = min(m_max0, len(nodes) - 1)
            top_k_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            top_k_global = nodes[top_k_idx]
            adj[nodes, :k] = top_k_global
            adj_count[nodes] = k

        # Step 3: Cross-cluster edges via batch matmul per cluster pair
        # For each cluster, compare to nearest clusters and add best matches
        centroid_sims = centers @ centers.T
        np.fill_diagonal(centroid_sims, -np.inf)
        n_cross = min(3, n_clusters - 1)

        for c_idx in range(n_clusters):
            c_nodes = cluster_nodes[c_idx]
            if len(c_nodes) == 0:
                continue

            top_clusters = np.argpartition(centroid_sims[c_idx], -n_cross)[-n_cross:]

            for tc_idx in top_clusters:
                tc_nodes_arr = cluster_nodes[int(tc_idx)]
                if len(tc_nodes_arr) == 0:
                    continue

                # Batch cross-cluster similarity
                cross_sims = vecs[c_nodes] @ vecs[tc_nodes_arr].T

                # Best match per node in c
                best_local = np.argmax(cross_sims, axis=1)
                best_global = tc_nodes_arr[best_local]

                # Add edges — skip per-element sim check, just append or skip
                for i_local in range(len(c_nodes)):
                    i_global = int(c_nodes[i_local])
                    c = adj_count[i_global]
                    if c < m_max0:
                        adj[i_global, c] = int(best_global[i_local])
                        adj_count[i_global] = c + 1

        self._entry_point = 0

    @staticmethod
    def _kmeans(vecs: np.ndarray, k: int, rng: np.random.Generator,
                max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering with cosine similarity."""
        n = len(vecs)
        # Random initialization
        indices = rng.choice(n, k, replace=False)
        centers = vecs[indices].copy()

        for _ in range(max_iter):
            # Assign: batch matmul
            sims = vecs @ centers.T  # (n, k)
            assignments = np.argmax(sims, axis=1)

            # Update centroids
            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    new_centers[c] = vecs[mask].mean(axis=0)
                    norm = np.linalg.norm(new_centers[c])
                    if norm > 0:
                        new_centers[c] /= norm

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        return assignments, centers

    def _build_upper_layers(self, vecs: np.ndarray, rng: np.random.Generator) -> None:
        """Build upper HNSW layers on top of the layer 0 graph.

        Assigns random layers to nodes and builds upper layer graphs using
        greedy search through the already-built layer 0.
        """
        n = self._n_nodes

        # Assign layers to all nodes
        for idx in range(n):
            level = int(-math.log(rng.random() + 1e-10) * self._ml)
            self._node_layers[idx] = level
            if level > self._max_layer:
                self._max_layer = level
                self._entry_point = idx

        # Build upper layers using beam search on layer 0 for neighbor finding
        if self._max_layer == 0:
            return

        for idx in range(n):
            level = self._node_layers[idx]
            if level == 0:
                continue

            q_vec = vecs[idx]

            # For each upper layer this node belongs to, find neighbors
            # using beam search on layer 0 (which has good connectivity)
            candidates = self._beam_search_0(q_vec, self._entry_point, self.ef_construction, vecs)

            for layer in range(1, level + 1):
                if layer not in self._upper:
                    self._upper[layer] = {}

                # Filter candidates to only include nodes that exist at this layer
                layer_nbrs = [nid for _, nid in candidates
                              if self._node_layers.get(nid, 0) >= layer and nid != idx]
                layer_nbrs = layer_nbrs[:self.m]

                self._upper[layer][idx] = layer_nbrs

                # Reverse edges
                for nbr in layer_nbrs:
                    if nbr not in self._upper[layer]:
                        self._upper[layer][nbr] = []
                    nbr_list = self._upper[layer][nbr]
                    if idx not in nbr_list:
                        nbr_list.append(idx)
                        if len(nbr_list) > self.m:
                            nbr_arr = np.array(nbr_list, dtype=np.int64)
                            sims = vecs[nbr_arr] @ vecs[nbr]
                            top_idx = np.argpartition(sims, -self.m)[-self.m:]
                            self._upper[layer][nbr] = nbr_arr[top_idx].tolist()

    # --- Core HNSW operations ---

    def _insert_node(self, idx: int, rng: np.random.Generator, vecs: np.ndarray) -> None:
        """Insert node by vector-array index."""
        level = int(-math.log(rng.random() + 1e-10) * self._ml)
        self._node_layers[idx] = level

        if self._entry_point is None:
            self._entry_point = idx
            self._max_layer = level
            for l in range(1, level + 1):
                if l not in self._upper:
                    self._upper[l] = {}
                self._upper[l][idx] = []
            return

        q_vec = vecs[idx]
        current = self._entry_point

        # Greedy descent through upper layers
        for layer in range(self._max_layer, level, -1):
            current = self._greedy_search_upper(q_vec, current, layer, vecs)

        # Insert at upper layers (level..1)
        for layer in range(min(level, self._max_layer), 0, -1):
            candidates = self._beam_search_upper(q_vec, current, layer, self.ef_construction, vecs)
            neighbors = [nid for _, nid in candidates[:self.m]]

            if layer not in self._upper:
                self._upper[layer] = {}
            self._upper[layer][idx] = neighbors

            layer_graph = self._upper[layer]
            for neighbor in neighbors:
                nbr_list = layer_graph.get(neighbor)
                if nbr_list is None:
                    layer_graph[neighbor] = [idx]
                    continue
                nbr_list.append(idx)
                if len(nbr_list) > self.m:
                    nbr_arr = np.array(nbr_list, dtype=np.int64)
                    sims = vecs[nbr_arr] @ vecs[neighbor]
                    top_idx = np.argpartition(sims, -self.m)[-self.m:]
                    layer_graph[neighbor] = nbr_arr[top_idx].tolist()

            if candidates:
                current = candidates[0][1]

        # Insert at layer 0 using numpy arrays
        candidates = self._beam_search_0(q_vec, current, self.ef_construction, vecs)
        neighbors = np.array([nid for _, nid in candidates[:self.m_max0]], dtype=np.int32)

        count = len(neighbors)
        self._adj[idx, :count] = neighbors
        self._adj_count[idx] = count

        # Reverse edges at layer 0 with pruning when full
        adj = self._adj
        adj_count = self._adj_count
        m_max0 = self.m_max0
        for nbr in neighbors:
            nbr = int(nbr)
            c = adj_count[nbr]
            if c < m_max0:
                adj[nbr, c] = idx
                adj_count[nbr] = c + 1
            else:
                # Full — replace worst neighbor if new edge is better
                nbrs_arr = adj[nbr, :m_max0]
                sims = vecs[nbrs_arr] @ vecs[nbr]
                worst_pos = int(np.argmin(sims))
                new_sim = float(vecs[idx] @ vecs[nbr])
                if new_sim > sims[worst_pos]:
                    adj[nbr, worst_pos] = idx

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = idx

    def _insert_node_fast(self, idx: int, rng: np.random.Generator,
                          vecs: np.ndarray, adj_width: int,
                          visited_gen: np.ndarray, gen: int) -> int:
        """Insert node with periodic overflow pruning and pre-allocated visited array.

        Reverse edges are appended into an overflow buffer. When a node's
        buffer fills, it is immediately pruned to m_max0 best neighbors.
        Returns updated generation counter.
        """
        level = int(-math.log(rng.random() + 1e-10) * self._ml)
        self._node_layers[idx] = level

        if self._entry_point is None:
            self._entry_point = idx
            self._max_layer = level
            for l in range(1, level + 1):
                if l not in self._upper:
                    self._upper[l] = {}
                self._upper[l][idx] = []
            return gen

        q_vec = vecs[idx]
        current = self._entry_point

        # Greedy descent through upper layers
        for layer in range(self._max_layer, level, -1):
            current = self._greedy_search_upper(q_vec, current, layer, vecs)

        # Insert at upper layers (level..1)
        for layer in range(min(level, self._max_layer), 0, -1):
            candidates = self._beam_search_upper(q_vec, current, layer, self.ef_construction, vecs)
            neighbors = [nid for _, nid in candidates[:self.m]]

            if layer not in self._upper:
                self._upper[layer] = {}
            self._upper[layer][idx] = neighbors

            layer_graph = self._upper[layer]
            for neighbor in neighbors:
                nbr_list = layer_graph.get(neighbor)
                if nbr_list is None:
                    layer_graph[neighbor] = [idx]
                    continue
                nbr_list.append(idx)
                if len(nbr_list) > self.m:
                    nbr_arr = np.array(nbr_list, dtype=np.int64)
                    sims = vecs[nbr_arr] @ vecs[neighbor]
                    top_idx = np.argpartition(sims, -self.m)[-self.m:]
                    layer_graph[neighbor] = nbr_arr[top_idx].tolist()

            if candidates:
                current = candidates[0][1]

        # Beam search on layer 0 with pre-allocated visited
        gen += 1
        candidates = self._beam_search_0_gen(q_vec, current, self.ef_construction,
                                              vecs, visited_gen, gen)
        neighbors = np.array([nid for _, nid in candidates[:self.m_max0]], dtype=np.int32)

        count = len(neighbors)
        self._adj[idx, :count] = neighbors
        self._adj_count[idx] = count

        # Reverse edges with periodic overflow pruning
        adj = self._adj
        adj_count = self._adj_count
        m_max0 = self.m_max0
        for nbr in neighbors:
            nbr = int(nbr)
            c = adj_count[nbr]
            if c < adj_width:
                adj[nbr, c] = idx
                adj_count[nbr] = c + 1
                # When overflow buffer fills, prune immediately
                if c + 1 >= adj_width:
                    nbrs_all = adj[nbr, :adj_width].copy()
                    sims = vecs[nbrs_all] @ vecs[nbr]
                    top_k = np.argpartition(sims, -m_max0)[-m_max0:]
                    adj[nbr, :m_max0] = nbrs_all[top_k]
                    adj[nbr, m_max0:] = -1
                    adj_count[nbr] = m_max0

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = idx

        return gen

    def _beam_search_0_gen(
        self, q_vec: np.ndarray, entry: int, ef: int, vecs: np.ndarray,
        visited_gen: np.ndarray, gen: int,
    ) -> list[tuple[float, int]]:
        """Beam search on layer 0 using pre-allocated visited generation counter."""
        adj = self._adj
        adj_count = self._adj_count

        entry_sim = float(q_vec @ vecs[entry])
        visited_gen[entry] = gen

        candidates = [(-entry_sim, entry)]
        results = [(entry_sim, entry)]
        worst_sim = entry_sim

        while candidates:
            neg_sim, current = heapq.heappop(candidates)
            current_sim = -neg_sim

            if current_sim < worst_sim and len(results) >= ef:
                break

            count = adj_count[current]
            if count == 0:
                continue

            nbrs = adj[current, :count]

            # Filter visited using generation counter (no allocation!)
            new_mask = visited_gen[nbrs] != gen
            new_nbrs = nbrs[new_mask]
            if len(new_nbrs) == 0:
                continue
            visited_gen[new_nbrs] = gen

            sims = vecs[new_nbrs] @ q_vec

            if len(results) >= ef:
                good = sims > worst_sim
                good_idx = np.where(good)[0]
            else:
                good_idx = np.arange(len(new_nbrs))

            for i in good_idx:
                sim_f = float(sims[i])
                nbr = int(new_nbrs[i])
                heapq.heappush(candidates, (-sim_f, nbr))
                heapq.heappush(results, (sim_f, nbr))
                if len(results) > ef:
                    heapq.heappop(results)
                worst_sim = results[0][0]

        results.sort(reverse=True)
        return results

    def _batch_prune(self, vecs: np.ndarray) -> None:
        """Prune overflow neighbors to m_max0 in a single batch pass."""
        n = self._n_nodes
        adj = self._adj
        adj_count = self._adj_count
        m_max0 = self.m_max0

        for i in range(n):
            c = adj_count[i]
            if c <= m_max0:
                continue
            # Prune: keep top m_max0 by similarity
            nbrs = adj[i, :c]
            sims = vecs[nbrs] @ vecs[i]
            top_k = np.argpartition(sims, -m_max0)[-m_max0:]
            adj[i, :m_max0] = nbrs[top_k]
            adj[i, m_max0:] = -1
            adj_count[i] = m_max0

        # Shrink adj to standard width
        self._adj = self._adj[:, :m_max0].copy()

    def _greedy_search_upper(self, q_vec: np.ndarray, entry: int, layer: int, vecs: np.ndarray) -> int:
        """Greedy search on upper layers (dict-based)."""
        current = entry
        current_sim = float(q_vec @ vecs[current])
        layer_graph = self._upper.get(layer, {})

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

    def _beam_search_upper(
        self, q_vec: np.ndarray, entry: int, layer: int, ef: int, vecs: np.ndarray
    ) -> list[tuple[float, int]]:
        """Beam search on upper layers (dict-based)."""
        entry_sim = float(q_vec @ vecs[entry])
        visited: set[int] = {entry}
        candidates = [(-entry_sim, entry)]
        results = [(entry_sim, entry)]
        worst_sim = entry_sim
        layer_graph = self._upper.get(layer, {})

        while candidates:
            neg_sim, current = heapq.heappop(candidates)
            if -neg_sim < worst_sim and len(results) >= ef:
                break
            neighbors = layer_graph.get(current)
            if not neighbors:
                continue
            new_nbrs = [n for n in neighbors if n not in visited]
            if not new_nbrs:
                continue
            visited.update(new_nbrs)
            nbr_arr = np.array(new_nbrs, dtype=np.int64)
            sims = vecs[nbr_arr] @ q_vec
            for i in range(len(new_nbrs)):
                sim_f = float(sims[i])
                if len(results) < ef or sim_f > worst_sim:
                    nbr = new_nbrs[i]
                    heapq.heappush(candidates, (-sim_f, nbr))
                    heapq.heappush(results, (sim_f, nbr))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst_sim = results[0][0]

        results.sort(reverse=True)
        return results

    def _beam_search_0(
        self, q_vec: np.ndarray, entry: int, ef: int, vecs: np.ndarray
    ) -> list[tuple[float, int]]:
        """Beam search on layer 0 using numpy arrays for speed."""
        n = self._n_nodes
        adj = self._adj
        adj_count = self._adj_count

        entry_sim = float(q_vec @ vecs[entry])
        visited = np.zeros(n, dtype=bool)
        visited[entry] = True

        candidates = [(-entry_sim, entry)]
        results = [(entry_sim, entry)]
        worst_sim = entry_sim

        while candidates:
            neg_sim, current = heapq.heappop(candidates)
            current_sim = -neg_sim

            if current_sim < worst_sim and len(results) >= ef:
                break

            count = adj_count[current]
            if count == 0:
                continue

            # Get neighbors from numpy array — no dict lookup, no conversion
            nbrs = adj[current, :count]

            # Filter visited using numpy boolean indexing
            new_mask = ~visited[nbrs]
            new_nbrs = nbrs[new_mask]
            if len(new_nbrs) == 0:
                continue
            visited[new_nbrs] = True

            # Batch similarity
            sims = vecs[new_nbrs] @ q_vec

            # Only process candidates better than worst
            if len(results) >= ef:
                good = sims > worst_sim
                good_idx = np.where(good)[0]
            else:
                good_idx = np.arange(len(new_nbrs))

            for i in good_idx:
                sim_f = float(sims[i])
                nbr = int(new_nbrs[i])
                heapq.heappush(candidates, (-sim_f, nbr))
                heapq.heappush(results, (sim_f, nbr))
                if len(results) > ef:
                    heapq.heappop(results)
                worst_sim = results[0][0]

        results.sort(reverse=True)
        return results

    def _refine_graph(self, vecs: np.ndarray, n_iters: int = 2,
                      max_cands: int = 64) -> None:
        """NN-descent refinement: check 2-hop neighbors, swap in better edges.

        Uses pre-allocated boolean array and caps candidates per node to
        control memory bandwidth at high dimensions.
        """
        n = self._n_nodes
        adj = self._adj
        adj_count = self._adj_count
        m_max0 = self.m_max0

        # Pre-allocate exclude mask once
        exclude = np.zeros(n, dtype=bool)
        rng = np.random.default_rng(42)

        for iteration in range(n_iters):
            updates = 0

            for i in range(n):
                count_i = adj_count[i]
                if count_i == 0:
                    continue

                nbrs_i = adj[i, :count_i]

                # Gather 2-hop candidates: flatten neighbors' neighbor arrays
                all_2hop = adj[nbrs_i].ravel()
                all_2hop = all_2hop[all_2hop >= 0]
                if len(all_2hop) == 0:
                    continue

                cands = np.unique(all_2hop)

                # Exclude self and current neighbors using pre-allocated array
                exclude[i] = True
                exclude[nbrs_i] = True
                cands = cands[~exclude[cands]]
                exclude[i] = False
                exclude[nbrs_i] = False

                if len(cands) == 0:
                    continue

                # Cap candidates to control memory bandwidth
                if len(cands) > max_cands:
                    cands = rng.choice(cands, max_cands, replace=False)

                # Compute similarities
                vec_i = vecs[i]
                cand_sims = vecs[cands] @ vec_i
                curr_sims = vecs[nbrs_i] @ vec_i

                # Early exit if no candidate beats worst current neighbor
                if cand_sims.max() <= curr_sims.min():
                    continue

                # Merge and keep top m_max0
                all_nodes = np.concatenate([nbrs_i, cands])
                all_sims = np.concatenate([curr_sims, cand_sims])

                if len(all_nodes) <= m_max0:
                    continue

                top_k = np.argpartition(all_sims, -m_max0)[-m_max0:]
                new_nbrs = all_nodes[top_k]

                adj[i, :m_max0] = new_nbrs
                adj_count[i] = m_max0
                updates += 1

            if updates == 0:
                break

    # --- Persistence ---

    def _save_graph(self, cursor: sqlite3.Cursor, doc_ids: list[int]) -> None:
        """Save graph as BLOBs in hnsw_meta for fast serialization."""
        n = self._n_nodes
        # Layer 0: save numpy arrays as raw bytes
        adj_data = self._adj[:n].tobytes()
        count_data = self._adj_count[:n].tobytes()
        for key, value in [
            ("hnsw_adj", adj_data),
            ("hnsw_adj_count", count_data),
            ("hnsw_adj_width", pickle.dumps(self._adj.shape[1])),
            ("hnsw_upper", pickle.dumps(self._upper)),
        ]:
            cursor.execute(
                "INSERT OR REPLACE INTO hnsw_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def _load_graph(self, cursor: sqlite3.Cursor) -> None:
        """Load graph from BLOBs in hnsw_meta."""
        cursor.execute("SELECT key, value FROM hnsw_meta WHERE key IN "
                       "('hnsw_adj', 'hnsw_adj_count', 'hnsw_adj_width', 'hnsw_upper')")
        meta = {row["key"]: row["value"] for row in cursor.fetchall()}

        if "hnsw_adj" not in meta:
            # Fallback: load from legacy edge table
            self._load_graph_legacy(cursor)
            return

        n = self._n_nodes
        adj_width = pickle.loads(meta["hnsw_adj_width"])
        self._adj = np.frombuffer(meta["hnsw_adj"], dtype=np.int32).reshape(n, adj_width).copy()
        self._adj_count = np.frombuffer(meta["hnsw_adj_count"], dtype=np.int32).copy()
        self._capacity = n
        self._upper = pickle.loads(meta["hnsw_upper"])

        # Rebuild node_layers from graph structure
        for i in range(n):
            if self._adj_count[i] > 0:
                self._node_layers.setdefault(i, 0)
        for layer in self._upper:
            for idx in self._upper[layer]:
                if idx not in self._node_layers or layer > self._node_layers[idx]:
                    self._node_layers[idx] = layer

        self._graph_loaded = True

    def _load_graph_legacy(self, cursor: sqlite3.Cursor) -> None:
        """Load graph from hnsw_edges table (backward compatibility)."""
        n = self._n_nodes
        self._adj = np.full((n, self.m_max0), -1, dtype=np.int32)
        self._adj_count = np.zeros(n, dtype=np.int32)
        self._capacity = n
        self._upper = {}

        cursor.execute("SELECT layer, src_id, dst_id FROM hnsw_edges")
        for row in cursor.fetchall():
            layer = row["layer"]
            src_idx = self._id_to_idx[row["src_id"]]
            dst_idx = self._id_to_idx[row["dst_id"]]

            if layer == 0:
                c = self._adj_count[src_idx]
                if c < self.m_max0:
                    self._adj[src_idx, c] = dst_idx
                    self._adj_count[src_idx] = c + 1
            else:
                if layer not in self._upper:
                    self._upper[layer] = {}
                if src_idx not in self._upper[layer]:
                    self._upper[layer][src_idx] = []
                self._upper[layer][src_idx].append(dst_idx)

        for i in range(n):
            if self._adj_count[i] > 0:
                self._node_layers.setdefault(i, 0)
        for layer in self._upper:
            for idx in self._upper[layer]:
                if idx not in self._node_layers or layer > self._node_layers[idx]:
                    self._node_layers[idx] = layer

        self._graph_loaded = True

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
