"""numba-compiled layer-0 beam search for the HNSW build path.

The pure-Python ``_beam_search_0_gen`` (called once per node during graph
construction) is ~85% of HNSW build time — interpreter overhead in a tight
``heapq`` loop, exactly the cost a compiled inner loop avoids. This module
gives the pure-Python ``mode="hnsw"`` that compiled inner loop via numba.

The algorithm is identical to ``HNSWStrategy._beam_search_0_gen_numpy``:
best-first search with a max-heap of candidates (expand highest-similarity
first) and a min-heap of results (keep the top-ef, with results[0] the worst).
Heaps live in caller-allocated buffers reused across the n insertions (mirrors
the pre-allocated ``visited_gen``), so build allocates them once rather than
per node.

numba is a hard dependency; this module still falls back to the numpy path if
the JIT kernel is unavailable (e.g. a numba import failure at runtime).
"""

import numpy as np

try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    _NUMBA_OK = False
    njit = None


if _NUMBA_OK:

    @njit(cache=True)
    def _heap_push_max(hsim, hidx, size, sim, idx):
        hsim[size] = sim
        hidx[size] = idx
        i = size
        size += 1
        while i > 0:
            parent = (i - 1) >> 1
            if hsim[i] <= hsim[parent]:
                break
            ts = hsim[i]
            hsim[i] = hsim[parent]
            hsim[parent] = ts
            ti = hidx[i]
            hidx[i] = hidx[parent]
            hidx[parent] = ti
            i = parent
        return size

    @njit(cache=True)
    def _heap_pop_max(hsim, hidx, size):
        rsim = hsim[0]
        ridx = hidx[0]
        size -= 1
        hsim[0] = hsim[size]
        hidx[0] = hidx[size]
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            best = i
            if left < size and hsim[left] > hsim[best]:
                best = left
            if right < size and hsim[right] > hsim[best]:
                best = right
            if best == i:
                break
            ts = hsim[i]
            hsim[i] = hsim[best]
            hsim[best] = ts
            ti = hidx[i]
            hidx[i] = hidx[best]
            hidx[best] = ti
            i = best
        return rsim, ridx, size

    @njit(cache=True)
    def _heap_push_min(hsim, hidx, size, sim, idx):
        hsim[size] = sim
        hidx[size] = idx
        i = size
        size += 1
        while i > 0:
            parent = (i - 1) >> 1
            if hsim[i] >= hsim[parent]:
                break
            ts = hsim[i]
            hsim[i] = hsim[parent]
            hsim[parent] = ts
            ti = hidx[i]
            hidx[i] = hidx[parent]
            hidx[parent] = ti
            i = parent
        return size

    @njit(cache=True)
    def _heap_pop_min(hsim, hidx, size):
        rsim = hsim[0]
        ridx = hidx[0]
        size -= 1
        hsim[0] = hsim[size]
        hidx[0] = hidx[size]
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            best = i
            if left < size and hsim[left] < hsim[best]:
                best = left
            if right < size and hsim[right] < hsim[best]:
                best = right
            if best == i:
                break
            ts = hsim[i]
            hsim[i] = hsim[best]
            hsim[best] = ts
            ti = hidx[i]
            hidx[i] = hidx[best]
            hidx[best] = ti
            i = best
        return rsim, ridx, size

    @njit(cache=True)
    def beam_search_0(
        q_vec, vecs, adj, adj_count, entry, ef,
        visited_gen, gen, cand_sim, cand_idx, res_sim, res_idx,
    ):
        """Layer-0 beam search. Returns the number of results, leaving them
        sorted descending by similarity in ``res_sim``/``res_idx[:size]``.

        The candidate/result heaps are written into the caller's reusable
        buffers; ``visited_gen`` is a generation-counter array (entry marked
        with ``gen``), so no per-search allocation is needed.
        """
        dim = vecs.shape[1]

        entry_sim = 0.0
        for d in range(dim):
            entry_sim += q_vec[d] * vecs[entry, d]
        visited_gen[entry] = gen

        cand_sim[0] = entry_sim
        cand_idx[0] = entry
        cand_size = 1

        res_sim[0] = entry_sim
        res_idx[0] = entry
        res_size = 1
        worst_sim = entry_sim

        while cand_size > 0:
            cur_sim, cur, cand_size = _heap_pop_max(cand_sim, cand_idx, cand_size)

            if cur_sim < worst_sim and res_size >= ef:
                break

            count = adj_count[cur]
            if count <= 0:
                continue

            # Match the numpy path: the "good neighbor" filter uses the result
            # set state at the start of this frontier expansion, not updated
            # mid-batch. While we have fewer than ef results, accept all.
            filter_mode = res_size >= ef
            batch_worst = worst_sim

            for p in range(count):
                nbr = adj[cur, p]
                if nbr < 0:
                    continue
                if visited_gen[nbr] == gen:
                    continue
                visited_gen[nbr] = gen

                sim = 0.0
                for d in range(dim):
                    sim += q_vec[d] * vecs[nbr, d]

                if (not filter_mode) or (sim > batch_worst):
                    cand_size = _heap_push_max(cand_sim, cand_idx, cand_size, sim, nbr)
                    res_size = _heap_push_min(res_sim, res_idx, res_size, sim, nbr)
                    if res_size > ef:
                        _, _, res_size = _heap_pop_min(res_sim, res_idx, res_size)
                        worst_sim = res_sim[0]

        # Selection sort descending by sim (res_size <= ef is small).
        for i in range(res_size):
            best = i
            for j in range(i + 1, res_size):
                if res_sim[j] > res_sim[best]:
                    best = j
            if best != i:
                ts = res_sim[i]
                res_sim[i] = res_sim[best]
                res_sim[best] = ts
                ti = res_idx[i]
                res_idx[i] = res_idx[best]
                res_idx[best] = ti

        return res_size

else:  # pragma: no cover
    beam_search_0 = None
