#!/usr/bin/env python3
"""
Profile filtered vector search on the Cohere benchmark data.

This is intentionally narrower than bench_modes.py: it instruments the filtered
search hot path and reports where latency is spent for a repeated filter.

Usage:
    uv run python benchmark/profile_filtered_search.py --n-vectors 30000
    uv run python benchmark/profile_filtered_search.py --modes hnsw --n-vectors 100000
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlitesearch.vector.index import VectorSearchIndex  # noqa: E402

DATASET_DIR = Path(
    os.environ.get("SQLITESEARCH_BENCH_DATA", "/data/vectordb_bench/dataset/cohere_medium_1m")
)
SEED = 42


DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "lsh": {"n_tables": 8, "hash_size": 16, "n_probe": 2},
    "ivf": {"n_probe_clusters": 8},
    "hnsw": {"m": 20, "ef_construction": 64, "ef_search": 300},
}


def load_data(n_vectors: int, n_queries: int) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading {n_vectors:,} train vectors and {n_queries:,} queries...", flush=True)
    t0 = time.perf_counter()

    pf = pq.ParquetFile(DATASET_DIR / "shuffle_train.parquet")
    train_embs = None
    loaded = 0
    remaining = n_vectors
    batch_size = min(50_000, n_vectors)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["emb"]):
        take = min(remaining, len(batch))
        values = batch.column("emb")[:take].to_pylist()
        chunk = np.array(values, dtype=np.float32)
        del values
        if train_embs is None:
            train_embs = np.empty((n_vectors, chunk.shape[1]), dtype=np.float32)
        train_embs[loaded : loaded + len(chunk)] = chunk
        loaded += len(chunk)
        remaining -= take
        print(f"  loaded train {loaded:,}/{n_vectors:,}", flush=True)
        if remaining <= 0:
            break

    if train_embs is None:
        train_embs = np.empty((0, 0), dtype=np.float32)
    elif loaded < n_vectors:
        train_embs = train_embs[:loaded]

    test_table = pq.read_table(DATASET_DIR / "test.parquet", columns=["emb"])
    test_embs = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)[:n_queries]

    print(
        f"Loaded in {time.perf_counter() - t0:.1f}s  dim={train_embs.shape[1]}",
        flush=True,
    )
    return train_embs, test_embs


def wrap_timer(timings: dict[str, float], counts: dict[str, int], name: str, obj, attr: str):
    original = getattr(obj, attr)

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        timings[name] += time.perf_counter() - t0
        counts[name] += 1
        return result

    setattr(obj, attr, timed)


def run_mode(
    mode: str,
    train_embs: np.ndarray,
    test_embs: np.ndarray,
    *,
    k: int,
    categories: int,
    broad_categories: int,
    config: dict[str, Any],
) -> None:
    n_vectors = len(train_embs)
    db_path = f"/tmp/sqlitesearch_profile_filtered_{mode}_{n_vectors}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    payload = [{"idx": i, "category": f"c{i % categories}"} for i in range(n_vectors)]
    filter_dict = {"category": [f"c{i}" for i in range(broad_categories)]}

    print(f"\nMode: {mode}", flush=True)
    print("Building payload and index...", flush=True)
    index = VectorSearchIndex(
        mode=mode,
        keyword_fields=["category"],
        id_field="idx",
        db_path=db_path,
        seed=SEED,
        **config,
    )

    t0 = time.perf_counter()
    index.fit(train_embs, payload)
    build_s = time.perf_counter() - t0

    # Warm the vector/index caches before instrumenting.
    index.search(test_embs[0], filter_dict=filter_dict, num_results=k, output_ids=True)

    timings = {name: 0.0 for name in ("count", "enum", "apply", "find", "rerank")}
    counts = {name: 0 for name in timings}
    wrap_timer(timings, counts, "count", index, "_count_filtered")
    wrap_timer(timings, counts, "enum", index, "_enumerate_filtered_ids")
    wrap_timer(timings, counts, "apply", index, "_apply_filters")
    wrap_timer(timings, counts, "find", index._strategy, "find_candidates")
    wrap_timer(timings, counts, "rerank", index, "_rerank")

    latencies = []
    for query in test_embs:
        t0 = time.perf_counter()
        index.search(query, filter_dict=filter_dict, num_results=k, output_ids=True)
        latencies.append(time.perf_counter() - t0)

    avg_ms = float(np.mean(latencies)) * 1000
    p99_ms = float(np.percentile(latencies, 99)) * 1000
    print(f"build_s={build_s:.1f}  avg_ms={avg_ms:.1f}  p99_ms={p99_ms:.1f}")
    for name in timings:
        if counts[name]:
            print(
                f"  {name:6s} calls={counts[name]:4d} "
                f"avg_ms={timings[name] / counts[name] * 1000:.1f} "
                f"total_ms/q={timings[name] / len(test_embs) * 1000:.1f}"
            )

    index.close()
    if os.path.exists(db_path):
        os.remove(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vectors", type=int, default=30_000)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--modes", nargs="+", choices=sorted(DEFAULT_CONFIGS), default=["lsh", "ivf", "hnsw"])
    parser.add_argument("--categories", type=int, default=10)
    parser.add_argument("--broad-categories", type=int, default=7)
    parser.add_argument("--hnsw-ef-construction", type=int, default=None)
    parser.add_argument("--hnsw-ef-search", type=int, default=None)
    args = parser.parse_args()

    train_embs, test_embs = load_data(args.n_vectors, args.n_queries)
    for mode in args.modes:
        config = DEFAULT_CONFIGS[mode].copy()
        if mode == "hnsw":
            if args.hnsw_ef_construction is not None:
                config["ef_construction"] = args.hnsw_ef_construction
            if args.hnsw_ef_search is not None:
                config["ef_search"] = args.hnsw_ef_search
        run_mode(
            mode,
            train_embs,
            test_embs,
            k=args.k,
            categories=args.categories,
            broad_categories=args.broad_categories,
            config=config,
        )


if __name__ == "__main__":
    main()
