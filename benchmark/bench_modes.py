#!/usr/bin/env python3
"""
Benchmark LSH vs IVF vs HNSW on the Cohere-768d dataset.

Measures recall@10, recall@100, search latency, insert time, and DB size
at different dataset sizes.

Usage:
    uv run python benchmark/bench_modes.py
    uv run python benchmark/bench_modes.py --n-vectors 100000
    uv run python benchmark/bench_modes.py --n-vectors 1000 10000 100000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlitesearch.vector.index import VectorSearchIndex

DATASET_DIR = Path("/tmp/vectordb_bench/dataset/cohere_medium_1m")
K = 100
N_QUERIES = 100
SEED = 42


def load_data(n_vectors):
    """Load Cohere dataset and compute brute-force ground truth."""
    print(f"Loading {n_vectors:,} vectors from Cohere dataset...")
    t0 = time.time()
    pf = pq.ParquetFile(DATASET_DIR / "shuffle_train.parquet")
    # Read only needed rows in batches for speed
    train_rows = []
    remaining = n_vectors
    for batch in pf.iter_batches(batch_size=min(100000, n_vectors), columns=["emb"]):
        take = min(remaining, len(batch))
        train_rows.extend(batch.column("emb")[:take].to_pylist())
        remaining -= take
        if remaining <= 0:
            break
    train_embs = np.array(train_rows, dtype=np.float32)
    del train_rows

    test_table = pq.read_table(DATASET_DIR / "test.parquet", columns=["emb"])
    test_embs = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)[:N_QUERIES]
    del test_table
    print(f"  Loaded in {time.time()-t0:.1f}s  dim={train_embs.shape[1]}  n={train_embs.shape[0]:,}")

    # Brute-force ground truth
    print(f"  Computing brute-force top-{K}...", end=" ", flush=True)
    t0 = time.time()
    train_norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
    train_norms = np.where(train_norms == 0, 1.0, train_norms)
    train_normalized = train_embs / train_norms

    test_norms = np.linalg.norm(test_embs, axis=1, keepdims=True)
    test_norms = np.where(test_norms == 0, 1.0, test_norms)
    test_normalized = test_embs / test_norms

    # Use batch matmul - precompute transpose once, then full matmul
    train_normalized_T = np.ascontiguousarray(train_normalized.T)
    sim_matrix = test_normalized @ train_normalized_T
    del train_normalized_T
    ground_truth = []
    for i in range(N_QUERIES):
        top_k_idx = np.argpartition(sim_matrix[i], -K)[-K:]
        top_k_idx = top_k_idx[np.argsort(sim_matrix[i, top_k_idx])[::-1]]
        ground_truth.append([int(x) for x in top_k_idx])
    del sim_matrix
    print(f"{time.time()-t0:.1f}s", flush=True)

    return train_embs, test_embs, ground_truth


def run_config(train_embs, test_embs, ground_truth, mode, **kwargs):
    """Test one configuration. Returns dict with metrics."""
    n_vectors = len(train_embs)
    label = kwargs.pop("label", mode)
    db_path = f"/tmp/sqlitesearch_bench_{mode}_{n_vectors}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    payload = [{"idx": i} for i in range(n_vectors)]

    # Build index
    t0 = time.time()
    index = VectorSearchIndex(
        mode=mode,
        keyword_fields=[],
        id_field="idx",
        db_path=db_path,
        seed=SEED,
        **kwargs,
    )
    index.fit(train_embs, payload)
    insert_time = time.time() - t0
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # Search
    latencies = []
    recalls_10 = []
    recalls_100 = []

    # Warmup
    for i in range(min(5, len(test_embs))):
        index.search(test_embs[i], num_results=K, output_ids=True)

    for i in range(len(test_embs)):
        t0 = time.time()
        results = index.search(test_embs[i], num_results=K, output_ids=True)
        latencies.append(time.time() - t0)

        result_ids = set(r["_id"] for r in results)
        gt = ground_truth[i]
        gt_set = set(gt)
        recalls_100.append(len(result_ids & gt_set) / min(K, len(gt)))

        result_ids_10 = set(r["_id"] for r in results[:10])
        gt_10 = set(gt[:10])
        recalls_10.append(len(result_ids_10 & gt_10) / min(10, len(gt_10)))

    index.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    return {
        "label": label,
        "mode": mode,
        "recall@10": round(float(np.mean(recalls_10)), 4),
        "recall@100": round(float(np.mean(recalls_100)), 4),
        "avg_lat_ms": round(float(np.mean(latencies)) * 1000, 1),
        "p99_lat_ms": round(float(np.percentile(latencies, 99)) * 1000, 1),
        "qps": round(1.0 / float(np.mean(latencies)), 1),
        "insert_s": round(insert_time, 1),
        "db_mb": round(db_size_mb, 1),
    }


# Configurations to benchmark
CONFIGS = [
    # LSH configs
    {"mode": "lsh", "label": "LSH 8t/16b", "n_tables": 8, "hash_size": 16},
    {"mode": "lsh", "label": "LSH 32t/8b", "n_tables": 32, "hash_size": 8},
    {"mode": "lsh", "label": "LSH 64t/6b", "n_tables": 64, "hash_size": 6},
    # IVF configs
    {"mode": "ivf", "label": "IVF auto/4p", "n_probe_clusters": 4},
    {"mode": "ivf", "label": "IVF auto/8p", "n_probe_clusters": 8},
    {"mode": "ivf", "label": "IVF auto/16p", "n_probe_clusters": 16},
    # HNSW default (m=20, ef_c=64)
    {"mode": "hnsw", "label": "HNSW ef200"},
    {"mode": "hnsw", "label": "HNSW ef300", "ef_search": 300},
    {"mode": "hnsw", "label": "HNSW ef500", "ef_search": 500},
    # HNSW fast build (ef_c=16)
    {"mode": "hnsw", "label": "HNSW-fast ef200", "ef_construction": 16},
    {"mode": "hnsw", "label": "HNSW-fast ef300", "ef_construction": 16, "ef_search": 300},
    {"mode": "hnsw", "label": "HNSW-fast ef500", "ef_construction": 16, "ef_search": 500},
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark LSH vs IVF vs HNSW")
    parser.add_argument(
        "--n-vectors",
        type=int,
        nargs="+",
        default=[10000],
        help="Number of vectors to test (default: 10000)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["lsh", "ivf", "hnsw"],
        help="Modes to benchmark (default: all)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Filter by config label substring (e.g. 'm16/ef100' 'auto/16p')",
    )
    args = parser.parse_args()

    for n_vectors in args.n_vectors:
        train_embs, test_embs, ground_truth = load_data(n_vectors)

        configs = [c for c in CONFIGS if c["mode"] in args.modes]
        if args.configs:
            configs = [
                c for c in configs
                if any(f in c["label"] for f in args.configs)
            ]

        print(f"\n{'='*95}")
        print(f"Benchmarking {n_vectors:,} vectors  |  {N_QUERIES} queries  |  top-{K}")
        print(f"{'='*95}")
        print(
            f"{'Config':<20} {'R@10':>6} {'R@100':>6}  "
            f"{'avg(ms)':>8} {'p99(ms)':>8} {'QPS':>7}  "
            f"{'insert':>7} {'DB(MB)':>7}"
        )
        print("-" * 95)

        results = []
        for cfg in configs:
            cfg = cfg.copy()
            r = run_config(train_embs, test_embs, ground_truth, **cfg)
            results.append(r)
            print(
                f"{r['label']:<20} {r['recall@10']:>6.4f} {r['recall@100']:>6.4f}  "
                f"{r['avg_lat_ms']:>8.1f} {r['p99_lat_ms']:>8.1f} {r['qps']:>7.1f}  "
                f"{r['insert_s']:>6.1f}s {r['db_mb']:>7.1f}"
            )

        # Summary
        print(f"\n{'='*60}")
        print("Best by recall@100:")
        results.sort(key=lambda r: -r["recall@100"])
        for r in results[:3]:
            print(
                f"  {r['label']:<20}  R@100={r['recall@100']:.4f}  "
                f"lat={r['avg_lat_ms']:.1f}ms  QPS={r['qps']:.1f}"
            )
        print("\nBest by QPS:")
        results.sort(key=lambda r: -r["qps"])
        for r in results[:3]:
            print(
                f"  {r['label']:<20}  QPS={r['qps']:.1f}  "
                f"R@100={r['recall@100']:.4f}  lat={r['avg_lat_ms']:.1f}ms"
            )


if __name__ == "__main__":
    main()
