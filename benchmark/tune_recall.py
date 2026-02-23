#!/usr/bin/env python3
"""
Tune LSH parameters for recall on Cohere-768d dataset.

Loads N vectors (default 25K), computes brute-force ground truth,
then tests different (n_tables, hash_size, n_probe) configurations.

Usage:
    uv run python benchmark/tune_recall.py
    uv run python benchmark/tune_recall.py --n-vectors 10000
    uv run python benchmark/tune_recall.py --n-vectors 100000
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
    train_table = pq.read_table(DATASET_DIR / "shuffle_train.parquet")
    train_embs = np.array(train_table.column("emb").to_pylist(), dtype=np.float32)[:n_vectors]

    test_table = pq.read_table(DATASET_DIR / "test.parquet")
    test_embs = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)[:N_QUERIES]
    print(f"  Loaded in {time.time()-t0:.1f}s (dim={train_embs.shape[1]})")

    # Compute brute-force ground truth
    print(f"Computing brute-force top-{K} for {N_QUERIES} queries...")
    t0 = time.time()
    # Normalize all vectors
    train_norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
    train_norms = np.where(train_norms == 0, 1.0, train_norms)
    train_normalized = train_embs / train_norms

    test_norms = np.linalg.norm(test_embs, axis=1, keepdims=True)
    test_norms = np.where(test_norms == 0, 1.0, test_norms)
    test_normalized = test_embs / test_norms

    # All-pairs cosine similarity: (n_queries, n_vectors)
    sim_matrix = test_normalized @ train_normalized.T
    # Ground truth: top-K indices for each query (ordered by similarity)
    ground_truth = []
    for i in range(N_QUERIES):
        top_k_idx = np.argpartition(sim_matrix[i], -K)[-K:]
        top_k_idx = top_k_idx[np.argsort(sim_matrix[i, top_k_idx])[::-1]]
        ground_truth.append([int(x) for x in top_k_idx])  # ordered list
    print(f"  Ground truth computed in {time.time()-t0:.1f}s")

    return train_embs, test_embs, ground_truth


def run_config(train_embs, test_embs, ground_truth, n_tables, hash_size, n_probe):
    """Test one LSH configuration. Returns dict with metrics."""
    n_vectors = len(train_embs)
    db_path = f"/tmp/sqlitesearch_tune_{n_vectors}_{n_tables}_{hash_size}_{n_probe}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    payload = [{"idx": i} for i in range(n_vectors)]

    index = VectorSearchIndex(
        keyword_fields=[],
        id_field="idx",
        n_tables=n_tables,
        hash_size=hash_size,
        n_probe=n_probe,
        db_path=db_path,
        seed=SEED,
    )

    # Insert
    t0 = time.time()
    index.fit(train_embs, payload)
    insert_time = time.time() - t0
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # Search
    latencies = []
    recalls_10 = []
    recalls_100 = []

    # Warmup
    for i in range(min(3, len(test_embs))):
        index.search(test_embs[i], num_results=K, output_ids=True)

    for i in range(len(test_embs)):
        t0 = time.time()
        results = index.search(test_embs[i], num_results=K, output_ids=True)
        latencies.append(time.time() - t0)

        result_ids = set(r["_id"] for r in results)
        gt = ground_truth[i]  # ordered list, best first
        gt_set = set(gt)
        recalls_100.append(len(result_ids & gt_set) / min(K, len(gt)))

        result_ids_10 = set(r["_id"] for r in results[:10])
        gt_10 = set(gt[:10])
        recalls_10.append(len(result_ids_10 & gt_10) / min(10, len(gt_10)))

    index.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    avg_lat = float(np.mean(latencies)) * 1000
    qps = 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0

    return {
        "n_tables": n_tables,
        "hash_size": hash_size,
        "n_probe": n_probe,
        "recall@10": round(float(np.mean(recalls_10)), 4),
        "recall@100": round(float(np.mean(recalls_100)), 4),
        "avg_lat_ms": round(avg_lat, 1),
        "qps": round(float(qps), 1),
        "insert_s": round(insert_time, 1),
        "db_mb": round(db_size_mb, 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Tune LSH recall parameters")
    parser.add_argument("--n-vectors", type=int, default=25000,
                        help="Number of vectors to index (default: 25000)")
    args = parser.parse_args()

    train_embs, test_embs, ground_truth = load_data(args.n_vectors)

    # Parameter grid: (n_tables, hash_size, n_probe)
    configs = [
        # Baseline
        (8, 16, 0),
        # Fewer bits, n_probe=0
        (8, 10, 0),
        (8, 8, 0),
        (8, 6, 0),
        (8, 4, 0),
        # More tables, n_probe=0
        (16, 10, 0),
        (16, 8, 0),
        (16, 6, 0),
        (32, 10, 0),
        (32, 8, 0),
        (32, 6, 0),
        (64, 10, 0),
        (64, 8, 0),
        (64, 6, 0),
    ]

    print(f"\n{'='*90}")
    print(f"Tuning LSH parameters on {args.n_vectors:,} Cohere vectors ({N_QUERIES} queries, top-{K})")
    print(f"{'='*90}")
    print(f"\n{'tables':>6} {'bits':>4} {'probe':>5}  "
          f"{'R@10':>6} {'R@100':>6}  "
          f"{'lat(ms)':>8} {'QPS':>6}  "
          f"{'insert':>7} {'DB(MB)':>7}")
    print("-" * 75)

    results = []
    for n_tables, hash_size, n_probe in configs:
        r = run_config(train_embs, test_embs, ground_truth, n_tables, hash_size, n_probe)
        results.append(r)
        print(f"{r['n_tables']:>6} {r['hash_size']:>4} {r['n_probe']:>5}  "
              f"{r['recall@10']:>6.4f} {r['recall@100']:>6.4f}  "
              f"{r['avg_lat_ms']:>8.1f} {r['qps']:>6.1f}  "
              f"{r['insert_s']:>6.1f}s {r['db_mb']:>6.0f}")

    # Highlight best configs
    print(f"\n{'='*50}")
    print("Best by recall@100 (above 0.85):")
    good = [r for r in results if r["recall@100"] >= 0.85]
    good.sort(key=lambda r: (-r["recall@100"], r["avg_lat_ms"]))
    for r in good[:5]:
        print(f"  tables={r['n_tables']} bits={r['hash_size']} probe={r['n_probe']}  "
              f"R@100={r['recall@100']:.4f}  lat={r['avg_lat_ms']:.1f}ms  QPS={r['qps']:.1f}")

    if not good:
        print("  None reached 0.85 recall. Showing top 5 by recall:")
        results.sort(key=lambda r: -r["recall@100"])
        for r in results[:5]:
            print(f"  tables={r['n_tables']} bits={r['hash_size']} probe={r['n_probe']}  "
                  f"R@100={r['recall@100']:.4f}  lat={r['avg_lat_ms']:.1f}ms  QPS={r['qps']:.1f}")


if __name__ == "__main__":
    main()
