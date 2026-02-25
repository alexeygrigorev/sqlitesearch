#!/usr/bin/env python3
"""
Test different ef_search values on a single HNSW build.
Builds once, then varies ef_search to find the recall/latency tradeoff.
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
    print(f"Loading {n_vectors:,} vectors from Cohere dataset...")
    t0 = time.time()
    pf = pq.ParquetFile(DATASET_DIR / "shuffle_train.parquet")
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
    print(f"  Loaded in {time.time()-t0:.1f}s  dim={train_embs.shape[1]}")

    # Ground truth
    print(f"  Computing brute-force top-{K}...", end=" ", flush=True)
    t0 = time.time()
    train_norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
    train_norms = np.where(train_norms == 0, 1.0, train_norms)
    train_normalized = train_embs / train_norms
    test_norms = np.linalg.norm(test_embs, axis=1, keepdims=True)
    test_norms = np.where(test_norms == 0, 1.0, test_norms)
    test_normalized = test_embs / test_norms
    sim_matrix = test_normalized @ np.ascontiguousarray(train_normalized.T)
    ground_truth = []
    for i in range(N_QUERIES):
        top_k_idx = np.argpartition(sim_matrix[i], -K)[-K:]
        top_k_idx = top_k_idx[np.argsort(sim_matrix[i, top_k_idx])[::-1]]
        ground_truth.append([int(x) for x in top_k_idx])
    del sim_matrix
    print(f"{time.time()-t0:.1f}s")

    return train_embs, test_embs, ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=100000)
    parser.add_argument("--ef-construction", type=int, default=16)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-search", type=int, nargs="+",
                        default=[100, 200, 300, 400, 500])
    args = parser.parse_args()

    train_embs, test_embs, ground_truth = load_data(args.n_vectors)

    db_path = f"/tmp/sqlitesearch_hnsw_tune_{args.n_vectors}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    payload = [{"idx": i} for i in range(args.n_vectors)]

    # Build once
    print(f"\nBuilding HNSW (m={args.m}, ef_c={args.ef_construction})...")
    t0 = time.time()
    index = VectorSearchIndex(
        mode="hnsw", keyword_fields=[], id_field="idx",
        m=args.m, ef_construction=args.ef_construction,
        db_path=db_path, seed=SEED,
    )
    index.fit(train_embs, payload)
    build_time = time.time() - t0
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Built in {build_time:.1f}s  DB={db_size_mb:.0f}MB")

    # Test different ef_search values
    print(f"\n{'ef_search':>10} {'R@10':>6} {'R@100':>6}  "
          f"{'avg(ms)':>8} {'p99(ms)':>8} {'QPS':>7}")
    print("-" * 55)

    for ef_s in args.ef_search:
        index._strategy.ef_search = ef_s

        # Warmup
        for i in range(3):
            index.search(test_embs[i], num_results=K, output_ids=True)

        latencies = []
        recalls_10 = []
        recalls_100 = []
        for i in range(N_QUERIES):
            t0 = time.time()
            results = index.search(test_embs[i], num_results=K, output_ids=True)
            latencies.append(time.time() - t0)

            result_ids = set(r["_id"] for r in results)
            gt = ground_truth[i]
            recalls_100.append(len(result_ids & set(gt)) / min(K, len(gt)))
            result_ids_10 = set(r["_id"] for r in results[:10])
            recalls_10.append(len(result_ids_10 & set(gt[:10])) / min(10, len(gt[:10])))

        r10 = float(np.mean(recalls_10))
        r100 = float(np.mean(recalls_100))
        avg_ms = float(np.mean(latencies)) * 1000
        p99_ms = float(np.percentile(latencies, 99)) * 1000
        qps = 1.0 / float(np.mean(latencies))
        print(f"{ef_s:>10} {r10:>6.4f} {r100:>6.4f}  "
              f"{avg_ms:>8.1f} {p99_ms:>8.1f} {qps:>7.1f}")

    index.close()
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    main()
