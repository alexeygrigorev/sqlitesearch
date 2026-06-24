#!/usr/bin/env python3
"""
Benchmark sqlitesearch vector search against VectorDBBench Cohere-1M dataset.

Dataset: Cohere Wikipedia-22-12 English embeddings
- 1,000,000 train vectors (768 dimensions, cosine similarity)
- Test queries with pre-computed ground truth (top-100 neighbors)

Methodology follows VectorDBBench:
1. Insert all training vectors
2. Serial search: measure recall@100 and p99 latency
3. Concurrent search: measure max QPS
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
from sqlitesearch.vector.index import VectorSearchIndex

# --- Configuration ---
DATASET_DIR = Path(os.environ.get("SQLITESEARCH_BENCH_DATA", "/data/vectordb_bench/dataset"))
COHERE_1M_DIR = DATASET_DIR / "cohere_medium_1m"
S3_BUCKET = "assets.zilliz.com"
S3_PREFIX = "benchmark/cohere_medium_1m"

K = 100  # Top-K for search (VDBBench default)

# Subsets to test for scaling analysis
SUBSET_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]

# LSH configurations to test (recall vs speed tradeoff)
LSH_CONFIGS = [
    {"n_tables": 8, "hash_size": 16, "n_probe": 2, "label": "default (8t/16h/p2)"},
    {"n_tables": 16, "hash_size": 16, "n_probe": 2, "label": "high-recall (16t/16h/p2)"},
    {"n_tables": 32, "hash_size": 12, "n_probe": 2, "label": "max-recall (32t/12h/p2)"},
]


def download_dataset():
    """Download Cohere-1M dataset from S3 if not cached."""
    files = ["shuffle_train.parquet", "test.parquet", "neighbors.parquet"]
    COHERE_1M_DIR.mkdir(parents=True, exist_ok=True)

    for fname in files:
        local_path = COHERE_1M_DIR / fname
        if local_path.exists():
            print(f"  [cached] {fname} ({local_path.stat().st_size / 1e6:.1f} MB)")
            continue

        print(f"  [downloading] {fname}...")
        import s3fs
        s3 = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
        s3_path = f"{S3_BUCKET}/{S3_PREFIX}/{fname}"
        s3.get(s3_path, str(local_path))
        print(f"  [done] {fname} ({local_path.stat().st_size / 1e6:.1f} MB)")


def load_dataset():
    """Load the Cohere-1M dataset from parquet files."""
    print("\nLoading dataset from parquet...")

    # Train vectors
    t0 = time.time()
    train_table = pq.read_table(COHERE_1M_DIR / "shuffle_train.parquet")
    train_ids = train_table.column("id").to_pylist()
    train_embs = np.array(train_table.column("emb").to_pylist(), dtype=np.float32)
    print(f"  Train: {len(train_ids)} vectors, dim={train_embs.shape[1]} ({time.time()-t0:.1f}s)")

    # Test queries
    t0 = time.time()
    test_table = pq.read_table(COHERE_1M_DIR / "test.parquet")
    test_ids = test_table.column("id").to_pylist()
    test_embs = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)
    print(f"  Test:  {len(test_ids)} queries, dim={test_embs.shape[1]} ({time.time()-t0:.1f}s)")

    # Ground truth
    t0 = time.time()
    neighbors_table = pq.read_table(COHERE_1M_DIR / "neighbors.parquet")
    neighbors = neighbors_table.column("neighbors_id").to_pylist()
    print(f"  Ground truth: {len(neighbors)} entries ({time.time()-t0:.1f}s)")

    return train_ids, train_embs, test_ids, test_embs, neighbors


def calc_recall(ground_truth_ids, result_ids, k=K):
    """Calculate recall@k: fraction of ground truth in results."""
    gt_set = set(ground_truth_ids[:k])
    result_set = set(result_ids[:k])
    return len(gt_set & result_set) / k


def run_benchmark(train_ids, train_embs, test_embs, neighbors,
                  n_subset, lsh_config, db_path):
    """Run a single benchmark configuration."""
    n_tables = lsh_config["n_tables"]
    hash_size = lsh_config["hash_size"]
    n_probe = lsh_config["n_probe"]
    label = lsh_config["label"]

    print(f"\n{'='*70}")
    print(f"Config: {label} | Subset: {n_subset:,} vectors")
    print(f"{'='*70}")

    # Use subset of training data
    subset_embs = train_embs[:n_subset]
    subset_ids = train_ids[:n_subset]

    # Create payload (minimal - just IDs)
    payload = [{"doc_id": int(did)} for did in subset_ids]

    # --- Stage 1: INSERT ---
    print(f"\n[INSERT] Indexing {n_subset:,} vectors (dim={subset_embs.shape[1]})...")
    # Remove old DB
    if os.path.exists(db_path):
        os.remove(db_path)

    index = VectorSearchIndex(
        keyword_fields=[],
        id_field="doc_id",
        n_tables=n_tables,
        hash_size=hash_size,
        n_probe=n_probe,
        db_path=db_path,
    )

    t_insert_start = time.time()
    index.fit(subset_embs, payload)
    t_insert = time.time() - t_insert_start

    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Insert time: {t_insert:.1f}s ({n_subset/t_insert:.0f} vec/s)")
    print(f"  DB size: {db_size_mb:.1f} MB")

    # --- Build ID mapping ---
    # VDBBench ground truth uses the original dataset IDs.
    # sqlitesearch returns doc_id (original ID) via output_ids=True.
    # We need to map: ground truth IDs reference positions in the FULL dataset.
    # But we're only using a subset. So we need to recompute ground truth
    # for the subset, OR filter ground truth to only include IDs in our subset.
    #
    # Actually, VDBBench ground truth neighbors are dataset IDs (the 'id' column).
    # For a subset, valid neighbors are only those in our subset.
    # We'll compute recall by checking how many of the ground truth top-K
    # that are IN our subset are also returned by our search.
    subset_id_set = set(int(x) for x in subset_ids)

    # --- Stage 2: SERIAL SEARCH (recall + latency) ---
    # Use up to 100 test queries (VDBBench uses all, but we limit for time)
    n_queries = min(len(test_embs), 100)
    print(f"\n[SEARCH] Running {n_queries} serial queries (top-{K})...")

    latencies = []
    recalls = []

    for i in range(n_queries):
        query = test_embs[i]
        gt_all = neighbors[i]  # Ground truth IDs for this query

        t0 = time.time()
        results = index.search(query, num_results=K, output_ids=True)
        latency = time.time() - t0
        latencies.append(latency)

        # Get returned IDs
        result_ids = [r["_id"] for r in results]

        # For subset benchmark: filter ground truth to IDs in our subset
        gt_in_subset = [gid for gid in gt_all if gid in subset_id_set]
        if len(gt_in_subset) > 0:
            # Recall = what fraction of available ground truth did we find?
            k_effective = min(K, len(gt_in_subset))
            gt_top_k = gt_in_subset[:k_effective]
            found = len(set(gt_top_k) & set(result_ids))
            recall = found / k_effective
        else:
            recall = 0.0
        recalls.append(recall)

        if (i + 1) % 20 == 0:
            avg_recall = np.mean(recalls)
            avg_lat = np.mean(latencies) * 1000
            print(f"  Query {i+1}/{n_queries}: avg recall={avg_recall:.4f}, "
                  f"avg latency={avg_lat:.1f}ms")

    mean_recall = np.mean(recalls)
    p99_latency = np.percentile(latencies, 99) * 1000  # ms
    p95_latency = np.percentile(latencies, 95) * 1000
    avg_latency = np.mean(latencies) * 1000
    serial_qps = 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0

    print("\n  Results:")
    print(f"    Recall@{K}:     {mean_recall:.4f}")
    print(f"    Avg latency:   {avg_latency:.1f} ms")
    print(f"    P95 latency:   {p95_latency:.1f} ms")
    print(f"    P99 latency:   {p99_latency:.1f} ms")
    print(f"    Serial QPS:    {serial_qps:.1f}")
    print(f"    Candidates:    {len(results)} returned per query (avg)")

    # Estimate concurrent QPS (single-process, but gives idea)
    # VDBBench measures true multi-process QPS, but for a fair single-machine
    # comparison we report serial QPS
    index.close()

    result = {
        "config": label,
        "n_vectors": n_subset,
        "dimension": subset_embs.shape[1],
        "n_tables": n_tables,
        "hash_size": hash_size,
        "n_probe": n_probe,
        "insert_time_s": round(t_insert, 1),
        "insert_rate_vec_s": round(n_subset / t_insert, 0),
        "db_size_mb": round(db_size_mb, 1),
        "n_queries": n_queries,
        "recall_at_100": round(mean_recall, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "p99_latency_ms": round(p99_latency, 1),
        "serial_qps": round(serial_qps, 1),
    }

    return result


def print_leaderboard_comparison(results):
    """Print comparison with VDBBench leaderboard."""
    print("\n" + "=" * 80)
    print("COMPARISON WITH VDBBench LEADERBOARD (Cohere-1M, 768d, cosine)")
    print("=" * 80)

    # Leaderboard data (from zilliz.com/vdbbench-leaderboard, 1M dataset)
    leaderboard = [
        {"name": "ZillizCloud-8cu-perf", "qps": 9704, "p99_ms": 2.5, "recall": 0.917},
        {"name": "Milvus-16c64g-sq8", "qps": 3465, "p99_ms": 2.2, "recall": 0.953},
        {"name": "OpenSearch-16c128g-fm", "qps": 3055, "p99_ms": 7.2, "recall": 0.907},
        {"name": "ElasticCloud-8c60g-fm", "qps": 1925, "p99_ms": 11.3, "recall": 0.896},
        {"name": "QdrantCloud-16c64g", "qps": 1242, "p99_ms": 6.4, "recall": 0.947},
        {"name": "Pinecone-p2.x8", "qps": 1147, "p99_ms": 13.7, "recall": 0.926},
    ]

    # Find best sqlitesearch result at 1M (or largest subset)
    best_result = None
    for r in results:
        if best_result is None or r["n_vectors"] > best_result["n_vectors"]:
            best_result = r
        elif r["n_vectors"] == best_result["n_vectors"] and r["recall_at_100"] > best_result["recall_at_100"]:
            best_result = r

    print(f"\n{'Database':<30} {'QPS':>8} {'P99 (ms)':>10} {'Recall@100':>12}")
    print("-" * 65)
    for entry in leaderboard:
        print(f"{entry['name']:<30} {entry['qps']:>8,} {entry['p99_ms']:>10.1f} {entry['recall']:>12.4f}")

    print("-" * 65)
    if best_result:
        name = f"sqlitesearch ({best_result['config']})"
        n = best_result['n_vectors']
        suffix = f" [{n//1000}K]" if n < 1_000_000 else ""
        print(f"{name + suffix:<30} {best_result['serial_qps']:>8.0f} "
              f"{best_result['p99_latency_ms']:>10.1f} {best_result['recall_at_100']:>12.4f}")

    print("\nNote: Leaderboard QPS is multi-process concurrent; sqlitesearch is serial (single-process).")
    print("      Leaderboard systems run on dedicated cloud hardware (8-16 cores, 32-128GB RAM).")
    print("      sqlitesearch is designed for small local projects, not 1M-scale benchmarks.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sqlitesearch LSH against the VectorDBBench Cohere dataset"
    )
    parser.add_argument(
        "--scales",
        default=",".join(str(size) for size in SUBSET_SIZES),
        help="Comma-separated subset sizes to benchmark",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Filter LSH configs by label substring",
    )
    args = parser.parse_args()

    subset_sizes = [int(value) for value in args.scales.split(",") if value]
    lsh_configs = LSH_CONFIGS
    if args.configs:
        lsh_configs = [
            config
            for config in LSH_CONFIGS
            if any(label_part in config["label"] for label_part in args.configs)
        ]

    print("VectorDBBench Comparison: sqlitesearch vs Leaderboard")
    print(f"Date: {datetime.now().isoformat()}")
    print("Dataset: Cohere-1M (768d, cosine)")

    # Step 1: Download dataset
    print("\n--- Downloading dataset ---")
    download_dataset()

    # Step 2: Load dataset
    train_ids, train_embs, test_ids, test_embs, neighbors = load_dataset()

    # Step 3: Run benchmarks
    all_results = []

    # Start with smaller subsets to show scaling behavior
    # Use a single LSH config first, then try others at a practical size
    for n_subset in subset_sizes:
        if n_subset > len(train_embs):
            print(f"\nSkipping {n_subset:,} (only {len(train_embs):,} vectors available)")
            continue

        for config in lsh_configs:
            # For large subsets, skip expensive configs to save time
            if n_subset >= 500_000 and config["n_tables"] > 16:
                print(f"\nSkipping {config['label']} at {n_subset:,} (too slow)")
                continue
            if n_subset >= 100_000 and config["n_tables"] > 16:
                print(f"\nSkipping {config['label']} at {n_subset:,} (too slow)")
                continue

            db_path = f"/tmp/sqlitesearch_bench_{n_subset}_{config['n_tables']}t.db"
            try:
                result = run_benchmark(
                    train_ids, train_embs, test_embs, neighbors,
                    n_subset, config, db_path
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n  ERROR: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up DB to save disk space
                if os.path.exists(db_path):
                    os.remove(db_path)

    # Step 4: Print scaling analysis
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    print(f"\n{'Config':<25} {'N vectors':>10} {'Insert (s)':>10} {'Recall':>8} "
          f"{'P99 (ms)':>10} {'QPS':>8} {'DB (MB)':>8}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['config']:<25} {r['n_vectors']:>10,} {r['insert_time_s']:>10.1f} "
              f"{r['recall_at_100']:>8.4f} {r['p99_latency_ms']:>10.1f} "
              f"{r['serial_qps']:>8.1f} {r['db_size_mb']:>8.1f}")

    # Step 5: Compare with leaderboard
    print_leaderboard_comparison(all_results)

    # Save results
    results_path = "/tmp/sqlitesearch_bench_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
