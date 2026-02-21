#!/usr/bin/env python3
"""
Benchmark sqlitesearch vector search using Cohere-1M dataset (768d, cosine).

Uses a fixed random seed so results are deterministic and comparable
across runs (before/after optimization).

Usage:
    uv run python benchmark/bench_cohere.py --output baseline.json
    uv run python benchmark/bench_cohere.py --output optimized.json
    uv run python benchmark/bench_cohere.py --compare baseline.json optimized.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlitesearch.vector.lsh import VectorSearchIndex

# --- Configuration ---
DATASET_DIR = Path("/tmp/vectordb_bench/dataset/cohere_medium_1m")
K = 100
SUBSET_SIZES = [1_000, 10_000, 100_000]
N_QUERIES = 100
SEED = 42  # Fixed seed for reproducible LSH projections


def load_dataset():
    """Load Cohere-1M dataset from parquet files."""
    print("Loading dataset...")
    t0 = time.time()
    train_table = pq.read_table(DATASET_DIR / "shuffle_train.parquet")
    train_ids = train_table.column("id").to_pylist()
    train_embs = np.array(train_table.column("emb").to_pylist(), dtype=np.float32)

    test_table = pq.read_table(DATASET_DIR / "test.parquet")
    test_embs = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)

    neighbors_table = pq.read_table(DATASET_DIR / "neighbors.parquet")
    neighbors = neighbors_table.column("neighbors_id").to_pylist()

    print(f"  Loaded {len(train_ids)} train, {len(test_embs)} test vectors "
          f"(dim={train_embs.shape[1]}) in {time.time()-t0:.1f}s")
    return train_ids, train_embs, test_embs, neighbors


def run_benchmark(train_ids, train_embs, test_embs, neighbors, n_subset):
    """Run benchmark at given scale. Returns metrics + per-query result IDs."""
    print(f"\n{'='*70}")
    print(f"Scale: {n_subset:,} vectors (8 tables, 16 hash bits, seed={SEED})")
    print(f"{'='*70}")

    subset_embs = train_embs[:n_subset]
    subset_ids = train_ids[:n_subset]
    payload = [{"doc_id": int(did)} for did in subset_ids]

    db_path = f"/tmp/sqlitesearch_bench_{n_subset}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    index = VectorSearchIndex(
        keyword_fields=[],
        id_field="doc_id",
        n_tables=8,
        hash_size=16,
        db_path=db_path,
        seed=SEED,
    )

    # --- INSERT ---
    print(f"\n[INSERT] {n_subset:,} vectors...")
    t_insert_start = time.time()
    index.fit(subset_embs, payload)
    t_insert = time.time() - t_insert_start
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Time: {t_insert:.2f}s ({n_subset/t_insert:.0f} vec/s)")
    print(f"  DB size: {db_size_mb:.1f} MB")

    # --- SEARCH ---
    subset_id_set = set(int(x) for x in subset_ids)
    n_queries = min(len(test_embs), N_QUERIES)
    print(f"\n[SEARCH] {n_queries} queries (top-{K})...")

    latencies = []
    recalls = []
    per_query_ids = []  # Save result IDs for correctness verification

    # Warmup
    for i in range(min(3, n_queries)):
        index.search(test_embs[i], num_results=K, output_ids=True)

    for i in range(n_queries):
        query = test_embs[i]
        gt_all = neighbors[i]

        t0 = time.time()
        results = index.search(query, num_results=K, output_ids=True)
        latency = time.time() - t0
        latencies.append(latency)

        result_ids = [r["_id"] for r in results]
        per_query_ids.append(result_ids)

        gt_in_subset = [gid for gid in gt_all if gid in subset_id_set]
        if gt_in_subset:
            k_eff = min(K, len(gt_in_subset))
            found = len(set(gt_in_subset[:k_eff]) & set(result_ids))
            recalls.append(found / k_eff)
        else:
            recalls.append(0.0)

    mean_recall = float(np.mean(recalls))
    avg_lat = float(np.mean(latencies)) * 1000
    p95_lat = float(np.percentile(latencies, 95)) * 1000
    p99_lat = float(np.percentile(latencies, 99)) * 1000
    serial_qps = 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0

    print(f"\n  Recall@{K}:   {mean_recall:.4f}")
    print(f"  Avg latency: {avg_lat:.1f} ms")
    print(f"  P95 latency: {p95_lat:.1f} ms")
    print(f"  P99 latency: {p99_lat:.1f} ms")
    print(f"  Serial QPS:  {serial_qps:.1f}")

    index.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    return {
        "n_vectors": n_subset,
        "dimension": int(subset_embs.shape[1]),
        "n_tables": 8,
        "hash_size": 16,
        "seed": SEED,
        "insert_time_s": round(t_insert, 2),
        "insert_rate_vec_s": round(n_subset / t_insert, 0),
        "db_size_mb": round(db_size_mb, 1),
        "n_queries": n_queries,
        "recall_at_100": round(mean_recall, 4),
        "avg_latency_ms": round(avg_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "p99_latency_ms": round(p99_lat, 1),
        "serial_qps": round(float(serial_qps), 1),
        "per_query_ids": per_query_ids,
    }


def print_results_table(results):
    """Print results in a compact table."""
    print(f"\n{'N vectors':>10} {'Insert(s)':>10} {'vec/s':>8} {'Recall':>8} "
          f"{'Avg(ms)':>8} {'P95(ms)':>8} {'P99(ms)':>8} {'QPS':>8} {'DB(MB)':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['n_vectors']:>10,} {r['insert_time_s']:>10.2f} "
              f"{r['insert_rate_vec_s']:>8.0f} {r['recall_at_100']:>8.4f} "
              f"{r['avg_latency_ms']:>8.1f} {r['p95_latency_ms']:>8.1f} "
              f"{r['p99_latency_ms']:>8.1f} {r['serial_qps']:>8.1f} "
              f"{r['db_size_mb']:>8.1f}")


def print_leaderboard_comparison(results):
    """Print comparison with VDBBench leaderboard."""
    leaderboard = [
        {"name": "ZillizCloud-8cu-perf", "qps": 9704, "p99_ms": 2.5, "recall": 0.917},
        {"name": "Milvus-16c64g-sq8", "qps": 3465, "p99_ms": 2.2, "recall": 0.953},
        {"name": "OpenSearch-16c128g-fm", "qps": 3055, "p99_ms": 7.2, "recall": 0.907},
        {"name": "ElasticCloud-8c60g-fm", "qps": 1925, "p99_ms": 11.3, "recall": 0.896},
        {"name": "QdrantCloud-16c64g", "qps": 1242, "p99_ms": 6.4, "recall": 0.947},
        {"name": "Pinecone-p2.x8", "qps": 1147, "p99_ms": 13.7, "recall": 0.926},
    ]

    best = max(results, key=lambda r: r["n_vectors"])

    print(f"\n{'='*65}")
    print(f"VDBBench Leaderboard Comparison (Cohere-1M, 768d, cosine)")
    print(f"{'='*65}")
    print(f"\n{'Database':<30} {'QPS':>8} {'P99(ms)':>8} {'Recall':>8}")
    print("-" * 58)
    for e in leaderboard:
        print(f"{e['name']:<30} {e['qps']:>8,} {e['p99_ms']:>8.1f} {e['recall']:>8.4f}")
    print("-" * 58)
    suffix = f" [{best['n_vectors']//1000}K]" if best['n_vectors'] < 1_000_000 else ""
    print(f"{'sqlitesearch' + suffix:<30} {best['serial_qps']:>8.0f} "
          f"{best['p99_latency_ms']:>8.1f} {best['recall_at_100']:>8.4f}")
    print(f"\nNote: Leaderboard = multi-process on cloud hardware; "
          f"sqlitesearch = serial single-process.")


def compare_results(file_a, file_b):
    """Compare two result files, checking both metrics and result IDs."""
    results_dir = Path(__file__).parent / "results"
    with open(results_dir / file_a) as f:
        a_results = json.load(f)
    with open(results_dir / file_b) as f:
        b_results = json.load(f)

    print(f"\n{'='*80}")
    print(f"COMPARISON: {file_a} vs {file_b}")
    print(f"{'='*80}")

    # Metrics comparison
    print(f"\n{'Metric':<15} {'N':>8}", end="")
    print(f"  {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 60)

    for ra in a_results:
        n = ra["n_vectors"]
        rb = next((r for r in b_results if r["n_vectors"] == n), None)
        if not rb:
            continue

        for metric, label in [
            ("insert_time_s", "Insert (s)"),
            ("recall_at_100", "Recall@100"),
            ("avg_latency_ms", "Avg lat(ms)"),
            ("p99_latency_ms", "P99 lat(ms)"),
            ("serial_qps", "QPS"),
            ("db_size_mb", "DB size(MB)"),
        ]:
            va, vb = ra[metric], rb[metric]
            if va != 0:
                pct = (vb - va) / abs(va) * 100
                sign = "+" if pct > 0 else ""
                print(f"{label:<15} {n:>8,}  {va:>10.2f} {vb:>10.2f} {sign}{pct:>9.1f}%")
            else:
                print(f"{label:<15} {n:>8,}  {va:>10.2f} {vb:>10.2f} {'N/A':>10}")
        print()

    # ID correctness verification
    print(f"\nResult ID verification:")
    for ra in a_results:
        n = ra["n_vectors"]
        rb = next((r for r in b_results if r["n_vectors"] == n), None)
        if not rb:
            continue
        ids_a = ra.get("per_query_ids", [])
        ids_b = rb.get("per_query_ids", [])
        if not ids_a or not ids_b:
            print(f"  {n:>8,}: no per-query IDs to compare")
            continue
        n_queries = min(len(ids_a), len(ids_b))
        identical = 0
        for i in range(n_queries):
            if ids_a[i] == ids_b[i]:
                identical += 1
        print(f"  {n:>8,}: {identical}/{n_queries} queries return identical IDs "
              f"({identical/n_queries*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sqlitesearch")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (e.g. baseline.json)")
    parser.add_argument("--scales", default=None,
                        help="Comma-separated scales (e.g. 1000,10000)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two result files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    scales = SUBSET_SIZES
    if args.scales:
        scales = [int(s) for s in args.scales.split(",")]

    train_ids, train_embs, test_embs, neighbors = load_dataset()

    all_results = []
    for n in scales:
        if n > len(train_embs):
            print(f"\nSkipping {n:,} (only {len(train_embs):,} available)")
            continue
        result = run_benchmark(train_ids, train_embs, test_embs, neighbors, n)
        all_results.append(result)

    print_results_table(all_results)
    print_leaderboard_comparison(all_results)

    if args.output:
        out_path = Path(__file__).parent / "results" / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
