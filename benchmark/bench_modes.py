#!/usr/bin/env python3
"""
Benchmark LSH vs IVF vs HNSW on the Cohere-768d dataset.

Measures recall@10, recall@100, search latency, insert time, and DB size
at different dataset sizes.

With --filtered, additionally measures filtered search: a synthetic
``category`` attribute is attached to each vector and three scenarios are
benchmarked per config — unfiltered, a narrow filter (selective, exercises
the planner's exact-scan branch) and a broad filter (exercises the
filtered-ANN branch). Recall is computed against brute-force top-k over the
*filtered subset*, so it stays meaningful under a filter. Use a large enough
--n-vectors (>= 30000) for ``broad`` to exceed the exact-scan threshold and
actually hit the filtered-ANN branch.

Usage:
    uv run python benchmark/bench_modes.py
    uv run python benchmark/bench_modes.py --n-vectors 100000
    uv run python benchmark/bench_modes.py --n-vectors 1000 10000 100000
    uv run python benchmark/bench_modes.py --filtered --n-vectors 30000
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

DATASET_DIR = Path(
    os.environ.get("SQLITESEARCH_BENCH_DATA", "/data/vectordb_bench/dataset/cohere_medium_1m")
)
K = 100
N_QUERIES = 100
SEED = 42

# --- filtered-search scenarios --------------------------------------------
# A synthetic category (idx % 10) is attached to every vector when --filtered.
#   narrow -> 1/10 of the corpus  -> exact-scan branch (selective filter)
#   broad  -> 7/10 of the corpus  -> filtered-ANN branch (when n large enough)
N_CATEGORIES = 10
FILTER_SCENARIOS = [
    ("unfiltered", None),
    ("narrow 1cat", {"category": "c0"}),
    ("broad 7cats", {"category": [f"c{i}" for i in range(7)]}),
]
# Default planner threshold (VectorSearchIndex._exact_filter_threshold). A
# scenario whose match count is <= this takes the exact branch, else the
# filtered-ANN branch. Read from the index at runtime when available.
DEFAULT_EXACT_THRESHOLD = 20000


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


def _normalize_rows(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return v / norms


def subset_ground_truth(train_normed, test_normed, mask, k):
    """Per-query top-k over the masked train subset.

    Returns a list (len = #queries) of row-index lists, sorted best-first —
    the same shape as ``load_data``'s ``ground_truth`` so recall@10 (@100)
    can reuse the same slicing.
    """
    global_idx = np.where(mask)[0]
    sub = train_normed[global_idx]
    sims = test_normed @ sub.T  # (Q, n_sub)
    gts = []
    for i in range(len(test_normed)):
        k_eff = min(k, sub.shape[0])
        if k_eff == 0:
            gts.append([])
            continue
        top = np.argpartition(sims[i], -k_eff)[-k_eff:]
        top = top[np.argsort(sims[i, top])[::-1]]
        gts.append([int(global_idx[t]) for t in top])
    return gts


def run_config(train_embs, test_embs, ground_truth, mode, filtered=False, **kwargs):
    """Test one configuration. Returns a list of per-scenario metric dicts."""
    n_vectors = len(train_embs)
    label = kwargs.pop("label", mode)
    db_path = f"/tmp/sqlitesearch_bench_{mode}_{n_vectors}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    payload = [{"idx": i} for i in range(n_vectors)]
    keyword_fields = []
    if filtered:
        for p in payload:
            p["category"] = f"c{p['idx'] % N_CATEGORIES}"
        keyword_fields = ["category"]

    # Build index
    t0 = time.time()
    index = VectorSearchIndex(
        mode=mode,
        keyword_fields=keyword_fields,
        id_field="idx",
        db_path=db_path,
        seed=SEED,
        **kwargs,
    )
    index.fit(train_embs, payload)
    insert_time = time.time() - t0
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # Ground truth per scenario. Unfiltered reuses the precomputed full-set GT;
    # filtered scenarios compute brute-force top-k over their subset.
    if filtered:
        train_normed = _normalize_rows(train_embs)
        test_normed = _normalize_rows(test_embs)
        cats = np.array([p["idx"] % N_CATEGORIES for p in payload])
        scenarios = [("unfiltered", None, ground_truth)]
        for slabel, flt in FILTER_SCENARIOS[1:]:
            wanted = (
                {int(c[1:]) for c in flt["category"]}
                if isinstance(flt["category"], list)
                else {int(flt["category"][1:])}
            )
            mask = np.isin(cats, list(wanted))
            scenarios.append((slabel, flt, subset_ground_truth(train_normed, test_normed, mask, K)))
    else:
        scenarios = [("unfiltered", None, ground_truth)]

    # Warmup
    for i in range(min(5, len(test_embs))):
        index.search(test_embs[i], num_results=K, output_ids=True)

    threshold = getattr(index, "_exact_filter_threshold", None) or DEFAULT_EXACT_THRESHOLD

    rows = []
    for slabel, flt, gts in scenarios:
        latencies = []
        recalls_10 = []
        recalls_100 = []
        for i in range(len(test_embs)):
            t0 = time.time()
            results = index.search(test_embs[i], filter_dict=flt, num_results=K, output_ids=True)
            latencies.append(time.time() - t0)

            res_ids = [r["_id"] for r in results]
            gt = gts[i]
            recalls_100.append(len(set(res_ids) & set(gt)) / min(K, len(gt)))
            recalls_10.append(len(set(res_ids[:10]) & set(gt[:10])) / min(10, len(gt[:10])))

        n_match = n_vectors if flt is None else int(
            np.isin(cats, [int(c[1:]) for c in flt["category"]]).sum()
            if isinstance(flt["category"], list)
            else (cats == int(flt["category"][1:])).sum()
        ) if filtered else n_vectors
        branch = "-" if flt is None else ("exact" if n_match <= threshold else "ann")

        rows.append({
            "label": label,
            "scenario": slabel,
            "n_match": n_match,
            "branch": branch,
            "recall@10": round(float(np.mean(recalls_10)), 4),
            "recall@100": round(float(np.mean(recalls_100)), 4),
            "avg_lat_ms": round(float(np.mean(latencies)) * 1000, 1),
            "p99_lat_ms": round(float(np.percentile(latencies, 99)) * 1000, 1),
            "qps": round(1.0 / float(np.mean(latencies)), 1),
            "insert_s": round(insert_time, 1),
            "db_mb": round(db_size_mb, 1),
        })

    index.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    return rows


# Configurations to benchmark
CONFIGS = [
    # LSH configs
    {"mode": "lsh", "label": "LSH 8t/16b/p2", "n_tables": 8, "hash_size": 16, "n_probe": 2},
    {"mode": "lsh", "label": "LSH 32t/8b/p2", "n_tables": 32, "hash_size": 8, "n_probe": 2},
    {"mode": "lsh", "label": "LSH 64t/6b/p2", "n_tables": 64, "hash_size": 6, "n_probe": 2},
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


def _fmt_unfiltered(rows):
    """Original leaderboard table (one row per config)."""
    print(
        f"{'Config':<20} {'R@10':>6} {'R@100':>6}  "
        f"{'avg(ms)':>8} {'p99(ms)':>8} {'QPS':>7}  "
        f"{'insert':>7} {'DB(MB)':>7}"
    )
    print("-" * 88)
    for r in rows:
        r = r[0]
        print(
            f"{r['label']:<20} {r['recall@10']:>6.4f} {r['recall@100']:>6.4f}  "
            f"{r['avg_lat_ms']:>8.1f} {r['p99_lat_ms']:>8.1f} {r['qps']:>7.1f}  "
            f"{r['insert_s']:>6.1f}s {r['db_mb']:>7.1f}"
        )


def _fmt_filtered(rows, threshold):
    """Filter-aware table (one row per config x scenario)."""
    print(
        f"{'Config':<20} {'scenario':<13} {'n_match':>9} {'branch':>7}  "
        f"{'R@10':>6} {'R@100':>6}  {'avg(ms)':>8} {'p99(ms)':>8} {'QPS':>7}"
    )
    print("-" * 100)
    for cfg_rows in rows:
        for r in cfg_rows:
            print(
                f"{r['label']:<20} {r['scenario']:<13} {r['n_match']:>9,} {r['branch']:>7}  "
                f"{r['recall@10']:>6.4f} {r['recall@100']:>6.4f}  "
                f"{r['avg_lat_ms']:>8.1f} {r['p99_lat_ms']:>8.1f} {r['qps']:>7.1f}"
            )
    print(f"\n(branch: exact when n_match <= threshold {threshold:,}, else ann)")


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
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Also benchmark filtered search (narrow + broad scenarios). "
        "Use --n-vectors >= 30000 so 'broad' hits the filtered-ANN branch.",
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

        print(f"\n{'=' * 95}")
        print(
            f"Benchmarking {n_vectors:,} vectors  |  {N_QUERIES} queries  |  top-{K}"
            + ("  |  FILTERED" if args.filtered else "")
        )
        print(f"{'=' * 95}")

        all_rows = []
        for cfg in configs:
            cfg = cfg.copy()
            rows = run_config(train_embs, test_embs, ground_truth, filtered=args.filtered, **cfg)
            all_rows.append(rows)
            for r in rows:
                print(
                    f"{r['label']:<20} {r['scenario']:<13}  "
                    f"R@10={r['recall@10']:.4f} R@100={r['recall@100']:.4f}  "
                    f"avg={r['avg_lat_ms']:.1f}ms QPS={r['qps']:.1f}"
                )

        threshold = DEFAULT_EXACT_THRESHOLD
        print(f"\n{'=' * 60}")
        if args.filtered:
            _fmt_filtered(all_rows, threshold)
            print("\nBest filtered recall@100 (broad scenario):")
            broad = [r for cfg in all_rows for r in cfg if r["scenario"] == "broad 7cats"]
            for r in sorted(broad, key=lambda r: -r["recall@100"])[:3]:
                print(
                    f"  {r['label']:<20}  R@100={r['recall@100']:.4f}  "
                    f"lat={r['avg_lat_ms']:.1f}ms  QPS={r['qps']:.1f}"
                )
        else:
            flat = [r[0] for r in all_rows]
            _fmt_unfiltered(all_rows)
            print("\nBest by recall@100:")
            for r in sorted(flat, key=lambda r: -r["recall@100"])[:3]:
                print(
                    f"  {r['label']:<20}  R@100={r['recall@100']:.4f}  "
                    f"lat={r['avg_lat_ms']:.1f}ms  QPS={r['qps']:.1f}"
                )
            print("\nBest by QPS:")
            for r in sorted(flat, key=lambda r: -r["qps"])[:3]:
                print(
                    f"  {r['label']:<20}  QPS={r['qps']:.1f}  "
                    f"R@100={r['recall@100']:.4f}  lat={r['avg_lat_ms']:.1f}ms"
                )


if __name__ == "__main__":
    main()
