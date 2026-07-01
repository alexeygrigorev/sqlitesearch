#!/usr/bin/env python3
"""Compare the vector search modes (lsh, lsh_int8, ivf, hnsw).

Runs each mode in its own subprocess, builds the index over a synthetic or real
Cohere-768d corpus, searches, and reports fit time, latency (avg/p99), recall
vs a brute-force ground truth, DB size, and resident/peak RSS.

WARNING -- run inside a memory-limited cgroup. This harness holds the full
corpus plus a brute-force ground-truth similarity matrix in RAM and can peak at
several GB (~2 GB at 100K Cohere-768d, ~10 GB at 1M). On a shared box that
drives the whole system into memory pressure, and the OOM killer then reclaims
*other* processes (your shell, tmux sessions, other agents), not the benchmark
that tripped it -- an unbounded run has killed tmux sessions this way. Cap it:

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=12G \
        uv run python benchmark/bench_vector_modes.py ...
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlitesearch import VectorSearchIndex  # noqa: E402

COHERE_DIR = Path(
    os.environ.get(
        "SQLITESEARCH_BENCH_DATA",
        "/data/vectordb_bench/dataset/cohere_medium_1m",
    )
)


def _normalize_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _ground_truth(vectors, queries, k):
    """Brute-force top-k per query.

    Normalizes the corpus in chunks instead of materializing a second full
    copy of ``vectors`` -- at 1M x 768d that copy alone is ~3 GB. Peak extra
    memory here is one chunk plus the (n_queries x n_vectors) similarity row
    block, not a duplicate of the whole corpus.
    """
    q = _normalize_rows(queries)
    n = len(vectors)
    sims = np.empty((len(q), n), dtype=np.float32)
    chunk = 50_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = _normalize_rows(vectors[start:end])
        sims[:, start:end] = q @ block.T
    out = []
    for i in range(len(queries)):
        top = np.argpartition(sims[i], -k)[-k:]
        top = top[np.argsort(sims[i, top])[::-1]]
        out.append([int(x) for x in top])
    return out


def _list_column_to_numpy(col):
    """Convert an Arrow large_list<float> column to a (n, dim) float32 array
    without going through Python objects (``.to_pylist()`` would briefly
    materialize one Python float per value -- ~34 GB at 1M x 768d)."""
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return flat.reshape(len(col), -1)


def _load_cohere(n_vectors, n_queries):
    """Load the first ``n_vectors`` real Cohere-768d vectors and ``n_queries``
    test queries from the VDBBench parquet dataset."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(COHERE_DIR / "shuffle_train.parquet")
    vectors = None
    filled = 0
    remaining = n_vectors
    for batch in pf.iter_batches(batch_size=min(100_000, n_vectors), columns=["emb"]):
        take = min(remaining, len(batch))
        block = _list_column_to_numpy(batch.column("emb")[:take])
        if vectors is None:
            # Pre-allocate the full array so we never hold two copies (a
            # running np.concatenate would peak at ~2x the corpus).
            vectors = np.empty((n_vectors, block.shape[1]), dtype=np.float32)
        vectors[filled:filled + take] = block
        filled += take
        remaining -= take
        if remaining <= 0:
            break

    test = pq.read_table(COHERE_DIR / "test.parquet", columns=["emb"])
    queries = _list_column_to_numpy(test.column("emb").combine_chunks())[:n_queries]
    return vectors, queries


def _load_vectors(args):
    """Return (vectors, queries) for the selected dataset."""
    if args.dataset == "cohere":
        return _load_cohere(args.n_vectors, args.n_queries)
    rng = np.random.default_rng(args.seed)
    vectors = rng.standard_normal(size=(args.n_vectors, args.dim)).astype(np.float32)
    queries = rng.standard_normal(size=(args.n_queries, args.dim)).astype(np.float32)
    return vectors, queries


def _cleanup(path):
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _peak_rss_mb():
    # Linux reports ru_maxrss in KiB. This benchmark is intended for the Linux
    # dev/CI environment used by the project.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_mode(mode, vectors, queries, ground_truth, args):
    fd, db_path = tempfile.mkstemp(prefix=f"sqlitesearch_{mode}_", suffix=".db")
    os.close(fd)
    os.unlink(db_path)

    payload = [{"idx": i, "category": f"c{i % 10}"} for i in range(len(vectors))]
    # Baseline RSS: interpreter + the harness's vector/query/ground-truth
    # arrays, measured before the index exists. rss_after_fit - baseline is
    # then the index's own resident memory (for Python modes this includes the
    # _cached_vectors copy; the disk-backed modes stay near zero).
    rss_baseline_mb = _rss_mb()
    index = VectorSearchIndex(
        mode=mode,
        keyword_fields=["category"],
        id_field="idx",
        db_path=db_path,
        backend=args.backend,
        seed=args.seed,
        **_mode_kwargs(mode, args),
    )

    t0 = time.perf_counter()
    index.fit(vectors, payload)
    fit_s = time.perf_counter() - t0
    cache_loaded_after_fit = index._cached_vectors is not None
    rss_after_fit_mb = _rss_mb()

    for query in queries[: min(5, len(queries))]:
        index.search(query, num_results=args.k, output_ids=True)

    latencies = []
    recalls = []
    for query, gt in zip(queries, ground_truth):
        t0 = time.perf_counter()
        results = index.search(query, num_results=args.k, output_ids=True)
        latencies.append(time.perf_counter() - t0)
        ids = [r["_id"] for r in results]
        recalls.append(len(set(ids) & set(gt)) / min(args.k, len(gt)))

    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    cache_loaded_after_search = index._cached_vectors is not None
    rss_after_search_mb = _rss_mb()
    peak_rss_mb = _peak_rss_mb()
    index.close()
    _cleanup(db_path)

    return {
        "mode": mode,
        "fit_s": fit_s,
        "avg_ms": float(np.mean(latencies)) * 1000,
        "p99_ms": float(np.percentile(latencies, 99)) * 1000,
        "recall": float(np.mean(recalls)),
        "db_mb": db_size_mb,
        "rss_baseline_mb": rss_baseline_mb,
        "rss_fit_mb": rss_after_fit_mb,
        "rss_search_mb": rss_after_search_mb,
        "peak_rss_mb": peak_rss_mb,
        "cache_after_fit": cache_loaded_after_fit,
        "cache_after_search": cache_loaded_after_search,
    }


def _mode_kwargs(mode, args):
    if "lsh" in mode:
        return {
            "n_tables": args.n_tables,
            "hash_size": args.hash_size,
            "n_probe": args.n_probe,
        }
    if mode.endswith("ivf"):
        return {"n_probe_clusters": args.n_probe_clusters}
    if mode.endswith("hnsw"):
        return {
            "m": args.hnsw_m,
            "ef_construction": args.hnsw_ef_construction,
            "ef_search": args.hnsw_ef_search,
        }
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=10_000)
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "cohere"],
        default="synthetic",
        help="synthetic standard-normal vectors, or the real Cohere-768d dataset",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--backend",
        choices=["sqlite3", "libsql"],
        default="sqlite3",
        help="storage backend: stdlib sqlite3 (default) or libsql local file",
    )
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-tables", type=int, default=8)
    parser.add_argument("--hash-size", type=int, default=16)
    parser.add_argument("--n-probe", type=int, default=2)
    parser.add_argument("--n-probe-clusters", type=int, default=8)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=80)
    parser.add_argument("--hnsw-ef-search", type=int, default=60)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["lsh", "lsh_int8", "ivf", "hnsw"],
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Run all modes in this process instead of one subprocess per mode.",
    )
    parser.add_argument("--_worker-mode", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    vectors, queries = _load_vectors(args)
    ground_truth = _ground_truth(vectors, queries, args.k)

    if args._worker_mode:
        print(json.dumps(run_mode(args._worker_mode, vectors, queries, ground_truth, args)))
        return

    if args.no_isolate:
        rows = [run_mode(mode, vectors, queries, ground_truth, args) for mode in args.modes]
    else:
        rows = [_run_isolated(mode, args) for mode in args.modes]

    print(
        f"{'mode':<10} {'fit_s':>8} {'avg_ms':>8} {'p99_ms':>8} "
        f"{'recall':>8} {'db_mb':>8} {'base':>7} {'rss_fit':>8} {'index':>7} "
        f"{'peak':>8} {'cache':>6}"
    )
    print("-" * 96)
    for row in rows:
        print(
            f"{row['mode']:<10} {row['fit_s']:>8.2f} {row['avg_ms']:>8.2f} "
            f"{row['p99_ms']:>8.2f} {row['recall']:>8.3f} {row['db_mb']:>8.1f} "
            f"{_fmt_mb(row['rss_baseline_mb']):>7} {_fmt_mb(row['rss_fit_mb']):>8} "
            f"{_fmt_index_mb(row):>7} {_fmt_mb(row['peak_rss_mb']):>8} "
            f"{str(row['cache_after_fit']):>6}"
        )
    print(
        "\nbase   = RSS before the index exists (interpreter + harness vector/GT arrays)\n"
        "rss_fit= RSS after fit()   index = rss_fit - base (the index's own RAM)\n"
        "cache  = whether the index keeps a full float32 vector copy in RAM"
    )


def _fmt_mb(value):
    if value is None:
        return "?"
    return f"{value:.0f}"


def _fmt_index_mb(row):
    base = row.get("rss_baseline_mb")
    fit = row.get("rss_fit_mb")
    if base is None or fit is None:
        return "?"
    return f"{fit - base:.0f}"


def _run_isolated(mode, args):
    cmd = [
        sys.executable,
        __file__,
        "--n-vectors",
        str(args.n_vectors),
        "--dataset",
        args.dataset,
        "--backend",
        args.backend,
        "--dim",
        str(args.dim),
        "--n-queries",
        str(args.n_queries),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--n-tables",
        str(args.n_tables),
        "--hash-size",
        str(args.hash_size),
        "--n-probe",
        str(args.n_probe),
        "--n-probe-clusters",
        str(args.n_probe_clusters),
        "--hnsw-m",
        str(args.hnsw_m),
        "--hnsw-ef-construction",
        str(args.hnsw_ef_construction),
        "--hnsw-ef-search",
        str(args.hnsw_ef_search),
        "--_worker-mode",
        mode,
    ]
    output = subprocess.check_output(cmd, text=True)
    return json.loads(output)


if __name__ == "__main__":
    main()
