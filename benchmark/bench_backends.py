#!/usr/bin/env python3
"""Benchmark the storage backends against each other.

Compares vector ingest (`fit`) time and search latency across:

* ``sqlite3``        -- stdlib, local file (baseline)
* ``libsql-local``   -- libSQL, local file (no sync)
* ``turso-local``    -- pyturso, in-process local engine
* ``libsql-replica`` -- libSQL embedded replica syncing to the local Turso
  emulator (``dev/turso_emulator.py``); reports the number of write
  round-trips, which is what makes naive bulk ingest slow on Turso (issue #3).

Backends whose optional dependency isn't installed are skipped. Vector search
is used because every backend supports it (pyturso has no FTS5).

Usage:
    uv run python benchmark/bench_backends.py
    uv run python benchmark/bench_backends.py --n 2000 --dim 128
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev"))

from sqlitesearch import VectorSearchIndex  # noqa: E402


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def _make_data(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    docs = [{"doc_id": i, "section": "a" if i % 2 else "b"} for i in range(n)]
    queries = vectors[:: max(1, n // 50)][:50]
    return vectors, docs, queries


def _index(db_path, mode, **kw):
    return VectorSearchIndex(
        keyword_fields=["section"], id_field="doc_id", db_path=db_path,
        mode=mode, seed=7, **kw,
    )


def _time_search(index, queries):
    t0 = time.perf_counter()
    for q in queries:
        index.search(q, num_results=10)
    return (time.perf_counter() - t0) / len(queries) * 1000.0  # ms/query


def bench_local(label, n, dim, mode, **kw):
    vectors, docs, queries = _make_data(n, dim)
    idx = _index(_fresh_db(), mode, **kw)
    t0 = time.perf_counter()
    idx.fit(vectors, docs)
    fit_s = time.perf_counter() - t0
    latency_ms = _time_search(idx, queries)
    return {"backend": label, "fit_s": fit_s, "search_ms": latency_ms, "round_trips": "-"}


def bench_replica(n, dim, mode):
    """libSQL embedded replica against the local Turso emulator."""
    try:
        import libsql  # noqa: F401
        from turso_emulator import EmulatorProcess
    except Exception as e:  # libsql not installed / emulator import failed
        print(f"  (skipping libsql-replica: {e})")
        return None

    vectors, docs, queries = _make_data(n, dim)
    # The emulator doesn't replicate frames back to the writing replica, so the
    # pre-write _is_empty() read can't be served; skip it. We're measuring the
    # write round-trips, which is the point.
    orig = VectorSearchIndex._is_empty
    VectorSearchIndex._is_empty = lambda self: True
    try:
        with EmulatorProcess(port=8300) as emu:
            # No id_field here: the upsert-dedup path reads ids back, which the
            # emulator can't replicate to the writing replica. The default
            # (lastrowid) ingest path is what we want to measure anyway.
            idx = VectorSearchIndex(
                keyword_fields=["section"], db_path=_fresh_db(), mode=mode, seed=7,
                backend="libsql", sync_url=emu.url, auth_token="x",
            )
            t0 = time.perf_counter()
            idx.fit(vectors, docs)
            fit_s = time.perf_counter() - t0
            round_trips = emu.stats()["push_count"]
    finally:
        VectorSearchIndex._is_empty = orig
    # Search latency is the local-file case (reads hit the local replica), so we
    # report ingest + round-trips here and leave search to the local rows above.
    return {"backend": "libsql-replica", "fit_s": fit_s, "search_ms": float("nan"),
            "round_trips": round_trips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of documents")
    ap.add_argument("--dim", type=int, default=64, help="vector dimension")
    ap.add_argument("--mode", default="lsh", choices=["lsh", "ivf", "hnsw"])
    args = ap.parse_args()

    print(f"Backends benchmark: n={args.n} dim={args.dim} mode={args.mode}\n")
    rows = [bench_local("sqlite3", args.n, args.dim, args.mode, backend="sqlite3")]

    try:
        import libsql  # noqa: F401
        rows.append(bench_local("libsql-local", args.n, args.dim, args.mode, backend="libsql"))
    except Exception:
        print("  (skipping libsql-local: libsql not installed)")

    try:
        import turso  # noqa: F401
        rows.append(bench_local("turso-local", args.n, args.dim, args.mode, backend="turso"))
    except Exception:
        print("  (skipping turso-local: pyturso not installed)")

    replica = bench_replica(args.n, args.dim, args.mode)
    if replica:
        rows.append(replica)

    print(f"\n{'backend':<16}{'fit (s)':>10}{'search (ms)':>14}{'write RTs':>12}")
    print("-" * 52)
    for r in rows:
        sm = "-" if r["search_ms"] != r["search_ms"] else f"{r['search_ms']:.3f}"  # nan check
        print(f"{r['backend']:<16}{r['fit_s']:>10.3f}{sm:>14}{str(r['round_trips']):>12}")
    print(
        "\nNote: libsql-replica forwards each write to the remote; 'write RTs' is\n"
        "the number of round-trips for fit(). Batched multi-row inserts (issue #3)\n"
        "keep it ~O(n/chunk) instead of O(n)."
    )


if __name__ == "__main__":
    main()
