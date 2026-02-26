# Benchmarks

Detailed results and analysis: **[WRITEUP.md](WRITEUP.md)**

## Scripts

| Script | What it does |
|--------|-------------|
| `bench_modes.py` | Compare LSH, IVF, and HNSW modes on the Cohere-768d dataset. Measures recall, latency, QPS, build time, and DB size. |
| `tune_recall.py` | Sweep LSH parameters (n_tables, hash_size, n_probe) to find the best recall/speed tradeoff. |
| `tune_hnsw_search.py` | Build one HNSW index, then sweep ef_search values to find the recall/latency curve. |
| `vector_bench_cohere.py` | Low-level vector search benchmark on Cohere-1M (used for LSH optimization iterations). |
| `run_benchmark.py` | Text search benchmark using Simple English Wikipedia. Compares against minsearch. |
| `run_full_benchmark.py` | Runs the text search benchmark at all dataset sizes (1K to 291K). Designed for AWS EC2. |

## Dataset

Vector benchmarks use the [Cohere Wikipedia-22-12 Medium (1M)](https://cohere.com/embed) dataset — 768-dimensional embeddings, the same dataset used by the [VDBBench leaderboard](https://zilliz.com/vdbbench-leaderboard). See [WRITEUP.md](WRITEUP.md) for download instructions.

## Quick start

```bash
# Compare all vector search modes at 100K
uv run python benchmark/bench_modes.py --n-vectors 100000

# HNSW recall/latency sweep at 1M
uv run python benchmark/tune_hnsw_search.py --n-vectors 1000000 --ef-search 200 300 500 1000

# Text search benchmark (1K docs)
uv run python benchmark/run_benchmark.py -n 1000 -q 50
```

## Results directory

`results/` contains raw JSON output from earlier text search and LSH optimization runs, kept for reference.
