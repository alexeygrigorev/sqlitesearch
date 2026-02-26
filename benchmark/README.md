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

## Running the benchmarks

### Vector search

Vector benchmarks require the Cohere-1M dataset. Download it to `/tmp/vectordb_bench/dataset/cohere_medium_1m/` (see [WRITEUP.md](WRITEUP.md) for S3 URLs).

```bash
# Compare all vector search modes at 100K
uv run python benchmark/bench_modes.py --n-vectors 100000

# Compare modes at 1M (takes ~30 min total)
uv run python benchmark/bench_modes.py --n-vectors 1000000

# Only HNSW configs
uv run python benchmark/bench_modes.py --n-vectors 100000 --modes hnsw

# Filter by config label
uv run python benchmark/bench_modes.py --n-vectors 1000000 --configs "HNSW-fast"

# HNSW recall/latency sweep (builds once, tests multiple ef_search)
uv run python benchmark/tune_hnsw_search.py --n-vectors 1000000 --ef-search 200 300 500 1000

# HNSW with custom m and ef_construction
uv run python benchmark/tune_hnsw_search.py --n-vectors 1000000 --m 20 --ef-construction 32 --ef-search 200 500

# LSH parameter tuning
uv run python benchmark/tune_recall.py --n-vectors 100000
```

### Text search

Text search benchmarks use Simple English Wikipedia (291K articles). Download the dataset first:

```bash
# Download and parse Wikipedia dump
pip install mwparserfromhell  # or: uv add mwparserfromhell
git clone --depth 1 https://github.com/alexeygrigorev/minsearch.git /tmp/minsearch
uv run python /tmp/minsearch/benchmark/download_wikipedia.py
uv run python /tmp/minsearch/benchmark/parse_wikipedia.py benchmark/data/simplewiki-*.xml.bz2

# Run benchmark (subset)
uv run python benchmark/run_benchmark.py -n 10000 -q 100

# Run full benchmark (all dataset sizes: 1K, 10K, 50K, 125K, 291K)
uv run python benchmark/run_full_benchmark.py
```

**CLI options for `run_benchmark.py`:**

```
-i, --input       Path to JSONL file (default: data/wikipedia_docs.jsonl)
-n, --num-docs    Max documents (default: all)
-q, --num-queries Number of queries (default: 100)
```

## Results directory

`results/` contains raw JSON output from earlier text search and LSH optimization runs, kept for reference.
