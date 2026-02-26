# Benchmarks

## Text Search

Dataset: Simple English Wikipedia (291K articles, ~1 GB JSONL).

| Docs | Indexing | DB size | Search latency | QPS |
|-----:|---------|--------:|---------------:|----:|
| 1K | 0.57s | 51 MB | 1.0ms | 970 |
| 10K | 6.5s | 346 MB | 1.7ms | 604 |
| 50K | 23.6s | 1,050 MB | 3.5ms | 290 |
| 125K | 60.4s | 2,549 MB | 5.6ms | 179 |
| 291K | 153.5s | 5,559 MB | 90.6ms | 11 |

At 125K docs, sqlitesearch is **2.4x faster** than minsearch (5.6ms vs 13.7ms) with persistence and minimal RAM usage.

## Vector Search at 100K

Dataset: [Cohere Wikipedia-22-12 Medium](https://cohere.com/embed) (768d embeddings), same as the [VDBBench leaderboard](https://zilliz.com/vdbbench-leaderboard).

| Mode | Config | R@10 | R@100 | Latency | QPS | Build | DB |
|------|--------|-----:|------:|--------:|----:|------:|---:|
| HNSW | ef_c64/ef_s300 | **0.939** | **0.937** | **5.5ms** | **181** | 161s | 547 MB |
| HNSW | ef_c16/ef_s300 | 0.918 | 0.917 | 5.6ms | 179 | 69s | 537 MB |
| IVF | 16 probes | 0.922 | 0.860 | 28.6ms | 35 | 39s | 399 MB |
| LSH | n_probe=2 | 0.950 | 0.890 | 181ms | 6 | 9s | 466 MB |

## Vector Search at 1M

| Mode | Config | R@10 | R@100 | Latency | QPS | Build | DB |
|------|--------|-----:|------:|--------:|----:|------:|---:|
| HNSW | ef_s=300 | 0.907 | 0.891 | **6.3ms** | **158** | 639s | 4,099 MB |
| HNSW | ef_s=1000 | 0.953 | 0.945 | 28.1ms | 36 | 639s | 4,099 MB |
| IVF | 16 probes | **0.944** | **0.923** | 219ms | 4.6 | **368s** | 3,974 MB |
| LSH | 64t/8b | 0.950 | 0.810 | 3,993ms | 0.3 | 567s | 8,300 MB |

HNSW is best for low-latency search (6ms, 158 QPS). IVF has the fastest build (6 min) and highest recall at default settings. LSH is impractical at 1M.

For detailed analysis, parameter tuning, and optimization notes, see **[WRITEUP.md](WRITEUP.md)**.

---

## Scripts

| Script | What it does |
|--------|-------------|
| `bench_modes.py` | Compare LSH, IVF, and HNSW on the Cohere-768d dataset |
| `tune_hnsw_search.py` | Build one HNSW index, sweep ef_search values |
| `tune_recall.py` | Sweep LSH parameters (n_tables, hash_size, n_probe) |
| `vector_bench_cohere.py` | Low-level vector search benchmark on Cohere-1M |
| `run_benchmark.py` | Text search benchmark using Simple English Wikipedia |
| `run_full_benchmark.py` | Full text search benchmark at all dataset sizes (1K-291K) |

## Running the benchmarks

### Vector search

Vector benchmarks require the Cohere-1M dataset at `/tmp/vectordb_bench/dataset/cohere_medium_1m/`. See [WRITEUP.md](WRITEUP.md) for download details.

```bash
# Compare all vector search modes at 100K
uv run python benchmark/bench_modes.py --n-vectors 100000

# Compare modes at 1M
uv run python benchmark/bench_modes.py --n-vectors 1000000

# Only HNSW configs
uv run python benchmark/bench_modes.py --n-vectors 100000 --modes hnsw

# HNSW recall/latency sweep (builds once, tests multiple ef_search)
uv run python benchmark/tune_hnsw_search.py --n-vectors 1000000 --ef-search 200 300 500 1000

# LSH parameter tuning
uv run python benchmark/tune_recall.py --n-vectors 100000
```

### Text search

Text search benchmarks use Simple English Wikipedia (291K articles):

```bash
# Download and parse Wikipedia dump
pip install mwparserfromhell
git clone --depth 1 https://github.com/alexeygrigorev/minsearch.git /tmp/minsearch
uv run python /tmp/minsearch/benchmark/download_wikipedia.py
uv run python /tmp/minsearch/benchmark/parse_wikipedia.py benchmark/data/simplewiki-*.xml.bz2

# Run benchmark (subset)
uv run python benchmark/run_benchmark.py -n 10000 -q 100

# Full benchmark (all dataset sizes)
uv run python benchmark/run_full_benchmark.py
```

## Results directory

`results/` contains raw JSON output from earlier text search and LSH optimization runs, kept for reference.
