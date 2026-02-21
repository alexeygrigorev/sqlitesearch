# sqlitesearch Benchmark

Benchmarks for `sqlitesearch.TextSearchIndex` using Simple English Wikipedia,
compared against [minsearch](https://github.com/alexeygrigorev/minsearch).

## Dataset

| Metric | Value |
|--------|-------|
| Source | Simple English Wikipedia dump |
| Documents | 291,737 articles |
| Text size | ~1 GB (JSONL) |
| Compressed | ~444 MB (.xml.bz2) |

## Setup

**Hardware**: EC2 t3.large (2 vCPU, 8 GB RAM), eu-west-1, gp3 storage.

**Tokenizer**: `Tokenizer(stop_words='english', stemmer=porter_stemmer)` —
same stopwords and stemming as minsearch.

## Results

### sqlitesearch (all cutting points)

| Docs | Indexing | DB size | Search avg | QPS |
|------|----------|---------|-----------|-----|
| 1K | 0.57s | 51 MB | 1.03 ms | 970 |
| 10K | 6.54s | 346 MB | 1.66 ms | 604 |
| 50K | 23.62s | 1,050 MB | 3.45 ms | 290 |
| 125K | 60.41s | 2,549 MB | 5.59 ms | 179 |
| 291K | 153.51s | 5,559 MB | 90.62 ms | 11.0 |

### minsearch (125K — max for 8 GB RAM)

| Metric | Regular Index | AppendableIndex |
|--------|---------------|-----------------|
| Indexing | 57.76s | 79.08s |
| Search avg | 1,043.73 ms | 13.66 ms |
| QPS | 1.0 | 73 |

minsearch with 291K docs does NOT fit in 8 GB RAM.

### Head-to-head: 125K docs

| Metric | minsearch Regular | minsearch Appendable | **sqlitesearch** |
|--------|-------------------|----------------------|-----------------|
| Indexing | 57.76s | 79.08s | **60.41s** |
| Search avg | 1,043.73 ms | 13.66 ms | **5.59 ms** |
| QPS | 1.0 | 73 | **179** |
| 291K docs | OOM | OOM | **Works (11 QPS)** |
| Persistence | No | No | **Yes (SQLite)** |
| RAM usage | ~6 GB | ~6 GB | **Minimal** |

### Key findings

1. **Search: sqlitesearch is 2.4x faster.** At 125K docs, sqlitesearch (5.6ms, 179 QPS)
   beats both minsearch AppendableIndex (14ms, 73 QPS) and Regular Index (1,044ms, 1 QPS).

2. **Indexing speed is comparable.** sqlitesearch (60s) is between minsearch Regular (58s)
   and Appendable (79s).

3. **sqlitesearch handles the full dataset.** 291K docs works on 8 GB RAM at 91ms/query.
   minsearch cannot fit 291K docs in memory.

4. **Persistence is the killer feature.** sqlitesearch indexes once and persists to disk.
   minsearch must re-index from scratch on every restart.

5. **Search scales sub-linearly up to 125K.** From 1K to 125K (125x more docs), search
   time goes from 1.0ms to 5.6ms (only 5.6x slower). The jump at 291K (91ms) is due to
   stemmer-generated terms that bypass stopword filtering — see "Potential improvements" below.

## Optimizations applied

### SQLite tuning
- `PRAGMA journal_mode=WAL` — Write-Ahead Logging for concurrent reads
- `PRAGMA synchronous=NORMAL` — relaxed durability for throughput
- `PRAGMA cache_size=-64000` — 64 MB page cache

### Batch inserts
- `executemany()` for both docs and FTS5 tables (vs one-by-one `execute()`)

### Stopwords removal
- Filter 174 common English words from queries before sending to FTS5
- Prevents OR queries like `all OR i OR ever OR wanted` from scanning the entire index

### Subquery optimization
- When no filters: rank inside the FTS5 subquery with LIMIT, then JOIN only top results
- Avoids joining 290K rows when only 10 are needed

### Impact of search optimizations (291K docs)

| Version | Search avg | QPS | vs baseline |
|---------|-----------|-----|-------------|
| Baseline (no optimizations) | 11,611 ms | 0.1 | — |
| + stopwords removal | 5,142 ms | 0.2 | 2.3x |
| + subquery optimization | 118 ms | 8.5 | **98x** |
| + stemmer (porter) | 91 ms | 11.0 | **128x** |

## Potential improvements

The 125K→291K jump (179 QPS → 11 QPS) is disproportionate. At 2.3x more docs,
search should be ~2-3x slower, not 16x. The root cause: the Porter stemmer
produces terms that are not in the stopword list, so high-frequency stems
like "want" (from "wanted") or "us" (from "use") match hundreds of thousands
of documents. FTS5 must score all of them before the LIMIT can take effect.

### 1. Stem the stopword list

**Impact: High | Effort: Low**

Currently stopwords are filtered _before_ stemming. So "wanted" is not a
stopword, passes through, gets stemmed to "want", and matches nearly every
document. Fix: also apply the stemmer to the stopword set so stemmed forms
are filtered too.

```python
# Current
self.stop_words = DEFAULT_ENGLISH_STOP_WORDS

# Improved
if self.stemmer:
    self.stop_words = {self.stemmer(w) for w in DEFAULT_ENGLISH_STOP_WORDS}
    self.stop_words |= DEFAULT_ENGLISH_STOP_WORDS
```

### 2. Term frequency pruning

**Impact: High | Effort: Medium**

Query FTS5's built-in vocab table to check how many documents a term appears
in. Drop terms that appear in more than X% of documents (e.g. 10%) — they
add little ranking value and are expensive to score.

```python
# FTS5 exposes term stats via a virtual table
cursor.execute("""
    SELECT doc_count FROM docs_fts_vocab
    WHERE term = ? AND col = '*'
""", [term])
```

### 3. Move stemming into FTS5 tokenizer

**Impact: High | Effort: Medium**

Currently stemming happens in Python: we tokenize the query, stem each term,
then build an OR query string. FTS5 then re-tokenizes that string with its own
tokenizer (unicode61). Moving stemming into FTS5's tokenizer layer (via
`tokenize='porter unicode61'`) means:
- Stemming happens at the C level during both indexing and querying
- No need to build synthetic OR queries in Python
- FTS5 can use its internal optimizations (e.g. prefix-compressed posting lists)

The `stemming=True` parameter already exists but isn't used with the Tokenizer.

### 4. Use AND instead of OR for multi-term queries

**Impact: Medium | Effort: Low**

Currently `"quick brown fox"` becomes `quick OR brown OR fox` — matching any
document with any of those terms. Switching to AND narrows the candidate set
dramatically (intersection vs union). Users who want OR can still pass FTS5
syntax directly.

### 5. Run FTS5 optimize after indexing

**Impact: Low-Medium | Effort: Low**

After batch `fit()`, merge FTS5's internal b-tree segments for faster reads:

```python
cursor.execute("INSERT INTO docs_fts(docs_fts) VALUES('optimize')")
```

This is a one-time cost after indexing that can speed up all subsequent queries.

### 6. Content-less FTS5 table

**Impact: Medium | Effort: Medium**

Currently text is stored twice: in `docs.doc_json` and inside FTS5's internal
tables. Using `content=''` or `content=docs` avoids the duplication, cutting
DB size roughly in half (~2.5 GB instead of 5.5 GB for 291K docs). Smaller DB
means better cache utilization and faster I/O.

### 7. Benchmark query analysis

Some of the 100 benchmark queries may be outliers (single high-frequency stem).
Logging per-query times and the generated FTS5 query would help identify
exactly which queries cause the 91ms average and guide targeted fixes.

## Running the benchmarks

### Local (subset)

```bash
cd sqlitesearch

# 1K docs
uv run python benchmark/run_benchmark.py -n 1000 -q 50

# 10K docs
uv run python benchmark/run_benchmark.py -n 10000 -q 100
```

### AWS (full dataset)

```bash
# Launch EC2 t3.large (eu-west-1, Ubuntu 24.04, 30 GB gp3)
ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and prepare data
git clone https://github.com/alexeygrigorev/sqlitesearch.git
cd sqlitesearch/benchmark
git clone --depth 1 https://github.com/alexeygrigorev/minsearch.git /tmp/minsearch
uv run python /tmp/minsearch/benchmark/download_wikipedia.py
uv run python /tmp/minsearch/benchmark/parse_wikipedia.py data/simplewiki-*.xml.bz2

# Run benchmark (all cutting points)
uv run python run_benchmark.py -q 100

# Or simplified version
uv run python run_full_benchmark.py
```

### CLI options (run_benchmark.py)

```
-i, --input       Path to JSONL file (default: data/wikipedia_docs.jsonl)
-n, --num-docs    Max documents (default: all)
-q, --num-queries Number of queries (default: 100)
```

### Cost

- t3.large: ~$0.08/hour
- Full benchmark: ~10-30 minutes
- Total cost: ~$0.02-0.04 per run
