# sqlitesearch Benchmark Results

## Recommendation

sqlitesearch is designed for **small, local projects** — no servers, no dependencies beyond numpy, just a single SQLite file.

| Use case | Scale | Verdict |
|----------|------:|---------|
| Text search | up to 125K docs | Excellent — 2.4x faster than minsearch |
| Text search | 291K docs | Works but slower (11 QPS) |
| Vector search | up to 100K vectors | Good — sub-100ms latency |
| Vector search | 1M+ vectors | Too slow — use a dedicated vector DB |

---

## Text Search Benchmark

### Dataset

Simple English Wikipedia: 291,737 articles (~1 GB JSONL).

### Results

| Docs | Indexing | DB size | Search avg | QPS |
|------|----------|---------|-----------|-----|
| 1K | 0.57s | 51 MB | 1.03 ms | 970 |
| 10K | 6.54s | 346 MB | 1.66 ms | 604 |
| 50K | 23.62s | 1,050 MB | 3.45 ms | 290 |
| 125K | 60.41s | 2,549 MB | 5.59 ms | 179 |
| 291K | 153.51s | 5,559 MB | 90.62 ms | 11 |

### Comparison: sqlitesearch vs minsearch (125K docs)

| Metric | minsearch Regular | minsearch Appendable | **sqlitesearch** |
|--------|-------------------|----------------------|-----------------|
| Indexing | 57.76s | 79.08s | **60.41s** |
| Search avg | 1,043.73 ms | 13.66 ms | **5.59 ms** |
| QPS | 1.0 | 73 | **179** |
| 291K docs | OOM | OOM | **Works (11 QPS)** |
| Persistence | No | No | **Yes (SQLite)** |
| RAM usage | ~6 GB | ~6 GB | **Minimal** |

### Text search optimizations applied

1. **SQLite WAL mode + cache tuning** — `journal_mode=WAL`, `synchronous=NORMAL`, 64 MB cache
2. **Batch inserts** — `executemany()` for both docs and FTS5 tables
3. **Stopwords removal** — filter 174 common English words before FTS5 query
4. **Subquery optimization** — rank inside FTS5 subquery with LIMIT, then JOIN only top results

**Impact at 291K docs**: baseline 11,611ms/query → optimized 91ms/query (**128x faster**).

---

## Vector Search Benchmark

### Dataset

Cohere-1M (from VDBBench): 1,000,000 Wikipedia embeddings, 768 dimensions, cosine similarity.
Pre-computed ground truth (top-100 neighbors) for 1,000 test queries.

### Results (optimized, seed=42, 8 tables, 16 hash bits)

| N vectors | Insert | vec/s | Recall@100 | Avg lat | P99 lat | QPS | DB size |
|----------:|-------:|------:|-----------:|--------:|--------:|----:|--------:|
| 1,000 | 0.19s | 5,367 | 0.1267 | 0.5ms | 0.9ms | 2,152 | 5 MB |
| 10,000 | 1.12s | 8,958 | 0.1923 | 6.2ms | 19.5ms | 162 | 47 MB |
| 100,000 | 11.88s | 8,419 | 0.1999 | 56ms | 117ms | 18 | 466 MB |
| 1,000,000 | 141s | 7,087 | 0.2253 | 365ms | 522ms | 3 | 4,666 MB |

### Baseline vs optimized (same seed=42)

| Scale | Metric | Before | After | Change |
|------:|--------|-------:|------:|-------:|
| 1K | QPS | 1,261 | 2,152 | **+71%** |
| 1K | P99 latency | 1.7ms | 0.9ms | **-47%** |
| 10K | Insert time | 1.39s | 1.12s | **-19%** |
| 100K | Insert time | 16.7s | 11.9s | **-29%** |
| 100K | Avg latency | 71.9ms | 56.0ms | **-22%** |
| 100K | P99 latency | 171.6ms | 117.1ms | **-32%** |
| All | Recall@100 | identical | identical | **0%** |

### Result ID verification (baseline vs optimized)

Using fixed seed=42, per-query result IDs were compared before and after optimization:

```
     1,000: 100/100 queries return identical IDs (100.0%)
    10,000:  99/100 queries return identical IDs  (99.0%)
   100,000:  99/100 queries return identical IDs  (99.0%)
```

The 1% difference is floating-point ordering of near-tied cosine scores. Recall is exactly the same.

### VDBBench leaderboard comparison (Cohere-1M)

```
Database                            QPS  P99(ms)   Recall
----------------------------------------------------------
ZillizCloud-8cu-perf              9,704      2.5   0.9170
Milvus-16c64g-sq8                 3,465      2.2   0.9530
OpenSearch-16c128g-fm             3,055      7.2   0.9070
ElasticCloud-8c60g-fm             1,925     11.3   0.8960
QdrantCloud-16c64g                1,242      6.4   0.9470
Pinecone-p2.x8                    1,147     13.7   0.9260
----------------------------------------------------------
sqlitesearch [100K]                  18    117.1   0.1999
sqlitesearch [1M]                     3    522.2   0.2253
```

Note: Leaderboard = multi-process on cloud hardware (8-16 cores, 32-128GB RAM). sqlitesearch = serial single-process.

### Vector search optimizations applied

1. **Vectorized batch hashing** — single matmul for all vectors x all tables during insert
2. **Vectorized query hashing** — single matmul for all 8 tables during search
3. **Consolidated LSH query** — single SQL with `GROUP BY doc_id ORDER BY hits DESC` (multi-probe ranking)
4. **Raw bytes instead of pickle** — `tobytes()`/`frombuffer()` for vector storage
5. **Vectorized cosine reranking** — matrix multiply + `argpartition` for top-K
6. **Chunked IN-queries** — avoids SQLite variable limit for large candidate sets
7. **Multi-probe candidate cap** — limits reranking to top 50K candidates by table-hit count

### Why 1M doesn't scale

At 1M with 8 tables / 16 hash bits, each query produces 30K-120K LSH candidates (3-12% of the dataset). Reranking requires fetching 3KB BLOBs for each candidate from a 4.7GB SQLite file — fundamentally I/O-bound. Purpose-built vector databases use HNSW or IVF indexes with in-memory graph traversal, which scales to millions of vectors.

---

All 102 existing tests pass after both text and vector optimizations.
