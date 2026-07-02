# sqlitesearch

A tiny, SQLite-backed search library for small-scale projects with up to 100,000 documents. It provides text search, vector search, and hybrid search - all stored in a single `.db` file with zero infrastructure.

sqlitesearch is a persistent sibling of [minsearch](https://github.com/alexeygrigorev/minsearch) - same API, but stores data on disk.

## Installation

```bash
uv add sqlitesearch
# or: pip install sqlitesearch
```

Optional extras:

- `sqlitesearch[libsql]` — back the index with libSQL / Turso Cloud (see [Storage backends](#storage-backends)).

## Text Search

Text search uses SQLite's FTS5 (Full-Text Search) extension with BM25 ranking.

### Basic Usage

```python
from sqlitesearch import TextSearchIndex

# Create an index
index = TextSearchIndex(
    text_fields=["title", "description"],
    keyword_fields=["category"],
    db_path="search.db"
)

# Index documents in bulk
documents = [
    {"id": 1, "title": "Python Tutorial", "description": "Learn Python basics", "category": "tutorial"},
    {"id": 2, "title": "Java Guide", "description": "Java programming guide", "category": "guide"},
]
index.fit(documents)

# Or add one at a time
index.add({"id": 3, "title": "Advanced Python", "description": "Deep dive into Python", "category": "tutorial"})

# Search
results = index.search("python programming")
for result in results:
    print(result["title"], result["score"])
```

### Filtering

```python
# Filter by keyword fields
results = index.search("python", filter_dict={"category": "tutorial"})

# Multi-value (IN/OR) keyword filter: match any of the listed values
results = index.search("python", filter_dict={"category": ["tutorial", "guide"]})

# Filter by numeric range
results = index.search("python", filter_dict={"price": [('>=', 50), ('<', 200)]})

# Exact numeric match
results = index.search("python", filter_dict={"price": 100})

# Filter by date range
from datetime import date
results = index.search("python", filter_dict={
    "created_at": [('>=', date(2024, 1, 1)), ('<', date(2024, 12, 31))]
})
```

### Field Boosting

```python
# Boost title matches higher than description
results = index.search("python", boost_dict={"title": 2.0, "description": 1.0})
```

### Tokenizer & Stemming

sqlitesearch uses a `Tokenizer` class for query processing (same interface as `minsearch.Tokenizer`). By default, English stop words are removed.

```python
from sqlitesearch import TextSearchIndex, Tokenizer

# Built-in Porter stemming: "running" matches "run", "courses" matches "course"
index = TextSearchIndex(
    text_fields=["title", "description"],
    stemming=True,  # disabled by default to match minsearch behavior
    db_path="search.db"
)

# Custom tokenizer: no stop words
index = TextSearchIndex(
    text_fields=["title", "description"],
    tokenizer=Tokenizer(),
    db_path="search.db"
)

# Custom tokenizer: custom stop words + custom stemmer (any callable(str) -> str)
from stemlite import porter_stemmer  # pip install "sqlitesearch[stemming]"

index = TextSearchIndex(
    text_fields=["title", "description"],
    tokenizer=Tokenizer(stop_words={"custom", "words"}, stemmer=porter_stemmer),
    db_path="search.db"
)

# Or pick a stemmer by name (porter / snowball / lancaster) via get_stemmer:
from stemlite import get_stemmer

index = TextSearchIndex(
    text_fields=["title", "description"],
    tokenizer=Tokenizer(stop_words="english", stemmer=get_stemmer("snowball")),
    db_path="search.db"
)
```

The Python-side `stemmer` is any `callable(str) -> str`; the optional
[stemlite](https://github.com/alexeygrigorev/stemlite) package provides
ready-made Porter, Snowball, and Lancaster stemmers. Install it with the
`stemming` extra: `pip install "sqlitesearch[stemming]"`.

### Custom ID Field

```python
index = TextSearchIndex(
    text_fields=["title", "description"],
    id_field="doc_id",
    db_path="search.db"
)

results = index.search("python", output_ids=True)
# Results will include 'id' field with the doc_id value
```

## Vector Search

Vector search supports four modes for approximate nearest neighbor search, all followed by exact cosine similarity reranking:

| Mode | Best for | How it works |
|------|----------|--------------|
| **HNSW** (default) | 10K-100K vectors | Hierarchical proximity graph traversal |
| **LSH** | Up to 100K vectors | Random hyperplane projections + bucket lookup |
| **LSH_INT8** | Up to 100K vectors, lower RAM | LSH + an int8-quantized, disk-backed shortlist cache (drops the float32 vector copy) |
| **IVF** | 10K-100K vectors | K-means clustering + nearest-cluster probe |

### HNSW (Hierarchical Navigable Small World) — default

Builds a multi-layer proximity graph. Highest recall and fastest search, but slower to build. This is the default mode.

```python
index = VectorSearchIndex(
    mode="hnsw",               # the default; shown for clarity
    m=16,                   # Max connections per node (more = better recall)
    ef_construction=200,    # Build-time beam width (more = better graph)
    ef_search=50,           # Search-time beam width (more = better recall)
    db_path="vectors.db"
)
```

The layer-0 beam search is compiled with numba (a hard dependency, installed automatically), which cuts build time substantially at scale.

For lower serving RAM, pass `disk_backed=True` to drop the float32 vector cache the index keeps by default and serve the graph-walk vectors from a file-backed memmap instead, so steady-state search RSS stays flat as the corpus grows (the OS can evict file-backed pages it can't evict anonymous RAM). Results are exact and unchanged:

```python
index = VectorSearchIndex(
    mode="hnsw",
    disk_backed=True,       # drop the vector cache; serve nav vectors from disk
    db_path="vectors.db"
)
```

### LSH

Each vector is hashed into one bucket per table using random hyperplane projections. At query time, LSH looks up buckets matching the query's hash to find candidates, then reranks them by exact cosine similarity. By default, `n_probe=2` enables multi-probe lookup, so LSH also checks neighboring buckets that differ by 1 or 2 bits — this dramatically improves recall because similar vectors that landed in an adjacent bucket (due to one projection going the other way) are still found.

```python
import numpy as np
from sqlitesearch import VectorSearchIndex

index = VectorSearchIndex(
    mode="lsh",
    keyword_fields=["category"],
    n_tables=8,      # Number of hash tables (more = better recall)
    hash_size=16,    # Bits per hash (more = better precision)
    n_probe=2,       # Multi-probe bit flips (0-2, higher = better recall)
    db_path="vectors.db"
)

vectors = np.random.rand(100, 384)
documents = [{"category": "test"} for _ in range(100)]
index.fit(vectors, documents)

query = np.random.rand(384)
results = index.search(query)
```

### LSH_INT8 (lower-RAM LSH)

Identical algorithm and results to plain LSH, but instead of holding the full float32 vector matrix in RAM it keeps an int8-quantized, disk-backed copy of the normalized vectors. LSH narrows candidates to a shortlist with a cheap int8 dot product, then the final top-k is an exact float32 cosine rerank over the shortlist's SQLite BLOBs — so recall matches plain LSH, at roughly a quarter of the index RAM (the int8 cache is also file-backed, so the OS can evict its pages under memory pressure). Use it when the corpus is large and RSS matters more than a little search latency.

```python
index = VectorSearchIndex(
    mode="lsh_int8",
    keyword_fields=["category"],
    n_tables=8,
    hash_size=16,
    n_probe=2,
    db_path="vectors.db"
)
```

### IVF (Inverted File Index)

Clusters vectors using k-means, then searches only the nearest clusters at query time. Good balance of build speed and recall.

```python
index = VectorSearchIndex(
    mode="ivf",
    n_clusters=None,        # Auto-scales (sqrt(n), capped at 256)
    n_probe_clusters=8,     # Clusters to search (more = better recall, slower)
    db_path="vectors.db"
)
```

Filtering works the same as text search - see the [Filtering](#filtering) section.

## Hybrid Search

Text and vector indexes can share the same database file, enabling hybrid search over a single `.db`. Pass an **`id_field`** to both indexes so they recognise the same documents and store them **once** in a shared `docs` table:

```python
from sqlitesearch import TextSearchIndex, VectorSearchIndex

docs = [{"doc_id": 1, "title": "Python", "description": "...", "category": "dev"}, ...]

# The vector index stores the documents (and their vectors)...
vector_index = VectorSearchIndex(keyword_fields=["category"], id_field="doc_id", db_path="hybrid.db")
vector_index.fit(query_vectors, docs)

# ...and the text index builds its full-text index over the *same* rows.
text_index = TextSearchIndex(text_fields=["title", "description"], keyword_fields=["category"],
                             id_field="doc_id", db_path="hybrid.db")
text_index.fit(docs)

text_results = text_index.search("python tutorial", output_ids=True)
vector_results = vector_index.search(query_vector, output_ids=True)

# Combine the two result lists by their shared id (e.g. reciprocal rank fusion).
```

**How it works:** both indexes use one `docs` table; the vector index fills a `vector_hash` column while the text index leaves it `NULL` and maintains the FTS5 table. When `id_field` is set, inserts **upsert by that id**, so fitting the same corpus into both indexes updates the same row instead of duplicating it — and results from both indexes line up by your `id`.

**Caveats:**
- The `id_field` is what ties the two indexes together. **Without it there is no key to deduplicate on**, so the same document fitted into both indexes is stored as two separate rows and you can only correlate results by re-matching your own fields. Provide `id_field` for hybrid use.
- Use a field name **other than the reserved `id`** (that name is the internal row id); e.g. `"doc_id"`.
- Configure both indexes with the same `keyword_fields` / `numeric_fields` / `date_fields` so the shared `docs` schema matches.

## Index Management

Both index types automatically persist to disk. Reopen an existing index by creating it with the same `db_path` - it's ready to search immediately. Use `index.clear()` to remove all documents.

## Storage backends

By default the index is a local SQLite file opened with Python's built-in `sqlite3`. One other backend is available via the `backend` parameter:

| `backend` | engine | extra | use |
|---|---|---|---|
| `"sqlite3"` (default) | stdlib `sqlite3` | — | local file |
| `"libsql"` | libSQL | `sqlitesearch[libsql]` | local file, or embedded replica synced to **Turso Cloud** |

### libSQL / Turso Cloud (remote persistence)

Back the index with [libSQL](https://github.com/tursodatabase/libsql) so the data persists even on hosts with an ephemeral disk — useful for free/serverless deployments.

```bash
pip install "sqlitesearch[libsql]"
```

```python
# Embedded replica: reads run against a local file that syncs to Turso Cloud.
index = VectorSearchIndex(
    keyword_fields=["category"],
    db_path="local-replica.db",           # local embedded-replica cache
    sync_url="libsql://your-db.turso.io",  # Turso database URL
    auth_token="...",                      # Turso auth token
)
```

Reads run against the local replica (fast). The same `backend` / `sync_url` / `auth_token` parameters are available on `TextSearchIndex`. Bulk ingest is batched into multi-row inserts so it stays fast even when writes are forwarded to the remote.

#### Deploy pattern: ingest once, serve read-only

For a free or serverless host with an ephemeral disk, split the work in two so the running app never rebuilds the index:

1. **Ingest (offline, once)** — build the index and write it straight to Turso. Batched `fit()` keeps this fast even over the network.
2. **Serve (every request)** — open the same index with `sync_url` set. On boot the embedded replica syncs the data down once; every search then reads the local file with no per-query round-trip.

```python
# ingest.py — run once, writes to Turso
index = VectorSearchIndex(keyword_fields=["category"], db_path="cache.db",
                          sync_url=URL, auth_token=TOKEN)
index.fit(vectors, docs)

# app.py — the running service, read-only
index = VectorSearchIndex(keyword_fields=["category"], db_path="cache.db",
                          sync_url=URL, auth_token=TOKEN)
results = index.search(query_vector)
```

Prefer to build the index offline (in CI, say) and ship a finished file? Build a plain local `.db`, then seed Turso from it in one shot:

```bash
turso db create my-index --from-file local.db
```

## When to Use

sqlitesearch is ideal when you want:
- Zero infrastructure (no external services)
- Data persistence across restarts
- Real search functionality for pet projects, demos, or prototypes
- Simple deployment (just a Python file and a `.db` file)

| Use case | Recommendation |
|----------|---------------|
| In-memory / experiments | [minsearch](https://github.com/alexeygrigorev/minsearch) (e.g., in notebooks) |
| Local projects, up to 100K docs/vectors | **sqlitesearch** |
| Production / high traffic / 1M+ vectors | Elasticsearch, Qdrant, Milvus, etc. |

## Benchmarks

We benchmarked sqlitesearch on [Simple English Wikipedia (291K articles)](benchmark/WRITEUP.md) for text search and the [Cohere-1M dataset (768d vectors)](benchmark/WRITEUP.md) for vector search.

Vector search is intended for local, memory-resident use up to about 100K
vectors. Larger corpora should use a dedicated vector database such as Qdrant,
Milvus, Elasticsearch/OpenSearch, or another disk/page-aware ANN system.

| Type | 1K | 10K | 100K |
|------|---:|----:|-----:|
| Text search QPS | 970 | 604 | 179 |
| Text search latency | 1ms | 2ms | 6ms |
| Vector search QPS | 333 | 39 | 6 |
| Vector search latency | 3ms | 26ms | 181ms |
| Vector recall@100 | 0.65 | 0.97 | 0.89 |

Vector search uses an in-memory vector cache for reranking and ANN search
structures. At 100K, recall (0.89 with LSH, higher with tuned HNSW/LSH)
is competitive for local use. See [benchmark/WRITEUP.md](benchmark/WRITEUP.md)
for full results and tuning notes.

## Architecture

Everything is persisted in a single SQLite database file. Text search uses FTS5
with BM25 ranking. Vector search stores vectors, payloads, filter columns, and
ANN metadata in SQLite, then warms in-memory vector/ANN caches for fast local
search. No separate server process or network service is required, but vector
search is not an out-of-core SQLite-page ANN engine.
