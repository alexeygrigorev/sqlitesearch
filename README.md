# sqlitesearch

A tiny, SQLite-backed search library for small-scale projects with up to 100,000 documents. It provides text search, vector search, and hybrid search - all stored in a single `.db` file with zero infrastructure.

sqlitesearch is a persistent sibling of [minsearch](https://github.com/alexeygrigorev/minsearch) - same API, but stores data on disk.

## Installation

```bash
uv add sqlitesearch
```

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
from minsearch.stemmers import porter_stemmer  # pip install minsearch

index = TextSearchIndex(
    text_fields=["title", "description"],
    tokenizer=Tokenizer(stop_words={"custom", "words"}, stemmer=porter_stemmer),
    db_path="search.db"
)
```

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

Vector search supports three modes for approximate nearest neighbor search, all followed by exact cosine similarity reranking:

| Mode | Best for | How it works |
|------|----------|--------------|
| **LSH** (default) | Up to 100K vectors | Random hyperplane projections + bucket lookup |
| **IVF** | 10K-500K vectors | K-means clustering + nearest-cluster probe |
| **HNSW** | 10K-1M+ vectors | Hierarchical proximity graph traversal |

### LSH (default)

Each vector is hashed into one bucket per table using random hyperplane projections. At query time, LSH looks up buckets matching the query's hash to find candidates, then reranks them by exact cosine similarity. With `n_probe > 0` (multi-probe), it also checks neighboring buckets that differ by 1 or 2 bits — this dramatically improves recall because similar vectors that landed in an adjacent bucket (due to one projection going the other way) are still found.

```python
import numpy as np
from sqlitesearch import VectorSearchIndex

index = VectorSearchIndex(
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

### HNSW (Hierarchical Navigable Small World)

Builds a multi-layer proximity graph. Highest recall and fastest search, but slower to build.

```python
index = VectorSearchIndex(
    mode="hnsw",
    m=16,                   # Max connections per node (more = better recall)
    ef_construction=200,    # Build-time beam width (more = better graph)
    ef_search=50,           # Search-time beam width (more = better recall)
    db_path="vectors.db"
)
```

Filtering works the same as text search - see the [Filtering](#filtering) section.

## Hybrid Search

Text and vector indexes can share the same database file, enabling hybrid search.

```python
from sqlitesearch import TextSearchIndex, VectorSearchIndex

text_index = TextSearchIndex(text_fields=["title", "description"], db_path="hybrid.db")
vector_index = VectorSearchIndex(db_path="hybrid.db")

text_results = text_index.search("python tutorial")
vector_results = vector_index.search(query_vector)

# Combine and deduplicate results based on your ranking strategy
```

## Index Management

Both index types automatically persist to disk. Reopen an existing index by creating it with the same `db_path` - it's ready to search immediately. Use `index.clear()` to remove all documents.

## When to Use

sqlitesearch is ideal when you want:
- Zero infrastructure (no external services)
- Data persistence across restarts
- Real search functionality for pet projects, demos, or prototypes
- Simple deployment (just a Python file and a `.db` file)

| Use case | Recommendation |
|----------|---------------|
| In-memory / experiments | [minsearch](https://github.com/alexeygrigorev/minsearch) (e.g., in notebooks) |
| Local projects, up to 100K docs | **sqlitesearch** |
| Production / high traffic / 1M+ | Elasticsearch, Qdrant, Milvus, etc. |

## Benchmarks

We benchmarked sqlitesearch on [Simple English Wikipedia (291K articles)](benchmark/WRITEUP.md) for text search and the [Cohere-1M dataset (768d vectors)](benchmark/WRITEUP.md) for vector search.

| Type | 1K | 10K | 100K |
|------|---:|----:|-----:|
| Text search QPS | 970 | 604 | 179 |
| Text search latency | 1ms | 2ms | 6ms |
| Vector search QPS | 333 | 39 | 6 |
| Vector search latency | 3ms | 26ms | 181ms |
| Vector recall@100 | 0.65 | 0.97 | 0.89 |

Vector search uses multi-probe LSH (`n_probe=2`) with in-memory vector cache for reranking. At 100K, recall (0.89) is competitive with cloud vector databases like ElasticCloud (0.90). For higher recall, use `n_tables=16` (0.95 recall). See [benchmark/WRITEUP.md](benchmark/WRITEUP.md) for full results, recall tuning, and VDBBench leaderboard comparison.

## Architecture

Everything lives in a single SQLite database file. Text search uses FTS5 with BM25 ranking. Vector search uses Locality-Sensitive Hashing (LSH) with random projections for fast candidate retrieval, followed by exact cosine similarity reranking via NumPy. No separate server process, no network communication - SQLite runs inside your Python process, reading and writing directly to the file.
