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

Vector search uses Locality-Sensitive Hashing (LSH) with random projections for fast approximate nearest neighbor search, followed by exact cosine similarity reranking.

```python
import numpy as np
from sqlitesearch import VectorSearchIndex

# Create an index
index = VectorSearchIndex(
    keyword_fields=["category"],
    n_tables=8,      # Number of hash tables (more = better recall)
    hash_size=16,    # Bits per hash (more = better precision)
    db_path="vectors.db"
)

# Index vectors with documents
vectors = np.random.rand(100, 384)  # 100 documents, 384 dimensions
documents = [{"category": "test"} for _ in range(100)]
index.fit(vectors, documents)

# Search
query = np.random.rand(384)
results = index.search(query)
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
| Vector search QPS | 2,152 | 162 | 18 |
| Vector search latency | 0.5ms | 6ms | 56ms |

At 100K, both text and vector search deliver sub-100ms latency. Beyond that, performance degrades: text search drops to 11 QPS at 291K docs, and vector search to 3 QPS at 1M vectors. See [benchmark/WRITEUP.md](benchmark/WRITEUP.md) for full results, methodology, and VDBBench leaderboard comparison.

## Architecture

Everything lives in a single SQLite database file. Text search uses FTS5 with BM25 ranking. Vector search uses Locality-Sensitive Hashing (LSH) with random projections for fast candidate retrieval, followed by exact cosine similarity reranking via NumPy. No separate server process, no network communication - SQLite runs inside your Python process, reading and writing directly to the file.
