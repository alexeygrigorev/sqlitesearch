# sqlitesearch

A tiny, SQLite-backed search library for small, local projects. sqlitesearch is a persistent sibling of [minsearch](https://github.com/alexeygrigorev/minsearch) - same API, but stores data on disk.

## Features

sqlitesearch provides:

- Text search using SQLite FTS5 with BM25 ranking
- Vector search using LSH (random projections) with exact cosine similarity reranking
- Hybrid search by combining text and vector results
- Single-file storage - everything in one `.db` file

## When to use

sqlitesearch works well for datasets **up to ~100K documents/vectors**. Beyond that, consider dedicated search engines.

| Use case | Recommendation |
|----------|---------------|
| In-memory / experiments | [minsearch](https://github.com/alexeygrigorev/minsearch) (e.g., in notebooks) |
| Local projects, up to 100K docs | **sqlitesearch** |
| Production / high traffic / 1M+ | Elasticsearch, Qdrant, Milvus, etc. |

**Why 100K?** We benchmarked sqlitesearch on [Simple English Wikipedia (291K articles)](benchmark/WRITEUP.md) for text search and the [Cohere-1M dataset (768d vectors)](benchmark/WRITEUP.md) for vector search. Results:

| Type | 1K | 10K | 100K |
|------|---:|----:|-----:|
| Text search QPS | 970 | 604 | 179 |
| Text search latency | 1ms | 2ms | 6ms |
| Vector search QPS | 2,152 | 162 | 18 |
| Vector search latency | 0.5ms | 6ms | 56ms |

At 100K, both text and vector search deliver sub-100ms latency. Beyond that, performance degrades: text search drops to 11 QPS at 291K docs, and vector search to 3 QPS at 1M vectors. See [benchmark/WRITEUP.md](benchmark/WRITEUP.md) for full results, methodology, and VDBBench leaderboard comparison.

sqlitesearch is ideal when you want:
- Zero infrastructure (no external services)
- Data persistence across restarts
- Real search functionality for pet projects, demos, or prototypes
- Simple deployment (just a Python file and a `.db` file)

## Architecture

sqlitesearch stores the entire search index in a single SQLite database file on disk, unlike server-based systems (e.g., PostgreSQL, Elasticsearch). This single file contains your data tables, index structures for fast lookup, and search metadata.

SQLite requires no separate server process. It runs inside your Python process, reading/writing directly to the file, eliminating network communication, background daemons, and distributed setup. You install the package and start using it. There is no cluster management, JVM tuning, or DevOps overhead.

Conceptually, it sits between two extremes:
- minsearch: Provides in-memory search without persistence
- Production search engines: Such as Elasticsearch or Qdrant, which are built for large-scale distributed systems

It fills the gap for projects that need real persistence and vector search without operational complexity.

## Installation

```bash
uv add sqlitesearch
```

## Technical Overview

sqlitesearch is built primarily on the Python standard library and two key components:

- SQLite: The fundamental database engine used for all data persistence
- NumPy: Essential for vector mathematics, including cosine similarity calculations

The library uses the following extensions and modules:

- SQLite FTS5: The extension used to power full-text search
- LSH (Locality-Sensitive Hashing): Implemented for approximate nearest neighbor search via random projections

## Text Search

Text search uses SQLite's FTS5 (Full-Text Search) extension with BM25 ranking.

### How It Works

- Documents (JSON) are stored in the `docs` table alongside keyword, numeric, and date fields
- Text fields are indexed in FTS5 using Unicode61 tokenizer (with optional Porter stemming)
- The `docs_fts` virtual table is the FTS5 index for text fields
- Other fields (keyword, numeric, date) are stored as columns for efficient filtering
- Uses FTS5 `MATCH` queries with BM25 ranking for scoring
- Supports field boosting to weight certain fields higher

### Basic Usage

```python
from sqlitesearch import TextSearchIndex

# Create an index
index = TextSearchIndex(
    text_fields=["title", "description"],
    keyword_fields=["category"],
    db_path="search.db"
)

# Index some documents
documents = [
    {"id": 1, "title": "Python Tutorial", "description": "Learn Python basics", "category": "tutorial"},
    {"id": 2, "title": "Java Guide", "description": "Java programming guide", "category": "guide"},
]
index.fit(documents)

# Search
results = index.search("python programming")
for result in results:
    print(result["title"], result["score"])
```

### Filtering

```python
# Filter by keyword fields
results = index.search(
    "python",
    filter_dict={"category": "tutorial"}
)

# Filter by price range
results = index.search(
    "python",
    filter_dict={"price": [('>=', 50), ('<', 200)]}
)

# Exact numeric match
results = index.search(
    "python",
    filter_dict={"price": 100}
)
```

```python
# Filter by date range
from datetime import date

results = index.search(
    "python",
    filter_dict={"created_at": [('>=', date(2024, 1, 1)), ('<', date(2024, 12, 31))]}
)
```

### Field Boosting

```python
# Boost title matches higher than description
results = index.search(
    "python",
    boost_dict={"title": 2.0, "description": 1.0}
)
```

### Stemming

Enable Porter stemming for better matching of word variants:

```python
# With stemming, "running" matches "run", "courses" matches "course", etc.
index = TextSearchIndex(
    text_fields=["title", "description"],
    stemming=True,
    db_path="search.db"
)
```

By default, stemming is disabled to match minsearch behavior.

### Custom Tokenizer

sqlitesearch uses a `Tokenizer` class for query processing that follows the same
interface as `minsearch.Tokenizer`. By default, English stop words are removed
from queries for better performance.

```python
from sqlitesearch import TextSearchIndex, Tokenizer

# Default: English stop words removed (recommended)
index = TextSearchIndex(
    text_fields=["title", "description"],
    db_path="search.db"
)

# No stop words
index = TextSearchIndex(
    text_fields=["title", "description"],
    tokenizer=Tokenizer(),
    db_path="search.db"
)

# Custom stop words
index = TextSearchIndex(
    text_fields=["title", "description"],
    tokenizer=Tokenizer(stop_words={"custom", "words"}),
    db_path="search.db"
)
```

You can also pass a stemmer callable to the tokenizer. For example, using
stemmers from [minsearch](https://github.com/alexeygrigorev/minsearch)
(`pip install minsearch`):

```python
from minsearch.stemmers import porter_stemmer
from sqlitesearch import TextSearchIndex, Tokenizer

index = TextSearchIndex(
    text_fields=["title", "description"],
    tokenizer=Tokenizer(stop_words='english', stemmer=porter_stemmer),
    db_path="search.db"
)
```

Any `callable(str) -> str` works as a stemmer.

### Adding Documents

```python
# Add documents one by one
index.add({
    "id": 3,
    "title": "Advanced Python",
    "description": "Deep dive into Python",
    "category": "tutorial"
})
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

Vector search uses Locality-Sensitive Hashing (LSH) with random projections
for fast approximate nearest neighbor search, followed by exact cosine
similarity reranking.

### How It Works

- The `docs` table stores document JSON, vectors (as raw bytes), and fields for filtering
- The `lsh_buckets` table holds hash buckets for LSH lookup
- The `metadata` table stores LSH configuration parameters (dimension, random projection vectors)

LSH Implementation:
- Random Projections: Generates a set of random vectors (`n_tables × hash_size`) from a Gaussian distribution
- Hashing: Converts each vector into a binary hash key by computing `sign(random_projection @ vector)`
- Multiple Hash Tables: Uses multiple hash tables (default 8) to enhance recall

Search Process (3 Steps):
1. LSH Candidate Finding: The query vector is hashed across all tables to quickly retrieve documents with matching hash keys
2. Filtering: Standard keyword, numeric, and date filters are applied to the LSH candidates
3. Exact Reranking: Exact cosine similarity is computed for filtered candidates using NumPy. Results are sorted by similarity and top results returned

This approach balances speed and accuracy for small-to-medium datasets. LSH rapidly narrows down candidates (approximate search), while exact cosine similarity provides accurate final ranking.

### Basic Usage

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

### Filtering

Vector search supports the same filtering options as text search (keyword, numeric range, and date range filters). See the [Text Search Filtering](#filtering) section for examples.

## Hybrid Search

The Text Search and Vector Search indexes can share the same underlying SQLite database file and the common `docs` table. This enables hybrid search by combining results from both text and vector indexes at query time.

```python
from sqlitesearch import TextSearchIndex, VectorSearchIndex

# Both indexes use the same database
text_index = TextSearchIndex(
    text_fields=["title", "description"],
    db_path="hybrid.db"
)

vector_index = VectorSearchIndex(
    db_path="hybrid.db"  # Same database
)

# Search both indexes and combine results
text_results = text_index.search("python tutorial")
vector_results = vector_index.search(query_vector)

# Combine and deduplicate results based on your ranking strategy
```

## Index Management

### Persistence

Both index types automatically persist to disk. You can reopen an existing index:

```python
# Open existing index
index = TextSearchIndex(
    text_fields=["title", "description"],
    db_path="search.db"
)
# Ready to search immediately
```

### Clearing the Index

```python
index.clear()  # Remove all documents
```
