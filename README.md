# sqlitesearch

A tiny, SQLite-backed search library for small, local projects. sqlitesearch is a persistent sibling of [minsearch](https://github.com/alexeygrigorev/minsearch) - same API, but stores data on disk.


sqlitesearch provides

- text search using SQLite FTS5 and persistent
- vector search using LSH (random projections) with exact reranking
- hybrid search (Todo explain)

It stores the index in a single SQLite file, making it perfect for applications
that need search functionality without running a separate search server.

## When to use

- minsearch - in-memory, for experiments and notebooks
- sqlitesearch - persistent (SQLite file), for pet projects and prototypes
- Postgres/Elasticsearch/Qdrant/etc - production workloads with high traffic


## Installation

```bash
uv add sqlitesearch
```

## Text Search

Text search uses SQLite's FTS5 (Full-Text Search) extension with BM25 ranking.

### Basic usage

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
```

### Field boosting

```python
# Boost title matches higher than description
results = index.search(
    "python",
    boost_dict={"title": 2.0, "description": 1.0}
)
```

### Adding documents

```python
# Add documents one by one
index.add({
    "id": 3,
    "title": "Advanced Python",
    "description": "Deep dive into Python",
    "category": "tutorial"
})
```

### Custom ID field

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

### Basic usage

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

```python
results = index.search(
    query,
    filter_dict={"category": "test"}
)
```

## Persistence

Both index types automatically persist to disk. You can reopen an existing index:

```python
# Open existing index
index = TextSearchIndex(
    text_fields=["title", "description"],
    db_path="search.db"
)
# Ready to search immediately
```

## Clearing the index

```python
index.clear()  # Remove all documents
```

## API Compatibility

The API is designed to match minsearch for easy migration:

- `fit(docs)` - Index documents (only if index is empty)
- `add(doc)` - Add a single document
- `search(query, filter_dict=None, boost_dict=None, num_results=10)` - Search
