#!/usr/bin/env python3
"""
Run full benchmark on Wikipedia dataset.
Designed for AWS EC2 (or similar) with the complete Simple Wikipedia dump.
"""

import sys
import time
import random
import json

sys.path.insert(0, '..')
from minsearch.stemmers import porter_stemmer
from sqlitesearch import TextSearchIndex, Tokenizer

import tempfile
import os


def cleanup_db(db_path):
    """Remove database file(s)."""
    for suffix in ['', '-wal', '-shm', '-journal']:
        path = db_path + suffix
        if os.path.exists(path):
            os.unlink(path)


def get_db_size(db_path):
    """Get the size of the database file(s) including WAL/journal."""
    total = 0
    for suffix in ['', '-wal', '-shm', '-journal']:
        path = db_path + suffix
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total


def main():
    print("=" * 70)
    print("FULL WIKIPEDIA BENCHMARK - sqlitesearch")
    print("=" * 70)

    # Load documents
    print("\nLoading documents from data/wikipedia_docs.jsonl...")
    docs = []
    with open('data/wikipedia_docs.jsonl', 'r') as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"Loaded {len(docs):,} documents")

    total_text_size = sum(len(d.get('text', '')) for d in docs)
    print(f"Total text size: {total_text_size / 1024 / 1024:.2f} MB")

    # Benchmark indexing
    print("\n" + "-" * 70)
    print("INDEXING")
    print("-" * 70)

    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    os.unlink(db_path)

    t1 = time.time()
    index = TextSearchIndex(
        text_fields=['text'],
        tokenizer=Tokenizer(stop_words='english', stemmer=porter_stemmer),
        db_path=db_path,
    )
    index.fit(docs)
    index_time = time.time() - t1

    db_size = get_db_size(db_path)

    print(f"Time: {index_time:.2f}s")
    print(f"DB size: {db_size / 1024 / 1024:.2f} MB")

    # Benchmark search
    print("\n" + "-" * 70)
    print("SEARCH BENCHMARK")
    print("-" * 70)

    random.seed(42)
    titles = [d.get('title', '') for d in docs if d.get('title')]
    queries = [q.split('(')[0].strip().lower() for q in random.sample(titles, 10)]

    # Warmup
    index.search('test')

    times = []
    for query in queries:
        start = time.time()
        index.search(query, num_results=10)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times) * 1000

    print(f"Queries: {len(queries)}")
    print(f"Average: {avg_time:.2f}ms ({1000/avg_time:.1f} QPS)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Documents: {len(docs):,}")
    print(f"Text size: {total_text_size / 1024 / 1024:.2f} MB")
    print(f"Indexing:  {index_time:.2f}s")
    print(f"DB size:   {db_size / 1024 / 1024:.2f} MB")
    print(f"Search:    {avg_time:.2f}ms avg ({1000/avg_time:.1f} QPS)")

    # Cleanup
    index.close()
    cleanup_db(db_path)


if __name__ == "__main__":
    main()
