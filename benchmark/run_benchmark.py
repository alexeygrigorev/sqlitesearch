#!/usr/bin/env python3
"""
Benchmark script for sqlitesearch TextSearchIndex.
Measures indexing time, search time, memory usage, and database size.

Uses the same Wikipedia dataset and methodology as the minsearch benchmark
for direct comparison.
"""

import json
import os
import random
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

# Add parent directory to path to import sqlitesearch
sys.path.insert(0, str(Path(__file__).parent.parent))

from minsearch.stemmers import porter_stemmer
from sqlitesearch import TextSearchIndex, Tokenizer


def measure_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def format_memory(bytes_val):
    """Format memory in human-readable format."""
    mb = bytes_val / (1024 * 1024)
    if mb < 1:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{mb:.2f} MB"


def format_size(bytes_val):
    """Format file size in human-readable format."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.2f} MB"


def create_index_fit(docs, db_path):
    """Create and fit a TextSearchIndex using batch fit()."""
    index = TextSearchIndex(
        text_fields=['text'],
        tokenizer=Tokenizer(stop_words='english', stemmer=porter_stemmer),
        db_path=db_path,
    )
    index.fit(docs)
    return index


def create_index_incremental(docs, db_path):
    """Create a TextSearchIndex and add documents one by one."""
    index = TextSearchIndex(
        text_fields=['text'],
        tokenizer=Tokenizer(stop_words='english', stemmer=porter_stemmer),
        db_path=db_path,
    )
    for doc in docs:
        index.add(doc)
    return index


def get_db_size(db_path):
    """Get the size of the database file(s) including WAL/journal."""
    total = 0
    for suffix in ['', '-wal', '-shm', '-journal']:
        path = db_path + suffix
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total


def cleanup_db(db_path):
    """Remove database file(s)."""
    for suffix in ['', '-wal', '-shm', '-journal']:
        path = db_path + suffix
        if os.path.exists(path):
            os.unlink(path)


def benchmark_indexing(docs, num_docs):
    """Benchmark indexing performance."""
    print("\n" + "=" * 70)
    print("INDEXING BENCHMARK")
    print("=" * 70)
    print(f"Documents: {len(docs):,}")

    total_chars = sum(len(str(doc.get('text', ''))) for doc in docs)
    print(f"Total text size: {format_memory(total_chars)}")

    print("\n" + "-" * 70)

    results = {}

    # --- Batch fit() ---
    print("\n1. TextSearchIndex (batch fit)")

    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    os.unlink(db_path)

    try:
        time_index, fit_time = measure_time(create_index_fit, docs, db_path)
        db_size = get_db_size(db_path)
        print(f"   Time:     {format_time(fit_time)}")
        print(f"   DB size:  {format_size(db_size)}")
    finally:
        time_index.close()
        cleanup_db(db_path)

    # Memory measurement (separate run)
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    os.unlink(db_path)

    try:
        fit_index, fit_mem = measure_memory(create_index_fit, docs, db_path)
        print(f"   Memory:   {format_memory(fit_mem)}")
        results['fit'] = {
            'index': fit_index,
            'db_path': db_path,
            'time': fit_time,
            'memory': fit_mem,
            'db_size': db_size,
        }
    except Exception:
        cleanup_db(db_path)
        raise

    # --- Incremental add() --- only for smaller datasets
    if num_docs is not None and num_docs <= 10000:
        print("\n2. TextSearchIndex (incremental add)")

        fd, db_path_inc = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        os.unlink(db_path_inc)

        try:
            inc_index, inc_time = measure_time(create_index_incremental, docs, db_path_inc)
            inc_db_size = get_db_size(db_path_inc)
            print(f"   Time:     {format_time(inc_time)}")
            print(f"   DB size:  {format_size(inc_db_size)}")
            print(f"   Ratio vs batch: {inc_time / fit_time:.2f}x slower")
        finally:
            inc_index.close()
            cleanup_db(db_path_inc)

        results['incremental_time'] = inc_time
    else:
        results['incremental_time'] = None

    return results


def benchmark_search(index, queries):
    """Benchmark search performance."""
    print("\n" + "=" * 70)
    print("SEARCH BENCHMARK")
    print("=" * 70)
    print(f"Queries: {len(queries)}")

    # Warm up
    index.search(queries[0])

    times = []
    for query in queries:
        _, search_time = measure_time(index.search, query, num_results=10)
        times.append(search_time)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"   Average: {format_time(avg_time)}")
    print(f"   Min:     {format_time(min_time)}")
    print(f"   Max:     {format_time(max_time)}")
    print(f"   QPS:     {1/avg_time:.2f} queries/second")

    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'qps': 1 / avg_time,
    }


def load_documents(input_path, num_docs=None):
    """Load documents from JSON or JSONL file."""
    docs = []

    if str(input_path).endswith('.jsonl'):
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
                if num_docs and len(docs) >= num_docs:
                    break
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        if num_docs:
            docs = docs[:num_docs]

    return docs


def generate_queries(docs, num_queries):
    """Generate search queries from document titles (same method as minsearch benchmark)."""
    titles = [doc['title'] for doc in docs if 'title' in doc]
    random.seed(42)
    sample_titles = random.sample(titles, min(num_queries, len(titles)))
    queries = [title.split('(')[0].strip().lower() for title in sample_titles]
    return queries


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark sqlitesearch TextSearchIndex')
    parser.add_argument('-i', '--input', default='data/wikipedia_docs.jsonl',
                        help='Path to JSON/JSONL file with documents')
    parser.add_argument('-n', '--num-docs', type=int, default=None,
                        help='Maximum number of documents to use (default: all)')
    parser.add_argument('-q', '--num-queries', type=int, default=100,
                        help='Number of search queries to benchmark (default: 100)')

    args = parser.parse_args()

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        script_dir = Path(__file__).parent
        input_path = script_dir / args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        print("\nPlease ensure the Wikipedia dataset is in benchmark/data/")
        sys.exit(1)

    # Load documents
    print(f"Loading documents from: {input_path}")
    docs = load_documents(input_path, args.num_docs)
    print(f"Loaded {len(docs):,} documents")

    # Run benchmark
    print("\n" + "=" * 70)
    print("SQLITESEARCH BENCHMARK - TextSearchIndex")
    print("=" * 70)

    # Indexing benchmark
    indexing_results = benchmark_indexing(docs, args.num_docs)

    # Generate queries
    print("\n" + "=" * 70)
    print("GENERATING SEARCH QUERIES")
    print("=" * 70)
    queries = generate_queries(docs, args.num_queries)
    print(f"Generated {len(queries)} search queries from document titles")

    # Search benchmark (using the fit index)
    fit_index = indexing_results['fit']['index']
    search_results = benchmark_search(fit_index, queries)

    # Clean up
    fit_index.close()
    cleanup_db(indexing_results['fit']['db_path'])

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nDocuments: {len(docs):,}")
    print(f"\nIndexing (batch fit):")
    print(f"  Time:    {format_time(indexing_results['fit']['time'])}")
    print(f"  Memory:  {format_memory(indexing_results['fit']['memory'])}")
    print(f"  DB size: {format_size(indexing_results['fit']['db_size'])}")
    if indexing_results.get('incremental_time'):
        print(f"\nIndexing (incremental add):")
        print(f"  Time:    {format_time(indexing_results['incremental_time'])}")
    print(f"\nSearch:")
    print(f"  Average: {format_time(search_results['avg_time'])}")
    print(f"  QPS:     {search_results['qps']:.2f}")

    # Save results
    results = {
        'num_docs': len(docs),
        'indexing_time': indexing_results['fit']['time'],
        'indexing_memory': indexing_results['fit']['memory'],
        'db_size': indexing_results['fit']['db_size'],
        'incremental_time': indexing_results.get('incremental_time'),
        'search_avg_time': search_results['avg_time'],
        'search_min_time': search_results['min_time'],
        'search_max_time': search_results['max_time'],
        'search_qps': search_results['qps'],
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "benchmark_results.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
