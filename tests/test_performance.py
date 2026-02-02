"""
Performance tests with 10,000 documents.
"""

import os
import tempfile
import time

import numpy as np
import pytest

from sqlitesearch import TextSearchIndex, VectorSearchIndex


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def large_docs():
    """Generate 10,000 sample documents."""
    rng = np.random.default_rng(42)

    topics = ["python", "javascript", "rust", "go", "java", "typescript", "c++", "swift", "kotlin", "ruby"]
    categories = ["tutorial", "reference", "example", "news", "guide"]

    docs = []
    for i in range(10000):
        topic = rng.choice(topics)
        category = rng.choice(categories)

        doc = {
            "id": int(i),
            "title": f"{topic.capitalize()} - {category} {i}",
            "description": f"This is a document about {topic} programming. {category} number {i}.",
            "topic": topic,
            "category": category,
            "views": int(rng.integers(0, 10000)),
        }
        docs.append(doc)

    return docs


@pytest.fixture
def large_vectors():
    """Generate 10,000 sample vectors (1024 dimensions)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(size=(10000, 1024), dtype=np.float32)


class TestTextSearchPerformance:
    """Performance tests for TextSearchIndex with 10,000 documents."""

    def test_index_10000_documents(self, large_docs, temp_db):
        """Test indexing 10,000 documents."""
        index = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["topic", "category"],
            db_path=temp_db
        )

        try:
            start = time.time()
            index.fit(large_docs)
            fit_time = time.time() - start

            print(f"\nFit 10,000 documents in {fit_time:.2f} seconds")

            # Should complete in reasonable time (< 30 seconds)
            assert fit_time < 30.0
        finally:
            index.close()

    def test_search_10000_documents(self, large_docs, temp_db):
        """Test searching over 10,000 documents."""
        index = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["topic", "category"],
            db_path=temp_db
        )
        index.fit(large_docs)

        try:
            # Test various searches
            queries = ["python tutorial", "javascript example", "rust reference", "go news"]

            for query in queries:
                start = time.time()
                results = index.search(query, num_results=10)
                search_time = time.time() - start

                print(f"Search '{query}' returned {len(results)} results in {search_time:.4f} seconds")

                # Should return results quickly (< 1 second)
                assert search_time < 1.0
                assert len(results) <= 10
                assert len(results) > 0
        finally:
            index.close()

    def test_search_with_filter_10000_documents(self, large_docs, temp_db):
        """Test searching with filters over 10,000 documents."""
        index = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["topic", "category"],
            db_path=temp_db
        )
        index.fit(large_docs)

        try:
            start = time.time()
            results = index.search(
                "python",
                filter_dict={"category": "tutorial"},
                num_results=5
            )
            search_time = time.time() - start

            print(f"\nFiltered search returned {len(results)} results in {search_time:.4f} seconds")

            # All results should be python tutorials
            for result in results:
                assert result["category"] == "tutorial"
        finally:
            index.close()

    def test_add_to_10000_documents(self, large_docs, temp_db):
        """Test adding documents to an index of 10,000."""
        index = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["topic"],
            db_path=temp_db
        )

        try:
            # Fit initial 10,000
            index.fit(large_docs)

            # Add 100 more documents
            new_docs = [
                {
                    "id": 10000 + i,
                    "title": f"New Document {i}",
                    "description": f"A newly added document about python",
                    "topic": "python",
                }
                for i in range(100)
            ]

            start = time.time()
            for doc in new_docs:
                index.add(doc)
            add_time = time.time() - start

            print(f"\nAdded 100 documents in {add_time:.2f} seconds")

            # Search should find the new documents
            results = index.search("New Document", num_results=10)
            assert len(results) > 0
        finally:
            index.close()

    def test_persistence_10000_documents(self, large_docs, temp_db):
        """Test that 10,000 documents persist correctly."""
        # Index and close
        index1 = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["topic"],
            db_path=temp_db
        )
        index1.fit(large_docs)
        index1.close()

        # Reopen and search
        index2 = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["topic"],
            db_path=temp_db
        )

        try:
            results = index2.search("python", num_results=10)
            assert len(results) > 0
            assert len(results) <= 10
        finally:
            index2.close()


class TestVectorSearchPerformance:
    """Performance tests for VectorSearchIndex with 10,000 documents and 1024-dim vectors."""

    def test_index_10000_vectors(self, large_vectors, large_docs, temp_db):
        """Test indexing 10,000 vectors (1024 dimensions)."""
        index = VectorSearchIndex(
            keyword_fields=["topic", "category"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )

        try:
            start = time.time()
            index.fit(large_vectors, large_docs)
            fit_time = time.time() - start

            print(f"\nFit 10,000 vectors (1024-dim) in {fit_time:.2f} seconds")

            # Should complete in reasonable time
            assert fit_time < 120.0
        finally:
            index.close()

    def test_search_10000_vectors(self, large_vectors, large_docs, temp_db):
        """Test searching over 10,000 vectors."""
        index = VectorSearchIndex(
            keyword_fields=["topic", "category"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )
        index.fit(large_vectors, large_docs)

        try:
            # Use first vector as query (should return itself as top result)
            query_vector = large_vectors[0]

            start = time.time()
            results = index.search(query_vector, num_results=10)
            search_time = time.time() - start

            print(f"Search returned {len(results)} results in {search_time:.4f} seconds")

            # Should return results quickly
            assert search_time < 1.0
            assert len(results) <= 10
            # First result should be the query vector itself
            assert results[0]["id"] == 0
        finally:
            index.close()

    def test_search_multiple_queries_10000_vectors(self, large_vectors, large_docs, temp_db):
        """Test multiple searches over 10,000 vectors."""
        index = VectorSearchIndex(
            keyword_fields=["topic"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )
        index.fit(large_vectors, large_docs)

        try:
            # Test 10 random queries
            rng = np.random.default_rng(123)
            total_time = 0.0

            for i in range(10):
                query_idx = rng.integers(0, 10000)
                query_vector = large_vectors[query_idx]

                start = time.time()
                results = index.search(query_vector, num_results=5)
                search_time = time.time() - start
                total_time += search_time

            avg_time = total_time / 10
            print(f"\nAverage search time over 10 queries: {avg_time:.4f} seconds")

            # Average should be fast
            assert avg_time < 0.5
        finally:
            index.close()

    def test_search_with_filter_10000_vectors(self, large_vectors, large_docs, temp_db):
        """Test searching with filters over 10,000 vectors."""
        index = VectorSearchIndex(
            keyword_fields=["topic", "category"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )
        index.fit(large_vectors, large_docs)

        try:
            query_vector = large_vectors[0]

            # Filter by topic
            start = time.time()
            results = index.search(
                query_vector,
                filter_dict={"topic": "python"},
                num_results=5
            )
            search_time = time.time() - start

            print(f"\nFiltered vector search returned {len(results)} results in {search_time:.4f} seconds")

            # All results should have topic=python
            for result in results:
                assert result["topic"] == "python"
        finally:
            index.close()

    def test_add_to_10000_vectors(self, large_vectors, large_docs, temp_db):
        """Test adding vectors to an index of 10,000."""
        index = VectorSearchIndex(
            keyword_fields=["topic"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )

        try:
            # Fit initial 10,000
            index.fit(large_vectors, large_docs)

            # Add 100 more vectors
            new_vectors = np.random.randn(100, 1024).astype(np.float32)
            new_docs = [
                {
                    "id": 10000 + i,
                    "title": f"New Vector Doc {i}",
                    "topic": "python",
                }
                for i in range(100)
            ]

            start = time.time()
            for vec, doc in zip(new_vectors, new_docs):
                index.add(vec, doc)
            add_time = time.time() - start

            print(f"\nAdded 100 vectors in {add_time:.2f} seconds")

            # Search should find the new documents
            query = new_vectors[0]
            results = index.search(query, num_results=10)
            assert len(results) > 0
        finally:
            index.close()

    def test_persistence_10000_vectors(self, large_vectors, large_docs, temp_db):
        """Test that 10,000 vectors persist correctly."""
        # Index and close
        index1 = VectorSearchIndex(
            keyword_fields=["topic"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )
        index1.fit(large_vectors, large_docs)
        index1.close()

        # Reopen and search
        index2 = VectorSearchIndex(
            keyword_fields=["topic"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )

        try:
            query_vector = large_vectors[0]
            results = index2.search(query_vector, num_results=10)
            assert len(results) > 0
            assert results[0]["id"] == 0
        finally:
            index2.close()


class TestCombinedPerformance:
    """Combined performance tests."""

    def test_combined_text_and_vector_10000(self, large_docs, large_vectors, temp_db):
        """Test both text and vector search on the same dataset (10k docs, 1024-dim vectors)."""
        # Text search
        text_index = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["topic"],
            db_path=temp_db + "_text.db"
        )

        vec_index = VectorSearchIndex(
            keyword_fields=["topic"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db + "_vector.db"
        )

        try:
            start = time.time()
            text_index.fit(large_docs)
            text_fit_time = time.time() - start

            text_results = text_index.search("python tutorial", num_results=10)

            # Vector search
            start = time.time()
            vec_index.fit(large_vectors, large_docs)
            vec_fit_time = time.time() - start

            vec_results = vec_index.search(large_vectors[0], num_results=10)

            print(f"\n--- Performance Summary for 10,000 documents ---")
            print(f"Text search fit time: {text_fit_time:.2f}s")
            print(f"Text search results: {len(text_results)}")
            print(f"Vector search fit time (1024-dim): {vec_fit_time:.2f}s")
            print(f"Vector search results: {len(vec_results)}")

            assert len(text_results) > 0
            assert len(vec_results) > 0
        finally:
            text_index.close()
            vec_index.close()
