"""
Comprehensive tests for VectorSearchIndex based on minsearch patterns.
"""

import os
import tempfile

import numpy as np
import pytest

from sqlitesearch import VectorSearchIndex


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
def sample_vectors():
    """Sample vectors for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(size=(100, 384), dtype=np.float32)


@pytest.fixture
def sample_payload():
    """Sample payload documents."""
    return [
        {"id": i, "category": "test" if i % 2 == 0 else "dev", "name": f"doc_{i}"}
        for i in range(100)
    ]


class TestVectorSearchIndexBasics:
    """Test basic VectorSearchIndex functionality."""

    def test_index_initialization(self, temp_db):
        """Test that index initializes correctly."""
        index = VectorSearchIndex(
            keyword_fields=["category"],
            n_tables=8,
            hash_size=16,
            db_path=temp_db
        )
        assert index.keyword_fields == ["category"]
        assert index.n_tables == 8
        assert index.hash_size == 16

    def test_fit_and_search(self, sample_vectors, sample_payload, temp_db):
        """Test basic fit and search functionality."""
        index = VectorSearchIndex(db_path=temp_db)
        index.fit(sample_vectors, sample_payload)

        query = sample_vectors[0]
        results = index.search(query, num_results=10)

        assert len(results) > 0
        assert len(results) <= 10
        # First result should have highest similarity
        assert results[0]["id"] == 0

    def test_num_results(self, sample_vectors, sample_payload, temp_db):
        """Test num_results parameter."""
        index = VectorSearchIndex(db_path=temp_db)
        index.fit(sample_vectors, sample_payload)

        query = sample_vectors[0]

        for n in [1, 5, 10]:
            results = index.search(query, num_results=n)
            assert len(results) <= n

    def test_output_ids(self, sample_vectors, sample_payload, temp_db):
        """Test output_ids parameter."""
        index = VectorSearchIndex(db_path=temp_db)
        index.fit(sample_vectors, sample_payload)

        query = sample_vectors[0]

        # Without output_ids
        results = index.search(query, output_ids=False)
        if results:
            assert "_id" not in results[0]

        # With output_ids
        results = index.search(query, output_ids=True)
        if results:
            assert "_id" in results[0]
            assert isinstance(results[0]["_id"], int)

    def test_empty_results(self, temp_db):
        """Test behavior when no results match."""
        vectors = np.random.randn(3, 64).astype(np.float32)
        payload = [
            {"id": 1, "category": "programming"},
            {"id": 2, "category": "data"},
            {"id": 3, "category": "ai"}
        ]

        index = VectorSearchIndex(keyword_fields=["category"], db_path=temp_db)
        index.fit(vectors, payload)

        query = np.random.randn(64).astype(np.float32)
        results = index.search(query, filter_dict={"category": "nonexistent"})

        assert len(results) == 0


class TestVectorSearchIndexFiltering:
    """Test keyword filtering functionality."""

    def test_keyword_filtering(self, temp_db):
        """Test keyword filtering."""
        np.random.seed(42)

        vectors = np.random.rand(5, 10).astype(np.float32)
        payload = [
            {"id": 1, "title": "Python Tutorial", "category": "programming", "level": "beginner"},
            {"id": 2, "title": "Data Science", "category": "data", "level": "intermediate"},
            {"id": 3, "title": "Machine Learning", "category": "ai", "level": "advanced"},
            {"id": 4, "title": "Web Dev", "category": "programming", "level": "intermediate"},
            {"id": 5, "title": "Statistics", "category": "data", "level": "beginner"}
        ]

        index = VectorSearchIndex(keyword_fields=["category", "level"], db_path=temp_db)
        index.fit(vectors, payload)

        # Use one of the existing vectors as query to ensure LSH finds it
        query_vector = vectors[0]  # This is the "Python Tutorial" vector
        results = index.search(query_vector, filter_dict={"category": "programming", "level": "beginner"})

        # Should return the programming + beginner doc (id=1)
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["category"] == "programming"
        assert results[0]["level"] == "beginner"

    def test_filter_combinations(self, temp_db):
        """Test different filter combinations."""
        vectors = np.random.randn(4, 64).astype(np.float32)
        payload = [
            {"id": 1, "category": "dev", "level": "beginner"},
            {"id": 2, "category": "data", "level": "intermediate"},
            {"id": 3, "category": "dev", "level": "advanced"},
            {"id": 4, "category": "data", "level": "beginner"}
        ]

        index = VectorSearchIndex(keyword_fields=["category", "level"], db_path=temp_db)
        index.fit(vectors, payload)

        query = vectors[0]

        # Test single filter
        results = index.search(query, filter_dict={"category": "dev"})
        assert all(doc["category"] == "dev" for doc in results)

        # Test multiple filters
        results = index.search(query, filter_dict={"category": "dev", "level": "beginner"})
        for doc in results:
            assert doc["category"] == "dev"
            assert doc["level"] == "beginner"

        # Test non-existent filter
        results = index.search(query, filter_dict={"category": "nonexistent"})
        assert len(results) == 0


class TestVectorSearchIndexAdd:
    """Test add() functionality."""

    def test_add_single_vector(self, temp_db):
        """Test adding a single vector."""
        index = VectorSearchIndex(db_path=temp_db)

        # Add first document
        vec1 = np.random.randn(64).astype(np.float32)
        doc1 = {"id": 1, "title": "Python"}
        index.add(vec1, doc1)

        query = vec1
        results = index.search(query)
        assert len(results) == 1
        assert results[0]["id"] == 1

        # Add second document
        vec2 = np.random.randn(64).astype(np.float32)
        doc2 = {"id": 2, "title": "Java"}
        index.add(vec2, doc2)

        results = index.search(vec1)
        assert len(results) >= 1

    def test_add_after_fit(self, temp_db):
        """Test adding vectors after fit."""
        vectors = np.random.randn(3, 64).astype(np.float32)
        payload = [{"id": i} for i in range(3)]

        index = VectorSearchIndex(db_path=temp_db)
        index.fit(vectors, payload)

        # Add another vector
        new_vec = np.random.randn(64).astype(np.float32)
        index.add(new_vec, {"id": 3})

        results = index.search(new_vec)
        assert len(results) >= 1

    def test_add_1d_vector(self, temp_db):
        """Test that add() accepts 1D vectors."""
        index = VectorSearchIndex(db_path=temp_db)

        vec = np.random.randn(64).astype(np.float32)
        doc = {"id": 1}

        # Should work with 1D vector
        index.add(vec, doc)

        query = vec
        results = index.search(query)
        assert len(results) == 1

    def test_fit_raises_error_when_not_empty(self, temp_db):
        """Test that fit() raises error when index is not empty."""
        vectors = np.random.randn(3, 64).astype(np.float32)
        payload = [{"id": i} for i in range(3)]

        index = VectorSearchIndex(db_path=temp_db)
        index.fit(vectors, payload)

        with pytest.raises(ValueError, match="Index already contains documents"):
            index.fit(vectors, payload)


class TestVectorSearchIndexClear:
    """Test clear() functionality."""

    def test_clear(self, temp_db):
        """Test clearing the index."""
        vectors = np.random.randn(5, 64).astype(np.float32)
        payload = [{"id": i} for i in range(5)]

        index = VectorSearchIndex(db_path=temp_db)
        index.fit(vectors, payload)

        assert len(index.search(vectors[0])) > 0

        index.clear()

        assert len(index.search(vectors[0])) == 0

        # Should be able to fit again after clear
        index.fit(vectors, payload)
        assert len(index.search(vectors[0])) > 0


class TestVectorSearchIndexIdField:
    """Test id_field functionality."""

    def test_id_field_with_custom_ids(self, temp_db):
        """Test using custom ID field."""
        vectors = np.random.randn(3, 64).astype(np.float32)
        payload = [
            {"doc_id": 100, "title": "Python"},
            {"doc_id": 200, "title": "Java"},
            {"doc_id": 300, "title": "C++"},
        ]

        index = VectorSearchIndex(
            keyword_fields=["category"],
            id_field="doc_id",
            db_path=temp_db
        )
        index.fit(vectors, payload)

        results = index.search(vectors[0], output_ids=True)
        assert len(results) >= 1
        # Should have doc_id in results
        assert results[0]["doc_id"] in [100, 200, 300]

    def test_id_field_without_custom_ids(self, temp_db):
        """Test without custom ID field."""
        vectors = np.random.randn(2, 64).astype(np.float32)
        payload = [
            {"title": "Python"},
            {"title": "Java"},
        ]

        index = VectorSearchIndex(id_field=None, db_path=temp_db)
        index.fit(vectors, payload)

        results = index.search(vectors[0], output_ids=True)
        assert len(results) >= 1
        assert "_id" in results[0]

    def test_id_field_in_keyword_fields(self, temp_db):
        """Test that id_field is automatically added to keyword_fields."""
        vectors = np.random.randn(2, 64).astype(np.float32)
        payload = [
            {"doc_id": "abc123", "title": "Python"},
            {"doc_id": "def456", "title": "Java"},
        ]

        index = VectorSearchIndex(
            keyword_fields=["category"],
            id_field="doc_id",
            db_path=temp_db
        )

        # id_field should be added to keyword_fields
        assert "doc_id" in index.keyword_fields

        index.fit(vectors, payload)

        # Should be able to filter by id_field
        results = index.search(vectors[0], filter_dict={"doc_id": "abc123"})
        assert len(results) == 1


class TestVectorSearchIndexSortingCorrectness:
    """Test that search results are correctly sorted by similarity."""

    def test_sorting_correctness(self, temp_db):
        """Test cosine similarity sorting."""
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0, 0.0]
        ], dtype=np.float32)

        payload = [
            {"id": 0, "title": "Most Similar"},
            {"id": 1, "title": "Second Most Similar"},
            {"id": 2, "title": "Third Most Similar"},
            {"id": 3, "title": "Fourth Most Similar"},
            {"id": 4, "title": "Least Similar"}
        ]

        # Use more hash tables for better recall on small datasets
        index = VectorSearchIndex(n_tables=16, hash_size=10, db_path=temp_db)
        index.fit(vectors, payload)

        # Query vector identical to vector 0
        query_vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        results = index.search(query_vector, num_results=5)

        # Check sorting - all should have positive similarity
        assert len(results) == 5
        assert results[0]["id"] == 0
        assert results[0]["title"] == "Most Similar"
        assert results[1]["id"] == 1
        assert results[1]["title"] == "Second Most Similar"


class TestVectorSearchIndexNoneValues:
    """Test handling of None values."""

    @pytest.fixture
    def docs_with_none(self):
        return [
            {'url': 'https://example.com/1', 'assignee_login': None, 'state': 'open'},
            {'url': 'https://example.com/2', 'assignee_login': 'JaneSmith', 'state': 'closed'},
            {'url': 'https://example.com/3', 'assignee_login': None, 'state': 'open'},
        ]

    def test_with_none_values(self, docs_with_none, temp_db):
        """Test handling None values in keyword fields."""
        vectors = np.random.randn(len(docs_with_none), 10).astype(np.float32)

        index = VectorSearchIndex(keyword_fields=["assignee_login", "state"], db_path=temp_db)
        index.fit(vectors, docs_with_none)

        # Use one of the existing vectors as query to ensure LSH finds results
        query = vectors[0]
        results = index.search(query)
        assert len(results) > 0

    def test_filter_with_none(self, docs_with_none, temp_db):
        """Test filtering with None values."""
        vectors = np.random.randn(len(docs_with_none), 10).astype(np.float32)

        index = VectorSearchIndex(keyword_fields=["assignee_login", "state"], db_path=temp_db)
        index.fit(vectors, docs_with_none)

        # Use one of the existing vectors that has assignee_login=None
        query = vectors[0]  # First doc has assignee_login=None
        results = index.search(query, filter_dict={"assignee_login": None})

        assert len(results) >= 1
        for result in results:
            assert result['assignee_login'] is None


class TestVectorSearchIndexValidation:
    """Test input validation."""

    def test_vectors_payload_mismatch(self, temp_db):
        """Test error when vectors and payload lengths don't match."""
        vectors = np.random.randn(3, 10).astype(np.float32)
        payload = [{"id": 1}, {"id": 2}]

        index = VectorSearchIndex(db_path=temp_db)

        with pytest.raises(ValueError, match="must match"):
            index.fit(vectors, payload)

    def test_2d_array_required(self, temp_db):
        """Test error when vectors are not 2D."""
        vectors = np.random.randn(384).astype(np.float32)  # 1D
        payload = [{"id": 0}]

        index = VectorSearchIndex(db_path=temp_db)

        with pytest.raises(ValueError, match="must be 2D array"):
            index.fit(vectors, payload)

    def test_dimension_mismatch(self, temp_db):
        """Test error on dimension mismatch."""
        vectors = np.random.randn(5, 384).astype(np.float32)
        payload = [{"id": i} for i in range(5)]

        index = VectorSearchIndex(db_path=temp_db)
        index.fit(vectors, payload)

        # Query with wrong dimension
        query = np.random.randn(128).astype(np.float32)

        with pytest.raises(ValueError, match="does not match index dimension"):
            index.search(query)


class TestVectorSearchIndexPersistence:
    """Test persistence across index instances."""

    def test_data_persists(self, temp_db):
        """Test that data persists across index instances."""
        vectors = np.random.randn(5, 64).astype(np.float32)
        payload = [{"id": i} for i in range(5)]

        # First index
        index1 = VectorSearchIndex(db_path=temp_db)
        index1.fit(vectors, payload)
        index1.close()

        # Second index with same db
        index2 = VectorSearchIndex(db_path=temp_db)

        results = index2.search(vectors[0])
        assert len(results) > 0


class TestVectorSearchIndexContextManager:
    """Test context manager functionality."""

    def test_context_manager(self, temp_db):
        """Test using index as context manager."""
        vectors = np.random.randn(5, 64).astype(np.float32)
        payload = [{"id": i} for i in range(5)]

        with VectorSearchIndex(db_path=temp_db) as index:
            index.fit(vectors, payload)
            query = vectors[0]
            results = index.search(query)
            assert len(results) > 0


class TestVectorSearchIndexLSHParameters:
    """Test different LSH parameter configurations."""

    def test_conervative_settings(self, temp_db):
        """Test conservative LSH settings (more tables, larger hash)."""
        vectors = np.random.randn(20, 64).astype(np.float32)
        payload = [{"id": i} for i in range(20)]

        index = VectorSearchIndex(n_tables=12, hash_size=20, db_path=temp_db)
        index.fit(vectors, payload)

        query = vectors[0]
        results = index.search(query)
        assert len(results) > 0

    def test_aggressive_settings(self, temp_db):
        """Test aggressive LSH settings (fewer tables, smaller hash)."""
        vectors = np.random.randn(20, 64).astype(np.float32)
        payload = [{"id": i} for i in range(20)]

        index = VectorSearchIndex(n_tables=4, hash_size=10, db_path=temp_db)
        index.fit(vectors, payload)

        query = vectors[0]
        results = index.search(query)
        assert len(results) > 0
