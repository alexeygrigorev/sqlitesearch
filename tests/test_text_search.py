"""
Comprehensive tests for TextSearchIndex based on minsearch patterns.
"""

import os
import tempfile

import pytest

from sqlitesearch import TextSearchIndex


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
def sample_docs():
    """Sample documents for testing."""
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language. It's easy to learn.",
            "section": "Programming",
            "course": "python-basics",
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI. It uses algorithms.",
            "section": "AI",
            "course": "ml-basics",
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality. Use pytest for Python.",
            "section": "Testing",
            "course": "python-basics",
        },
    ]


@pytest.fixture
def text_fields():
    return ["question", "text", "section"]


@pytest.fixture
def keyword_fields():
    return ["course"]


class TestTextSearchIndexBasics:
    """Test basic TextSearchIndex functionality."""

    def test_index_initialization(self, text_fields, keyword_fields, temp_db):
        """Test that index initializes correctly."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        assert index.text_fields == text_fields
        assert index.keyword_fields == keyword_fields
        assert index.db_path == temp_db

    def test_fit_and_search(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test basic fit and search functionality."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        # Test search with different queries
        results = index.search("python")
        assert len(results) > 0

        results = index.search("machine learning")
        assert len(results) > 0

        results = index.search("testing")
        assert len(results) > 0

        results = index.search("nonexistent")
        assert len(results) == 0

    def test_search_with_filters(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test search with keyword filters."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        results = index.search("python", filter_dict={"course": "python-basics"})
        assert len(results) > 0
        assert all(doc["course"] == "python-basics" for doc in results)

        results = index.search("python", filter_dict={"course": "nonexistent"})
        assert len(results) == 0

    def test_search_with_multi_value_filter(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test search with multi-value (IN/OR) keyword filters."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        # List value -> IN / OR within the field
        results = index.search(
            "python", filter_dict={"course": ["python-basics", "ml-basics"]}
        )
        assert len(results) > 0
        assert all(doc["course"] in {"python-basics", "ml-basics"} for doc in results)

        # A non-matching value in the list is simply ignored
        results = index.search(
            "python", filter_dict={"course": ["python-basics", "nonexistent"]}
        )
        assert len(results) > 0
        assert all(doc["course"] == "python-basics" for doc in results)

        # Tuple and set work the same as list
        results = index.search("python", filter_dict={"course": ("python-basics",)})
        assert all(doc["course"] == "python-basics" for doc in results)
        results = index.search("python", filter_dict={"course": {"python-basics"}})
        assert all(doc["course"] == "python-basics" for doc in results)

        # Empty list matches nothing
        results = index.search("python", filter_dict={"course": []})
        assert len(results) == 0

    def test_search_with_boosts(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test search with field boosts."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        results = index.search("python", boost_dict={"question": 2.0})
        assert len(results) > 0

    def test_num_results(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test num_results parameter."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        for n in [1, 2, 5]:
            results = index.search("python", num_results=n)
            assert len(results) <= n

    def test_output_ids(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test output_ids parameter."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        # Without output_ids
        results = index.search("python")
        if results:
            assert "id" not in results[0]

        # With output_ids
        results = index.search("python", output_ids=True)
        if results:
            assert "id" in results[0]
            assert isinstance(results[0]["id"], int)


class TestTextSearchIndexAdd:
    """Test add() functionality for incremental indexing."""

    def test_add_single_document(self, text_fields, keyword_fields, temp_db):
        """Test adding a single document."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)

        # Add first document
        doc1 = {"question": "Python Programming", "text": "Learn Python", "course": "CS101"}
        index.add(doc1)

        results = index.search("python")
        assert len(results) == 1
        assert results[0]["question"] == "Python Programming"

        # Add second document
        doc2 = {"question": "Data Science", "text": "Python for data", "course": "CS102"}
        index.add(doc2)

        results = index.search("python")
        assert len(results) == 2

    def test_add_after_fit(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test adding documents after fit."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs[:2])

        # Add third document
        index.add(sample_docs[2])

        results = index.search("testing")
        assert len(results) == 1
        assert "pytest" in results[0]["text"]

    def test_fit_raises_error_when_not_empty(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test that fit() raises error when index is not empty."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs[:2])

        with pytest.raises(ValueError, match="Index already contains documents"):
            index.fit(sample_docs)


class TestTextSearchIndexClear:
    """Test clear() functionality."""

    def test_clear(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test clearing the index."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        assert len(index.search("python")) > 0

        index.clear()

        assert len(index.search("python")) == 0

        # Should be able to fit again after clear
        index.fit(sample_docs)
        assert len(index.search("python")) > 0


class TestTextSearchIndexIdField:
    """Test id_field functionality."""

    def test_id_field_with_custom_ids(self, temp_db):
        """Test using custom ID field."""
        docs = [
            {"doc_id": 100, "title": "Python", "category": "dev"},
            {"doc_id": 200, "title": "Java", "category": "dev"},
            {"doc_id": 300, "title": "C++", "category": "dev"},
        ]

        index = TextSearchIndex(
            text_fields=["title"],
            keyword_fields=["category"],
            id_field="doc_id",
            db_path=temp_db
        )
        index.fit(docs)

        results = index.search("python", output_ids=True)
        assert len(results) == 1
        assert results[0]["doc_id"] == 100
        assert results[0]["id"] == 100

    def test_id_field_without_custom_ids(self, temp_db):
        """Test without custom ID field (auto-generated IDs)."""
        docs = [
            {"title": "Python", "category": "dev"},
            {"title": "Java", "category": "dev"},
        ]

        index = TextSearchIndex(
            text_fields=["title"],
            keyword_fields=["category"],
            id_field=None,
            db_path=temp_db
        )
        index.fit(docs)

        results = index.search("python", output_ids=True)
        assert len(results) == 1
        # id should be the database row id (1 or greater)
        assert isinstance(results[0]["id"], int)
        assert results[0]["id"] >= 1

    def test_id_field_in_keyword_fields(self, temp_db):
        """Test that id_field is automatically added to keyword_fields."""
        docs = [
            {"doc_id": "abc123", "title": "Python"},
            {"doc_id": "def456", "title": "Java"},
        ]

        index = TextSearchIndex(
            text_fields=["title"],
            keyword_fields=["category"],
            id_field="doc_id",
            db_path=temp_db
        )

        # id_field should be added to keyword_fields
        assert "doc_id" in index.keyword_fields

        index.fit(docs)

        # Should be able to filter by id_field
        results = index.search("python", filter_dict={"doc_id": "abc123"})
        assert len(results) == 1


class TestTextSearchIndexEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_docs(self, text_fields, keyword_fields, temp_db):
        """Test behavior with empty document list."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit([])

        results = index.search("python")
        assert len(results) == 0

    def test_empty_fields(self, text_fields, keyword_fields, temp_db):
        """Test behavior with empty field values."""
        docs = [
            {"question": "", "text": "", "section": "", "course": ""},
            {"question": "Python", "text": "Programming", "section": "Dev", "course": "CS101"},
        ]

        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(docs)

        results = index.search("python")
        assert len(results) > 0

    def test_empty_query(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test search with empty query."""
        index = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index.fit(sample_docs)

        results = index.search("")
        # FTS5 with empty query may return all docs or none depending on implementation
        # Just verify it doesn't crash
        assert isinstance(results, list)

    def test_special_characters(self, temp_db):
        """Test search with special characters."""
        docs = [
            {"title": "Python-Programming", "description": "Learn Python (programming)"},
            {"title": "Data-Science", "description": "Python for data-science"},
        ]

        index = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=[],
            db_path=temp_db
        )
        index.fit(docs)

        results = index.search("python-programming")
        assert len(results) > 0


class TestTextSearchIndexNoneValues:
    """Test handling of None values in keyword fields."""

    @pytest.fixture
    def docs_with_none(self):
        return [
            {
                'url': 'https://example.com/1',
                'user_login': 'DouweM',
                'assignee_login': None,
                'state': 'open',
                'body': 'Description of tools'
            },
            {
                'url': 'https://example.com/2',
                'user_login': 'JohnDoe',
                'assignee_login': 'JaneSmith',
                'state': 'closed',
                'body': 'Some other description'
            },
            {
                'url': 'https://example.com/3',
                'user_login': 'AliceWonder',
                'assignee_login': None,
                'state': 'open',
                'body': 'Another tools with no assignee'
            },
        ]

    def test_index_with_none_values(self, docs_with_none, temp_db):
        """Test index can handle None values in keyword fields."""
        index = TextSearchIndex(
            text_fields=["body"],
            keyword_fields=["assignee_login", "state"],
            db_path=temp_db
        )

        index.fit(docs_with_none)
        results = index.search("tools")
        assert len(results) > 0

    def test_search_with_none_filter(self, docs_with_none, temp_db):
        """Test search with None value in filter."""
        index = TextSearchIndex(
            text_fields=["body"],
            keyword_fields=["assignee_login", "state"],
            db_path=temp_db
        )

        index.fit(docs_with_none)

        results = index.search("tools", filter_dict={"assignee_login": None})
        assert len(results) == 2
        for result in results:
            assert result['assignee_login'] is None

    def test_none_in_text_field(self, docs_with_none, temp_db):
        """Test handling None values in text fields."""
        # Add a doc with None text field
        docs_with_none.append({
            'url': 'https://example.com/4',
            'user_login': 'TestUser',
            'assignee_login': None,
            'state': 'open',
            'body': None
        })

        index = TextSearchIndex(
            text_fields=["body"],
            keyword_fields=["assignee_login", "state"],
            db_path=temp_db
        )

        index.fit(docs_with_none)
        results = index.search("tools")
        # Should not crash and should find docs with non-None body
        assert len(results) >= 2


class TestTextSearchIndexPersistence:
    """Test persistence across index instances."""

    def test_data_persists(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test that data persists across index instances."""
        # First index
        index1 = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        index1.fit(sample_docs)
        index1.close()

        # Second index with same db
        index2 = TextSearchIndex(text_fields, keyword_fields, db_path=temp_db)
        results = index2.search("python")
        assert len(results) > 0


class TestTextSearchIndexContextManager:
    """Test context manager functionality."""

    def test_context_manager(self, text_fields, keyword_fields, sample_docs, temp_db):
        """Test using index as context manager."""
        with TextSearchIndex(text_fields, keyword_fields, db_path=temp_db) as index:
            index.fit(sample_docs)
            results = index.search("python")
            assert len(results) > 0


class TestTextSearchIndexBoostAffectsRanking:
    """Test that boost parameters affect search result ranking."""

    def test_boost_affects_ranking(self, temp_db):
        """Test that boost parameters affect the ranking of search results."""
        docs = [
            {
                "title": "Introduction to Programming",
                "description": "Python is a popular programming language. Python is used in many applications.",
                "course": "CS101",
            },
            {
                "title": "Python for Beginners",
                "description": "Learn the basics of programming",
                "course": "CS102",
            },
            {
                "title": "Advanced Topics",
                "description": "Python is essential for data science. Python is used in machine learning.",
                "course": "CS103",
            },
        ]

        index = TextSearchIndex(
            text_fields=["title", "description"],
            keyword_fields=["course"],
            db_path=temp_db
        )
        index.fit(docs)

        # Search without boost
        results_no_boost = index.search("python")
        assert len(results_no_boost) > 0

        # Search with title boost
        results_title_boost = index.search("python", boost_dict={"title": 10.0})
        assert len(results_title_boost) > 0

        # When title is boosted, documents with "Python" in title should rank higher
        title_boosted_first = results_title_boost[0]
        assert "Python" in title_boosted_first["title"]


class TestTextSearchIndexFilterCombinations:
    """Test different combinations of filters."""

    def test_multiple_filters(self, temp_db):
        """Test search with multiple keyword filters."""
        docs = [
            {"title": "Python Programming", "course": "CS101", "level": "beginner"},
            {"title": "Data Science", "course": "CS102", "level": "intermediate"},
            {"title": "Machine Learning", "course": "CS101", "level": "advanced"},
            {"title": "Web Development", "course": "CS102", "level": "beginner"},
        ]

        index = TextSearchIndex(
            text_fields=["title"],
            keyword_fields=["course", "level"],
            db_path=temp_db
        )
        index.fit(docs)

        # Test single filter
        results = index.search("programming", filter_dict={"course": "CS101"})
        assert len(results) >= 1
        assert all(doc["course"] == "CS101" for doc in results)

        # Test multiple filters
        results = index.search("programming", filter_dict={"course": "CS101", "level": "beginner"})
        # Should return only CS101 + beginner (if any match)
        for doc in results:
            assert doc["course"] == "CS101"
            assert doc["level"] == "beginner"

        # Test non-existent filter
        results = index.search("programming", filter_dict={"course": "nonexistent"})
        assert len(results) == 0

    def test_filter_with_non_existent_keyword(self, temp_db):
        """Test filtering with a keyword that's not in keyword_fields."""
        docs = [
            {"title": "Python Programming", "course": "CS101"},
            {"title": "Data Science", "course": "CS102"},
        ]

        index = TextSearchIndex(
            text_fields=["title"],
            keyword_fields=["course"],
            db_path=temp_db
        )
        index.fit(docs)

        # Filter by keyword not in keyword_fields should be ignored
        results = index.search("python", filter_dict={"nonexistent": "value"})
        # Should still return results because filter is ignored
        assert len(results) >= 1
