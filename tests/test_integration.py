"""
Integration tests comparing minsearch and sqlitesearch outputs.

Ensures sqlitesearch produces similar results to minsearch for text search.
"""

import os
import tempfile

import pytest
import requests
from minsearch import Index as MinsearchIndex

from sqlitesearch import TextSearchIndex


@pytest.fixture
def sample_docs():
    """Fetch documents from the LLM RAG workshop."""
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url, timeout=30)
    documents_raw = docs_response.json()

    # Process documents
    documents = []
    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    return documents


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


class TestMinsearchComparison:
    """Compare outputs between minsearch and sqlitesearch."""

    def test_basic_search_comparison(self, sample_docs, temp_db):
        """Test that both libraries return results for the same query."""
        # Create indices with same configuration
        minsearch_index = MinsearchIndex(
            text_fields=["question", "text", "section"],
            keyword_fields=["course"]
        )

        sqlite_index = TextSearchIndex(
            text_fields=["question", "text", "section"],
            keyword_fields=["course"],
            db_path=temp_db
        )

        # Fit both indices
        minsearch_index.fit(sample_docs)
        sqlite_index.fit(sample_docs)

        # Test query
        query = "Can I join the course after it has started?"
        filter_dict = {"course": "data-engineering-zoomcamp"}

        minsearch_results = minsearch_index.search(
            query,
            filter_dict=filter_dict,
            num_results=5
        )
        sqlite_results = sqlite_index.search(
            query,
            filter_dict=filter_dict,
            num_results=5
        )

        # Both should return results
        assert len(minsearch_results) > 0, "Minsearch should return results"
        assert len(sqlite_results) > 0, "Sqlitesearch should return results"

        # Both should return same number of results (up to num_results)
        assert len(minsearch_results) <= 5
        assert len(sqlite_results) <= 5

        # All results should be from the correct course
        for result in minsearch_results:
            assert result["course"] == "data-engineering-zoomcamp"
        for result in sqlite_results:
            assert result["course"] == "data-engineering-zoomcamp"

        sqlite_index.close()

    def test_boosted_search_comparison(self, sample_docs, temp_db):
        """Test that boosting works similarly in both libraries."""
        minsearch_index = MinsearchIndex(
            text_fields=["question", "text", "section"],
            keyword_fields=["course"]
        )

        sqlite_index = TextSearchIndex(
            text_fields=["question", "text", "section"],
            keyword_fields=["course"],
            db_path=temp_db
        )

        minsearch_index.fit(sample_docs)
        sqlite_index.fit(sample_docs)

        query = "How do I register for the course?"
        filter_dict = {"course": "data-engineering-zoomcamp"}
        boost_dict = {"question": 3.0}

        minsearch_results = minsearch_index.search(
            query,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=5
        )
        sqlite_results = sqlite_index.search(
            query,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=5
        )

        # Both should return results
        assert len(minsearch_results) > 0
        assert len(sqlite_results) > 0

        # Results should be relevant - check that at least some results contain course-related terms
        course_keywords = ["course", "register", "join", "zoomcamp", "start"]
        assert any(
            any(keyword in r.get("question", "").lower() or keyword in r.get("text", "").lower()
                for keyword in course_keywords)
            for r in minsearch_results
        ), "Minsearch results should be relevant"
        assert any(
            any(keyword in r.get("question", "").lower() or keyword in r.get("text", "").lower()
                for keyword in course_keywords)
            for r in sqlite_results
        ), "Sqlitesearch results should be relevant"

        sqlite_index.close()

    def test_multiple_queries_comparison(self, sample_docs, temp_db):
        """Test multiple queries to ensure consistent behavior."""
        minsearch_index = MinsearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"]
        )

        sqlite_index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"],
            db_path=temp_db
        )

        minsearch_index.fit(sample_docs)
        sqlite_index.fit(sample_docs)

        queries = [
            "course prerequisites",
            "project requirements",
            "certificate",
            "zoomcamp schedule"
        ]

        for query in queries:
            minsearch_results = minsearch_index.search(query, num_results=3)
            sqlite_results = sqlite_index.search(query, num_results=3)

            # Both should return results for each query
            assert len(minsearch_results) > 0, f"No minsearch results for: {query}"
            assert len(sqlite_results) > 0, f"No sqlitesearch results for: {query}"

        sqlite_index.close()

    def test_no_results_scenario(self, sample_docs, temp_db):
        """Test that both return empty when nothing matches."""
        minsearch_index = MinsearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"]
        )

        sqlite_index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"],
            db_path=temp_db
        )

        minsearch_index.fit(sample_docs)
        sqlite_index.fit(sample_docs)

        # Query with non-existent course filter
        query = "general question"
        filter_dict = {"course": "nonexistent-course"}

        minsearch_results = minsearch_index.search(query, filter_dict=filter_dict)
        sqlite_results = sqlite_index.search(query, filter_dict=filter_dict)

        # Both should return empty
        assert len(minsearch_results) == 0
        assert len(sqlite_results) == 0

        sqlite_index.close()

    def test_different_courses(self, sample_docs, temp_db):
        """Test filtering by different courses."""
        minsearch_index = MinsearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"]
        )

        sqlite_index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"],
            db_path=temp_db
        )

        minsearch_index.fit(sample_docs)
        sqlite_index.fit(sample_docs)

        # Get unique courses from sample docs
        courses = set(doc.get("course") for doc in sample_docs if doc.get("course"))

        for course in list(courses)[:3]:  # Test first 3 courses
            query = "course information"
            filter_dict = {"course": course}

            minsearch_results = minsearch_index.search(query, filter_dict=filter_dict, num_results=3)
            sqlite_results = sqlite_index.search(query, filter_dict=filter_dict, num_results=3)

            # All results should be from the specified course
            for result in minsearch_results:
                assert result.get("course") == course
            for result in sqlite_results:
                assert result.get("course") == course

        sqlite_index.close()

    def test_result_fields_match(self, sample_docs, temp_db):
        """Test that results contain the same fields."""
        minsearch_index = MinsearchIndex(
            text_fields=["question", "text", "section"],
            keyword_fields=["course"]
        )

        sqlite_index = TextSearchIndex(
            text_fields=["question", "text", "section"],
            keyword_fields=["course"],
            db_path=temp_db
        )

        minsearch_index.fit(sample_docs)
        sqlite_index.fit(sample_docs)

        query = "homework"
        minsearch_results = minsearch_index.search(query, num_results=1)
        sqlite_results = sqlite_index.search(query, num_results=1)

        if minsearch_results and sqlite_results:
            minsearch_keys = set(minsearch_results[0].keys())
            sqlite_keys = set(sqlite_results[0].keys())

            # Should have the same essential fields
            essential_fields = {"question", "text", "section", "course"}
            assert essential_fields.issubset(minsearch_keys)
            assert essential_fields.issubset(sqlite_keys)

        sqlite_index.close()

    def test_persistence_comparison(self, sample_docs, temp_db):
        """Test that sqlitesearch persists data while minsearch doesn't."""
        # Create and fit sqlitesearch index
        sqlite_index1 = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"],
            db_path=temp_db
        )
        sqlite_index1.fit(sample_docs)
        sqlite_index1.close()

        # Create new sqlitesearch instance with same db - should have data
        sqlite_index2 = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["course"],
            db_path=temp_db
        )

        results = sqlite_index2.search("course", num_results=5)
        assert len(results) > 0, "Sqlitesearch should persist data across instances"

        sqlite_index2.close()

        # Minsearch is in-memory - would need to re-fit for new instance
        # This is the key difference between the libraries
