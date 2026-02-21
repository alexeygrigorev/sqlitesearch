"""
TextSearchIndex - Persistent full-text search using SQLite FTS5.

This module provides a text search index backed by SQLite's FTS5 (Full-Text Search)
extension, enabling efficient BM25 ranking and boolean queries.
"""

import json
import sqlite3
import threading
from datetime import date, datetime
from typing import Any, Optional

from sqlitesearch.operators import OPERATORS, is_range_filter
from sqlitesearch.tokenizer import Tokenizer


class TextSearchIndex:
    """
    A persistent text search index using SQLite FTS5.

    This index stores documents in a SQLite database and uses FTS5 for efficient
    full-text search with BM25 ranking.

    API matches minsearch.Index for easy migration:
    - __init__(text_fields, keyword_fields=None, numeric_fields=None, date_fields=None, id_field=None)
    - fit(docs) - Index documents (only if index is empty)
    - add(doc) - Add a single document to existing index
    - search(query, filter_dict=None, boost_dict=None, num_results=10, output_ids=False)

    Example:
        >>> index = TextSearchIndex(
        ...     text_fields=["title", "description"],
        ...     keyword_fields=["category"],
        ...     numeric_fields=["price", "rating"],
        ...     date_fields=["created_at"],
        ...     id_field="id",
        ...     db_path="search.db"
        ... )
        >>> index.fit([{"id": 1, "title": "Hello", "description": "World", "price": 100}])
        >>> results = index.search("hello", filter_dict={"price": [('>=', 50), ('<', 200)]})
    """

    def __init__(
        self,
        text_fields: list[str],
        keyword_fields: Optional[list[str]] = None,
        numeric_fields: Optional[list[str]] = None,
        date_fields: Optional[list[str]] = None,
        id_field: Optional[str] = None,
        db_path: str = "sqlitesearch.db",
        stemming: bool = False,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize the TextSearchIndex.

        Args:
            text_fields: List of field names to index with FTS5.
            keyword_fields: List of field names for exact filtering (not full-text searched).
            numeric_fields: List of field names for numeric range filtering.
            date_fields: List of field names for date range filtering.
            id_field: Field name to use as document ID. If None, auto-generates IDs.
            db_path: Path to the SQLite database file.
            stemming: If True, use Porter stemmer for better matching (e.g., "running" matches "run").
            tokenizer: Tokenizer instance for query processing. If None, uses a default
                tokenizer with English stop words. Pass Tokenizer() for no stop words,
                or Tokenizer(stop_words={'custom', 'words'}) for custom stop words.
        """
        self.text_fields = text_fields
        self.keyword_fields = list(keyword_fields) if keyword_fields is not None else []
        self.numeric_fields = list(numeric_fields) if numeric_fields is not None else []
        self.date_fields = list(date_fields) if date_fields is not None else []
        self.id_field = id_field
        self.db_path = db_path
        self.stemming = stemming
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer(stop_words='english')
        self._local = threading.local()

        # Add id_field to keyword_fields if provided and not already there
        if self.id_field and self.id_field not in self.keyword_fields:
            self.keyword_fields.append(self.id_field)

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Build keyword column definitions
        keyword_cols = []
        for field in self.keyword_fields:
            keyword_cols.append(f', "{field}" TEXT')
        keyword_sql = "\n".join(keyword_cols)

        # Build numeric column definitions
        numeric_cols = []
        for field in self.numeric_fields:
            numeric_cols.append(f', "{field}" REAL')
        numeric_sql = "\n".join(numeric_cols)

        # Build date column definitions (store as ISO 8601 strings for comparison)
        date_cols = []
        for field in self.date_fields:
            date_cols.append(f', "{field}" TEXT')
        date_sql = "\n".join(date_cols)

        # Create main documents table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_json TEXT NOT NULL{keyword_sql}{numeric_sql}{date_sql}
            )
        """)

        # Create FTS5 virtual table
        # Note: tokenizer applies to both indexing AND querying
        fts_columns = ["docid"] + [f'"{col}"' for col in self.text_fields]
        fts_col_list = ", ".join(fts_columns)

        if self.stemming:
            tokenizer = "tokenize='porter unicode61'"
        else:
            tokenizer = "tokenize='unicode61'"

        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                {fts_col_list},
                {tokenizer}
            )
        """)

        # Create indexes on keyword fields for faster filtering
        for field in self.keyword_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{field} ON docs ("{field}")')

        # Create indexes on numeric fields for faster filtering
        for field in self.numeric_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_num_{field} ON docs ("{field}")')

        # Create indexes on date fields for faster filtering
        for field in self.date_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_date_{field} ON docs ("{field}")')

        conn.commit()

    def _is_empty(self) -> bool:
        """Check if the index is empty."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM docs")
        row = cursor.fetchone()
        return row["count"] == 0

    def fit(self, docs: list[dict[str, Any]]) -> "TextSearchIndex":
        """
        Index the provided documents.

        Only works if the index is empty. Use add() to append documents.

        Args:
            docs: List of documents to index. Each document is a dictionary.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If the index already contains documents.
        """
        if not self._is_empty():
            raise ValueError(
                "Index already contains documents. "
                "Use clear() to reset the index or add() to append documents."
            )

        return self._add_docs(docs)

    def add(self, doc: dict[str, Any]) -> "TextSearchIndex":
        """
        Add a single document to the index.

        Args:
            doc: Document to add. Must be a dictionary.

        Returns:
            self for method chaining.
        """
        return self._add_docs([doc])

    def _add_docs(self, docs: list[dict[str, Any]]) -> "TextSearchIndex":
        """Internal method to add documents to the index."""
        if not docs:
            return self

        conn = self._get_conn()
        cursor = conn.cursor()

        # Build column lists including keyword, numeric, and date fields
        filter_cols = (
            [f'"{field}"' for field in self.keyword_fields] +
            [f'"{field}"' for field in self.numeric_fields] +
            [f'"{field}"' for field in self.date_fields]
        )
        all_cols = ["doc_json"] + filter_cols
        col_names = ", ".join(all_cols)
        placeholders = ", ".join(["?"] * len(all_cols))

        fts_col_names = ", ".join(
            ["docid"] + [f'"{field}"' for field in self.text_fields]
        )
        fts_placeholders = ", ".join(["?"] * (1 + len(self.text_fields)))

        # Prepare all rows
        doc_rows = []
        fts_text_per_doc = []

        for doc in docs:
            # Convert date/datetime objects to ISO format for JSON serialization
            doc_for_json = {}
            for key, value in doc.items():
                if isinstance(value, (date, datetime)):
                    doc_for_json[key] = value.isoformat()
                else:
                    doc_for_json[key] = value
            doc_json = json.dumps(doc_for_json)

            keyword_vals = [doc.get(field) for field in self.keyword_fields]
            numeric_vals = [doc.get(field) for field in self.numeric_fields]

            date_vals = []
            for field in self.date_fields:
                value = doc.get(field)
                if isinstance(value, (date, datetime)):
                    date_vals.append(value.isoformat())
                else:
                    date_vals.append(value)

            doc_rows.append([doc_json] + keyword_vals + numeric_vals + date_vals)
            fts_text_per_doc.append(
                [str(doc.get(field, "")) for field in self.text_fields]
            )

        # Batch insert into docs table
        doc_insert_sql = f"INSERT INTO docs ({col_names}) VALUES ({placeholders})"
        cursor.executemany(doc_insert_sql, doc_rows)

        # Get the range of inserted IDs (AUTOINCREMENT guarantees sequential)
        cursor.execute("SELECT MAX(id) FROM docs")
        last_id = cursor.fetchone()[0]
        first_id = last_id - len(docs) + 1

        # Batch insert into FTS5 table
        fts_rows = []
        for i, fts_text in enumerate(fts_text_per_doc):
            fts_rows.append([first_id + i] + fts_text)

        fts_insert_sql = (
            f"INSERT INTO docs_fts ({fts_col_names}) VALUES ({fts_placeholders})"
        )
        cursor.executemany(fts_insert_sql, fts_rows)

        conn.commit()
        return self

    def clear(self) -> "TextSearchIndex":
        """
        Clear all documents from the index.

        Returns:
            self for method chaining.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM docs")
        cursor.execute("DELETE FROM docs_fts")

        conn.commit()
        return self

    def search(
        self,
        query: str,
        filter_dict: Optional[dict[str, Any]] = None,
        boost_dict: Optional[dict[str, float]] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search the index with the given query.

        Args:
            query: The search query string. Supports FTS5 query syntax.
            filter_dict: Dictionary of filters. Can include:
                - Keyword fields: {"field": "value"} for exact match
                - Numeric fields: {"field": [('>=', 100), ('<', 200)]} for range filters
                - Numeric fields: {"field": 100} for exact match
                - Date fields: {"field": [('>=', date(...)), ('<', date(...))]} for range filters
                - Any field: {"field": None} for null/missing values
            boost_dict: Dictionary of boost scores for text fields.
            num_results: Maximum number of results to return.
            output_ids: If True, adds an 'id' field with the document ID.

        Returns:
            List of documents matching the search criteria, ranked by relevance.
        """
        if filter_dict is None:
            filter_dict = {}
        if boost_dict is None:
            boost_dict = {}

        # Handle empty query - return empty results
        if not query or not query.strip():
            return []

        conn = self._get_conn()
        cursor = conn.cursor()

        # Build FTS5 query with boosts
        fts_query = self._build_fts_query(query, boost_dict)

        if not fts_query or not fts_query.strip():
            return []

        # Build WHERE clause for filters (keyword, numeric, date)
        where_clauses = []
        where_params = []

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                # Keyword field filters (exact match)
                if value is None:
                    where_clauses.append(f'd."{field}" IS NULL')
                else:
                    where_clauses.append(f'd."{field}" = ?')
                    where_params.append(value)

            elif field in self.numeric_fields:
                # Numeric field filters (exact match or range)
                where_clauses, where_params = self._add_numeric_filter(
                    where_clauses, where_params, field, value
                )

            elif field in self.date_fields:
                # Date field filters (exact match or range)
                where_clauses, where_params = self._add_date_filter(
                    where_clauses, where_params, field, value
                )

        where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        if where_clauses:
            # With filters: need to join before limiting
            search_query = f"""
                SELECT
                    f.docid,
                    d.doc_json,
                    bm25(docs_fts) AS score
                FROM docs_fts f
                JOIN docs d ON f.docid = d.id
                WHERE docs_fts MATCH ?{where_sql}
                ORDER BY score
                LIMIT ?
            """
            cursor.execute(search_query, [fts_query] + where_params + [num_results])
        else:
            # No filters: rank in FTS5 first, then join only top results
            search_query = f"""
                SELECT
                    top.docid,
                    d.doc_json
                FROM (
                    SELECT docid, bm25(docs_fts) AS score
                    FROM docs_fts
                    WHERE docs_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                ) top
                JOIN docs d ON top.docid = d.id
                ORDER BY top.score
            """
            cursor.execute(search_query, [fts_query, num_results])

        rows = cursor.fetchall()

        results = []
        for row in rows:
            doc = json.loads(row["doc_json"])
            # Convert ISO date strings back to date/datetime objects
            doc = self._convert_dates(doc)
            if output_ids:
                # Use id_field value if available, otherwise use database id
                if self.id_field:
                    doc_id = doc.get(self.id_field)
                    # Try to convert to int if possible
                    if doc_id is not None and str(doc_id).isdigit():
                        doc_id = int(doc_id)
                else:
                    doc_id = row["docid"]
                doc["id"] = doc_id
            results.append(doc)

        return results

    def _convert_dates(self, doc: dict[str, Any]) -> dict[str, Any]:
        """
        Convert ISO date strings back to date/datetime objects for date_fields.

        Args:
            doc: Document with potentially ISO formatted date strings.

        Returns:
            Document with date fields converted back to date/datetime objects.
        """
        if not self.date_fields:
            return doc

        for field in self.date_fields:
            if field in doc and doc[field] is not None:
                value = doc[field]
                if isinstance(value, str):
                    # Check if string contains time component (has 'T' or ' ')
                    has_time = 'T' in value or ' ' in value

                    if has_time:
                        # Parse as datetime
                        try:
                            doc[field] = datetime.fromisoformat(value)
                        except ValueError:
                            pass
                    else:
                        # Parse as date only
                        try:
                            doc[field] = date.fromisoformat(value)
                        except ValueError:
                            pass
        return doc

    def _add_numeric_filter(
        self,
        where_clauses: list[str],
        where_params: list[Any],
        field: str,
        value: Any,
    ) -> tuple[list[str], list[Any]]:
        """
        Add a numeric filter to the WHERE clause.

        Supports:
        - None/missing values: {"field": None}
        - Exact match: {"field": 100}
        - Range filters: {"field": [('>=', 100), ('<', 200)]}

        Returns:
            Tuple of (updated where_clauses, updated where_params).
        """
        if value is None:
            where_clauses.append(f'd."{field}" IS NULL')
        elif is_range_filter(value):
            # Range filter: [('>=', 100), ('<', 200)]
            for op, op_value in value:
                if op in OPERATORS and op_value is not None:
                    where_clauses.append(f'd."{field}" {op} ?')
                    where_params.append(op_value)
        else:
            # Exact match
            where_clauses.append(f'd."{field}" = ?')
            where_params.append(value)

        return where_clauses, where_params

    def _add_date_filter(
        self,
        where_clauses: list[str],
        where_params: list[Any],
        field: str,
        value: Any,
    ) -> tuple[list[str], list[Any]]:
        """
        Add a date filter to the WHERE clause.

        Supports:
        - None/missing values: {"field": None}
        - Exact match: {"field": date(...)} or {"field": "2024-01-15"}
        - Range filters: {"field": [('>=', date(...)), ('<', date(...))]}

        Returns:
            Tuple of (updated where_clauses, updated where_params).
        """
        if value is None:
            where_clauses.append(f'd."{field}" IS NULL')
        elif is_range_filter(value):
            # Range filter: [('>=', date(...)), ('<', date(...))]
            for op, op_value in value:
                if op in OPERATORS and op_value is not None:
                    # Convert date/datetime to ISO format string for comparison
                    if isinstance(op_value, (date, datetime)):
                        op_value = op_value.isoformat()
                    where_clauses.append(f'd."{field}" {op} ?')
                    where_params.append(op_value)
        else:
            # Exact match - convert date/datetime to ISO format
            if isinstance(value, (date, datetime)):
                value = value.isoformat()
            where_clauses.append(f'd."{field}" = ?')
            where_params.append(value)

        return where_clauses, where_params

    def _build_fts_query(self, query: str, boost_dict: dict[str, float]) -> str:
        """
        Build an FTS5 query with boost weights.

        Args:
            query: The raw query string.
            boost_dict: Field -> boost weight mapping.

        Returns:
            An FTS5 query string.
        """
        query_terms = self._extract_query_terms(query)

        # Note: empty queries are handled in search() method
        if not boost_dict:
            # OR query - any term matches (better recall)
            return " OR ".join(query_terms)

        # Build boosted query for each field
        parts = []

        for field in self.text_fields:
            boost = boost_dict.get(field, 1.0)
            if boost == 0:
                continue

            # Use OR within field for better recall
            field_query = " OR ".join(query_terms)
            parts.append(f'"{field}":({field_query})')

        return " OR ".join(parts) if parts else " OR ".join(query_terms)

    def _extract_query_terms(self, query: str) -> list[str]:
        """
        Extract search terms from a query string using the configured tokenizer.

        Uses self.tokenizer to split text, remove stop words, and optionally stem.
        Falls back to raw terms if all tokens are filtered out.
        """
        tokens = [t for t in self.tokenizer.tokenize(query) if t]
        if tokens:
            return tokens
        # Fallback: if all terms were stop words or empty, use raw terms
        import re
        raw = re.findall(r'\w+', query.lower())
        return raw if raw else [query]

    def _escape_fts_query(self, query: str) -> str:
        """
        Escape special FTS5 characters in a query.

        FTS5 special characters: " ( ) [ ] * & | - +
        """
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")

    def __enter__(self) -> "TextSearchIndex":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
