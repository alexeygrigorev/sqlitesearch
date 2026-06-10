"""Connection layer for sqlitesearch.

Lets the library talk to either:

* the standard library ``sqlite3`` module against a local file (default), or
* ``libsql`` (Turso) -- a local file, or an embedded replica that syncs to a
  remote Turso database for free persistent storage on ephemeral hosts.

The rest of the codebase is written against the ``sqlite3`` API and reads
columns by name (``row["id"]``), which relies on ``row_factory = sqlite3.Row``.
The ``libsql`` client returns plain tuples and has no ``row_factory``, so this
module wraps a libsql connection in thin adapters that make its rows behave
like ``sqlite3.Row`` (named *and* positional access). That keeps every
``row["..."]`` access in the strategies working unchanged.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA cache_size=-64000",
)

# Conservative cap below SQLite's historical SQLITE_MAX_VARIABLE_NUMBER (999)
# so a multi-row INSERT never exceeds the bound on any backend/version.
_MAX_SQL_VARS = 900

# For the libsql embedded replica each statement is a network round-trip, so we
# want the largest safe multi-row INSERT. SQLite's modern limit is 32766; 30000
# leaves margin and the libsql client accepts it. For local sqlite3 the chunk
# size barely affects speed (no network), so the conservative cap stays -- it's
# also safe on old SQLite.
_MAX_SQL_VARS_NETWORK = 30000


def max_sql_vars(backend: str) -> int:
    """Bound-variable budget per statement for a backend (see #13)."""
    return _MAX_SQL_VARS_NETWORK if backend == "libsql" else _MAX_SQL_VARS


def _chunk_rows(rows: list, n_cols: int, max_vars: int = _MAX_SQL_VARS):
    max_rows = max(1, max_vars // max(1, n_cols))
    for i in range(0, len(rows), max_rows):
        yield rows[i : i + max_rows]


def bulk_insert(cursor, table: str, columns: list, rows: list, max_vars: int = _MAX_SQL_VARS) -> None:
    """Insert ``rows`` using chunked multi-row ``INSERT`` statements.

    Equivalent to ``cursor.executemany`` but collapses each chunk of rows into a
    single statement. Over the libsql/Turso embedded-replica backend, every
    statement is a network round-trip to the remote primary, so an N-row
    ``executemany`` becomes N round-trips. Batching turns that into ~N/chunk
    round-trips (see issue #3); ``max_vars`` controls the chunk size. ``columns``
    are pre-formatted column tokens; ``rows`` is a list of equal-length value
    sequences.
    """
    if not rows:
        return
    n_cols = len(columns)
    col_sql = ", ".join(columns)
    row_ph = "(" + ", ".join(["?"] * n_cols) + ")"
    for chunk in _chunk_rows(rows, n_cols, max_vars):
        values_sql = ", ".join([row_ph] * len(chunk))
        flat = [v for row in chunk for v in row]
        cursor.execute(f"INSERT INTO {table} ({col_sql}) VALUES {values_sql}", flat)


def bulk_insert_returning_ids(cursor, table: str, columns: list, rows: list, max_vars: int = _MAX_SQL_VARS) -> list:
    """Like :func:`bulk_insert`, returning the autoincrement ids of the inserted
    rows in order.

    Relies on a single writer and ``INTEGER PRIMARY KEY`` rowids being assigned
    consecutively within each chunk (true for SQLite/libsql), so the chunk's ids
    are ``[lastrowid - len(chunk) + 1 .. lastrowid]``.
    """
    ids: list = []
    if not rows:
        return ids
    n_cols = len(columns)
    col_sql = ", ".join(columns)
    row_ph = "(" + ", ".join(["?"] * n_cols) + ")"
    for chunk in _chunk_rows(rows, n_cols, max_vars):
        values_sql = ", ".join([row_ph] * len(chunk))
        flat = [v for row in chunk for v in row]
        cursor.execute(f"INSERT INTO {table} ({col_sql}) VALUES {values_sql}", flat)
        last = cursor.lastrowid
        ids.extend(range(last - len(chunk) + 1, last + 1))
    return ids


def bulk_upsert(cursor, table: str, columns: list, rows: list, conflict_col: str, update_cols: list, max_vars: int = _MAX_SQL_VARS) -> None:
    """Chunked multi-row ``INSERT ... ON CONFLICT(conflict_col) DO UPDATE``.

    Used for shared/hybrid files (issue #2): documents are keyed by the user's
    ``id_field`` so fitting the same corpus into both the text and vector index
    updates the *same* row instead of duplicating it. ``conflict_col`` and the
    members of ``update_cols`` must be pre-quoted column tokens.
    """
    if not rows:
        return
    n_cols = len(columns)
    col_sql = ", ".join(columns)
    row_ph = "(" + ", ".join(["?"] * n_cols) + ")"
    set_sql = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
    for chunk in _chunk_rows(rows, n_cols, max_vars):
        values_sql = ", ".join([row_ph] * len(chunk))
        flat = [v for row in chunk for v in row]
        cursor.execute(
            f"INSERT INTO {table} ({col_sql}) VALUES {values_sql} "
            f"ON CONFLICT({conflict_col}) DO UPDATE SET {set_sql}",
            flat,
        )


def fetch_ids_by_key(cursor, table: str, id_col: str, key_values: list, max_vars: int = _MAX_SQL_VARS) -> dict:
    """Return ``{str(key_value): internal_id}`` for the given key column values.

    Keys are normalised to ``str`` because the id column has TEXT affinity, so
    e.g. an integer id ``100`` is stored (and read back) as ``"100"``; callers
    look up with ``str(value)`` to avoid an int/str mismatch.
    """
    out: dict = {}
    keys = list(key_values)
    for i in range(0, len(keys), max_vars):
        chunk = keys[i : i + max_vars]
        ph = ",".join(["?"] * len(chunk))
        cursor.execute(f"SELECT id, {id_col} FROM {table} WHERE {id_col} IN ({ph})", chunk)
        for row in cursor.fetchall():
            out[str(row[1])] = row[0]
    return out


class _Row:
    """A tuple that also supports ``row["column"]`` access, like sqlite3.Row."""

    __slots__ = ("_values", "_cols")

    def __init__(self, values: tuple, cols: dict[str, int]):
        self._values = values
        self._cols = cols

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._values[self._cols[key]]
        return self._values[key]

    def keys(self):
        return list(self._cols.keys())

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"_Row({self._values!r})"


class _CursorWrapper:
    """Wraps a libsql cursor so fetch* return :class:`_Row` objects."""

    def __init__(self, cursor):
        self._cursor = cursor

    def _cols(self) -> dict[str, int]:
        desc = self._cursor.description
        return {d[0]: i for i, d in enumerate(desc)} if desc else {}

    def execute(self, *args, **kwargs):
        self._cursor.execute(*args, **kwargs)
        return self

    def executemany(self, *args, **kwargs):
        self._cursor.executemany(*args, **kwargs)
        return self

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return _Row(tuple(row), self._cols())

    def fetchall(self):
        cols = self._cols()
        return [_Row(tuple(r), cols) for r in self._cursor.fetchall()]

    def __iter__(self):
        cols = self._cols()
        for r in self._cursor:
            yield _Row(tuple(r), cols)

    def __getattr__(self, name):
        # lastrowid, description, rowcount, etc. fall through to the real cursor
        return getattr(self._cursor, name)


class _ConnectionWrapper:
    """Wraps a libsql connection so cursors yield :class:`_Row` objects."""

    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return _CursorWrapper(self._conn.cursor())

    def execute(self, *args, **kwargs):
        return _CursorWrapper(self._conn.execute(*args, **kwargs))

    def sync(self):
        """Push local writes to / pull updates from the remote Turso replica."""
        sync = getattr(self._conn, "sync", None)
        if sync is not None:
            sync()

    def __getattr__(self, name):
        # commit, rollback, close, etc. fall through to the real connection
        return getattr(self._conn, name)


def connect(
    db_path: str,
    *,
    backend: str = "sqlite3",
    sync_url: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> Any:
    """Open a connection for sqlitesearch.

    Args:
        db_path: Local database file path. With libsql this is the local file
            (or local embedded-replica file when ``sync_url`` is set).
        backend: ``"sqlite3"`` (default, stdlib) or ``"libsql"`` (local file or,
            with ``sync_url``, a Turso Cloud embedded replica). Providing
            ``sync_url`` implies ``"libsql"``.
        sync_url: Turso database URL (``libsql://...``) for an embedded
            replica. Reads hit the local file; writes sync to Turso.
        auth_token: Turso auth token (used with ``sync_url``).

    Returns:
        A connection object exposing the sqlite3 API with ``sqlite3.Row``-style
        row access, regardless of backend.
    """
    if sync_url:
        backend = "libsql"

    if backend == "libsql":
        import libsql

        if sync_url:
            raw = libsql.connect(db_path, sync_url=sync_url, auth_token=auth_token)
            raw.sync()
        else:
            raw = libsql.connect(db_path)
        conn: Any = _ConnectionWrapper(raw)
    elif backend == "sqlite3":
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    for pragma in _PRAGMAS:
        try:
            conn.execute(pragma)
        except Exception:
            # Some pragmas may be unsupported / no-ops on a given backend.
            pass

    return conn
