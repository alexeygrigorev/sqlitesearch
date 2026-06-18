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

Design note - why an embedded replica, not a direct remote connection
---------------------------------------------------------------------
With libsql, every statement against a remote database is a network round-trip
to the Turso primary. Vector search is read-heavy: each query runs several
index probes plus a rerank lookup, so a plain remote connection would put
network latency on every one of those reads and make search slow.

The embedded replica fixes the read side -- reads are served from a local
SQLite file at disk speed. Writes still go *through* to the Turso primary on
commit: libsql forwards each statement, and ``commit()`` is write-through, so
an ingested row is durable on the remote immediately (verified against
libsql 0.1.x). ``bulk_insert`` batches writes only to cut the *number* of
round-trips, not to defer them. ``ConnectionWrapper.sync()`` is a
read-freshness call that pulls changes *down* from the primary -- it is not a
durability flush, so ingest does not depend on calling it.

So the win is local-file read speed with Turso's durable, restart-surviving
storage. The local replica file is therefore disposable -- ``connect`` defaults
it to an ephemeral temp file when a ``libsql://`` URL is passed as ``db_path``,
because the source of truth is always the remote primary, not the local copy.

Note on remotes: the embedded replica bootstraps by pulling a snapshot through
the Turso sync API (``GET /info`` + ``GET /export``). Turso Cloud serves that,
so a ``libsql://...turso.io`` URL works. A stock self-hosted ``sqld`` currently
speaks only Hrana and does not serve ``/export``, so the embedded replica will
not bootstrap against it -- use Turso Cloud (or a server that serves the sync
API) for the replica, or talk to a bare ``sqld`` in pure-remote mode instead.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from typing import Any
from urllib.parse import urlparse

# URL schemes that mean "this is a remote Turso/libSQL database, not a local
# file". When ``db_path`` uses one of these, sqlitesearch sets up the embedded
# replica transparently (local temp file + sync) instead of asking the caller
# to wire up ``sync_url``/``auth_token``/a replica path by hand.
_REMOTE_SCHEMES = frozenset({"libsql", "https", "http", "wss", "ws"})


def is_remote_url(target: Any) -> bool:
    """True when ``target`` is a remote libSQL/Turso URL rather than a path."""
    if not isinstance(target, str) or "://" not in target:
        return False
    return target.split("://", 1)[0].lower() in _REMOTE_SCHEMES


def default_replica_path(sync_url: str) -> str:
    """Pick an ephemeral local file to back an embedded replica of ``sync_url``.

    The file lives in the system temp dir, so the caller never has to name,
    create, or gitignore it. It is derived deterministically from the URL, so
    every connection in a process reuses the same replica. On an ephemeral host
    the temp file is gone after a restart and the replica simply re-syncs from
    Turso on the next boot - which is what would happen on that disk anyway.
    """
    name = urlparse(sync_url).netloc or sync_url
    digest = hashlib.sha1(sync_url.encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() else "-" for c in name)[:40].strip("-")
    return os.path.join(tempfile.gettempdir(), f"sqlitesearch-{safe}-{digest}.db")


def resolve_remote_target(
    db_path: str,
    replica_path: str | None = None,
) -> tuple[str, str | None]:
    """Resolve a possibly-remote ``db_path`` into ``(local_db_path, sync_url)``.

    Pure (no I/O). A remote URL becomes the sync target and is backed by a local
    embedded-replica file (``replica_path`` or an ephemeral temp file). A local
    path passes through with no sync target.
    """
    sync_url: str | None = None
    if is_remote_url(db_path):
        sync_url = db_path
        db_path = replica_path or default_replica_path(sync_url)
    return db_path, sync_url


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
    auth_token: str | None = None,
    replica_path: str | None = None,
) -> Any:
    """Open a connection for sqlitesearch.

    Args:
        db_path: Either a local database file path, or a remote Turso/libSQL
            URL (``libsql://...``). A URL is handled transparently: it becomes
            the sync target for a local embedded replica, so reads stay local
            and writes sync to the remote.
        backend: ``"sqlite3"`` (default, stdlib) or ``"libsql"``. A remote
            ``db_path`` implies ``"libsql"`` automatically.
        auth_token: Auth token for an authenticated remote database (e.g.
            Turso). Not needed for a local file or a local replica, and not
            needed for unauthenticated libsql servers. The caller passes it
            explicitly when required; the library never reads it from the
            environment.
        replica_path: Optional local file for the embedded replica. Defaults to
            an ephemeral temp file derived from the URL, which the caller never
            has to manage.

    Returns:
        A connection object exposing the sqlite3 API with ``sqlite3.Row``-style
        row access, regardless of backend.
    """
    # Transparent remote: a URL given as db_path is a remote database. Use it as
    # the sync target and back it with a local embedded replica (an ephemeral
    # temp file unless replica_path is given).
    db_path, sync_url = resolve_remote_target(db_path, replica_path)

    if sync_url:
        backend = "libsql"

    # Make sure the database file's directory exists before opening it - for a
    # local file or the embedded replica's local file alike - so callers never
    # have to mkdir by hand.
    if db_path and db_path != ":memory:":
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if backend == "libsql":
        import libsql

        if sync_url:
            # auth_token is optional: omit it when None so an unauthenticated
            # remote works without a token, and to avoid libsql 0.1.x raising a
            # TypeError on auth_token=None.
            kwargs: dict[str, Any] = {"sync_url": sync_url}
            if auth_token is not None:
                kwargs["auth_token"] = auth_token
            raw = libsql.connect(db_path, **kwargs)
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
