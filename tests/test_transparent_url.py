"""Transparent libSQL/Turso URLs: passing a ``libsql://...`` URL as ``db_path``
sets up the embedded replica without the caller wiring sync_url/replica/auth."""

import tempfile

import pytest

from sqlitesearch.connection import (
    default_replica_path,
    is_remote_url,
    resolve_remote_target,
)


@pytest.mark.parametrize(
    "target",
    [
        "libsql://my-db.turso.io",
        "https://my-db.turso.io",
        "http://localhost:8080",
        "wss://my-db.turso.io",
        "ws://localhost:8080",
        "LIBSQL://My-DB.turso.io",  # scheme is case-insensitive
    ],
)
def test_is_remote_url_positive(target):
    assert is_remote_url(target)


@pytest.mark.parametrize(
    "target",
    [
        "faq.db",
        "/tmp/faq.db",
        ":memory:",
        "sqlitesearch_vectors.db",
        "./data/faq-replica.db",
        "",
        None,
        123,
    ],
)
def test_is_remote_url_negative(target):
    assert not is_remote_url(target)


def test_default_replica_path_is_deterministic_and_in_tempdir():
    url = "libsql://faq-db.turso.io"
    p1 = default_replica_path(url)
    p2 = default_replica_path(url)
    assert p1 == p2  # same URL -> same ephemeral file (reused across connections)
    assert p1.startswith(tempfile.gettempdir())
    assert p1.endswith(".db")


def test_default_replica_path_differs_per_url():
    a = default_replica_path("libsql://db-a.turso.io")
    b = default_replica_path("libsql://db-b.turso.io")
    assert a != b


def test_resolve_remote_target_url_only():
    db_path, sync_url = resolve_remote_target("libsql://my-db.turso.io")
    assert sync_url == "libsql://my-db.turso.io"
    assert db_path == default_replica_path("libsql://my-db.turso.io")
    assert not is_remote_url(db_path)  # resolved to a local replica file


def test_resolve_remote_target_url_with_explicit_replica():
    db_path, sync_url = resolve_remote_target(
        "libsql://my-db.turso.io", replica_path="/var/data/replica.db"
    )
    assert sync_url == "libsql://my-db.turso.io"
    assert db_path == "/var/data/replica.db"


def test_resolve_remote_target_local_path_passthrough():
    db_path, sync_url = resolve_remote_target("faq.db")
    assert db_path == "faq.db"
    assert sync_url is None


def test_local_index_uses_sqlite3_backend():
    from sqlitesearch import VectorSearchIndex

    ix = VectorSearchIndex(mode="lsh", db_path=":memory:")
    assert ix.backend == "sqlite3"
