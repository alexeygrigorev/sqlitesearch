"""Local Turso emulator -- a tiny stand-in for Turso Cloud's sync endpoint.

The ``libsql`` Python client (>=0.1.x) talks to a remote sync target with
Turso's protocol, which self-hosted ``libsql-server`` (sqld) does **not** serve
in a way this client accepts (it 404s on ``/export``). So there is no
off-the-shelf Docker image an embedded replica (``sync_url=...``) can sync
against locally. This module implements just enough of the protocol to let an
embedded replica work fully offline -- no Turso account, no network.

Protocol (reverse-engineered from ``libsql/src/sync.rs`` + Hrana-over-HTTP):

Read / replica bootstrap
* ``GET  /info``                -> ``{"current_generation": <u32>}``
* ``GET  /export/{generation}`` -> raw SQLite database file bytes (snapshot)
* ``GET  /sync/{gen}/{a}/{b}``  -> 200 + binary WAL frames, **or** 400 +
  ``{"generation": N}`` to signal end-of-generation (caught up). We always
  report caught-up: the single writer's data is captured server-side and
  handed back via ``/export``.

Writes (the part that makes bulk ingest slow -- see issue #3)
* ``POST /v3/pipeline``         -> Hrana pipeline. Every ``execute()`` the
  client runs against an embedded replica is **forwarded here** as an HTTP
  round-trip. We execute the statements against a real server-side SQLite
  database and return Hrana-shaped results. ``push_count`` counts these
  round-trips so a test can prove how many a bulk ingest takes.

This is *not* a complete Turso: same-replica local read-after-write isn't
frame-replicated back (reads on the writing replica may be stale), but a fresh
replica that bootstraps via ``/export`` sees all committed data, and the write
round-trip count is exact -- which is what issue #3 needs.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _ServerDB:
    """Authoritative server-side SQLite database written to via Hrana."""

    def __init__(self) -> None:
        self.generation = 1
        self.replication_index = 0
        self.push_count = 0          # POST /v3/pipeline round-trips
        self.statement_count = 0     # individual execute requests
        self.lock = threading.RLock()

        fd_path = tempfile.mkstemp(suffix=".emu.db")
        import os

        os.close(fd_path[0])
        self._path = fd_path[1]
        # isolation_level=None => autocommit; BEGIN/COMMIT in the SQL stream
        # drive transactions, so conn.in_transaction reflects real tx state
        # (the client asks for that via get_autocommit).
        self.conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)

    # --- Hrana value <-> Python conversion -------------------------------
    @staticmethod
    def _to_py(val: dict):
        t = val.get("type")
        if t == "null":
            return None
        if t == "integer":
            return int(val["value"])
        if t == "float":
            return float(val["value"])
        if t == "text":
            return val["value"]
        if t == "blob":
            # libsql sends base64 without padding; restore it before decoding.
            s = val.get("base64", "")
            return base64.b64decode(s + "=" * (-len(s) % 4))
        return None

    @staticmethod
    def _to_hrana(v):
        if v is None:
            return {"type": "null"}
        if isinstance(v, bool):
            return {"type": "integer", "value": str(int(v))}
        if isinstance(v, int):
            return {"type": "integer", "value": str(v)}
        if isinstance(v, float):
            return {"type": "float", "value": v}
        if isinstance(v, (bytes, bytearray)):
            return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode()}
        return {"type": "text", "value": str(v)}

    def _execute(self, stmt: dict) -> dict:
        sql = stmt.get("sql", "")
        args = [self._to_py(a) for a in stmt.get("args", [])]
        named = {a["name"]: self._to_py(a["value"]) for a in stmt.get("named_args", [])}
        params = named if named else args
        cur = self.conn.execute(sql, params)
        self.statement_count += 1
        want_rows = stmt.get("want_rows", True)
        cols, rows = [], []
        if cur.description:
            cols = [{"name": d[0], "decltype": None} for d in cur.description]
            if want_rows:
                rows = [[self._to_hrana(c) for c in row] for row in cur.fetchall()]
        self.replication_index += 1
        return {
            "type": "execute",
            "result": {
                "cols": cols,
                "rows": rows,
                "affected_row_count": cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0,
                "last_insert_rowid": str(cur.lastrowid) if cur.lastrowid else None,
                "replication_index": str(self.replication_index),
                "rows_read": len(rows),
                "rows_written": 0,
                "query_duration_ms": 0.0,
            },
        }

    def pipeline(self, body: dict) -> dict:
        """Run a Hrana pipeline request, returning the Hrana response dict."""
        with self.lock:
            self.push_count += 1
            results = []
            for req in body.get("requests", []):
                rtype = req.get("type")
                try:
                    if rtype == "execute":
                        results.append({"type": "ok", "response": self._execute(req["stmt"])})
                    elif rtype == "describe":
                        results.append({"type": "ok", "response": self._describe(req)})
                    elif rtype == "batch":
                        results.append({"type": "ok", "response": self._batch(req["batch"])})
                    elif rtype == "get_autocommit":
                        results.append(
                            {
                                "type": "ok",
                                "response": {
                                    "type": "get_autocommit",
                                    "is_autocommit": not self.conn.in_transaction,
                                },
                            }
                        )
                    else:  # close, open_stream, sequence, store_sql, ...
                        results.append({"type": "ok", "response": {"type": rtype or "noop"}})
                except Exception as e:  # surface SQL errors in Hrana shape
                    results.append(
                        {"type": "error", "error": {"message": str(e), "code": "SQLITE_ERROR"}}
                    )
            return {"baton": None, "base_url": None, "results": results}

    def _describe(self, req: dict) -> dict:
        """Describe a statement. The client routes reads locally and writes to
        the remote based on ``is_readonly``, so getting that right matters."""
        sql = (req.get("sql") or "").lstrip()
        head = sql[:8].upper()
        is_readonly = head.startswith(("SELECT", "WITH")) or head.startswith("PRAGMA")
        is_explain = head.startswith("EXPLAIN")
        n_params = sql.count("?")
        return {
            "type": "describe",
            "result": {
                "params": [{"name": None} for _ in range(n_params)],
                "cols": [],
                "is_explain": is_explain,
                "is_readonly": is_readonly,
            },
        }

    def _batch(self, batch: dict) -> dict:
        step_results = []
        for step in batch.get("steps", []):
            try:
                res = self._execute(step["stmt"])
                step_results.append({"type": "ok", "response": res})
            except Exception as e:
                step_results.append(
                    {"type": "error", "error": {"message": str(e), "code": "SQLITE_ERROR"}}
                )
        return {
            "type": "batch",
            "result": {
                "step_results": [r["response"]["result"] if r["type"] == "ok" else None for r in step_results],
                "step_errors": [r["error"] if r["type"] == "error" else None for r in step_results],
            },
        }

    def export_bytes(self) -> bytes:
        with self.lock:
            self.conn.commit()
            with open(self._path, "rb") as f:
                return f.read()


class _Handler(BaseHTTPRequestHandler):
    server_version = "TursoEmulator/0.2"

    @property
    def db(self) -> _ServerDB:
        return self.server.db  # type: ignore[attr-defined]

    def log_message(self, *args) -> None:
        pass

    def _send_json(self, obj: dict, code: int = 200) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, data: bytes, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parts = [p for p in self.path.split("/") if p]
        db = self.db
        if parts == ["info"]:
            self._send_json({"current_generation": db.generation})
        elif parts == ["__stats__"]:
            # Out-of-band stats endpoint so a parent process (the emulator runs
            # as a subprocess to avoid a GIL deadlock with the Rust client) can
            # read how many write round-trips happened.
            self._send_json(
                {
                    "push_count": db.push_count,
                    "statement_count": db.statement_count,
                    "generation": db.generation,
                }
            )
        elif len(parts) == 2 and parts[0] == "export":
            self._send_bytes(db.export_bytes())
        elif parts and parts[0] == "sync":
            # 400 + {"generation": N} == end-of-generation (caught up).
            self._send_json({"generation": db.generation}, code=400)
        else:
            self._send_json({"error": "not found", "path": self.path}, code=404)

    def do_POST(self) -> None:
        parts = [p for p in self.path.split("/") if p]
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        if parts[:2] == ["v3", "pipeline"] or parts[-1] == "pipeline":
            body = json.loads(raw or b"{}")
            self._send_json(self.db.pipeline(body))
        elif parts and parts[0] == "sync":
            # Frame push path (unused in Hrana-write mode); ack politely.
            self._send_json(
                {"status": "ok", "generation": self.db.generation, "max_frame_no": 0, "baton": None}
            )
        else:
            self._send_json({"error": "not found", "path": self.path}, code=404)


class TursoEmulator:
    """Context-managed local Turso sync server.

    Example::

        with TursoEmulator() as emu:
            conn = libsql.connect("replica.db", sync_url=emu.url, auth_token="x")
            ...
            print(emu.push_count)  # write round-trips
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.db = _ServerDB()
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.db = self.db  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def push_count(self) -> int:
        return self.db.push_count

    @property
    def statement_count(self) -> int:
        return self.db.statement_count

    def start(self) -> "TursoEmulator":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "TursoEmulator":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class EmulatorProcess:
    """Run the emulator in a **subprocess** and talk to it over HTTP.

    The ``libsql`` Rust client holds the GIL while blocking on its sync HTTP
    call, which starves a same-process Python server thread (deadlock). Running
    the emulator in its own process avoids that. Use this from tests/benchmarks::

        with EmulatorProcess() as emu:
            idx = VectorSearchIndex(..., db_path=emu.url, replica_path="replica.db")
            idx.fit(vectors, docs)
            print(emu.stats()["push_count"])  # write round-trips
    """

    def __init__(self, port: int = 8088) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self._proc = None

    def start(self) -> "EmulatorProcess":
        import subprocess
        import time
        import urllib.request

        self._proc = subprocess.Popen(
            [sys.executable, __file__, str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                urllib.request.urlopen(f"{self.url}/info", timeout=0.5).read()
                return self
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("emulator did not become ready")

    def stats(self) -> dict:
        import json as _json
        import urllib.request

        with urllib.request.urlopen(f"{self.url}/__stats__", timeout=2) as r:
            return _json.loads(r.read())

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def __enter__(self) -> "EmulatorProcess":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    emu = TursoEmulator(port=port).start()
    print(f"Turso emulator listening on {emu.url}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        emu.stop()
