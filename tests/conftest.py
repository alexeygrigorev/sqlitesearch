"""Shared pytest fixtures.

Holds the real-Docker ``sqld`` fixture used by the ``@pytest.mark.integration``
suite. It mirrors the repo's ``EmulatorProcess`` style -- a small subprocess
manager around ``docker run`` / ``docker rm`` -- rather than pulling in
``testcontainers``. That keeps the dev dependency surface unchanged (the
integration job only needs the ``docker`` CLI, which CI runners already have)
and the lifecycle is a handful of lines, so the heavier library buys nothing.

The fixture is import-safe on machines without Docker: it only ever runs when an
``integration``-marked test requests it, and it calls ``pytest.skip`` (never
errors) when the ``docker`` CLI is missing, the daemon is down, or the image
cannot be pulled. That lets ``uv run pytest`` (which excludes ``integration``)
stay green everywhere, and ``uv run pytest -m integration`` skip cleanly instead
of failing where Docker is absent.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import pytest

# Official libsql-server (aka sqld). Serves Hrana over HTTP on 8080 in-container.
SQLD_IMAGE = "ghcr.io/tursodatabase/libsql-server:latest"


def _docker_available() -> bool:
    """True when the docker CLI exists and its daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _free_port() -> int:
    """Grab an OS-assigned free TCP port for the container's host mapping."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pull_image() -> None:
    """Pull the sqld image, skipping the test if the pull fails (e.g. no network)."""
    proc = subprocess.run(
        ["docker", "pull", SQLD_IMAGE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not pull {SQLD_IMAGE}: {proc.stdout.strip()[-400:]}")


class SqldContainer:
    """A real ``sqld`` (libsql-server) container managed over the docker CLI.

    Lifecycle mirrors ``dev/turso_emulator.EmulatorProcess``: ``start`` runs the
    container detached and polls until it answers HTTP, ``stop`` force-removes
    it. ``url`` is the host-side ``http://127.0.0.1:<port>`` an embedded replica
    or a remote libsql client points at.
    """

    def __init__(self) -> None:
        self.port = _free_port()
        self.name = f"sqlitesearch-sqld-{uuid.uuid4().hex[:10]}"
        self.url = f"http://127.0.0.1:{self.port}"
        self._cid: str | None = None

    def start(self, ready_timeout: float = 45.0) -> SqldContainer:
        run = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--name", self.name,
                "-p", f"{self.port}:8080",
                SQLD_IMAGE,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            text=True,
        )
        if run.returncode != 0:
            pytest.skip(f"docker run failed for {SQLD_IMAGE}: {run.stdout.strip()[-400:]}")
        self._cid = run.stdout.strip()
        self._wait_ready(ready_timeout)
        return self

    def _wait_ready(self, timeout: float) -> None:
        # sqld answers GET / once HTTP is up; any HTTP response (even 4xx) means
        # the listener is serving, which is all we need before opening a client.
        deadline = time.time() + timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                urllib.request.urlopen(self.url + "/", timeout=2).read()
                return
            except urllib.error.HTTPError:
                return  # listener is up; it just doesn't like GET /
            except Exception as e:  # ConnectionRefused while still booting
                last_err = e
                time.sleep(0.3)
        self.stop()
        pytest.skip(f"sqld container did not become ready in {timeout}s: {last_err!r}")

    def stop(self) -> None:
        if self._cid is None:
            return
        # --rm cleans up on stop; `docker rm -f` is the belt-and-suspenders path.
        subprocess.run(
            ["docker", "rm", "-f", self.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        self._cid = None


@pytest.fixture
def sqld_container():
    """Yield a running real ``sqld`` container, or skip when Docker is unavailable."""
    if not _docker_available():
        pytest.skip("docker CLI/daemon not available; skipping real-Docker integration test")
    _pull_image()
    container = SqldContainer().start()
    try:
        yield container
    finally:
        container.stop()
