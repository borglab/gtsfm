"""Tests for best-effort Dask runner cleanup."""

from gtsfm.runner import _shutdown_dask_client


class _FakeClient:
    def __init__(self, *, shutdown_error: Exception | None = None, close_error: Exception | None = None) -> None:
        self.shutdown_error = shutdown_error
        self.close_error = close_error
        self.shutdown_called = False
        self.close_timeout: int | None = None

    def shutdown(self) -> None:
        self.shutdown_called = True
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def close(self, *, timeout: int) -> None:
        self.close_timeout = timeout
        if self.close_error is not None:
            raise self.close_error


def test_shutdown_dask_client_uses_graceful_shutdown() -> None:
    client = _FakeClient()

    _shutdown_dask_client(client)  # type: ignore[arg-type]

    assert client.shutdown_called
    assert client.close_timeout is None


def test_shutdown_dask_client_force_closes_after_timeout() -> None:
    client = _FakeClient(shutdown_error=TimeoutError())

    _shutdown_dask_client(client)  # type: ignore[arg-type]

    assert client.shutdown_called
    assert client.close_timeout == 2


def test_shutdown_dask_client_never_raises_cleanup_error() -> None:
    client = _FakeClient(shutdown_error=TimeoutError(), close_error=TimeoutError())

    _shutdown_dask_client(client)  # type: ignore[arg-type]

    assert client.shutdown_called
    assert client.close_timeout == 2
