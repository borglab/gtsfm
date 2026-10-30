"""Ray adapter contract tests without real SSH, a Ray cluster, or GPU jobs."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visualization.ray_backend import RayWorkspace


@pytest.fixture
def spec() -> dict:
    """Return a direct Linux connection with separate Ray and project environments."""
    return {
        "ssh_host": "gpu-linux",
        "ssh_username": "researcher",
        "ssh_workspace": "~/research/gtsfm",
        "ssh_python": ".venv/bin/python",
        "ssh_ray_binary": "~/.local/bin/ray",
    }


def workspace(spec: dict, tmp_path: Path) -> RayWorkspace:
    """Build an isolated workspace with an explicit native Ray executable."""
    return RayWorkspace(spec, tmp_path, local_ray_binary="/tools/ray")


def ready_workspace(spec: dict, tmp_path: Path) -> RayWorkspace:
    """Provide a connected fake SDK, leaving native CLI operations unstarted."""
    result = workspace(spec, tmp_path)
    result.client = Mock()
    result.client.submit_job.side_effect = lambda **kw: kw["submission_id"]
    return result


def test_launcher_config_is_stable_and_never_installs_or_stops_other_runtimes(spec, tmp_path):
    """Use native local-provider settings and private ports across reconnects."""
    first = workspace(spec, tmp_path)
    second = workspace({**spec, "ssh_workspace": "/another/project"}, tmp_path)
    try:
        assert first.config_path == second.config_path
        assert first.configuration == second.configuration
        assert first.gcs_port != 6379
        assert first.dashboard_port != 8265
        config = first.configuration
        assert config["provider"] == {"type": "local", "head_ip": "gpu-linux", "worker_ips": []}
        assert config["max_workers"] == 0
        assert config["setup_commands"] == config["initialization_commands"] == []
        command = config["head_start_ray_commands"][0]
        assert '"$HOME"/.local/bin/ray start --head' in command
        assert "--object-store-memory=134217728" in command
        assert "--dashboard-host=127.0.0.1" in command
        assert "ray stop" not in command and "pip install" not in command
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"ssh_wsl_distribution": "Ubuntu"}, "direct Linux SSH"),
        ({"ssh_port": 2222}, "OpenSSH host alias"),
        ({"ssh_host": "-oProxyCommand=bad"}, "Linux SSH host"),
        ({"ssh_username": ""}, "Linux SSH username"),
    ],
)
def test_invalid_native_connections_fail_before_launch(spec, tmp_path, change, message):
    """Reject unsupported Windows-shell and connection options before side effects."""
    with pytest.raises(ValueError, match=message):
        workspace({**spec, **change}, tmp_path)


def test_submit_uses_project_python_remote_home_and_public_sdk_resources(spec, tmp_path):
    """Keep the GTSFM environment intact and quote all user arguments."""
    result = ready_workspace(spec, tmp_path)
    try:
        job_id = result.submit(
            ["~/research/gtsfm/.venv/bin/python", "-u", "-m", "gtsfm.runner", "a'; touch /tmp/oops"],
            env={"OMP_NUM_THREADS": "2"},
            submission_id="gtsfm-test",
            memory_bytes=123456,
        )
        assert job_id == "gtsfm-test"
        kwargs = result.client.submit_job.call_args.kwargs
        assert kwargs["entrypoint"].startswith('cd "$HOME"/research/gtsfm && export PYTHONPATH="$PWD')
        assert 'exec "$HOME"/research/gtsfm/.venv/bin/python -u -m gtsfm.runner' in kwargs["entrypoint"]
        assert "'a'\"'\"'; touch /tmp/oops'" in kwargs["entrypoint"]
        assert kwargs["runtime_env"] == {"env_vars": {"OMP_NUM_THREADS": "2"}}
        assert kwargs["entrypoint_num_gpus"] == kwargs["entrypoint_num_cpus"] == 1
        assert kwargs["entrypoint_memory"] == 123456
    finally:
        result.close()
    result.client.stop_job.assert_called_once_with("gtsfm-test")


def test_uncertain_submission_is_cancelled_without_replay(spec, tmp_path):
    """A lost SDK response retains the known submission ID for cleanup."""
    result = ready_workspace(spec, tmp_path)
    result.client.submit_job.side_effect = ConnectionError("response lost")
    with pytest.raises(ConnectionError):
        result.submit(["sleep", "60"], submission_id="ours")
    result.close()
    result.close()
    result.client.submit_job.assert_called_once()
    result.client.stop_job.assert_called_once_with("ours")
    with pytest.raises(RuntimeError, match="closed"):
        result.submit(["true"])


def test_only_owned_jobs_can_be_cancelled(spec, tmp_path):
    """Preserve unrelated Ray Jobs on the shared dedicated runtime."""
    result = ready_workspace(spec, tmp_path)
    try:
        with pytest.raises(ValueError, match="does not own"):
            result.cancel("someone-elses-job")
        result.client.stop_job.assert_not_called()
    finally:
        result.close()


def test_inspection_uses_zero_gpu_and_failure_still_stops_owned_job(spec, tmp_path):
    """Inspection errors include the native logs and trigger cleanup."""
    result = ready_workspace(spec, tmp_path)
    result.client.get_job_info.return_value = SimpleNamespace(status="FAILED")
    result.client.get_job_logs.return_value = "missing project dependency"
    try:
        with pytest.raises(RuntimeError, match="missing project dependency"):
            result.run_command(["python", "-c", "raise ImportError()"])
        assert result.client.submit_job.call_args.kwargs["entrypoint_num_gpus"] == 0
        result.client.stop_job.assert_called_once()
    finally:
        result.close()


def test_upload_and_results_use_native_rsync_contents_semantics(spec, tmp_path, monkeypatch):
    """Create the remote parent and transfer input/output through Ray's CLI."""
    result = ready_workspace(spec, tmp_path)
    commands = Mock()
    inspection = Mock()
    monkeypatch.setattr(result, "_run_cli", commands)
    monkeypatch.setattr(result, "run_command", inspection)
    source = tmp_path / "photos"
    source.mkdir()
    target = tmp_path / "download"
    try:
        result.upload_directory(source, "~/research/gtsfm/.gtsfm/staged/job")
        result.download_directory("~/research/gtsfm/results/run", target)
        inspection.assert_called_once_with(["mkdir", "-p", "~/research/gtsfm/.gtsfm/staged/job"], cwd="~")
        assert commands.call_args_list[0].args == (
            "rsync-up",
            str(result.config_path),
            str(source) + "/",
            "~/research/gtsfm/.gtsfm/staged/job/",
        )
        assert commands.call_args_list[1].args == (
            "rsync-down",
            str(result.config_path),
            "~/research/gtsfm/results/run/",
            str(target) + "/",
        )
        assert target.is_dir()
    finally:
        result.close()


def test_open_uses_native_cli_and_preserves_configuration(spec, tmp_path, monkeypatch):
    """Reuse the launcher configuration and a native dashboard tunnel."""
    result = workspace(spec, tmp_path)
    commands = Mock()
    tunnel = Mock()
    monkeypatch.setattr(result, "_run_cli", commands)
    monkeypatch.setattr(result, "_start_cli", Mock(return_value=tunnel))
    monkeypatch.setattr(result, "_connect", Mock())
    with result:
        assert commands.call_args_list[0].args[:4] == ("up", str(result.config_path), "-y", "--no-restart")
        assert commands.call_args_list[1].args[0] == "exec"
        assert " status --address=" in commands.call_args_list[1].args[2]
        assert " || " in commands.call_args_list[1].args[2]
        assert json.loads(result.config_path.read_text()) == result.configuration
        assert result.config_path.stat().st_mode & 0o777 == 0o600
        assert result._start_cli.call_args.args[:2] == ("dashboard", str(result.config_path))
    assert "stop" not in [call.args[0] for call in commands.call_args_list]


def test_close_during_launcher_startup_prevents_dashboard_and_jobs(spec, tmp_path, monkeypatch):
    """Cancel can interrupt ray up without waiting for the startup timeout."""
    result = workspace(spec, tmp_path)
    launched = threading.Event()
    stopped = threading.Event()
    process = Mock()

    def wait(timeout):
        """Represent a launcher waiting for authentication."""
        launched.set()
        assert stopped.wait(5)
        return -15

    def start(*args):
        """Register the fake native process as the real starter does."""
        with result._lock:
            result._ensure_open()
            result._processes.add(process)
            return process

    process.wait.side_effect = wait
    monkeypatch.setattr(result, "_start_cli", start)
    monkeypatch.setattr(result, "_stop_process", lambda _: stopped.set())
    errors = []

    def open_workspace():
        """Record the expected close-during-startup error from the worker thread."""
        try:
            result.__enter__()
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=open_workspace)
    thread.start()
    assert launched.wait(5)
    result.close()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == ["Ray workspace is closed"]
    assert result.tunnel is None
    with pytest.raises(RuntimeError, match="closed"):
        result.__enter__()


def test_native_launcher_failure_retains_stderr(spec, tmp_path, monkeypatch):
    """Expose SSH/authentication startup diagnostics to the local job log."""
    result = workspace(spec, tmp_path)
    process = Mock()
    process.wait.return_value = 1
    monkeypatch.setattr(result, "_start_cli", Mock(return_value=process))
    result._log.write("Tailscale additional authentication required: https://login.tailscale.com/check/test\n")
    with pytest.raises(RuntimeError, match="Tailscale additional authentication"):
        result.__enter__()
    assert "https://login.tailscale.com/check/test" in result.stderr_tail


def test_concurrent_close_waits_for_owner_to_finish_sdk_stop_and_transport_reap(spec, tmp_path, monkeypatch):
    """A shutdown following background cancel must wait for actual cleanup."""
    result = ready_workspace(spec, tmp_path)
    result.submit(["sleep", "60"], submission_id="ours")
    stopping, release, second_done = threading.Event(), threading.Event(), threading.Event()
    process = Mock()
    result._processes.add(process)
    reaped = []

    def stop(job_id):
        """Hold the native stop request to expose a concurrent close call."""
        stopping.set()
        assert release.wait(5)

    result.client.stop_job.side_effect = stop
    monkeypatch.setattr(result, "_stop_process", lambda p: reaped.append(p))
    first = threading.Thread(target=result.close)
    second = threading.Thread(target=lambda: (result.close(), second_done.set()))
    first.start()
    assert stopping.wait(5)
    second.start()
    assert not second_done.wait(0.05)
    release.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert second_done.is_set() and reaped == [process]
    result.client.stop_job.assert_called_once_with("ours")


@pytest.mark.parametrize("command", ["up", "exec", "dashboard", "rsync-up", "rsync-down"])
def test_installed_ray_supports_native_commands_used_by_adapter(spec, tmp_path, command):
    """Verify actual public CLI names against the optional installed dependency."""
    if importlib.util.find_spec("ray") is None:
        pytest.skip("optional Ray dependency is not installed")
    result = RayWorkspace(spec, tmp_path)
    try:
        completed = subprocess.run(
            [result.local_ray_binary, command, "--help"], capture_output=True, text=True, timeout=30
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Usage:" in completed.stdout
    finally:
        result.close()


@pytest.fixture
def native_log_stream(monkeypatch):
    """Use Ray's actual public SDK log iterator with an in-memory WebSocket transport."""
    aiohttp = pytest.importorskip("aiohttp")
    sdk = pytest.importorskip("ray.job_submission")
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"ray_version": "2.59.0"}
    monkeypatch.setattr("requests.request", Mock(return_value=response))
    state = SimpleNamespace(messages=asyncio.Queue(), closed=False, url=None, close_code=1000)

    class WebSocket:
        @property
        def close_code(self):
            return state.close_code

        async def receive(self):
            return await state.messages.get()

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            state.closed = True

        async def ws_connect(self, url, **kwargs):
            state.url = url
            return WebSocket()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    state.client = sdk.JobSubmissionClient("http://127.0.0.1:8265")
    state.text = lambda text: SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=text)
    state.eof = SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)
    return state


def test_native_sdk_tail_preserves_chunks_and_closes_session_at_eof(spec, tmp_path, native_log_stream):
    """Follow the real SDK generator through normal WebSocket EOF."""
    result = workspace(spec, tmp_path)
    result.client = native_log_stream.client

    async def consume():
        for chunk in ("first\n", "part", "ial\n"):
            native_log_stream.messages.put_nowait(native_log_stream.text(chunk))
        native_log_stream.messages.put_nowait(native_log_stream.eof)
        return [chunk async for chunk in result.tail_logs("ours")]

    try:
        assert asyncio.run(consume()) == ["first\n", "part", "ial\n"]
        assert native_log_stream.url.endswith("/api/jobs/ours/logs/tail")
        assert native_log_stream.closed
    finally:
        result.close()


def test_native_sdk_tail_unexpected_eof_propagates_while_open(spec, tmp_path, native_log_stream):
    """An open workspace must report unexpected native stream failures."""
    result = workspace(spec, tmp_path)
    result.client = native_log_stream.client
    native_log_stream.close_code = 1006

    async def consume():
        native_log_stream.messages.put_nowait(native_log_stream.eof)
        return [chunk async for chunk in result.tail_logs("ours")]

    try:
        with pytest.raises(RuntimeError, match="closed unexpectedly"):
            asyncio.run(consume())
        assert native_log_stream.closed
    finally:
        result.close()


def test_native_sdk_tail_releases_lock_and_ends_when_workspace_transport_closes(
    spec, tmp_path, monkeypatch, native_log_stream
):
    """A pending receive cannot hold up close; native transport EOF ends the reader."""
    result = workspace(spec, tmp_path)
    result.client = native_log_stream.client
    native_log_stream.close_code = 1006
    process = Mock()
    result._processes.add(process)

    async def consume():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(
            result,
            "_stop_process",
            lambda _: loop.call_soon_threadsafe(native_log_stream.messages.put_nowait, native_log_stream.eof),
        )
        task = asyncio.create_task(_collect_logs(result))
        await asyncio.sleep(0)
        try:
            await asyncio.wait_for(asyncio.to_thread(result.close), timeout=2)
            assert await asyncio.wait_for(task, timeout=2) == []
        finally:
            if not task.done():
                native_log_stream.messages.put_nowait(native_log_stream.eof)
                await task

    asyncio.run(consume())
    assert native_log_stream.closed
    assert result._close_complete.is_set()


async def _collect_logs(workspace):
    """Collect chunks from a workspace while preserving native generator cleanup."""
    return [chunk async for chunk in workspace.tail_logs("ours")]


def test_closing_consumer_closes_native_sdk_log_session(spec, tmp_path, native_log_stream):
    """An early consumer exit releases the SDK's WebSocket session immediately."""
    result = workspace(spec, tmp_path)
    result.client = native_log_stream.client

    async def consume_one():
        native_log_stream.messages.put_nowait(native_log_stream.text("first chunk"))
        stream = result.tail_logs("ours")
        assert await anext(stream) == "first chunk"
        assert not native_log_stream.closed
        await stream.aclose()
        assert native_log_stream.closed

    try:
        asyncio.run(consume_one())
    finally:
        result.close()
