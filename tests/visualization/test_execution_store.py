"""Persistence transactions must survive independent Studio store instances."""

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor

from visualization.execution import ExecutionConfiguration, ExecutionStore


class SlowExecutionStore(ExecutionStore):
    """Expose stale read/modify/write races without coordinating inside a lock."""

    def _read(self):
        rows = super()._read()
        time.sleep(0.02)
        return rows


def _save_configuration(path, name, start):
    start.wait(timeout=10)
    SlowExecutionStore(path).save(ExecutionConfiguration(name=name, settings={"ssh_host": name}))


def test_independent_instances_do_not_lose_concurrent_saves(tmp_path):
    path = tmp_path / "settings.json"
    names = [f"Machine {index}" for index in range(8)]
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = [
            pool.submit(SlowExecutionStore(path).save, ExecutionConfiguration(name=name, settings={})) for name in names
        ]
        for future in futures:
            future.result(timeout=10)
    assert {row["name"] for row in ExecutionStore(path).list()} == set(names)


def test_independent_processes_do_not_lose_concurrent_saves(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "settings.json"
    start = context.Event()
    names = [f"Process {index}" for index in range(4)]
    processes = [context.Process(target=_save_configuration, args=(path, name, start)) for name in names]
    try:
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        assert {row["name"] for row in ExecutionStore(path).list()} == set(names)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_concurrent_delete_does_not_erase_another_instances_save(tmp_path):
    path = tmp_path / "settings.json"
    ExecutionStore(path).save(ExecutionConfiguration(name="Remove me", settings={}))
    with ThreadPoolExecutor(max_workers=2) as pool:
        remove = pool.submit(SlowExecutionStore(path).delete, "Remove me")
        save = pool.submit(SlowExecutionStore(path).save, ExecutionConfiguration(name="Keep me", settings={}))
        remove.result(timeout=10)
        save.result(timeout=10)
    assert [row["name"] for row in ExecutionStore(path).list()] == ["Keep me"]
