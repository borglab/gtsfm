#!/usr/bin/env python3
"""Independent bounded native-Dask probes; no reconstruction or GPU launches.

Each case starts a fresh subprocess with isolated DASK_CONFIG and records real worker
state. Acceptance is declared in docs/benchmarks/native_dask_verification.md.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest():
    files = [
        "gtsfm/runner.py",
        "gtsfm/configs/cluster_local.yaml",
        "gtsfm/utils/dask_stats.py",
        "visualization/runtime.py",
        "visualization/app.py",
        "visualization/execution_profiles.py",
        "visualization/frontend/src/main.tsx",
        "visualization/static/studio.js",
        "scripts/verify_native_dask.py",
    ]
    return {name: digest(ROOT / name) if (ROOT / name).exists() else None for name in files}


def worker_state(dask_worker=None):
    import dask
    import os

    return {
        "pid": os.getpid(),
        "nthreads": dask_worker.state.nthreads,
        "memory_limit": dask_worker.memory_manager.memory_limit,
        "memory_target": dask_worker.memory_manager.memory_target_fraction,
        "local_directory": dask_worker.local_directory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config_target": dask.config.get("distributed.worker.memory.target"),
        "connect_timeout": dask.config.get("distributed.comm.timeouts.connect"),
        "tcp_timeout": dask.config.get("distributed.comm.timeouts.tcp"),
        "worker_ttl": dask.config.get("distributed.scheduler.worker-ttl"),
    }


def failure_child(case, directory):
    """Inject an exception around real workers; observe cleanup before safety teardown."""
    import psutil
    import gtsfm.runner as module

    report = {"case": case, "source_before": source_manifest(), "checks": {}}
    clusters = []
    identities = []
    runner = module.GtsfmRunner(["--cluster_config", str(directory / "local-cluster.yaml")])
    instantiate = module.instantiate

    def observe_instantiate(*args, **kwargs):
        cluster = instantiate(*args, **kwargs)
        clusters.append(cluster)
        for worker in cluster.workers.values():
            pid = worker.process.pid
            identities.append((pid, psutil.Process(pid).create_time()))
        report["descendant_identities"] = [(p.pid, p.create_time()) for p in psutil.Process().children(recursive=True)]
        return cluster

    module.instantiate = observe_instantiate
    sentinel = "native-independent-injected-failure"

    def fail(*args, **kwargs):
        raise RuntimeError(sentinel)

    if case == "hydra-client-failure":
        module.Client = fail
    else:
        runner._construct_scene_optimizer = fail
    try:
        runner.run()
        report["checks"]["original_failure_preserved"] = False
    except Exception as exc:
        report["error"] = repr(exc)
        report["checks"]["original_failure_preserved"] = str(exc) == sentinel
    finally:
        deadline = time.monotonic() + 5
        while True:
            remaining = [
                pid for pid, born in identities if psutil.pid_exists(pid) and psutil.Process(pid).create_time() == born
            ]
            if not remaining or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        report["worker_identities"] = identities
        report["remaining_before_verifier_cleanup"] = remaining
        report["checks"]["production_failure_cleanup"] = len(identities) == 2 and not remaining
        for cluster in clusters:
            cluster.close(timeout=10)
        report["source_after"] = source_manifest()
        report["checks"]["source_unchanged_during_probe"] = report["source_before"] == report["source_after"]
        report["verdict"] = "VERIFIED" if all(report["checks"].values()) else "NOT VERIFIED"
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(report["verdict"] != "VERIFIED")


def studio_child(directory):
    """Exercise the public ASGI request contract; intercept reconstruction dispatch only."""
    from fastapi.testclient import TestClient
    from visualization.app import create_app
    from visualization.runtime import JobManager, build_runner_args

    report = {
        "source_before": source_manifest(),
        "checks": {},
        "scope": "Actual FastAPI/JobManager request routing with dispatch intercepted; no reconstruction proof.",
    }
    manager = JobManager(directory / "results")
    captured = {}
    event = threading.Event()

    def capture(job, env):
        captured["command"] = job.command
        captured["env"] = {k: v for k, v in env.items() if k.startswith(("DASK_", "CUDA_", "GTSFM_SELECTED"))}
        event.set()

    manager._run = capture
    app = create_app(directory / "results", job_manager=manager)
    dataset = directory / "dataset"
    dataset.mkdir()
    spec = {
        "name": "independent",
        "config_name": "sift_front_end",
        "loader": "colmap",
        "dataset_dir": str(dataset),
        "splat_implementation": "none",
        "hardware": "cpu",
        "cluster_config": str(directory / "local-cluster.yaml"),
        "num_workers": 1,
        "threads_per_worker": 1,
        "worker_memory_limit": "1GiB",
        "advanced_overrides": ["image_pairs_generator.retriever.min_score=0.4"],
    }
    with TestClient(app) as client:
        response = client.post("/api/jobs", json=spec)
        report["status"] = response.status_code
        report["response"] = response.json()
        report["checks"]["job_accepted"] = response.status_code == 202 and event.wait(10)
        report["checks"]["no_profile_routes"] = not any(
            "execution-profiles" in getattr(r, "path", "") for r in app.routes
        )
        schema = client.get("/openapi.json").json()
        bundle = client.get("/static/studio.js")
        report["served_bundle_sha256"] = hashlib.sha256(bundle.content).hexdigest()
        report["checks"]["served_bundle_has_no_profile_contract"] = bundle.status_code == 200 and not any(
            word in bundle.text
            for word in ("/api/execution-profiles", "resolved_execution_profile", "execution_profile_name")
        )
        report["checks"]["no_profile_request_schema"] = not any(
            "execution_profile" in key for key in schema["components"]["schemas"]["RunRequest"]["properties"]
        )
    report["captured"] = captured
    if "command" in captured:
        command = captured["command"]
        report["checks"]["existing_cluster_config_reaches_runner"] = (
            command[command.index("--cluster_config") + 1] == spec["cluster_config"]
        )
        report["checks"]["hydra_override_reaches_runner"] = spec["advanced_overrides"][0] in command
        report["checks"]["native_env_inherited"] = captured["env"].get("DASK_CONFIG") == os.environ["DASK_CONFIG"]
        report["checks"]["cpu_visibility_preserved"] = captured["env"].get("CUDA_VISIBLE_DEVICES") == ""
    manual = {**spec, "cluster_config": "", "worker_memory_limit": "512 MB"}
    argv, env = build_runner_args(manual, directory / "manual")
    report["manual"] = {"argv": argv, "env": env}
    report["checks"]["manual_controls_unchanged"] = (
        "--cluster_config" not in argv and argv[argv.index("--worker_memory_limit") + 1] == "512 MB"
    )
    report["checks"]["no_profile_persistence"] = not any((directory / "results").rglob("*profile*.json"))
    report["source_after"] = source_manifest()
    report["checks"]["source_unchanged_during_probe"] = report["source_before"] == report["source_after"]
    report["verdict"] = "VERIFIED" if all(report["checks"].values()) else "NOT VERIFIED"
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(report["verdict"] != "VERIFIED")


def legacy_child(directory):
    """Inspect actual shipped legacy YAML routing without making SSH/GPU connections."""
    import gtsfm.runner as module

    report = {
        "source_before": source_manifest(),
        "checks": {},
        "scope": "Real legacy YAML and runner routing; cluster/client constructors stubbed, no SSH or CUDA proof.",
    }
    calls = []

    class Client:
        dashboard_link = "stub://routing-only"

        def __init__(self, cluster):
            self.cluster = cluster

        def scheduler_info(self):
            return {"workers": {"tcp://worker-one": {}, "tcp://worker-two": {}}}

        def forward_logging(self):
            pass

    def ssh(hosts, **kwargs):
        calls.append({"hosts": hosts, "kwargs": kwargs})
        return object()

    module.Client = Client
    module.SSHCluster = ssh
    runner = module.GtsfmRunner(
        [
            "--cluster_config",
            "cluster.yaml",
            "--num_workers",
            "2",
            "--threads_per_worker",
            "3",
            "--worker_memory_limit",
            "512MiB",
        ]
    )
    try:
        runner._create_dask_client()
        report["checks"]["legacy_ssh_strings"] = calls[0]["hosts"] == [
            "host1.cc.gatech.edu",
            "host1.cc.gatech.edu",
            "host2.cc.gatech.edu",
            "host3.cc.gatech.edu",
        ]
        report["checks"]["legacy_manual_controls"] = calls[0]["kwargs"]["worker_options"] == {
            "n_workers": 2,
            "nthreads": 3,
            "memory_limit": "512MiB",
        }
        report["checks"]["legacy_io_worker_selection"] = runner._io_worker == "tcp://worker-one"
        cuda = module.GtsfmRunner(["--cluster_config", "cluster_gpu.yaml"])
        workers = []
        cuda._create_local_cuda_cluster = lambda configs: workers.extend(configs) or object()
        cuda._create_dask_client()
        report["checks"]["legacy_gpu_routing"] = len(workers) == 2 and all(
            w["host"] == "localhost" and w["gpus"] == 1 for w in workers
        )
    except Exception:
        report["exception"] = traceback.format_exc()
    report["ssh_calls"] = calls
    report["source_after"] = source_manifest()
    report["checks"]["source_unchanged_during_probe"] = report["source_before"] == report["source_after"]
    report["verdict"] = "VERIFIED" if all(report["checks"].values()) and "exception" not in report else "NOT VERIFIED"
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(report["verdict"] != "VERIFIED")


def child(case, directory):
    if case == "legacy":
        return legacy_child(directory)
    if case == "studio":
        return studio_child(directory)
    if case.endswith("failure"):
        return failure_child(case, directory)
    import dask
    import distributed
    import psutil
    from distributed import Client, LocalCluster
    from distributed.worker_memory import parse_memory_limit

    report = {
        "case": case,
        "pid": os.getpid(),
        "dask": dask.__version__,
        "distributed": distributed.__version__,
        "source_before": source_manifest(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("DASK_")},
        "checks": {},
        "workers": {},
        "closed": False,
    }
    checks = report["checks"]
    checks["pinned_version"] = dask.__version__ == distributed.__version__ == "2025.9.1"
    report["upstream_source"] = {
        "LocalCluster": digest(Path(inspect.getfile(LocalCluster))),
        "parse_memory_limit": digest(Path(inspect.getfile(parse_memory_limit))),
        "dask_config": digest(Path(dask.config.__file__)),
    }
    parsed = {
        str(x): parse_memory_limit(x, 2, logger=logging.getLogger(__name__))
        for x in (None, "0", "512MiB", "512 MB", "auto", "0.25")
    }
    report["standard_memory_parser"] = parsed
    checks["standard_parser_units"] = parsed["512MiB"] == 536870912 and parsed["512 MB"] == 512000000
    checks["standard_parser_unlimited"] = parsed["None"] is None and parsed["0"] is None
    cluster = client = None
    identities = []
    try:
        if case.startswith(("runner", "hydra")):
            from gtsfm.runner import GtsfmRunner

            argv = [
                "--num_workers",
                "2",
                "--threads_per_worker",
                "2",
                "--worker_memory_limit",
                "512MiB",
                "--dashboard_port",
                ":0",
            ]
            if case.startswith("hydra"):
                # Deliberately conflicting manual arguments must not override the file.
                argv = [
                    "--num_workers",
                    "1",
                    "--threads_per_worker",
                    "1",
                    "--worker_memory_limit",
                    "1GiB",
                    "--dashboard_port",
                    ":0",
                    "--cluster_config",
                    str(directory / "local-cluster.yaml"),
                ]
            runner = GtsfmRunner(argv)
            report["argv"] = argv
            client = runner._create_dask_client()
            cluster = client.cluster
        else:
            cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit="512MiB", dashboard_address=None)
            client = Client(cluster)
        client.wait_for_workers(2, timeout=25)
        report["workers"] = client.run(worker_state)
        identities = [(p["pid"], psutil.Process(p["pid"]).create_time()) for p in report["workers"].values()]
        report["descendant_identities"] = [(p.pid, p.create_time()) for p in psutil.Process().children(recursive=True)]
        report["scheduler"] = {
            a: {"nthreads": w["nthreads"], "memory_limit": w["memory_limit"]}
            for a, w in client.scheduler_info()["workers"].items()
        }
        env_case = "env" in case
        expected_target = 0.67 if env_case else 0.61
        expected_scratch = directory / ("env-scratch" if env_case else "yaml-scratch")
        if case == "hydra-scratch":
            expected_scratch = directory / "constructor-scratch"
        checks["two_distinct_worker_processes"] = (
            len(identities) == 2
            and len({p for p, _ in identities}) == 2
            and all(p != os.getpid() for p, _ in identities)
        )
        checks["effective_threads_memory"] = all(
            w["nthreads"] == 2 and w["memory_limit"] == (None if case == "hydra-zero" else 536870912)
            for w in report["workers"].values()
        )
        checks["native_target_precedence"] = all(
            w["memory_target"] == expected_target and w["config_target"] == expected_target
            for w in report["workers"].values()
        )
        expected_timeout = "30s" if case.endswith("old-default") else ("127s" if env_case else "123s")
        if case == "runner-defaults":
            expected_timeout = "300s"
        checks["native_timeout_precedence"] = all(
            w["connect_timeout"] == expected_timeout for w in report["workers"].values()
        )
        if case.startswith(("runner", "hydra")):
            checks["project_defaults_preserved"] = all(
                w["tcp_timeout"] == "300s" and w["worker_ttl"] is None for w in report["workers"].values()
            )
        checks["native_scratch_precedence"] = all(
            Path(w["local_directory"]).is_relative_to(expected_scratch) for w in report["workers"].values()
        )
        checks["explicit_cpu_visibility"] = all(w["cuda_visible_devices"] == "" for w in report["workers"].values())
        report["task_result"] = client.submit(sum, [1, 2, 3, 4]).result(timeout=10)
        checks["actual_task_execution"] = report["task_result"] == 10
        if case.startswith(("runner", "hydra")):
            from gtsfm.runner import _shutdown_dask_client

            _shutdown_dask_client(client)
        else:
            client.close(timeout=10)
            cluster.close(timeout=10)
        report["closed"] = True
    except Exception:
        report["exception"] = traceback.format_exc()
    finally:
        for obj in (client, cluster):
            if obj is not None:
                try:
                    obj.close(timeout=10)
                except Exception:
                    report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        deadline = time.monotonic() + 5
        remaining = []
        while time.monotonic() < deadline:
            remaining = [
                pid for pid, born in identities if psutil.pid_exists(pid) and psutil.Process(pid).create_time() == born
            ]
            if not remaining:
                break
            time.sleep(0.05)
        report["remaining_worker_pids"] = remaining
        checks["worker_process_cleanup"] = bool(identities) and not remaining
        report["source_after"] = source_manifest()
        checks["source_unchanged_during_probe"] = report["source_before"] == report["source_after"]
        report["verdict"] = "VERIFIED" if all(checks.values()) and "exception" not in report else "NOT VERIFIED"
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["verdict"] == "VERIFIED" else 1


def run(directory, cases):
    import psutil

    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for case in cases:
        folder = (directory / case).resolve()
        folder.mkdir()
        cfg = folder / "config"
        cfg.mkdir()
        (cfg / "native.yaml").write_text(
            f'temporary-directory: {json.dumps(str(folder / "yaml-scratch"))}\n'
            "distributed:\n  comm:\n    timeouts:\n      connect: 123s\n"
            "  worker:\n    memory:\n      target: 0.61\n"
            "      spill: 0.75\n      pause: 0.90\n      terminate: 0.95\n"
            # Deliberate inert keys; native config is not a LocalCluster kwargs schema.
            "  local-cluster:\n    n-workers: 9\n    threads-per-worker: 9\n"
        )
        if case == "runner-defaults":
            policy = cfg / "native.yaml"
            policy.write_text(policy.read_text().replace("  comm:\n    timeouts:\n      connect: 123s\n", ""))
        if case.startswith("hydra") or case == "studio":
            config_text = (
                "_target_: distributed.LocalCluster\nn_workers: 2\nthreads_per_worker: 2\n"
                "memory_limit: ${oc.env:NATIVE_TEST_MEMORY}\ndashboard_address: null\n"
            )
            if case == "hydra-scratch":
                config_text += f'local_directory: {json.dumps(str(folder / "constructor-scratch"))}\n'
            (folder / "local-cluster.yaml").write_text(config_text)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DASK_")}
        env.pop("GTSFM_API_KEY", None)
        env.pop("GTSFM_LIVE_DIR", None)
        env.update(
            DASK_CONFIG=str(cfg),
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NATIVE_TEST_MEMORY="512MiB",
        )
        if case == "hydra-zero":
            env["NATIVE_TEST_MEMORY"] = "0"
        if "env" in case:
            env["DASK_DISTRIBUTED__WORKER__MEMORY__TARGET"] = "0.67"
            env["DASK_TEMPORARY_DIRECTORY"] = str(folder / "env-scratch")
            env["DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "30s" if case.endswith("old-default") else "127s"
        command = [sys.executable, str(Path(__file__).resolve()), "--child", case, "--artifacts", str(folder)]
        with (folder / "process.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            timed_out = False
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        receipt = {
            "case": case,
            "command": command,
            "pid": process.pid,
            "returncode": process.returncode,
            "timed_out": timed_out,
            "result": str(folder / "result.json"),
        }
        if (folder / "result.json").exists():
            identities = json.loads((folder / "result.json").read_text()).get("descendant_identities", [])
            deadline = time.monotonic() + 5
            while True:
                remaining = [
                    pid
                    for pid, born in identities
                    if psutil.pid_exists(pid) and psutil.Process(pid).create_time() == born
                ]
                if not remaining or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            receipt["remaining_descendants_after_case_exit"] = remaining
        results.append(receipt)
        (folder / "process.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    (directory / "processes.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(r["returncode"] != 0 or r.get("remaining_descendants_after_case_exit") for r in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "native-yaml",
            "native-env",
            "runner-yaml",
            "runner-defaults",
            "runner-env",
            "runner-env-old-default",
            "hydra-yaml",
            "hydra-env",
            "hydra-scratch",
            "hydra-zero",
            "hydra-client-failure",
            "hydra-pipeline-failure",
            "studio",
            "legacy",
        ],
    )
    parser.add_argument("--child")
    args = parser.parse_args()
    raise SystemExit(child(args.child, args.artifacts) if args.child else run(args.artifacts, args.cases))
