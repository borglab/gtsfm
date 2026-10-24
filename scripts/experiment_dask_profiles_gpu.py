"""Bounded, sequential manual/profile trials on an isolated Linux source snapshot.

Run this actual file, never Python stdin, beneath a systemd CPU/RAM/wall limit.
This harness does not change production code or synthesize reconstruction outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


INSTRUMENTATION = r'''
import os, sys, json, time, threading, random, atexit
from pathlib import Path
root = Path(os.environ["GPU_EXPERIMENT_EVIDENCE"])
root.mkdir(parents=True, exist_ok=True)
pid = os.getpid()
def emit(value):
    with (root / ("python-%s.jsonl" % pid)).open("a") as f:
        f.write(json.dumps(dict(time=time.time(), pid=pid, **value), default=str) + "\n")
try:
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    import hydra
    from omegaconf import OmegaConf
    original_compose = hydra.compose
    def compose(*args, **kwargs):
        cfg = original_compose(*args, **kwargs)
        name = "hydra-%s-%s.yaml" % (pid, time.time_ns())
        (root / name).write_text(OmegaConf.to_yaml(cfg, resolve=True))
        emit(dict(kind="hydra", file=name, args=args, kwargs=kwargs))
        return cfg
    hydra.compose = compose
    emit(dict(kind="startup", argv=sys.argv, orig_argv=sys.orig_argv,
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
              cuda_visible_devices_is_set="CUDA_VISIBLE_DEVICES" in os.environ,
              selected_device=os.environ.get("GTSFM_SELECTED_DEVICE"), seed=42))
    stop = threading.Event()
    def sample():
        while not stop.wait(1):
            row = dict(kind="torch", cuda_initialized=torch.cuda.is_initialized())
            if torch.cuda.is_initialized():
                row.update(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                           peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            row["models"] = {k: getattr(v, "__file__", None) for k,v in list(sys.modules.items())
                             if k in ("vggt.models.vggt", "thirdparty.megaloc.megaloc", "__main__", "gtsfm.scene_optimizer", "gtsfm.utils.dask_stats")}
            emit(row)
    threading.Thread(target=sample, daemon=True).start()
    atexit.register(lambda: (stop.set(), emit(dict(kind="exit"))))
except Exception as exc:
    emit(dict(kind="instrumentation_error", error=repr(exc)))
'''


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8,
    )
    values = [int(x.strip()) for x in result.stdout.strip().split(",")]
    return dict(memory_used_mib=values[0], temperature_c=values[1], utilization_percent=values[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--arm", choices=["manual", "profile", "cancel"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=900)
    args = parser.parse_args()
    base = args.base.resolve()
    source = base / "source"
    evidence = base / "evidence" / args.trial
    evidence.mkdir(parents=True, exist_ok=False)
    os.chdir(source)
    sys.path.insert(0, str(source))
    # All production result caches live below this root; never reuse them across trials.
    cache = source / "cache"
    if cache.exists():
        cache.rename(evidence / "cache-before-trial")
    cache.mkdir()
    instrument = base / "instrumentation"
    instrument.mkdir(exist_ok=True)
    (instrument / "sitecustomize.py").write_text(INSTRUMENTATION)
    os.environ.update({
        "GPU_EXPERIMENT_EVIDENCE": str(evidence),
        "PYTHONPATH": str(instrument) + os.pathsep + str(source),
        "PYTHONHASHSEED": "42", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
        "DASK_DISTRIBUTED__WORKER__DAEMON": "False", "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
    from visualization.execution_profiles import ExecutionProfileStore, apply_profile
    from visualization.runtime import JobManager, build_runner_args

    scratch = base / "scratch"
    scratch.mkdir(exist_ok=True)
    profile = dict(name="rtx3080-fixed", execution_mode="local", workers=1, threads_per_worker=2,
                   host_memory_limit="18GiB", selected_gpus=["cuda:0"], scratch_directory=str(scratch))
    store_path = base / "profiles.json"
    store = ExecutionProfileStore(store_path)
    if not store_path.exists():
        store.save(profile)
    # New store instance reads persisted bytes, including in B1/B2 and cancellation.
    store = ExecutionProfileStore(store_path)
    common = dict(name=args.trial, config_name="vggt_sift_frontend_megaloc", loader="olsson",
                  dataset_dir=str(args.dataset.resolve()), max_resolution=518,
                  dashboard_port="127.0.0.1:0", splat_implementation="none", execution_target="local")
    manual = dict(common, num_workers=1, threads_per_worker=2, worker_memory_limit="18GiB",
                  dask_tmpdir=str(scratch), hardware="cuda:0")
    saved = dict(common, execution_profile=profile["name"])
    output = base / "results" / "runs" / (args.trial + "-" + args.trial)
    expanded, resolved = apply_profile(saved, store)
    manual_args, manual_env = build_runner_args(manual, output)
    saved_args, saved_env = build_runner_args(expanded, output)
    assert (manual_args, manual_env) == (saved_args, saved_env)
    dump(evidence / "contract.json", dict(manual=manual, profile_request=saved, profile=resolved,
         persisted_bytes=store_path.read_text(), argv=manual_args, environment=manual_env,
         cache_policy="Empty source/cache for every trial; warm read-only model weights; no prior pipeline results",
         dataset=[dict(name=p.name, size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in sorted((args.dataset / "images").glob("*")) if p.is_file()]))
    manager = None
    process = None
    log = None
    start = time.time()
    if args.arm == "manual":
        output.mkdir(parents=True)
        live = evidence / "live"
        live.mkdir()
        env = dict(os.environ, **manual_env, GTSFM_LIVE_DIR=str(live), GTSFM_LIVE_PREVIEW_INTERVAL="250")
        command = [sys.executable, "-m", "gtsfm.runner", *manual_args]
        log = (evidence / "runner.log").open("w")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        root_pid = process.pid
    else:
        manager = JobManager(base / "results", profile_store=store)
        job_record = manager.start(saved, job_id=args.trial)
        dump(evidence / "start.json", job_record)
        job = manager.get(args.trial)
        deadline = time.monotonic() + 30
        while job.process is None and job.status not in ("failed", "completed") and time.monotonic() < deadline:
            time.sleep(.1)
        if job.process is None:
            dump(evidence / "launch-failure.json", job.public())
            raise RuntimeError("JobManager did not launch runner")
        process = job.process
        root_pid = process.pid
        live = Path(job.live_root)
        command = job.command
    seen = {}
    termination_reason = None
    cancelled = False
    with (evidence / "telemetry.jsonl").open("w") as stream:
        while True:
            row = dict(time=time.time(), elapsed_seconds=time.time()-start, gpu=gpu_snapshot(), processes=[])
            try:
                processes = [psutil.Process(root_pid)] + psutil.Process(root_pid).children(recursive=True)
                for proc in processes:
                    try:
                        data = dict(pid=proc.pid, ppid=proc.ppid(), created=proc.create_time(), rss=proc.memory_info().rss,
                                    cpu_percent=proc.cpu_percent(), cpu_times=proc.cpu_times()._asdict(), argv=proc.cmdline(), status=proc.status())
                        row["processes"].append(data)
                        seen[proc.pid] = data
                        if not (evidence / ("process-%s.json" % proc.pid)).exists():
                            selected = {k:v for k,v in proc.environ().items() if k.startswith(("CUDA", "GTSFM", "DASK", "OMP", "MKL", "OPENBLAS", "PYTHON", "GPU_EXPERIMENT", "TORCH"))}
                            dump(evidence / ("process-%s.json" % proc.pid), dict(**data, environment=selected))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.NoSuchProcess:
                pass
            row["rss_sum"] = sum(p["rss"] for p in row["processes"])
            row["system_available_bytes"] = psutil.virtual_memory().available
            cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
            row["cgroup"] = {name: (cgroup / name).read_text().strip() for name in ("memory.current", "memory.peak", "memory.max", "cpu.max") if (cgroup / name).exists()}
            dask_path = live / "dask.json"
            if dask_path.exists():
                try: row["dask"] = json.loads(dask_path.read_text())
                except (ValueError, OSError): pass
            stream.write(json.dumps(row)+"\n"); stream.flush()
            if args.arm == "cancel" and not cancelled and row.get("dask", {}).get("running_tasks", 0) > 0 and row["elapsed_seconds"] >= 15:
                dump(evidence / "cancel-response.json", manager.cancel(args.trial))
                cancelled = True
                termination_reason = "requested_cancellation_during_active_reconstruction"
            if row["gpu"]["temperature_c"] >= 80 or row["gpu"]["memory_used_mib"] >= 9700 or row["system_available_bytes"] < 4*1024**3 or time.time()-start > args.wall_seconds:
                termination_reason = "thermal_memory_or_wall_cutoff"
                if manager: manager.cancel(args.trial)
                else: os.killpg(root_pid, signal.SIGTERM)
                try: process.wait(timeout=8)
                except subprocess.TimeoutExpired: os.killpg(root_pid, signal.SIGKILL)
            if process.poll() is not None:
                break
            time.sleep(1)
    process.wait(timeout=10)
    if log: log.close()
    time.sleep(7)
    remaining = []
    for pid, identity in seen.items():
        try:
            proc = psutil.Process(pid)
            if proc.create_time() == identity["created"]:
                remaining.append(dict(pid=pid, status=proc.status(), argv=proc.cmdline()))
        except psutil.NoSuchProcess: pass
    if manager:
        dump(evidence / "final-job.json", manager.get(args.trial).public())
        shutil.copy2(manager.runtime_root / (args.trial + ".log"), evidence / "runner.log")
        shutil.copytree(output, evidence / "output", dirs_exist_ok=True)
        shutil.copytree(live, evidence / "live", dirs_exist_ok=True)
    else:
        shutil.copytree(output, evidence / "output", dirs_exist_ok=True)
    dump(evidence / "summary.json", dict(trial=args.trial, arm=args.arm, command=command, root_pid=root_pid,
         started=start, ended=time.time(), wall_seconds=time.time()-start-7, return_code=process.returncode,
         termination_reason=termination_reason, observed_pids=seen, remaining_processes=remaining,
         post_pid_checks={str(pid): {"exists": psutil.pid_exists(pid)} for pid in seen},
         final_process_snapshot=subprocess.run(["ps", "-eo", "pid,ppid,pgid,sid,args"], capture_output=True, text=True).stdout,
         final_gpu=gpu_snapshot(), compute_processes=subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], capture_output=True,text=True).stdout))
    if cache.exists(): cache.rename(evidence / "pipeline-cache")
    print((evidence / "summary.json").read_text(), flush=True)
    return 0 if process.returncode == 0 or (args.arm == "cancel" and cancelled and not remaining) else 1


if __name__ == "__main__":
    raise SystemExit(main())
