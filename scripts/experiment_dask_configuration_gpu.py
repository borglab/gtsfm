"""Bounded, sequential manual/native-configuration trials on an isolated Linux source snapshot.

Run this actual file, never Python stdin, beneath a systemd CPU/RAM/wall limit.
This harness does not change production code or synthesize reconstruction outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import secrets
import socket
import threading
import traceback
import urllib.error
import urllib.request
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
import os, sys, json, time, threading, random, atexit, contextlib
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
    # Observe only the top-level runner Client; worker/nested clients remain untouched.
    if "gtsfm.runner" in sys.orig_argv and "-m" in sys.orig_argv:
        from distributed import Client
        original_client_init = Client.__init__
        observing_client = threading.local()
        def build_observer():
            def observe_worker(dask_worker=None):
                import dask, os
                return dict(pid=os.getpid(), address=dask_worker.address, nthreads=dask_worker.nthreads,
                            memory_limit_bytes=dask_worker.memory_manager.memory_limit,
                            local_directory=dask_worker.local_directory,
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            selected_device=os.environ.get("GTSFM_SELECTED_DEVICE"),
                            config={key:dask.config.get(key, default=None) for key in (
                                "distributed.worker.daemon", "distributed.worker.memory.target",
                                "distributed.worker.memory.spill", "distributed.worker.memory.pause",
                                "distributed.worker.memory.terminate", "temporary-directory",
                                "distributed.comm.timeouts.connect", "distributed.comm.timeouts.tcp")})
            return observe_worker
        def client_init(self, *args, **kwargs):
            original_client_init(self, *args, **kwargs)
            if (threading.current_thread() is not threading.main_thread()
                    or getattr(observing_client, "active", False) or self.cluster is None):
                return
            observing_client.active = True
            try:
                self.wait_for_workers(1, timeout=30)
                workers = self.run(build_observer())
                emit(dict(kind="dask_workers", workers=workers,
                          scheduler=self.scheduler_info(), dashboard=self.dashboard_link))
                expected_scratch = Path(os.environ["GPU_EXPERIMENT_EXPECTED_SCRATCH"]).resolve()
                controls_match = len(workers) == 1 and len(self.cluster.workers) == 1 and all(
                    worker["nthreads"] == 2 and worker["memory_limit_bytes"] == 18 * 1024**3
                    and Path(worker["local_directory"]).resolve().is_relative_to(expected_scratch)
                    and worker["cuda_visible_devices"] == "0"
                    and worker["config"]["distributed.worker.daemon"] is False
                    for worker in workers.values())
                emit(dict(kind="worker_control_guard", passed=controls_match))
                if not controls_match:
                    raise RuntimeError("Observed worker controls differ from frozen 1 worker / 2 threads / 18GiB / CUDA0 / scratch contract")
            except Exception as exc:
                emit(dict(kind="dask_observation_error", error=repr(exc)))
                cluster = self.cluster
                with contextlib.suppress(Exception):
                    self.close()
                with contextlib.suppress(Exception):
                    cluster.close()
                raise
            finally:
                observing_client.active = False
        Client.__init__ = client_init
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(base, evidence, phase):
    manifests = {}
    for name, prefix in [('source-file-manifest.json', base / 'source'), ('asset-manifest.json', Path('/'))]:
        expected = json.loads((base / name).read_text())
        observed = {name: sha256(prefix / name) if (prefix / name).is_file() else None for name in expected}
        manifests[name] = dict(expected=expected, observed=observed, unchanged=observed == expected)
    manifests['dataset'] = {str(p.relative_to(base / 'dataset-first4')): sha256(p)
                            for p in sorted((base / 'dataset-first4').rglob('*')) if p.is_file()}
    dump(evidence / f'manifest-{phase}.json', manifests)
    if not all(v['unchanged'] for k, v in manifests.items() if k != 'dataset'):
        raise RuntimeError(f'Pinned source or assets drifted at {phase}')
    return manifests


def request(url, token, method='GET', payload=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


@contextlib.contextmanager
def api_server(manager, evidence):
    import uvicorn
    from visualization.app import create_app
    token = secrets.token_urlsafe(32)
    previous = os.environ.get('GTSFM_API_KEY')
    os.environ['GTSFM_API_KEY'] = token
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(manager.results_root, job_manager=manager),
                                         host='127.0.0.1', port=port, access_log=False, log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{port}'
    receipt = dict(url=url, pid=os.getpid(), thread_name=thread.name)
    try:
        deadline = time.monotonic() + 30
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        if not server.started:
            raise TimeoutError('Loopback API did not start')
        unauthorized, _ = request(url + '/api/jobs', '')
        assert unauthorized == 401
        receipt['unauthenticated_status'] = unauthorized
        dump(evidence / 'api-server-started.json', receipt)
        yield url, token, receipt
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
        receipt['thread_stopped'] = not thread.is_alive()
        with socket.socket() as check:
            receipt['port_closed'] = check.connect_ex(('127.0.0.1', port)) != 0
        dump(evidence / 'api-server.json', receipt)
        if previous is None:
            os.environ.pop('GTSFM_API_KEY', None)
        else:
            os.environ['GTSFM_API_KEY'] = previous
        if thread.is_alive():
            raise RuntimeError('Owned API server thread did not stop')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--arm', choices=['manual', 'native'], required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, default=900)
    args = parser.parse_args()
    base = args.base.resolve()
    if not (base / 'SOURCE_FREEZE_AND_GPU_GRANTED.json').is_file():
        raise RuntimeError('Explicit source freeze and GPU grant receipt required')
    if args.dataset.resolve() != (base / 'dataset-first4').resolve():
        raise ValueError('--dataset must resolve to the pinned dataset-first4 directory')
    source = base / 'source'
    evidence = base / 'evidence' / args.trial
    evidence.mkdir(parents=True, exist_ok=False)
    started_wall, started = time.time(), time.monotonic()
    before = verify_files(base, evidence, 'before')
    os.chdir(source)
    sys.path.insert(0, str(source))
    cache = source / 'cache'
    if cache.exists():
        cache.rename(evidence / 'cache-before-trial')
    cache.mkdir()
    instrument = base / 'instrumentation'
    instrument.mkdir(exist_ok=True)
    instrumentation_path = instrument / 'sitecustomize.py'
    if instrumentation_path.exists() and instrumentation_path.read_text() != INSTRUMENTATION:
        raise RuntimeError('Instrumentation changed within the series')
    instrumentation_path.write_text(INSTRUMENTATION)
    os.environ.update({
        'GPU_EXPERIMENT_EVIDENCE': str(evidence),
        'GPU_EXPERIMENT_EXPECTED_SCRATCH': str(base / 'scratch'),
        'PYTHONPATH': str(instrument) + os.pathsep + str(source),
        'PYTHONHASHSEED': '42', 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
        'DASK_CONFIG': str(base / 'dask-policy'), 'DASK_DISTRIBUTED__WORKER__DAEMON': 'False',
        'TORCHINDUCTOR_COMPILE_THREADS': '1',
    })
    from visualization.runtime import JobManager
    scratch = base / 'scratch'
    scratch.mkdir(exist_ok=True)
    common = dict(name=args.trial, config_name='vggt_sift_frontend_megaloc', loader='olsson',
                  dataset_dir=str(args.dataset.resolve()), max_resolution=518, dashboard_port='127.0.0.1:0',
                  splat_implementation='none', execution_target='local', hardware='cuda:0',
                  dask_tmpdir=str(scratch), live_preview_interval=250)
    manual = dict(common, num_workers=1, threads_per_worker=2, worker_memory_limit='18GiB')
    native = dict(common, num_workers=3, threads_per_worker=1, worker_memory_limit='4GiB',
                  cluster_config=str(base / 'local-cluster.yaml'))
    spec = manual if args.arm == 'manual' else native
    dump(evidence / 'contract.json', dict(request=spec, arm=args.arm,
         native_yaml=(base / 'local-cluster.yaml').read_text(),
         dask_policy=(base / 'dask-policy' / 'policy.yaml').read_text(),
         environment={k:os.environ[k] for k in ('PYTHONPATH','PYTHONHASHSEED','PYTHONDONTWRITEBYTECODE',
             'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','DASK_CONFIG',
             'DASK_DISTRIBUTED__WORKER__DAEMON','TORCHINDUCTOR_COMPILE_THREADS')},
         instrumentation_sha256=sha256(instrumentation_path), executing_harness=str(Path(__file__).resolve()),
         executing_harness_sha256=sha256(__file__),
         cache_policy='Empty source/cache each trial; warm pinned model weights; no prior pipeline results'))
    manager = JobManager(base / 'results')
    process = job = None
    seen = {}
    termination_reason = error = None
    runner_started = runner_ended = None
    try:
        with api_server(manager, evidence) as (url, token, receipt):
            try:
                status, record = request(url + '/api/jobs', token, 'POST', spec)
                dump(evidence / 'start.json', dict(status=status, response=record, request=spec))
                if status != 202:
                    raise RuntimeError(f'POST /api/jobs failed: {status} {record}')
                job = manager.get(record['id'])
                deadline = time.monotonic() + 45
                while job.process is None and job.status not in ('failed','completed') and time.monotonic() < deadline:
                    time.sleep(.1)
                if job.process is None:
                    raise RuntimeError(f'Runner did not launch: {job.public()}')
                process = job.process
                runner_started = time.monotonic()
                live = Path(job.live_root)
                root_pid = process.pid
                with (evidence / 'telemetry.jsonl').open('w') as stream:
                    while True:
                        row = dict(time=time.time(), elapsed_seconds=time.monotonic()-started,
                                   runner_elapsed_seconds=time.monotonic()-runner_started, gpu=gpu_snapshot(), processes=[])
                        try:
                            processes = [psutil.Process(root_pid)] + psutil.Process(root_pid).children(recursive=True)
                            for proc in processes:
                                try:
                                    data = dict(pid=proc.pid, ppid=proc.ppid(), created=proc.create_time(),
                                                pgid=os.getpgid(proc.pid), rss=proc.memory_info().rss,
                                                cpu_times=proc.cpu_times()._asdict(), argv=proc.cmdline(), status=proc.status())
                                    row['processes'].append(data)
                                    seen[proc.pid] = data
                                    if not (evidence / f'process-{proc.pid}.json').exists():
                                        env = {k:v for k,v in proc.environ().items()
                                               if k.startswith(('CUDA','GTSFM','DASK','OMP','MKL','OPENBLAS','PYTHON','GPU_EXPERIMENT','TORCH'))
                                               and not k.endswith(('KEY','TOKEN','SECRET'))}
                                        dump(evidence / f'process-{proc.pid}.json', dict(**data, environment=env))
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except psutil.NoSuchProcess:
                            pass
                        row['rss_sum'] = sum(p['rss'] for p in row['processes'])
                        row['system_available_bytes'] = psutil.virtual_memory().available
                        cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
                        row['cgroup'] = {name:(cgroup/name).read_text().strip() for name in
                                         ('memory.current','memory.peak','memory.max','cpu.max') if (cgroup/name).exists()}
                        dask_path = live / 'dask.json'
                        if dask_path.exists():
                            try:
                                row['dask'] = json.loads(dask_path.read_text())
                            except (ValueError, OSError):
                                pass
                        stream.write(json.dumps(row)+'\n'); stream.flush()
                        if (row['gpu']['temperature_c'] >= 80 or row['gpu']['memory_used_mib'] >= 9700
                                or row['system_available_bytes'] < 4*1024**3 or time.monotonic()-started > args.wall_seconds):
                            termination_reason = 'thermal_memory_or_wall_cutoff'
                            dump(evidence / 'cancel-response.json', request(url + f'/api/jobs/{job.id}/cancel', token, 'POST'))
                            process.wait(timeout=15)
                        if process.poll() is not None:
                            runner_ended = time.monotonic()
                            break
                        time.sleep(1)
            finally:
                if job is not None and job.status in ('queued', 'running'):
                    dump(evidence / 'cleanup-cancel-response.json', request(url + f'/api/jobs/{job.id}/cancel', token, 'POST'))
                    if job.process is not None:
                        job.process.wait(timeout=15)
    except Exception:
        error = traceback.format_exc()
        (evidence / 'error.txt').write_text(error)
    finally:
        settling_start = time.monotonic()
        time.sleep(7)
        remaining = []
        for pid, identity in seen.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == identity['created']:
                    remaining.append(dict(pid=pid, status=proc.status(), argv=proc.cmdline()))
            except psutil.NoSuchProcess:
                pass
        if job is not None:
            dump(evidence / 'final-job.json', job.public())
            log_path = manager.runtime_root / f'{job.id}.log'
            if log_path.exists():
                shutil.copy2(log_path, evidence / 'runner.log')
            shutil.copytree(job.output_root, evidence / 'output', dirs_exist_ok=True)
            shutil.copytree(job.live_root, evidence / 'live', dirs_exist_ok=True)
        if cache.exists():
            cache.rename(evidence / 'pipeline-cache')
        after = verify_files(base, evidence, 'after')
        drift = before != after
        summary = dict(trial=args.trial, arm=args.arm, command=job.command if job else None,
            executing_harness_sha256=sha256(__file__),
            root_pid=process.pid if process else None, started=started_wall, ended=time.time(),
            harness_wall_seconds=time.monotonic()-started,
            runner_wall_seconds=runner_ended-runner_started if runner_ended is not None else None,
            settling_seconds=time.monotonic()-settling_start, return_code=process.returncode if process else None,
            termination_reason=termination_reason, error=error, drift=drift, observed_pids=seen,
            remaining_processes=remaining,
            post_pid_checks={str(pid):dict(exists=psutil.pid_exists(pid)) for pid in seen},
            final_process_snapshot=subprocess.run(['ps','-eo','pid,ppid,pgid,sid,args'],capture_output=True,text=True).stdout,
            final_gpu=gpu_snapshot(), compute_processes=subprocess.run(
                ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],
                capture_output=True,text=True).stdout)
        dump(evidence / 'summary.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('observed_pids','final_process_snapshot')}, default=str), flush=True)
    return 0 if process is not None and process.returncode == 0 and not any((error,termination_reason,remaining,drift)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
