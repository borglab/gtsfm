#!/usr/bin/env python3
"""Bounded independent acceptance probes; see docs/benchmarks/dask_profiles_experiment.md.

config uses actual loopback HTTP servers with injected, real temporary profile storage.
compare checks four explicitly selected COLMAP text exports, never starts reconstruction.
Neither command establishes GPU execution, cancellation, internal bitwise equality, or speedup.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_FILES = [
    "visualization/execution_profiles.py",
    "visualization/app.py",
    "visualization/runtime.py",
    "gtsfm/runner.py",
    "scripts/verify_dask_profiles.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(url, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"non_json_error": raw[:1000]}
        return exc.code, body


@contextlib.contextmanager
def server(directory, number):
    """Real app/store dependency injection; no user's home or saved profiles are changed."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code = """
import sys, uvicorn
from pathlib import Path
from visualization.app import create_app
from visualization.runtime import JobManager
from visualization.execution_profiles import ExecutionProfileStore
base=Path(sys.argv[1]); store=ExecutionProfileStore(base/'profiles.json')
manager=JobManager(base/'results',profile_store=store)
app=create_app(base/'results',job_manager=manager)
uvicorn.run(app,host='127.0.0.1',port=int(sys.argv[2]),access_log=False,log_level='warning')
"""
    env = dict(os.environ)
    env.pop("GTSFM_API_KEY", None)
    url = f"http://127.0.0.1:{port}"
    with (directory / f"server-{number}.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(directory), str(port)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {directory / f'server-{number}.log'}")
                try:
                    status, body = request(url + "/api/execution-profiles")
                    if status == 200:
                        if Path(body["path"]).resolve() != (directory / "profiles.json").resolve():
                            raise RuntimeError("API is not bound to the temporary store; refusing all mutations")
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Temporary server did not become ready in 20 seconds")
            receipt = {"pid": process.pid, "url": url, "returncode": None}
            yield url, receipt
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if "receipt" in locals():
                receipt["returncode"] = process.returncode


def check_configuration(directory):
    source_before = {name: digest(ROOT / name) for name in SOURCE_FILES}
    from visualization.execution_profiles import ExecutionProfileStore, apply_profile
    from visualization.runtime import build_runner_args

    directory.mkdir(parents=True, exist_ok=True)
    # Refuse reuse of a directory containing profile state, including earlier verifier runs.
    if (directory / "profiles.json").exists():
        raise ValueError("Use a fresh artifact directory; profiles.json already exists")
    scratch = directory / "scratch"
    scratch.mkdir()
    scratch_file = directory / "scratch-is-a-file"
    scratch_file.write_text("not a directory\n")
    readonly_scratch = directory / "readonly-scratch"
    readonly_scratch.mkdir()
    readonly_scratch.chmod(0o555)
    profile = {
        "workers": 2,
        "threads_per_worker": 2,
        "host_memory_limit": "512MiB",
        "selected_gpus": [],
        "scratch_directory": str(scratch.resolve()),
        "execution_mode": "local",
    }
    observations = []

    def record(label, url, method="GET", payload=None):
        status, body = request(url, method, payload)
        observations.append(
            {
                "label": label,
                "url": url,
                "method": method,
                "payload": json.loads(json.dumps(payload)),
                "status": status,
                "body": body,
            }
        )
        return status, body

    with server(directory, 1) as (url, server1):
        endpoint = url + "/api/execution-profiles/independent"
        created = record("create", endpoint, "PUT", profile)
        profile["threads_per_worker"] = 3
        updated = record("update", endpoint, "PUT", profile)
    persisted = (directory / "profiles.json").read_text()
    with server(directory, 2) as (url, server2):
        endpoint = url + "/api/execution-profiles/independent"
        reloaded = record("fresh-process-read", endpoint)
        record("fresh-process-list", url + "/api/execution-profiles")
        invalid = {
            "zero-workers": {"workers": 0},
            "negative-workers": {"workers": -1},
            "boolean-workers": {"workers": True},
            "fraction-workers": {"workers": 1.5},
            "fraction-string-workers": {"workers": "1.5"},
            "zero-threads": {"threads_per_worker": 0},
            "boolean-threads": {"threads_per_worker": True},
            "fraction-threads": {"threads_per_worker": 1.5},
            "zero-memory": {"host_memory_limit": "0GB"},
            "negative-memory": {"host_memory_limit": "-1GB"},
            "bad-memory": {"host_memory_limit": "lots"},
            "bad-mode": {"execution_mode": "unknown"},
            "unsupported-remote-mode": {"execution_mode": "remote"},
            "unknown-field": {"typo_workers": 5},
            "future-schema": {"schema_version": 99},
            "relative-scratch": {"scratch_directory": "relative"},
            "file-scratch": {"scratch_directory": str(scratch_file)},
            "unwritable-scratch": {"scratch_directory": str(readonly_scratch)},
            "unwritable-parent-scratch": {"scratch_directory": str(readonly_scratch / "new" / "nested")},
            "malformed-gpu": {"workers": 1, "selected_gpus": ["not-a-gpu"]},
            "negative-gpu": {"workers": 1, "selected_gpus": ["cuda:-1"]},
            "duplicate-gpu": {"workers": 1, "selected_gpus": ["cuda:0", "cuda:0"]},
            "gpu-multiple-workers": {"workers": 2, "selected_gpus": ["cuda:0"]},
        }
        rejected = {}
        try:
            for name, overrides in invalid.items():
                response = record(name, url + "/api/execution-profiles/invalid", "PUT", {**profile, **overrides})
                rejected[name] = 400 <= response[0] < 500
        finally:
            readonly_scratch.chmod(0o755)
        missing = record("missing-profile", url + "/api/execution-profiles/no-such-profile")
        store = ExecutionProfileStore(directory / "profiles.json")
        spec = {
            "config_name": "vggt_sift_frontend_megaloc",
            "loader": "olsson",
            "dataset_dir": str(directory),
            "splat_implementation": "none",
            "hardware": "cpu",
        }
        unavailable_gpu = record(
            "save-unavailable-gpu",
            url + "/api/execution-profiles/unavailable-gpu",
            "PUT",
            {**profile, "workers": 1, "selected_gpus": ["cuda:99999"]},
        )
        if 400 <= unavailable_gpu[0] < 500:
            gpu_rejected = True
        else:
            gpu_launch = record(
                "launch-unavailable-gpu", url + "/api/jobs", "POST", {**spec, "execution_profile": "unavailable-gpu"}
            )
            gpu_rejected = 400 <= gpu_launch[0] < 500 and "available" in str(gpu_launch[1]).lower()
        remote_spec = {**spec, "execution_target": "remote", "remote_endpoint": "not-a-url"}
        remote_profile = record(
            "unsupported-remote-profile-target",
            url + "/api/jobs",
            "POST",
            {**remote_spec, "execution_profile": "independent"},
        )
        remote_manual = record("manual-remote-url-validation", url + "/api/jobs", "POST", remote_spec)
        remote_boundary = (
            400 <= remote_profile[0] < 500
            and "profile" in str(remote_profile[1]).lower()
            and 400 <= remote_manual[0] < 500
            and "url" in str(remote_manual[1]).lower()
        )
        conflicts = {}
        for field, value in {
            "num_workers": 4,
            "threads_per_worker": 7,
            "worker_memory_limit": "8GB",
            "dask_tmpdir": str(directory / "different"),
            "hardware": "cuda:99999",
            "cluster_config": "incompatible.yaml",
            "input_worker": "tcp://127.0.0.1:1",
            "execution_profile_name": "different-profile-alias",
        }.items():
            probe = {
                **spec,
                "dataset_dir": str(directory / "missing-dataset"),
                "execution_profile": "independent",
                field: value,
            }
            response = record("conflict-" + field, url + "/api/jobs", "POST", probe)
            conflicts[field] = 400 <= response[0] < 500 and field in str(response[1])
        for field in ("num_workers", "threads_per_worker"):
            probe = {
                **spec,
                "dataset_dir": str(directory / "missing-dataset"),
                "execution_profile": "independent",
                field: True,
            }
            response = record("boolean-run-" + field, url + "/api/jobs", "POST", probe)
            conflicts["boolean-" + field] = 400 <= response[0] < 500 and field in str(response[1])
        resolved, saved = apply_profile({**spec, "execution_profile": "independent"}, store)
        manual = {
            **spec,
            "num_workers": 2,
            "threads_per_worker": 3,
            "worker_memory_limit": "512MiB",
            "dask_tmpdir": str(scratch.resolve()),
        }
        manual_args, manual_env = build_runner_args(manual, directory / "same-output")
        saved_args, saved_env = build_runner_args(resolved, directory / "same-output")
        equivalence = {
            "manual_args": manual_args,
            "profile_args": saved_args,
            "manual_env": manual_env,
            "profile_env": saved_env,
            "resolved_profile": saved,
            "equal": manual_args == saved_args and manual_env == saved_env,
        }

        def save_batch(urls, prefix):
            def save_one(index):
                name = f"{prefix}-{index:02}"
                endpoint = urls[index % len(urls)] + f"/api/execution-profiles/{name}"
                return name, record(name, endpoint, "PUT", profile)[0]

            with ThreadPoolExecutor(max_workers=12) as pool:
                return dict(pool.map(save_one, range(16)))

        concurrent_responses = save_batch([url], "parallel")
        parallel_list = record("after-concurrent-writes", url + "/api/execution-profiles")
        with server(directory, 4) as (second_url, server4):
            cross_responses = save_batch([url, second_url], "cross")
            cross_list = record("after-two-process-writes", url + "/api/execution-profiles")
        deleted = record("delete", endpoint, "DELETE")
    with server(directory, 3) as (url, server3):
        absent = record("fresh-process-after-delete", url + "/api/execution-profiles/independent")
        restarted_list = record("fresh-process-concurrent-writes", url + "/api/execution-profiles")
        for item in restarted_list[1].get("items", []):
            record("cleanup-" + item["name"], url + "/api/execution-profiles/" + item["name"], "DELETE")
    accepted = {name for name, status in concurrent_responses.items() if status == 200}
    listed = {item["name"] for item in parallel_list[1].get("items", [])}
    restarted = {item["name"] for item in restarted_list[1].get("items", [])}
    concurrency = {
        "responses": concurrent_responses,
        "accepted": sorted(accepted),
        "missing_immediate": sorted(accepted - listed),
        "missing_after_restart": sorted(accepted - restarted),
    }
    concurrent_ok = len(accepted) == 16 and not (accepted - listed) and not (accepted - restarted)
    cross_accepted = {name for name, status in cross_responses.items() if status == 200}
    cross_listed = {item["name"] for item in cross_list[1].get("items", [])}
    cross_ok = len(cross_accepted) == 16 and not (cross_accepted - cross_listed) and not (cross_accepted - restarted)
    cross_process = {
        "responses": cross_responses,
        "accepted": sorted(cross_accepted),
        "missing_immediate": sorted(cross_accepted - cross_listed),
        "missing_after_restart": sorted(cross_accepted - restarted),
    }
    persistence = (
        created[0] == updated[0] == reloaded[0] == 200
        and updated[1] == reloaded[1]
        and server1["pid"] != server2["pid"]
        and deleted[0] == 200
        and absent[0] == missing[0] == 404
    )
    corrupted = {}
    corrupt_path = directory / "corrupt-store.json"
    for label, contents in {
        "broken-json": "{",
        "future-schema": '{"schema_version":99,"profiles":{}}',
        "bad-entry": '{"schema_version":1,"profiles":{"broken":null}}',
    }.items():
        corrupted[label] = {}
        for operation in ("list", "get", "save", "delete"):
            corrupt_path.write_text(contents)
            try:
                other = ExecutionProfileStore(corrupt_path)
                if operation == "list":
                    other.list()
                elif operation == "get":
                    other.get("broken")
                elif operation == "save":
                    other.save({"name": "fresh"})
                else:
                    other.delete("broken")
                rejected_corruption = False
            except ValueError:
                rejected_corruption = True
            corrupted[label][operation] = rejected_corruption and corrupt_path.read_text() == contents
    source_after = {name: digest(ROOT / name) for name in SOURCE_FILES}
    return {
        "claim": "isolated API persistence, invalid schema rejection, and resolved argv/env equality",
        "persistence": "VERIFIED" if persistence else "NOT VERIFIED",
        "invalid_config": "VERIFIED" if all(rejected.values()) else "NOT VERIFIED",
        "unavailable_gpu_rejection": "VERIFIED" if gpu_rejected else "NOT VERIFIED",
        "corrupt_store": "VERIFIED" if all(all(v.values()) for v in corrupted.values()) else "NOT VERIFIED",
        "remote_boundary_rejection": "VERIFIED" if remote_boundary else "NOT VERIFIED",
        "conflicting_run_controls": "VERIFIED" if all(conflicts.values()) else "NOT VERIFIED",
        "run_controls_rejected": conflicts,
        "concurrent_persistence": "VERIFIED" if concurrent_ok else "NOT VERIFIED",
        "cross_process_persistence": "VERIFIED" if cross_ok else "NOT VERIFIED",
        "cross_process": cross_process,
        "concurrency": concurrency,
        "argv_environment_equivalence": "VERIFIED" if equivalence["equal"] else "NOT VERIFIED",
        "server_processes": [server1, server2, server4, server3],
        "persisted_bytes_after_update": persisted,
        "corruption_rejected_without_mutation": corrupted,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "source_stable": source_before == source_after,
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ["fastapi", "uvicorn", "hydra-core", "omegaconf"]
        },
        "invalid_rejected": rejected,
        "equivalence": equivalence,
        "http": observations,
        "scope": (
            "Actual HTTP and argument resolution with real injected temporary storage; no job, GPU, or cleanup proof."
        ),
    }


def read_model(directory, *, preserve_point_ids=False):
    import numpy as np

    def finite(values):
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite geometry")
        return result

    def rows(name):
        return [
            line.split()
            for line in (directory / name).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    cameras = {}
    for row in rows("cameras.txt"):
        key = int(row[0])
        if key in cameras:
            raise ValueError("Duplicate camera ID")
        cameras[key] = (tuple(row[1:4]), finite(row[4:]))
    images, ids, observations, camera_ids = {}, {}, {}, set()
    lines = iter((directory / "images.txt").read_text().splitlines())
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        row = line.split(maxsplit=9)
        if len(row) != 10:
            raise ValueError("Malformed image header")
        image_id, camera_id, name = int(row[0]), int(row[8]), row[9]
        if name in images or image_id in ids:
            raise ValueError("Duplicate image name/ID")
        q = finite(row[1:5])
        if not np.isclose(np.linalg.norm(q), 1, atol=1e-6):
            raise ValueError("Invalid camera quaternion")
        w, x, y, z = q / np.linalg.norm(q)
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        data = next(lines).split()
        if len(data) % 3:
            raise ValueError("Malformed image observations")
        obs = [(float(data[i]), float(data[i + 1]), int(data[i + 2])) for i in range(0, len(data), 3)]
        finite([[x, y] for x, y, _ in obs])
        images[name] = {"center": -rotation.T @ finite(row[5:8]), "rotation": rotation, "camera": cameras[camera_id]}
        camera_ids.add(camera_id)
        ids[image_id] = name
        observations[image_id] = obs
    if camera_ids != cameras.keys():
        raise ValueError("Unreferenced camera intrinsics")
    tracks, point_ids, linked = {}, set(), set()
    for row in rows("points3D.txt"):
        pid = int(row[0])
        if pid in point_ids:
            raise ValueError("Duplicate point ID")
        point_ids.add(pid)
        if (len(row) - 8) % 2 or len(row) < 12:
            raise ValueError("Malformed or unobserved/dense point; sparse-track comparison requires >=2 observations")
        key = []
        for index in range(8, len(row), 2):
            image_id, obs_index = int(row[index]), int(row[index + 1])
            if obs_index < 0:
                raise ValueError("Negative observation index")
            x, y, observed_pid = observations[image_id][obs_index]
            if observed_pid != pid:
                raise ValueError("Inconsistent point/image observation linkage")
            link = (image_id, obs_index, pid)
            if link in linked:
                raise ValueError("Duplicate observation linkage")
            linked.add(link)
            key.append((ids[image_id], round(x, 3), round(y, 3)))
        key = tuple(sorted(key))
        if len(set(key)) != len(key):
            raise ValueError("Ambiguous duplicate track/observation identity")
        # Complete byte equality establishes stable IDs without a correspondence guess.
        # Otherwise preserve the strict, predeclared observation-only identity rule.
        if preserve_point_ids:
            key = (key, pid)
        if key in tracks:
            raise ValueError("Ambiguous duplicate track/observation identity")
        tracks[key] = (finite(row[1:4]), float(finite([row[7]])[0]), tuple(int(x) for x in row[4:7]))
    expected = {
        (image_id, index, pid)
        for image_id, obs in observations.items()
        for index, (_, _, pid) in enumerate(obs)
        if pid >= 0
    }
    if linked != expected:
        raise ValueError("Incomplete point/image observation linkage")
    if not tracks or len(images) < 3:
        raise ValueError("Need nonempty sparse tracks and >=3 registered cameras")
    return images, tracks


def compare_models(baseline, treatment):
    import numpy as np

    ai, at = baseline
    bi, bt = treatment
    identity = {
        "same_registered_images": ai.keys() == bi.keys(),
        "same_track_observations": at.keys() == bt.keys(),
        "baseline_cameras": len(ai),
        "treatment_cameras": len(bi),
        "baseline_tracks": len(at),
        "treatment_tracks": len(bt),
    }
    if not all((identity["same_registered_images"], identity["same_track_observations"])):
        return {**identity, "pass": False}
    names = sorted(ai)
    target = np.array([ai[name]["center"] for name in names])
    source = np.array([bi[name]["center"] for name in names])
    diagonal = float(np.linalg.norm(np.ptp(target, axis=0)))
    x, y = source - source.mean(0), target - target.mean(0)
    if diagonal <= 0 or np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("Degenerate camera centers: no unique similarity alignment")
    u, singular, vt = np.linalg.svd(x.T @ y / len(x))
    signs = np.array([1, 1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ np.diag(signs) @ u.T
    scale = float((singular * signs).sum() / (x * x).sum() * len(x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale")
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    center_error = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1) / diagonal
    angle, intrinsic_ok, metadata_ok = [], True, True
    for name in names:
        relative = ai[name]["rotation"] @ (bi[name]["rotation"] @ rotation.T).T
        angle.append(float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))))
        ac, bc = ai[name]["camera"], bi[name]["camera"]
        metadata_ok &= ac[0] == bc[0]
        intrinsic_ok &= ac[1].shape == bc[1].shape and bool(np.allclose(ac[1], bc[1], atol=1e-6, rtol=1e-6))
    keys = sorted(at)
    pa, pb = np.array([at[k][0] for k in keys]), np.array([bt[k][0] for k in keys])
    point_error = np.linalg.norm(scale * pb @ rotation.T + translation - pa, axis=1) / diagonal
    reproj = np.abs(np.array([at[k][1] - bt[k][1] for k in keys]))
    colors_ok = all(at[k][2] == bt[k][2] for k in keys)
    maxima = {
        "camera_center_over_D": float(center_error.max()),
        "rotation_deg": max(angle),
        "point_over_D": float(point_error.max()),
        "exported_reprojection_px": float(reproj.max()),
    }
    distributions = {
        name: {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
        for name, values in [
            ("camera_center_over_D", center_error),
            ("rotation_deg", angle),
            ("point_over_D", point_error),
            ("exported_reprojection_px", reproj),
        ]
    }
    passed = (
        metadata_ok
        and intrinsic_ok
        and colors_ok
        and maxima["camera_center_over_D"] <= 1e-5
        and maxima["rotation_deg"] <= 0.001
        and maxima["point_over_D"] <= 1e-4
        and maxima["exported_reprojection_px"] <= 0.011
    )
    return {
        **identity,
        "same_camera_metadata": bool(metadata_ok),
        "intrinsics_within_tolerance": bool(intrinsic_ok),
        "same_track_colors": colors_ok,
        "max_errors": maxima,
        "error_distributions": distributions,
        "baseline_diagonal": diagonal,
        "sim3": {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()},
        "pass": bool(passed),
    }


def compare_trials(models, pairs):
    labels = ["A1", "A2", "B1", "B2"]
    fingerprints = [
        tuple(digest(path / name) for name in ("cameras.txt", "images.txt", "points3D.txt")) + (digest(pair),)
        for path, pair in zip(models, pairs)
    ]
    exact_exports = len(set(fingerprints)) == 1
    loaded = [read_model(path, preserve_point_ids=exact_exports) for path in models]
    named_pairs = []
    for path in pairs:
        identities = [tuple(line.split()[1:]) for line in path.read_text().splitlines() if line.strip()]
        if not identities or any(len(pair) != 2 for pair in identities):
            raise ValueError("Need nonempty unambiguous similarity_named_pairs.txt (filenames without spaces)")
        named_pairs.append(sorted(identities))
    comparisons = {}
    for i, j in [(0, 1), (0, 2), (0, 3), (2, 3)]:
        result = compare_models(loaded[i], loaded[j])
        result["same_retrieved_pair_identities"] = named_pairs[i] == named_pairs[j]
        result["pass"] &= result["same_retrieved_pair_identities"]
        comparisons[f"{labels[i]}:{labels[j]}"] = result
    repeatable = comparisons["A1:A2"]["pass"]
    verdict = (
        "INCONCLUSIVE"
        if not repeatable
        else ("VERIFIED" if all(x["pass"] for x in comparisons.values()) else "NOT VERIFIED")
    )
    return {
        "claim": "exported sparse reconstruction equivalence at predeclared precision",
        "verdict": verdict,
        "exact_export_bytes": exact_exports,
        "track_identity_mode": (
            "Full export/pair bytes identical: preserve exact point IDs and all observation multiplicity"
            if exact_exports
            else "Canonical observation identities; ambiguous duplicates fail visibly"
        ),
        "baseline_repeatable": repeatable,
        "comparisons": comparisons,
        "artifacts": {
            label: {
                "model": str(path),
                "pairs": str(pair),
                "sha256": {name: digest(path / name) for name in ("cameras.txt", "images.txt", "points3D.txt")},
                "pairs_sha256": digest(pair),
            }
            for label, path, pair in zip(labels, models, pairs)
        },
        "scope": (
            "Requires separate manifest/cache/source/worker review; "
            "does not establish internal bitwise losslessness or speedup."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("config", help="real isolated API persistence, schema, and argv probes")
    config.add_argument("--artifacts", required=True, type=Path)
    compare = sub.add_parser("compare", help="explicit exports and named-pair files ordered A1 A2 B1 B2")
    compare.add_argument("--models", nargs=4, required=True, type=Path)
    compare.add_argument("--pairs", nargs=4, required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.artifacts / "configuration.json" if args.command == "config" else args.output
    if output.exists():
        parser.error(f"Report already exists; preserve prior evidence and choose a fresh path: {output}")
    if args.command == "config" and args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("Configuration artifacts directory must be empty or absent")
    try:
        result = (
            check_configuration(args.artifacts.resolve())
            if args.command == "config"
            else compare_trials(args.models, args.pairs)
        )
    except Exception as exc:
        result = {"verdict": "INCONCLUSIVE", "error": f"{type(exc).__name__}: {exc}"}
    result["command"] = [sys.executable, *sys.argv]
    if result.get("source_stable") is False:
        result["verdict"] = "INCONCLUSIVE"
        result["error"] = "Shared source changed during verification; rerun after the implementation settles"
        for key in (
            "persistence",
            "invalid_config",
            "corrupt_store",
            "argv_environment_equivalence",
            "remote_boundary_rejection",
            "concurrent_persistence",
            "cross_process_persistence",
            "conflicting_run_controls",
            "unavailable_gpu_rejection",
        ):
            result[key] = "INCONCLUSIVE"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                **{
                    k: v
                    for k, v in result.items()
                    if k
                    in {
                        "verdict",
                        "error",
                        "persistence",
                        "invalid_config",
                        "corrupt_store",
                        "argv_environment_equivalence",
                        "remote_boundary_rejection",
                        "concurrent_persistence",
                        "cross_process_persistence",
                        "conflicting_run_controls",
                        "unavailable_gpu_rejection",
                    }
                },
            }
        )
    )
    failed = result.get("verdict") in {"INCONCLUSIVE", "NOT VERIFIED"} or "NOT VERIFIED" in result.values()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
