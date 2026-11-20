"""Source provenance shared by local runs and deployed workspaces."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any


def source_identity(root: Path | None = None) -> dict[str, Any]:
    """Identify application source, including uncommitted checkout changes.

    Args:
        root: Checkout root; defaults to this package's parent.

    Returns:
        Commit, dirty-state and content fingerprint, when available.
    """
    if os.environ.get("GTSFM_SOURCE_COMMIT"):
        return {
            "commit": os.environ["GTSFM_SOURCE_COMMIT"],
            "dirty": os.environ.get("GTSFM_SOURCE_DIRTY") == "1",
            "fingerprint": os.environ.get("GTSFM_SOURCE_FINGERPRINT", ""),
        }
    root = root or Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5).strip()
        paths = subprocess.check_output(
            [
                "git",
                "ls-files",
                "-co",
                "--exclude-standard",
                "gtsfm",
                "visualization",
                "thirdparty",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=root,
            text=True,
            timeout=5,
        ).splitlines()
        digest = hashlib.sha256()
        for name in sorted(set(paths)):
            path = root / name
            if not path.is_file() or path.is_symlink() or "node_modules" in path.parts:
                continue
            if path.suffix not in {".py", ".yaml", ".toml", ".lock", ".js", ".css", ".html", ".ts", ".tsx"}:
                continue
            digest.update(name.encode())
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").hexdigest().encode())
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, timeout=5))
        return {"commit": commit, "dirty": dirty, "fingerprint": digest.hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "fingerprint": None}
