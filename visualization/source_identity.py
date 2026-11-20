"""Source provenance shared by local runs and deployed workspaces."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

_SOURCE_PATHS = ("gtsfm", "visualization", "thirdparty", "pyproject.toml", "uv.lock")
_SOURCE_SUFFIXES = {".py", ".yaml", ".toml", ".lock", ".js", ".css", ".html", ".ts", ".tsx"}


def source_identity(root: Path | None = None) -> dict[str, Any]:
    """Identify the application files used from a checkout or installed package.

    Args:
        root: Checkout or package installation root. Defaults to this package's parent.

    Returns:
        Commit, dirty-state and content fingerprint, when available.
    """
    if os.environ.get("GTSFM_SOURCE_COMMIT"):
        return {
            "commit": os.environ["GTSFM_SOURCE_COMMIT"],
            "dirty": os.environ.get("GTSFM_SOURCE_DIRTY") == "1",
            "fingerprint": os.environ.get("GTSFM_SOURCE_FINGERPRINT", ""),
        }
    root = (root or Path(__file__).parent.parent).resolve()
    commit = dirty = None
    try:
        if (root / ".git").exists():
            git_args = {"cwd": root, "text": True, "timeout": 5, "stderr": subprocess.DEVNULL}
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], **git_args).strip()
            names = subprocess.check_output(
                ["git", "ls-files", "-zco", "--exclude-standard", "--", *_SOURCE_PATHS], **git_args
            ).split("\0")
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], **git_args))
        else:
            names = []
            for name in _SOURCE_PATHS:
                path = root / name
                if path.is_dir() and not path.is_symlink():
                    names.extend(child.relative_to(root).as_posix() for child in path.rglob("*"))
                else:
                    names.append(name)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "fingerprint": None}

    digest = hashlib.sha256()
    hashed_files = 0
    try:
        for name in sorted(set(names)):
            path = root / name
            if path.suffix not in _SOURCE_SUFFIXES or "node_modules" in path.parts:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            digest.update(name.encode())
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").hexdigest().encode())
            hashed_files += 1
    except OSError:
        return {"commit": commit, "dirty": dirty, "fingerprint": None}
    return {"commit": commit, "dirty": dirty, "fingerprint": digest.hexdigest() if hashed_files else None}
