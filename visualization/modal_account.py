"""Connect Studio through Modal's native browser login and local profiles."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class ModalCredentials:
    """Server-side credentials; never include these in API responses or logs."""

    profile: str
    token_id: str = field(repr=False)
    token_secret: str = field(repr=False)


def profile_credentials(profile: str = "") -> ModalCredentials:
    """Read a fresh native Modal profile without changing the active account.

    Args:
        profile: Selected profile, or the native active profile when empty.

    Returns:
        Credentials for the selected profile.
    """
    from modal.config import user_config_path

    path = Path(user_config_path).expanduser()
    data = tomllib.loads(path.read_text()) if path.exists() else {}
    selected = (
        profile
        or os.environ.get("MODAL_PROFILE")
        or next((name for name, values in data.items() if values.get("active")), "default")
    )
    values = data.get(selected, {})
    token_id = str(values.get("token_id", ""))
    token_secret = str(values.get("token_secret", ""))
    if not profile:
        token_id = os.environ.get("MODAL_TOKEN_ID", token_id)
        token_secret = os.environ.get("MODAL_TOKEN_SECRET", token_secret)
    if not token_id or not token_secret:
        raise ValueError("Connect your Modal account to continue.")
    return ModalCredentials(selected, token_id, token_secret)


def native_profiles() -> list[str]:
    """Return available profile names, without exposing their credentials."""
    from modal.config import user_config_path

    path = Path(user_config_path).expanduser()
    data = tomllib.loads(path.read_text()) if path.exists() else {}
    return sorted(name for name, values in data.items() if isinstance(values, dict) and values.get("token_id"))


async def verify_profile(credentials: ModalCredentials) -> str:
    """Verify a profile with Modal and return its workspace name.

    Args:
        credentials: Private native profile credentials.

    Returns:
        Verified workspace name.
    """
    from modal import Client, Workspace

    async def verify() -> str:
        client = await Client.from_credentials.aio(credentials.token_id, credentials.token_secret)
        try:
            workspace = Workspace.from_context(client=client)
            await workspace.hydrate.aio()
            return workspace.name or credentials.profile
        finally:
            close = getattr(client, "close", None) or client._close
            await close.aio()

    return await asyncio.wait_for(verify(), timeout=20)


class ModalAccount:
    """Keep browser authorization in the Studio host, outside frontend state."""

    def __init__(self) -> None:
        """Initialize a disconnected account without making network requests."""
        self._lock = threading.RLock()
        self._status = "disconnected"
        self._profile = ""
        self._workspace = ""
        self._url = ""
        self._code = ""
        self._error = ""
        self._process: subprocess.Popen[str] | None = None
        self._closed = threading.Event()

    def public(self) -> dict[str, Any]:
        """Return connection metadata only; credentials stay on the host."""
        with self._lock:
            try:
                profiles = native_profiles()
            except (ValueError, OSError, ImportError):
                profiles = []
            return {
                "status": self._status,
                "profile": self._profile,
                "workspace": self._workspace,
                "profiles": profiles,
                "login_url": self._url if self._status == "signing_in" else "",
                "code": self._code if self._status == "signing_in" else "",
                "error": self._error,
            }

    async def connect(self, profile: str = "", *, sign_in: bool = False) -> dict[str, Any]:
        """Reuse an existing profile or start Modal's supported browser flow.

        Args:
            profile: Native profile to connect.
            sign_in: Explicitly request a fresh browser login.

        Returns:
            Sanitized connection status.
        """
        with self._lock:
            if self._closed.is_set():
                raise ValueError("Studio is shutting down.")
            if self._status == "signing_in":
                return self.public()
        if not sign_in:
            try:
                credentials = profile_credentials(profile)
            except (ValueError, OSError, ImportError):
                pass
            else:
                try:
                    workspace = await verify_profile(credentials)
                except Exception:
                    with self._lock:
                        self._status = "error"
                        self._error = "Could not verify this Modal account. Retry or sign in again."
                    return self.public()
                with self._lock:
                    self._status = "connected"
                    self._profile = credentials.profile
                    self._workspace = workspace
                    self._error = ""
                return self.public()
        with self._lock:
            self._status = "signing_in"
            self._url = self._code = self._error = ""
        threading.Thread(target=self._login, daemon=True, name="gtsfm-modal-login").start()
        return self.public()

    def _login(self) -> None:
        """Run the native CLI flow and expose only its authorization link."""
        env = os.environ.copy()
        # Modal recognizes non-GUI browser handlers and prints its authorization URL.
        # Studio presents that URL in the user's existing browser instead.
        env.update({"BROWSER": "gtsfm-browser-link", "TERM": "dumb", "NO_COLOR": "1", "COLUMNS": "4096"})
        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
            env.pop(key, None)
        process = None
        timer = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "modal", "token", "new", "--no-activate"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with self._lock:
                self._process = process
                if self._closed.is_set():
                    process.terminate()
            timer = threading.Timer(300, process.terminate)
            timer.daemon = True
            timer.start()
            profile = ""
            assert process.stdout is not None
            for raw in process.stdout:
                line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).strip()
                match = re.search(r"https://[^\s\x1b]+", line)
                if match:
                    url = match[0].rstrip(".)")
                    if urlparse(url).hostname in {"modal.com", "www.modal.com"}:
                        with self._lock:
                            self._url = url
                if "Enter this code:" in line:
                    with self._lock:
                        self._code = line.split("Enter this code:", 1)[1].strip()
                match = re.search(r"in profile\s+(.+?)\.\s*$", line)
                if match:
                    profile = match[1]
            if process.wait() != 0:
                raise ValueError("Modal sign-in was cancelled or expired. Connect again to retry.")
            credentials = profile_credentials(profile)
            workspace = asyncio.run(verify_profile(credentials))
            with self._lock:
                self._profile = credentials.profile
                self._workspace = workspace
                self._status = "connected"
        except Exception:
            with self._lock:
                self._status = "error"
                self._error = "Modal sign-in did not complete. Connect again to retry."
        finally:
            if timer:
                timer.cancel()
            with self._lock:
                self._process = None
                self._url = self._code = ""

    def close(self) -> None:
        """Cancel an unfinished login without revoking or editing existing profiles."""
        self._closed.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
