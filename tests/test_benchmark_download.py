"""Dataset preparation must fail closed when a download is not an archive."""

import io
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/download_single_benchmark.sh"


@pytest.fixture
def downloader(tmp_path):
    """Run the real shell entrypoint with isolated local download responses."""
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    with zipfile.ZipFile(fixtures / "2011205_rc3.zip", "w") as archive:
        archive.writestr("2011205_rc3/images/frame.png", b"fixture image")
    with tarfile.open(fixtures / "cache_rc3_deep.tar.gz", "w:gz") as archive:
        data = b"fixture cache"
        member = tarfile.TarInfo("cache/matcher/pair.pbz2")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    wget = binaries / "wget"
    wget.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, shutil, sys\n"
        "args = sys.argv[1:]\n"
        "url = args[-1]\n"
        "with open(os.environ['DOWNLOAD_TRACE'], 'a') as trace: trace.write(url + '\\n')\n"
        "if os.environ['DOWNLOAD_MODE'] == 'failure': sys.exit(8)\n"
        "name = url.rsplit('/', 1)[1]\n"
        "target = args[args.index('-O') + 1] if '-O' in args else name\n"
        "if os.environ['DOWNLOAD_MODE'] == 'html' and '.zip' in name:\n"
        "    pathlib.Path(target).write_text('<html>Dropbox preview</html>')\n"
        "else:\n"
        "    shutil.copyfile(pathlib.Path(os.environ['DOWNLOAD_FIXTURES']) / name.split('?')[0], target)\n"
    )
    wget.chmod(0o755)
    sleep = binaries / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)

    def run(mode):
        env = {
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "DOWNLOAD_FIXTURES": str(fixtures),
            "DOWNLOAD_TRACE": str(tmp_path / "requests.txt"),
            "DOWNLOAD_MODE": mode,
        }
        return subprocess.run(
            ["bash", str(SCRIPT), "2011205_rc3", "wget"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    return tmp_path, run


def test_asteroid_download_requests_archive_and_extracts_it(downloader):
    """Request the binary Dropbox response and retain the expected dataset layout."""
    root, run = downloader
    result = run("valid")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2011205_rc3.zip?dl=1" in (root / "requests.txt").read_text()
    assert (root / "benchmarks/2011205_rc3/images/frame.png").read_bytes() == b"fixture image"


def test_html_download_cannot_be_reported_as_success(downloader):
    """An HTML preview must stop preparation before the runner receives a missing dataset."""
    _, run = downloader
    result = run("html")
    assert result.returncode != 0
    assert "processed successfully" not in result.stdout


def test_exhausted_download_retries_preserve_failure(downloader):
    """A failed downloader must not turn into a successful preparation result."""
    root, run = downloader
    result = run("failure")
    assert result.returncode != 0
    assert len((root / "requests.txt").read_text().splitlines()) == 5
    assert "processed successfully" not in result.stdout
