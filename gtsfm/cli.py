"""Installed command-line entry point for GTSFM."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gtsfm",
        description="Run GTSFM from a browser, the terminal, or inspect available hardware.",
    )
    parser.add_argument("--version", action="version", version="GTSFM 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Open the browser-based GTSFM workspace")
    run_parser.add_argument("--host", default="127.0.0.1", help="Address for the local web server")
    run_parser.add_argument("--port", type=int, default=5173, help="Port for the local web server")
    run_parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Directory containing existing and newly-created runs",
    )
    run_parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")

    execute_parser = commands.add_parser("execute", help="Run the existing terminal pipeline")
    execute_parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed unchanged to the GTSFM pipeline",
    )

    hardware_parser = commands.add_parser("hardware", help="Show compute devices detected on this machine")
    hardware_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    viz_parser = commands.add_parser("viz", help="Open the workspace directly in results-viewing mode")
    viz_parser.add_argument("--host", default="127.0.0.1")
    viz_parser.add_argument("--port", type=int, default=5173)
    viz_parser.add_argument("--results", type=Path, default=Path("results"))
    viz_parser.add_argument("--no-browser", action="store_true")
    return parser


def _serve(host: str, port: int, results: Path, open_browser: bool, initial_view: str) -> None:
    import uvicorn

    from visualization.app import create_app

    app = create_app(results.resolve())
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}/?view={initial_view}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"GTSFM workspace: {url}")
    print(f"Results directory: {results.resolve()}")
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="warning")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed ``gtsfm`` command."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {"run", "viz"}:
        _serve(
            host=args.host,
            port=args.port,
            results=args.results,
            open_browser=not args.no_browser,
            initial_view="results" if args.command == "viz" else "run",
        )
        return 0

    if args.command == "execute":
        from gtsfm.runner import GtsfmRunner

        runner_args = list(args.runner_args)
        if runner_args[:1] == ["--"]:
            runner_args = runner_args[1:]
        GtsfmRunner(override_args=runner_args).run()
        return 0

    if args.command == "hardware":
        from visualization.runtime import detect_hardware

        payload = detect_hardware()
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(payload["summary"])
            for device in payload["devices"]:
                suffix = f" ({device['memory']})" if device.get("memory") else ""
                print(f"- {device['label']}{suffix}: {device['status']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
