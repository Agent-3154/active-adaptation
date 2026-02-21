#!/usr/bin/env python3
"""
Minimal server that serves the latest checkpoints for runs.

Assumptions
-----------
- Training runs are stored under a root directory, by date, as in Hydra's default:
    <root>/<YYYY-MM-DD>/<HH-MM-SS>-<task>-<algo>/
- Each run directory contains a symlink or file named ``checkpoint_latest.pt``,
  created by ``train_ppo.py``.

This server:
- Lists today's runs (by date folder) that have ``checkpoint_latest.pt``.
- Exposes download links of the form:
    /download/<YYYY-MM-DD>/<HH-MM-SS>-<task>-<algo>

Usage
-----
  python scripts/checkpoint_server.py [--root ./outputs] [--port 8765] [--bind 127.0.0.1]
"""

import argparse
import datetime
import html
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def get_today_runs(root: Path) -> list[tuple[Path, Path]]:
    """
    Find run directories created today and their latest checkpoint file.

    Expects each run dir to have a ``checkpoint_latest.pt`` symlink/file.
    Returns list of (run_dir, latest_checkpoint_path) for runs that have it.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    date_dir = root / today
    if not date_dir.is_dir():
        return []

    result: list[tuple[Path, Path]] = []
    for run_dir in date_dir.iterdir():
        if not run_dir.is_dir():
            continue
        latest = run_dir / "wandb" / "latest-run" / "files" / "checkpoint_latest.pt"
        if not latest.exists():
            continue
        result.append((run_dir, latest))
    return sorted(result, key=lambda x: x[0].name)


class CheckpointHandler(BaseHTTPRequestHandler):
    @property
    def _root(self) -> Path:
        # Set by the custom HTTPServer subclass below
        return self.server.root  # type: ignore[attr-defined]

    def _resolve_run_path(self, run_id: str) -> Path | None:
        """Resolve URL-encoded run id to a path under root; return None if invalid."""
        raw = unquote(run_id, errors="strict")
        if not raw or ".." in raw or raw.startswith("/"):
            return None
        path = (self._root / raw).resolve()
        try:
            path.relative_to(self._root)
        except ValueError:
            return None
        return path if path.is_dir() else None

    def _send_plain(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file_headers(self, path: Path, filename: str | None = None) -> None:
        """Send 200 and headers for a checkpoint file (Last-Modified so clients can skip re-download)."""
        if not path.is_file():
            self._send_plain("Not found", 404)
            return
        st = path.stat()
        size = st.st_size
        last_modified = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(st.st_mtime))
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Last-Modified", last_modified)
        if filename:
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{html.escape(filename)}"',
            )
        self.end_headers()

    def _send_file(self, path: Path, filename: str | None = None) -> None:
        if not path.is_file():
            self._send_plain("Not found", 404)
            return
        self._file_headers(path, filename)
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # Index page: list today's runs
        if path == "/":
            runs = get_today_runs(self._root)
            today = datetime.date.today().strftime("%Y-%m-%d")
            rows: list[str] = []
            for run_dir, ckpt in runs:
                rel = run_dir.relative_to(self._root)
                run_id = str(rel).replace(os.sep, "/")
                display_name = ckpt.resolve().name
                rows.append(
                    f'<tr><td>{html.escape(run_dir.name)}</td>'
                    f'<td><a href="/download/{html.escape(run_id)}">'
                    f'{html.escape(display_name)}</a></td></tr>'
                )
            if rows:
                table = (
                    '<table border="1"><tr><th>Run</th><th>Latest checkpoint</th></tr>\n'
                    + "\n".join(rows)
                    + "\n</table>"
                )
            else:
                table = "<p>No runs with checkpoints for today.</p>"

            body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Checkpoints — {today}</title></head>
<body>
<h1>Today's runs ({today})</h1>
{table}
</body></html>"""
            self._send_html(body)
            return

        # Download latest checkpoint for a specific run
        if path.startswith("/download/"):
            run_id = path[len("/download/") :].lstrip("/")
            run_dir = self._resolve_run_path(run_id)
            if run_dir is None:
                self._send_plain("Invalid run", 400)
                return
            latest = run_dir / "wandb" / "latest-run" / "files" / "checkpoint_latest.pt"
            if not latest.exists():
                self._send_plain("No checkpoint found", 404)
                return
            filename = latest.resolve().name if latest.is_symlink() else latest.name
            self._send_file(latest, filename)
            return

        self._send_plain("Not found", 404)

    def do_HEAD(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """HEAD for /download/<run_id>: same headers as GET so clients can check Last-Modified without downloading."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path.startswith("/download/"):
            run_id = path[len("/download/") :].lstrip("/")
            run_dir = self._resolve_run_path(run_id)
            if run_dir is None:
                self._send_plain("Invalid run", 400)
                return
            latest = run_dir / "wandb" / "latest-run" / "files" / "checkpoint_latest.pt"
            if not latest.exists():
                self._send_plain("No checkpoint found", 404)
                return
            filename = latest.resolve().name if latest.is_symlink() else latest.name
            self._file_headers(latest, filename)
            return
        self._send_plain("Not found", 404)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Print to stdout; you can customize or silence logging here.
        print(format % args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve latest checkpoints from run directories for download."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing date folders (default: outputs)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--bind",
        type=str,
        default="127.0.0.1",
        help="Address to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    class Server(HTTPServer):
        def __init__(self, server_address, RequestHandlerClass, root_path: Path):
            super().__init__(server_address, RequestHandlerClass)
            self.root = root_path

    server = Server((args.bind, args.port), CheckpointHandler, root)
    print(f"Serving checkpoints from {root} at http://{args.bind}:{args.port}/")
    print("Today's runs with checkpoints will be listed on the index page.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

