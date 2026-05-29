"""CLI entry point — headless web server with a browser uPlot frontend."""

from __future__ import annotations

import argparse
from pathlib import Path

from jetson_imu_tui.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="jetson-imu-tui")
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file")
    parser.add_argument("--host", default=None, help="Bind host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="Port (default from config)")
    # Accepted for backwards compatibility; serving is now the only mode.
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    from jetson_imu_tui.web_server import run_server

    run_server(cfg, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
