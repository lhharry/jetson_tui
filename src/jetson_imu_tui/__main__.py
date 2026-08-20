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
    parser.add_argument(
        "--lan", action="store_true",
        help="Serve on the network (bind ::) instead of loopback only",
    )
    # Accepted for backwards compatibility; serving is now the only mode.
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # The config binds loopback so a Jetson in the field exposes nothing by default; --lan is
    # the deliberate opt-out. An explicit --host wins over both, so a specific address is never
    # silently widened.
    host = args.host or ("::" if args.lan else None)
    from jetson_imu_tui.web_server import run_server

    run_server(cfg, host=host, port=args.port)


if __name__ == "__main__":
    main()
