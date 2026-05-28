"""CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from jetson_imu_tui.app import JetsonImuApp
from jetson_imu_tui.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="jetson-imu-tui")
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    JetsonImuApp(cfg).run()


if __name__ == "__main__":
    main()
