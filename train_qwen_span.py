#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dz2_causal.train import load_train_config, train  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train from JSON config")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "train_qwen35_08b.json"),
        help="Path to JSON config file.",
    )
    cli_args = parser.parse_args()
    args = load_train_config(cli_args.config)
    train(args)
