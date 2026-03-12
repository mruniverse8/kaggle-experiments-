from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dinov2_pipeline import run_sanity_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run random-init sanity checks for DinoV2 Dogs-vs-Cats training pipeline"
    )
    parser.add_argument("--paths-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    args = parser.parse_args()

    result = run_sanity_check(
        paths_cfg_path=args.paths_config,
        exp_cfg_path=args.experiment_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
