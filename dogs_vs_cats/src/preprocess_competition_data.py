from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dinov2_pipeline import run_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Dogs-vs-Cats competition zips into manifests")
    parser.add_argument("--paths-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    result = run_preprocess(
        paths_cfg_path=args.paths_config,
        exp_cfg_path=args.experiment_config,
        force_extract=args.force_extract,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
