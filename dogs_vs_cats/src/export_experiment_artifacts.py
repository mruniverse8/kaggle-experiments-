from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

JSONDict = Dict[str, Any]


def _read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _to_path(value: str) -> Path:
    return Path(value)


def _resolve_export_paths(paths_cfg: JSONDict) -> Dict[str, Path]:
    work_dir = _to_path(paths_cfg["work_dir"])
    reports_dir = _to_path(paths_cfg["reports_dir"])
    checkpoints_dir = _to_path(paths_cfg["checkpoints_dir"])
    artifacts_output_dir = _to_path(paths_cfg.get("artifacts_output_dir", str(work_dir / "output")))

    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "work_dir": work_dir,
        "reports_dir": reports_dir,
        "checkpoints_dir": checkpoints_dir,
        "artifacts_output_dir": artifacts_output_dir,
    }


def _copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required artifact does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_optional_file(src: Optional[Path], dst: Path) -> Optional[Path]:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _parent_bucket(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent in {"metrics", "plots", "predictions", "reports", "checkpoints"}:
        return parent
    return "files"


def _atomic_replace_dir(tmp_dir: Path, target_dir: Path) -> None:
    backup_dir: Optional[Path] = None
    if target_dir.exists():
        backup_dir = target_dir.parent / f".backup_{target_dir.name}_{_now_stamp()}"
        target_dir.replace(backup_dir)

    try:
        tmp_dir.replace(target_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            backup_dir.replace(target_dir)
        raise
    finally:
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)


def _build_manifest(
    *,
    experiment_name: str,
    exported_at: str,
    paths_config: str,
    experiment_config: str,
    training_summary: JSONDict,
    file_map: JSONDict,
) -> JSONDict:
    final_val_metrics = training_summary.get("final_val_metrics", {})
    summary_fields = {
        "best_checkpoint": training_summary.get("best_checkpoint", ""),
        "last_checkpoint": training_summary.get("last_checkpoint", ""),
        "monitor": training_summary.get("monitor", ""),
        "best_monitor": training_summary.get("best_monitor"),
        "best_epoch": training_summary.get("best_epoch"),
    }
    return {
        "schema_version": "1.0",
        "experiment_name": experiment_name,
        "exported_at": exported_at,
        "source": {
            "paths_config": str(paths_config),
            "experiment_config": str(experiment_config),
        },
        "final_val_metrics": {
            "val_auc": final_val_metrics.get("val_auc"),
            "val_logloss": final_val_metrics.get("val_logloss"),
            "val_accuracy": final_val_metrics.get("val_accuracy"),
            "val_loss": final_val_metrics.get("val_loss"),
        },
        "training_summary": summary_fields,
        "files": file_map,
    }


def _update_bundle_index(output_root: Path, manifest: JSONDict) -> Path:
    index_path = output_root / "bundle_index.json"
    if index_path.exists():
        index = _read_json(index_path)
    else:
        index = {
            "schema_version": "1.0",
            "exported_at": _now_stamp(),
            "experiments": [],
        }

    exp_name = manifest["experiment_name"]
    metrics = manifest.get("final_val_metrics", {})
    entry = {
        "experiment_name": exp_name,
        "manifest_path": f"{exp_name}/manifest.json",
        "val_auc": metrics.get("val_auc"),
        "val_logloss": metrics.get("val_logloss"),
        "val_accuracy": metrics.get("val_accuracy"),
    }

    experiments = [row for row in index.get("experiments", []) if row.get("experiment_name") != exp_name]
    experiments.append(entry)
    experiments = sorted(experiments, key=lambda row: str(row.get("experiment_name", "")))

    index["schema_version"] = "1.0"
    index["exported_at"] = _now_stamp()
    index["experiments"] = experiments
    _write_json(index_path, index)
    return index_path


def export_artifacts(
    *,
    paths_config: str,
    experiment_config: str,
    output_root: str = "",
) -> JSONDict:
    paths_cfg = _read_json(Path(paths_config))
    exp_cfg = _read_json(Path(experiment_config))
    paths = _resolve_export_paths(paths_cfg)

    exp_name = str(exp_cfg.get("experiment_name", "dinov2_experiment"))
    reports_dir = paths["reports_dir"]
    summary_path = reports_dir / f"{exp_name}_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found: {summary_path}")

    summary = _read_json(summary_path)
    destination_root = Path(output_root) if output_root else paths["artifacts_output_dir"]
    destination_root.mkdir(parents=True, exist_ok=True)

    tmp_exp_dir = destination_root / f".tmp_{exp_name}_{_now_stamp()}"
    if tmp_exp_dir.exists():
        shutil.rmtree(tmp_exp_dir)
    tmp_exp_dir.mkdir(parents=True, exist_ok=True)

    file_map: JSONDict = {}

    # Checkpoints
    best_checkpoint_src = Path(summary.get("best_checkpoint", ""))
    last_checkpoint_src = Path(summary.get("last_checkpoint", "")) if summary.get("last_checkpoint", "") else None
    best_checkpoint_dst = tmp_exp_dir / "checkpoints" / "best.pt"
    _copy_required_file(best_checkpoint_src, best_checkpoint_dst)
    file_map["best_checkpoint"] = "checkpoints/best.pt"

    if last_checkpoint_src is None:
        fallback_last = paths["checkpoints_dir"] / exp_name / "last.pt"
        last_checkpoint_src = fallback_last if fallback_last.exists() else None
    last_checkpoint_dst = _copy_optional_file(last_checkpoint_src, tmp_exp_dir / "checkpoints" / "last.pt")
    if last_checkpoint_dst is not None:
        file_map["last_checkpoint"] = "checkpoints/last.pt"

    # Summary files referenced in report
    files_from_summary = summary.get("files", {})
    for key in sorted(files_from_summary.keys()):
        src = Path(str(files_from_summary[key]))
        bucket = _parent_bucket(src)
        dst = tmp_exp_dir / bucket / src.name
        _copy_required_file(src, dst)
        file_map[key] = str(dst.relative_to(tmp_exp_dir))

    # Include source training summary JSON in reports.
    copied_summary_path = tmp_exp_dir / "reports" / summary_path.name
    _copy_required_file(summary_path, copied_summary_path)
    file_map["training_summary_json"] = str(copied_summary_path.relative_to(tmp_exp_dir))

    # Include configs used for this experiment.
    exp_cfg_dst = tmp_exp_dir / "configs" / "experiment_config.json"
    paths_cfg_dst = tmp_exp_dir / "configs" / "paths_config.json"
    _copy_required_file(Path(experiment_config), exp_cfg_dst)
    _copy_required_file(Path(paths_config), paths_cfg_dst)
    file_map["experiment_config"] = str(exp_cfg_dst.relative_to(tmp_exp_dir))
    file_map["paths_config"] = str(paths_cfg_dst.relative_to(tmp_exp_dir))

    manifest = _build_manifest(
        experiment_name=exp_name,
        exported_at=_now_stamp(),
        paths_config=paths_config,
        experiment_config=experiment_config,
        training_summary=summary,
        file_map=file_map,
    )
    _write_json(tmp_exp_dir / "manifest.json", manifest)

    target_exp_dir = destination_root / exp_name
    _atomic_replace_dir(tmp_exp_dir, target_exp_dir)

    index_path = _update_bundle_index(destination_root, manifest)
    return {
        "experiment_name": exp_name,
        "exported_dir": str(target_exp_dir),
        "manifest_path": str(target_exp_dir / "manifest.json"),
        "bundle_index_path": str(index_path),
        "files_count": int(len(file_map)),
        "exported_at": _now_stamp(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment artifacts into reusable Kaggle bundle format")
    parser.add_argument("--paths-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    result = export_artifacts(
        paths_config=args.paths_config,
        experiment_config=args.experiment_config,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
