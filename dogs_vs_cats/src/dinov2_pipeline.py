from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoConfig, AutoModel


JSONDict = Dict[str, Any]


def _read_json(path: Union[str, Path]) -> JSONDict:
    return json.loads(Path(path).read_text())


def _write_json(path: Union[str, Path], payload: JSONDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_path(value: Union[str, Path]) -> Path:
    return value if isinstance(value, Path) else Path(value)


def resolve_paths(paths_cfg: JSONDict) -> Dict[str, Path]:
    paths: Dict[str, Path] = {
        "competition_dir": _to_path(paths_cfg["competition_dir"]),
        "train_zip": _to_path(paths_cfg["train_zip"]),
        "test_zip": _to_path(paths_cfg["test_zip"]),
        "sample_submission_csv": _to_path(paths_cfg["sample_submission_csv"]),
        "work_dir": _to_path(paths_cfg["work_dir"]),
        "extracted_dir": _to_path(paths_cfg["extracted_dir"]),
        "train_extracted_dir": _to_path(paths_cfg["train_extracted_dir"]),
        "test_extracted_dir": _to_path(paths_cfg["test_extracted_dir"]),
        "manifests_dir": _to_path(paths_cfg["manifests_dir"]),
        "checkpoints_dir": _to_path(paths_cfg["checkpoints_dir"]),
        "evaluation_dir": _to_path(paths_cfg["evaluation_dir"]),
        "metrics_dir": _to_path(paths_cfg["metrics_dir"]),
        "plots_dir": _to_path(paths_cfg["plots_dir"]),
        "predictions_dir": _to_path(paths_cfg["predictions_dir"]),
        "reports_dir": _to_path(paths_cfg["reports_dir"]),
    }

    for key in (
        "work_dir",
        "extracted_dir",
        "train_extracted_dir",
        "test_extracted_dir",
        "manifests_dir",
        "checkpoints_dir",
        "evaluation_dir",
        "metrics_dir",
        "plots_dir",
        "predictions_dir",
        "reports_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _contains_jpg_files(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(directory.rglob("*.jpg"))


def _extract_zip_if_needed(zip_path: Path, extract_to: Path, force: bool = False) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if _contains_jpg_files(extract_to) and not force:
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_to)


def _candidate_competition_dirs(base_dir: Path) -> List[Path]:
    dirs: List[Path] = [base_dir]
    as_posix = base_dir.as_posix()
    if "/kaggle/input/competitions/" in as_posix:
        dirs.append(Path(as_posix.replace("/kaggle/input/competitions/", "/kaggle/input/")))
    elif "/kaggle/input/" in as_posix and "/kaggle/input/competitions/" not in as_posix:
        dirs.append(Path(as_posix.replace("/kaggle/input/", "/kaggle/input/competitions/")))

    slug = base_dir.name
    if slug:
        dirs.append(Path("/kaggle/input") / slug)
        dirs.append(Path("/kaggle/input/competitions") / slug)

    # Include discovered dirs that contain the competition slug.
    for parent in (Path("/kaggle/input"), Path("/kaggle/input/competitions")):
        if parent.exists() and slug:
            for match in parent.glob(f"*{slug}*"):
                if match.is_dir():
                    dirs.append(match)

    # Keep order while removing duplicates.
    unique: List[Path] = []
    seen = set()
    for directory in dirs:
        key = directory.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(directory)
    return unique


def _resolve_input_file(
    configured_path: Path,
    competition_dir: Path,
    candidate_names: Iterable[str],
    description: str,
) -> Path:
    if configured_path.exists():
        return configured_path

    tried: List[Path] = [configured_path]
    candidate_dirs = _candidate_competition_dirs(competition_dir)
    candidate_dirs.append(configured_path.parent)

    for directory in candidate_dirs:
        for name in candidate_names:
            candidate = directory / name
            tried.append(candidate)
            if candidate.exists():
                return candidate

    tried_display = "\n".join(f"  - {path}" for path in tried)
    raise FileNotFoundError(
        f"{description} was not found.\n"
        f"Configured path: {configured_path}\n"
        f"Tried these locations:\n{tried_display}\n"
        "Update paths_kaggle.json or mount the competition dataset."
    )


def _resolve_optional_input_file(
    configured_path: Optional[Path],
    competition_dir: Path,
    candidate_names: Iterable[str],
) -> Optional[Path]:
    if configured_path is not None and configured_path.exists():
        return configured_path

    candidate_dirs = _candidate_competition_dirs(competition_dir)
    if configured_path is not None:
        candidate_dirs.append(configured_path.parent)

    for directory in candidate_dirs:
        for name in candidate_names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _iter_image_files(root: Path) -> Iterable[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in valid_suffixes:
            yield path


def _collect_train_records_from_dir(train_root: Path) -> List[Dict[str, Any]]:
    if not train_root.exists():
        raise FileNotFoundError(f"Train directory not found: {train_root}")

    class_dirs: Dict[Path, int] = {}
    for child in sorted(train_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name in {"cat", "cats"}:
            class_dirs[child] = 0
        elif name in {"dog", "dogs"}:
            class_dirs[child] = 1

    records: List[Dict[str, Any]] = []
    if class_dirs:
        for folder, label in class_dirs.items():
            for image_path in _iter_image_files(folder):
                records.append(
                    {
                        "id": image_path.stem,
                        "filepath": str(image_path),
                        "label": int(label),
                        "filename": image_path.name,
                    }
                )
    else:
        for image_path in _iter_image_files(train_root):
            label = _parse_train_label(image_path.name)
            if label is None:
                continue
            parts = image_path.stem.split(".")
            image_id = parts[1] if len(parts) > 1 else image_path.stem
            records.append(
                {
                    "id": str(image_id),
                    "filepath": str(image_path),
                    "label": int(label),
                    "filename": image_path.name,
                }
            )
    return records


def _collect_test_records_from_dir(test_root: Path) -> List[Dict[str, Any]]:
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for image_path in _iter_image_files(test_root):
        image_id = image_path.stem
        if image_id in seen_ids:
            image_id = image_path.relative_to(test_root).as_posix().replace("/", "_")
        seen_ids.add(image_id)
        records.append(
            {
                "id": str(image_id),
                "filepath": str(image_path),
                "filename": image_path.name,
            }
        )
    return records


def _locate_image_root(extract_dir: Path, preferred: str) -> Path:
    candidate = extract_dir / preferred
    if candidate.exists():
        return candidate

    direct_jpgs = list(extract_dir.glob("*.jpg"))
    if direct_jpgs:
        return extract_dir

    nested = [p for p in extract_dir.rglob(preferred) if p.is_dir()]
    if nested:
        return nested[0]

    for folder in extract_dir.rglob("*"):
        if folder.is_dir() and any(folder.glob("*.jpg")):
            return folder

    raise FileNotFoundError(f"Could not find image folder '{preferred}' under {extract_dir}")


def _parse_train_label(filename: str) -> Optional[int]:
    lower = filename.lower()
    if lower.startswith("cat."):
        return 0
    if lower.startswith("dog."):
        return 1
    return None


def build_manifests(paths_cfg: JSONDict, exp_cfg: JSONDict, force_extract: bool = False) -> Dict[str, Any]:
    paths = resolve_paths(paths_cfg)
    direct_train_dir = Path(paths_cfg.get("train_dir", "")) if paths_cfg.get("train_dir") else None
    direct_test_dir = Path(paths_cfg.get("test_dir", "")) if paths_cfg.get("test_dir") else None

    source_mode = "zip_competition"
    if direct_train_dir is not None and direct_test_dir is not None and direct_train_dir.exists() and direct_test_dir.exists():
        source_mode = "direct_dirs"
        train_root = direct_train_dir
        test_root = direct_test_dir
        train_records = _collect_train_records_from_dir(train_root)
        test_records = _collect_test_records_from_dir(test_root)
        resolved_train_zip = None
        resolved_test_zip = None
    else:
        paths["train_zip"] = _resolve_input_file(
            configured_path=paths["train_zip"],
            competition_dir=paths["competition_dir"],
            candidate_names=["train.zip"],
            description="Train zip",
        )
        paths["test_zip"] = _resolve_input_file(
            configured_path=paths["test_zip"],
            competition_dir=paths["competition_dir"],
            candidate_names=["test.zip"],
            description="Test zip",
        )

        _extract_zip_if_needed(paths["train_zip"], paths["train_extracted_dir"], force=force_extract)
        _extract_zip_if_needed(paths["test_zip"], paths["test_extracted_dir"], force=force_extract)

        train_root = _locate_image_root(paths["train_extracted_dir"], "train")
        try:
            test_root = _locate_image_root(paths["test_extracted_dir"], "test")
        except FileNotFoundError:
            test_root = _locate_image_root(paths["test_extracted_dir"], "test1")

        train_records = _collect_train_records_from_dir(train_root)
        test_records = _collect_test_records_from_dir(test_root)
        resolved_train_zip = paths["train_zip"]
        resolved_test_zip = paths["test_zip"]

    sample_submission_path = _resolve_optional_input_file(
        configured_path=Path(paths_cfg.get("sample_submission_csv", "")) if paths_cfg.get("sample_submission_csv") else None,
        competition_dir=paths["competition_dir"],
        candidate_names=["sample_submission.csv", "sampleSubmission.csv"],
    )
    if sample_submission_path is not None:
        paths["sample_submission_csv"] = sample_submission_path

    if not train_records:
        raise RuntimeError(f"No labeled training images found under {train_root}")

    if not test_records:
        raise RuntimeError(f"No test images found under {test_root}")

    df_train = pd.DataFrame(train_records)
    df_test = pd.DataFrame(test_records)

    val_ratio = float(exp_cfg.get("data", {}).get("val_ratio", 0.2))
    seed = int(exp_cfg.get("seed", 42))

    tr_idx, va_idx = train_test_split(
        np.arange(len(df_train)),
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=df_train["label"].values,
    )

    df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
    df_va = df_train.iloc[va_idx].reset_index(drop=True)

    train_manifest = paths["manifests_dir"] / "train_manifest.csv"
    val_manifest = paths["manifests_dir"] / "val_manifest.csv"
    test_manifest = paths["manifests_dir"] / "test_manifest.csv"

    df_tr.to_csv(train_manifest, index=False)
    df_va.to_csv(val_manifest, index=False)
    df_test.to_csv(test_manifest, index=False)

    summary: Dict[str, Any] = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "test_manifest": str(test_manifest),
        "train_count": int(len(df_train)),
        "train_split_count": int(len(df_tr)),
        "val_split_count": int(len(df_va)),
        "test_count": int(len(df_test)),
        "source_mode": source_mode,
        "label_distribution": {
            "train_label_0": int((df_train["label"] == 0).sum()),
            "train_label_1": int((df_train["label"] == 1).sum()),
            "split_train_label_0": int((df_tr["label"] == 0).sum()),
            "split_train_label_1": int((df_tr["label"] == 1).sum()),
            "split_val_label_0": int((df_va["label"] == 0).sum()),
            "split_val_label_1": int((df_va["label"] == 1).sum()),
        },
        "train_root": str(train_root),
        "test_root": str(test_root),
        "resolved_inputs": {
            "train_zip": str(resolved_train_zip) if resolved_train_zip is not None else "",
            "test_zip": str(resolved_test_zip) if resolved_test_zip is not None else "",
            "train_dir": str(direct_train_dir) if direct_train_dir is not None else "",
            "test_dir": str(direct_test_dir) if direct_test_dir is not None else "",
            "sample_submission_csv": str(sample_submission_path) if sample_submission_path is not None else "",
        },
        "created_at": _now_stamp(),
    }

    summary_path = paths["reports_dir"] / "preprocess_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


class ImageManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        image_transform: transforms.Compose,
        has_labels: bool,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest).reset_index(drop=True)
        else:
            self.df = manifest.reset_index(drop=True)
        self.image_transform = image_transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image_path = Path(row["filepath"])
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to open image: {image_path}") from exc

        item: Dict[str, Any] = {
            "images": self.image_transform(image),
            "id": str(row.get("id", image_path.stem)),
            "filepath": str(image_path),
        }
        if self.has_labels:
            item["labels"] = float(row["label"])
        return item


def _collate_image_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([sample["images"] for sample in batch], dim=0)
    output: Dict[str, Any] = {
        "images": images,
        "ids": [sample["id"] for sample in batch],
        "filepaths": [sample["filepath"] for sample in batch],
    }
    if "labels" in batch[0]:
        output["labels"] = torch.tensor([sample["labels"] for sample in batch], dtype=torch.float32)
    return output


def _build_transforms(exp_cfg: JSONDict) -> Tuple[transforms.Compose, transforms.Compose]:
    data_cfg = exp_cfg.get("data", {})
    aug_cfg = exp_cfg.get("augmentation", {})

    image_size = int(data_cfg.get("image_size", 224))
    train_aug = aug_cfg.get("train", {})
    eval_aug = aug_cfg.get("eval", {})
    norm_cfg = aug_cfg.get("normalize", {})

    crop_scale = train_aug.get("random_resized_crop_scale", [0.7, 1.0])
    color_jitter = train_aug.get("color_jitter", [0.15, 0.15, 0.15, 0.05])
    flip_p = float(train_aug.get("horizontal_flip_p", 0.5))
    random_erasing_p = float(train_aug.get("random_erasing_p", 0.1))

    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    eval_resize = int(eval_aug.get("resize", 256))
    eval_crop = int(eval_aug.get("center_crop", image_size))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(float(crop_scale[0]), float(crop_scale[1]))),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ColorJitter(
                brightness=float(color_jitter[0]),
                contrast=float(color_jitter[1]),
                saturation=float(color_jitter[2]),
                hue=float(color_jitter[3]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=random_erasing_p),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(eval_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def _resolve_backbone_layers(backbone: nn.Module) -> List[nn.Module]:
    candidates: List[Any] = []
    if hasattr(backbone, "encoder"):
        encoder = getattr(backbone, "encoder")
        for attr_name in ("layer", "layers", "block", "blocks"):
            if hasattr(encoder, attr_name):
                candidates.append(getattr(encoder, attr_name))
    for attr_name in ("layer", "layers", "block", "blocks"):
        if hasattr(backbone, attr_name):
            candidates.append(getattr(backbone, attr_name))

    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return list(candidate)
    return []


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _load_backbone(
    backbone_name: str,
    weights_source: str,
    local_weights_path: str,
    random_init: bool,
) -> nn.Module:
    trust_remote_code = False
    if random_init:
        cfg = AutoConfig.from_pretrained(backbone_name, trust_remote_code=trust_remote_code)
        return AutoModel.from_config(cfg, trust_remote_code=trust_remote_code)

    if weights_source == "local_only":
        if not local_weights_path:
            raise ValueError("weights_source='local_only' requires local_weights_path")
        return AutoModel.from_pretrained(local_weights_path, local_files_only=True, trust_remote_code=trust_remote_code)

    try:
        return AutoModel.from_pretrained(backbone_name, trust_remote_code=trust_remote_code)
    except Exception as first_exc:
        if local_weights_path:
            return AutoModel.from_pretrained(
                local_weights_path,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        raise RuntimeError(
            "Failed to auto-download backbone weights and no local fallback path was provided"
        ) from first_exc


class DinoBinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embed_dim: Optional[int],
        head_dropout: float,
        weights_source: str,
        local_weights_path: str,
        random_init: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = _load_backbone(
            backbone_name=backbone_name,
            weights_source=weights_source,
            local_weights_path=local_weights_path,
            random_init=random_init,
        )

        hidden_size = embed_dim
        if hidden_size is None:
            hidden_size = int(
                getattr(self.backbone.config, "hidden_size", 0)
                or getattr(self.backbone.config, "embed_dim", 0)
            )
        if not hidden_size:
            raise RuntimeError("Could not infer embed_dim from backbone; set model.embed_dim in config")

        self.head_norm = nn.LayerNorm(hidden_size)
        self.head_dropout = nn.Dropout(p=float(head_dropout))
        self.head_fc = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled: torch.Tensor
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            pooled = outputs[0][:, 0, :]

        logits = self.head_fc(self.head_dropout(self.head_norm(pooled))).squeeze(-1)
        return logits


def apply_freeze_policy(model: nn.Module, exp_cfg: JSONDict) -> None:
    base_model = _unwrap_model(model)
    freeze_cfg = exp_cfg.get("freeze", {})

    freeze_backbone = bool(freeze_cfg.get("freeze_backbone", False))
    trainable_last_n = int(freeze_cfg.get("trainable_last_n_blocks", -1))

    for param in base_model.backbone.parameters():
        param.requires_grad = True

    if freeze_backbone:
        for param in base_model.backbone.parameters():
            param.requires_grad = False
    elif trainable_last_n > 0:
        for param in base_model.backbone.parameters():
            param.requires_grad = False
        layers = _resolve_backbone_layers(base_model.backbone)
        if layers:
            for layer in layers[-trainable_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        for maybe_norm in ("layernorm", "norm", "post_layernorm"):
            if hasattr(base_model.backbone, maybe_norm):
                for param in getattr(base_model.backbone, maybe_norm).parameters():
                    param.requires_grad = True

    for module in (base_model.head_norm, base_model.head_fc):
        for param in module.parameters():
            param.requires_grad = True


def _get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    base_model = _unwrap_model(model)
    return [param for param in base_model.parameters() if param.requires_grad]


def _build_optimizer(model: nn.Module, exp_cfg: JSONDict) -> torch.optim.Optimizer:
    base_model = _unwrap_model(model)
    opt_cfg = exp_cfg.get("optimization", {})

    lr_backbone = float(opt_cfg.get("lr_backbone", 1e-5))
    lr_head = float(opt_cfg.get("lr_head", 3e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    betas = tuple(float(v) for v in opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

    backbone_params = [p for p in base_model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in list(base_model.head_norm.parameters()) + list(base_model.head_fc.parameters()) if p.requires_grad]

    groups: List[Dict[str, Any]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})

    if not groups:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")

    return torch.optim.AdamW(groups, weight_decay=weight_decay, betas=betas, eps=eps)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_ratio: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_updates = max(1, int(total_updates))
    warmup_updates = max(1, int(total_updates * warmup_ratio))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    lambdas = []
    for base_lr in base_lrs:
        min_ratio = float(min_lr) / base_lr if base_lr > 0 else 0.0
        min_ratio = float(min(max(min_ratio, 0.0), 1.0))

        def _lambda(step: int, warmup: int = warmup_updates, total: int = total_updates, ratio: float = min_ratio) -> float:
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return ratio + (1.0 - ratio) * cosine

        lambdas.append(_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def _resolve_device_and_parallel(exp_cfg: JSONDict) -> Tuple[torch.device, List[int]]:
    use_dp = bool(exp_cfg.get("use_data_parallel", False))
    requested_ids = [int(gid) for gid in exp_cfg.get("parallel_gpu_ids", [0, 1])]

    if torch.cuda.is_available():
        available = list(range(torch.cuda.device_count()))
        valid = [gid for gid in requested_ids if gid in available]
        if not valid:
            valid = available

        if use_dp and len(valid) >= 2:
            primary = valid[0]
            return torch.device(f"cuda:{primary}"), valid

        return torch.device(f"cuda:{valid[0]}"), [valid[0]]

    return torch.device("cpu"), []


def _resolve_amp_dtype(exp_cfg: JSONDict) -> Optional[torch.dtype]:
    amp_name = str(exp_cfg.get("amp_dtype", "fp16")).lower()
    if amp_name == "fp16":
        return torch.float16
    if amp_name == "bf16":
        return torch.bfloat16
    return None


def _is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _grad_norm_l2(model: nn.Module) -> float:
    base_model = _unwrap_model(model)
    total = 0.0
    for param in base_model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(torch.sum(grad * grad).item())
    return float(math.sqrt(total))


def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    return {
        "val_auc": auc,
        "val_logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    criterion: Optional[nn.Module] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    losses: List[float] = []
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    ids_all: List[str] = []
    paths_all: List[str] = []
    logits_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            labels = batch.get("labels")
            labels_t: Optional[torch.Tensor] = None
            if labels is not None:
                labels_t = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=bool(use_amp and device.type == "cuda" and amp_dtype is not None),
            ):
                logits = model(images)
                if criterion is not None and labels_t is not None:
                    loss = criterion(logits, labels_t)
                    losses.append(float(loss.detach().item()))

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(probs)
            logits_all.append(logits.detach().cpu().numpy())
            ids_all.extend(batch["ids"])
            paths_all.extend(batch["filepaths"])
            if labels is not None:
                labels_all.append(labels.numpy())

    if probs_all:
        y_prob = np.concatenate(probs_all, axis=0)
        y_logit = np.concatenate(logits_all, axis=0)
    else:
        y_prob = np.array([], dtype=float)
        y_logit = np.array([], dtype=float)

    pred_df = pd.DataFrame(
        {
            "id": ids_all,
            "filepath": paths_all,
            "logit": y_logit,
            "p_dog": y_prob,
            "p_cat": 1.0 - y_prob,
            "pred_label": (y_prob >= threshold).astype(int),
        }
    )

    metrics: Dict[str, float] = {}
    if labels_all:
        y_true = np.concatenate(labels_all, axis=0).astype(int)
        metrics.update(_compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold))
        metrics["val_loss"] = float(np.mean(losses)) if losses else float("nan")
        pred_df["label_true"] = y_true

    return metrics, pred_df


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: Path,
    meta: Dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "meta": meta,
        },
        checkpoint_path,
    )


def _load_model_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    _unwrap_model(model).load_state_dict(payload["model_state_dict"])


def _save_basic_plots(
    train_hist: pd.DataFrame,
    eval_hist: pd.DataFrame,
    val_pred_df: pd.DataFrame,
    plots_dir: Path,
    exp_name: str,
) -> Dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}

    if not train_hist.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(train_hist["global_step"], train_hist["train_loss"], label="train_loss")
        ax.set_title("Training Loss")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        path = plots_dir / f"{exp_name}_train_loss.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["train_loss_plot"] = str(path)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(train_hist["global_step"], train_hist["grad_norm"], label="grad_norm")
        ax.set_title("Gradient Norm")
        ax.set_xlabel("Global step")
        ax.set_ylabel("L2 norm")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        path = plots_dir / f"{exp_name}_grad_norm.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["grad_norm_plot"] = str(path)

    if not eval_hist.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(eval_hist["global_step"], eval_hist["val_auc"], marker="o")
        axes[0].set_title("Validation AUC")
        axes[0].set_xlabel("Global step")
        axes[0].set_ylabel("AUC")
        axes[0].grid(alpha=0.25)

        axes[1].plot(eval_hist["global_step"], eval_hist["val_logloss"], marker="o", color="tab:red")
        axes[1].set_title("Validation Logloss")
        axes[1].set_xlabel("Global step")
        axes[1].set_ylabel("Logloss")
        axes[1].grid(alpha=0.25)

        path = plots_dir / f"{exp_name}_val_metrics.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["val_metrics_plot"] = str(path)

    if "label_true" in val_pred_df.columns and len(val_pred_df) > 0:
        cm = confusion_matrix(val_pred_df["label_true"].astype(int), val_pred_df["pred_label"].astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1], labels=["cat", "dog"])
        ax.set_yticks([0, 1], labels=["cat", "dog"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Validation Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
        path = plots_dir / f"{exp_name}_val_confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["val_confusion_matrix_plot"] = str(path)

    return out


def _prepare_dataloaders(paths: Dict[str, Path], exp_cfg: JSONDict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_manifest = paths["manifests_dir"] / "train_manifest.csv"
    val_manifest = paths["manifests_dir"] / "val_manifest.csv"
    test_manifest = paths["manifests_dir"] / "test_manifest.csv"

    if not (train_manifest.exists() and val_manifest.exists() and test_manifest.exists()):
        raise FileNotFoundError(
            "Manifest files are missing. Run preprocess first to create train/val/test manifests."
        )

    data_cfg = exp_cfg.get("data", {})
    batch_train = int(data_cfg.get("batch_size_train", 8))
    batch_eval = int(data_cfg.get("batch_size_eval", 16))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    train_tf, eval_tf = _build_transforms(exp_cfg)

    ds_train = ImageManifestDataset(train_manifest, image_transform=train_tf, has_labels=True)
    ds_val = ImageManifestDataset(val_manifest, image_transform=eval_tf, has_labels=True)
    ds_test = ImageManifestDataset(test_manifest, image_transform=eval_tf, has_labels=False)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_image_batch,
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_image_batch,
        drop_last=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_image_batch,
        drop_last=False,
    )
    return dl_train, dl_val, dl_test


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    exp_cfg: JSONDict,
    checkpoints_dir: Path,
) -> Dict[str, Any]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = _build_optimizer(model, exp_cfg)

    training_cfg = exp_cfg.get("training", {})
    sched_cfg = exp_cfg.get("scheduler", {})
    eval_cfg = exp_cfg.get("evaluation", {})
    opt_cfg = exp_cfg.get("optimization", {})

    epochs = int(training_cfg.get("epochs", 1))
    grad_accum = max(1, int(training_cfg.get("grad_accum_steps", 1)))
    log_every = max(1, int(training_cfg.get("log_every_steps", 20)))
    eval_every = max(1, int(training_cfg.get("eval_every_steps", 200)))
    mixed_precision = bool(training_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(exp_cfg)
    use_scaler = bool(mixed_precision and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    threshold = float(eval_cfg.get("decision_threshold", 0.5))
    monitor = str(eval_cfg.get("monitor", "val_auc"))
    grad_clip = float(opt_cfg.get("grad_clip_norm", 1.0))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = max(1, updates_per_epoch * epochs)
    scheduler = _build_scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_ratio=float(sched_cfg.get("warmup_ratio", 0.1)),
        min_lr=float(sched_cfg.get("min_lr", 1e-6)),
    )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / "best.pt"
    last_ckpt_path = checkpoints_dir / "last.pt"

    best_score = -float("inf")
    best_epoch = -1
    no_improve_count = 0
    global_step = 0

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    running_loss = 0.0
    running_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=bool(mixed_precision and amp_dtype is not None),
            ):
                logits = model(images)
                loss = criterion(logits, labels)
                scaled_loss = loss / grad_accum

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += float(loss.detach().item())
            running_count += 1

            if (batch_idx % grad_accum == 0) or (batch_idx == len(train_loader)):
                if use_scaler:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(_get_trainable_params(model), grad_clip)
                grad_norm = _grad_norm_l2(model)

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                current_lr = max(float(group["lr"]) for group in optimizer.param_groups)

                if global_step % log_every == 0:
                    avg_loss = running_loss / max(1, running_count)
                    train_rows.append(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": float(avg_loss),
                            "grad_norm": float(grad_norm),
                            "lr": float(current_lr),
                        }
                    )
                    print(
                        f"[train] epoch={epoch} step={global_step} "
                        f"loss={avg_loss:.5f} grad_norm={grad_norm:.4f} lr={current_lr:.3e}"
                    )
                    running_loss = 0.0
                    running_count = 0

                if global_step % eval_every == 0:
                    metrics, _ = evaluate_model(
                        model=model,
                        loader=val_loader,
                        device=device,
                        threshold=threshold,
                        use_amp=mixed_precision,
                        amp_dtype=amp_dtype,
                        criterion=criterion,
                    )
                    eval_row = {"epoch": epoch, "global_step": global_step, **metrics}
                    eval_rows.append(eval_row)
                    print(
                        f"[eval] epoch={epoch} step={global_step} "
                        f"val_loss={metrics.get('val_loss', float('nan')):.5f} "
                        f"val_auc={metrics.get('val_auc', float('nan')):.5f} "
                        f"val_logloss={metrics.get('val_logloss', float('nan')):.5f} "
                        f"val_accuracy={metrics.get('val_accuracy', float('nan')):.5f}"
                    )

        epoch_metrics, _ = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            threshold=threshold,
            use_amp=mixed_precision,
            amp_dtype=amp_dtype,
            criterion=criterion,
        )
        eval_rows.append({"epoch": epoch, "global_step": global_step, "event": "epoch_end", **epoch_metrics})

        monitor_value = float(epoch_metrics.get(monitor, float("nan")))
        if math.isnan(monitor_value):
            is_better = False
        elif monitor == "val_logloss":
            is_better = (best_epoch < 0) or (monitor_value < best_score)
        else:
            is_better = (best_epoch < 0) or (monitor_value > best_score)

        if is_better:
            best_score = monitor_value
            best_epoch = epoch
            no_improve_count = 0
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=best_ckpt_path,
                meta={
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_score": best_score,
                    "monitor": monitor,
                },
            )
            print(f"[checkpoint] new best at epoch={epoch} {monitor}={monitor_value:.6f}")
        else:
            no_improve_count += 1

        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=last_ckpt_path,
            meta={
                "epoch": epoch,
                "global_step": global_step,
            },
        )

        if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
            print(
                f"[early-stop] patience reached ({early_stopping_patience}). "
                f"Stopping at epoch={epoch}."
            )
            break

    if not best_ckpt_path.exists():
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=best_ckpt_path,
            meta={"epoch": best_epoch, "global_step": global_step, "best_score": best_score, "monitor": monitor},
        )

    return {
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "train_history": train_rows,
        "eval_history": eval_rows,
        "best_monitor": best_score,
        "best_epoch": best_epoch,
        "monitor": monitor,
    }


def run_training(paths_cfg_path: Union[str, Path], exp_cfg_path: Union[str, Path]) -> Dict[str, Any]:
    paths_cfg = _read_json(paths_cfg_path)
    exp_cfg = _read_json(exp_cfg_path)

    paths = resolve_paths(paths_cfg)
    exp_name = str(exp_cfg.get("experiment_name", "dinov2_experiment"))
    set_seed(int(exp_cfg.get("seed", 42)))

    train_loader, val_loader, test_loader = _prepare_dataloaders(paths=paths, exp_cfg=exp_cfg)

    model_cfg = exp_cfg.get("model", {})
    backbone_name = str(model_cfg.get("backbone_name", "facebook/dinov2-base"))

    model = DinoBinaryClassifier(
        backbone_name=backbone_name,
        embed_dim=model_cfg.get("embed_dim"),
        head_dropout=float(model_cfg.get("head_dropout", 0.0)),
        weights_source=str(exp_cfg.get("weights_source", "auto_download_with_fallback")),
        local_weights_path=str(exp_cfg.get("local_weights_path", "")),
        random_init=False,
    )

    gradient_checkpointing = bool(exp_cfg.get("gradient_checkpointing", False))
    if gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    device, dp_ids = _resolve_device_and_parallel(exp_cfg)
    model.to(device)
    apply_freeze_policy(model, exp_cfg)

    if device.type == "cuda":
        print(f"Using CUDA device: {device}")
        for gid in dp_ids:
            free_b, total_b = torch.cuda.mem_get_info(gid)
            print(
                f"[cuda:{gid}] free={free_b/(1024**3):.2f}GB total={total_b/(1024**3):.2f}GB"
            )

    if bool(exp_cfg.get("use_data_parallel", False)) and len(dp_ids) >= 2 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=dp_ids, output_device=dp_ids[0])
        print(f"DataParallel enabled on GPUs: {dp_ids}")

    run_dir_ckpt = paths["checkpoints_dir"] / exp_name
    train_state = _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        exp_cfg=exp_cfg,
        checkpoints_dir=run_dir_ckpt,
    )

    best_ckpt = Path(train_state["best_checkpoint"])
    _load_model_state(model, best_ckpt, device=device)

    threshold = float(exp_cfg.get("evaluation", {}).get("decision_threshold", 0.5))
    mixed_precision = bool(exp_cfg.get("training", {}).get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(exp_cfg)

    criterion = nn.BCEWithLogitsLoss()
    val_metrics, val_pred_df = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        threshold=threshold,
        use_amp=mixed_precision,
        amp_dtype=amp_dtype,
        criterion=criterion,
    )
    _, test_pred_df = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        threshold=threshold,
        use_amp=mixed_precision,
        amp_dtype=amp_dtype,
        criterion=None,
    )

    train_hist_df = pd.DataFrame(train_state["train_history"])
    eval_hist_df = pd.DataFrame(train_state["eval_history"])

    metrics_csv = paths["metrics_dir"] / f"{exp_name}_metrics.csv"
    train_hist_csv = paths["metrics_dir"] / f"{exp_name}_train_history.csv"
    eval_hist_csv = paths["metrics_dir"] / f"{exp_name}_eval_history.csv"

    pd.DataFrame([val_metrics]).to_csv(metrics_csv, index=False)
    train_hist_df.to_csv(train_hist_csv, index=False)
    eval_hist_df.to_csv(eval_hist_csv, index=False)

    val_pred_csv = paths["predictions_dir"] / f"{exp_name}_val_predictions.csv"
    test_pred_csv = paths["predictions_dir"] / f"{exp_name}_test_predictions.csv"
    submission_csv = paths["predictions_dir"] / f"{exp_name}_submission.csv"

    val_pred_df.to_csv(val_pred_csv, index=False)
    test_pred_df.to_csv(test_pred_csv, index=False)

    if paths["sample_submission_csv"].is_file():
        sample = pd.read_csv(paths["sample_submission_csv"])
        merged = sample[["id"]].copy()
        test_tmp = test_pred_df[["id", "p_dog"]].rename(columns={"p_dog": "label"}).copy()
        merged["id"] = merged["id"].astype(str)
        test_tmp["id"] = test_tmp["id"].astype(str)
        merged = merged.merge(test_tmp, how="left", on="id")
        merged["label"] = merged["label"].fillna(0.5)
        merged.to_csv(submission_csv, index=False)
    else:
        sub = test_pred_df[["id", "p_dog"]].rename(columns={"p_dog": "label"}).copy()
        sub.to_csv(submission_csv, index=False)

    plot_paths = _save_basic_plots(
        train_hist=train_hist_df,
        eval_hist=eval_hist_df,
        val_pred_df=val_pred_df,
        plots_dir=paths["plots_dir"],
        exp_name=exp_name,
    )

    summary: Dict[str, Any] = {
        "experiment_name": exp_name,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": train_state["last_checkpoint"],
        "monitor": train_state["monitor"],
        "best_monitor": train_state["best_monitor"],
        "best_epoch": train_state["best_epoch"],
        "final_val_metrics": val_metrics,
        "files": {
            "metrics_csv": str(metrics_csv),
            "train_history_csv": str(train_hist_csv),
            "eval_history_csv": str(eval_hist_csv),
            "val_predictions_csv": str(val_pred_csv),
            "test_predictions_csv": str(test_pred_csv),
            "submission_csv": str(submission_csv),
            **plot_paths,
        },
        "created_at": _now_stamp(),
    }

    summary_path = paths["reports_dir"] / f"{exp_name}_training_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def run_sanity_check(paths_cfg_path: Union[str, Path], exp_cfg_path: Union[str, Path]) -> Dict[str, Any]:
    paths_cfg = _read_json(paths_cfg_path)
    exp_cfg = _read_json(exp_cfg_path)
    paths = resolve_paths(paths_cfg)

    set_seed(int(exp_cfg.get("seed", 42)))
    train_manifest = paths["manifests_dir"] / "train_manifest.csv"
    val_manifest = paths["manifests_dir"] / "val_manifest.csv"

    if not (train_manifest.exists() and val_manifest.exists()):
        raise FileNotFoundError("Sanity check requires train/val manifests. Run preprocessing first.")

    df_train = pd.read_csv(train_manifest)
    df_val = pd.read_csv(val_manifest)

    sanity_cfg = exp_cfg.get("sanity", {})
    train_samples = int(sanity_cfg.get("train_samples", 96))
    val_samples = int(sanity_cfg.get("val_samples", 96))
    smoke_steps = int(sanity_cfg.get("smoke_steps", 5))

    df_train_small = df_train.sample(n=min(train_samples, len(df_train)), random_state=int(exp_cfg.get("seed", 42)))
    df_val_small = df_val.sample(n=min(val_samples, len(df_val)), random_state=int(exp_cfg.get("seed", 42)) + 1)

    train_tf, eval_tf = _build_transforms(exp_cfg)

    ds_train = ImageManifestDataset(df_train_small, image_transform=train_tf, has_labels=True)
    ds_val = ImageManifestDataset(df_val_small, image_transform=eval_tf, has_labels=True)

    batch_size = max(1, int(sanity_cfg.get("batch_size", 8)))
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate_image_batch,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate_image_batch,
    )

    model_cfg = exp_cfg.get("model", {})
    model = DinoBinaryClassifier(
        backbone_name=str(model_cfg.get("backbone_name", "facebook/dinov2-base")),
        embed_dim=model_cfg.get("embed_dim"),
        head_dropout=float(model_cfg.get("head_dropout", 0.0)),
        weights_source=str(exp_cfg.get("weights_source", "auto_download_with_fallback")),
        local_weights_path=str(exp_cfg.get("local_weights_path", "")),
        random_init=True,
    )

    device, _ = _resolve_device_and_parallel({"use_data_parallel": False})
    model.to(device)
    apply_freeze_policy(model, exp_cfg)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = _build_optimizer(model, exp_cfg)

    threshold = float(exp_cfg.get("evaluation", {}).get("decision_threshold", 0.5))

    baseline_metrics, _ = evaluate_model(
        model=model,
        loader=dl_val,
        device=device,
        threshold=threshold,
        use_amp=False,
        amp_dtype=None,
        criterion=criterion,
    )

    first_param_before: Optional[torch.Tensor] = None
    for param in _get_trainable_params(model):
        first_param_before = param.detach().clone()
        break

    loss_values: List[float] = []
    grad_norm_values: List[float] = []

    model.train()
    for step_idx, batch in enumerate(dl_train, start=1):
        if step_idx > smoke_steps:
            break

        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        if not _is_finite_tensor(loss):
            break

        loss.backward()
        grad_norm = _grad_norm_l2(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_values.append(float(loss.detach().item()))
        grad_norm_values.append(float(grad_norm))

    final_metrics, _ = evaluate_model(
        model=model,
        loader=dl_val,
        device=device,
        threshold=threshold,
        use_amp=False,
        amp_dtype=None,
        criterion=criterion,
    )

    first_param_after: Optional[torch.Tensor] = None
    for param in _get_trainable_params(model):
        first_param_after = param.detach().clone()
        break

    parameter_delta = 0.0
    if first_param_before is not None and first_param_after is not None:
        parameter_delta = float(torch.mean(torch.abs(first_param_after - first_param_before)).item())

    checks = {
        "has_data_rows": bool(len(df_train_small) > 0 and len(df_val_small) > 0),
        "labels_binary": bool(df_train_small["label"].isin([0, 1]).all() and df_val_small["label"].isin([0, 1]).all()),
        "loss_values_finite": bool(len(loss_values) > 0 and np.isfinite(loss_values).all()),
        "grad_norms_finite": bool(len(grad_norm_values) > 0 and np.isfinite(grad_norm_values).all()),
        "parameter_delta_positive": bool(parameter_delta > 0.0),
        "baseline_metrics_in_range": bool(
            (0.0 <= baseline_metrics.get("val_accuracy", 0.0) <= 1.0)
            and (0.0 <= baseline_metrics.get("val_logloss", 1.0e6))
        ),
    }

    passed = bool(all(checks.values()))

    summary = {
        "passed": passed,
        "checks": checks,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "smoke_steps_executed": int(len(loss_values)),
        "smoke_loss_values": loss_values,
        "smoke_grad_norm_values": grad_norm_values,
        "parameter_delta": parameter_delta,
        "train_sample_count": int(len(df_train_small)),
        "val_sample_count": int(len(df_val_small)),
        "created_at": _now_stamp(),
    }

    report_path = paths["reports_dir"] / f"sanity_random_init_{exp_cfg.get('experiment_name', 'exp')}.json"
    _write_json(report_path, summary)
    summary["summary_path"] = str(report_path)
    return summary


def run_preprocess(paths_cfg_path: Union[str, Path], exp_cfg_path: Union[str, Path], force_extract: bool = False) -> Dict[str, Any]:
    paths_cfg = _read_json(paths_cfg_path)
    exp_cfg = _read_json(exp_cfg_path)
    return build_manifests(paths_cfg=paths_cfg, exp_cfg=exp_cfg, force_extract=force_extract)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="DinoV2 Dogs-vs-Cats utilities")
    parser.add_argument("--mode", choices=["preprocess", "sanity", "train"], required=True)
    parser.add_argument("--paths-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    if args.mode == "preprocess":
        result = run_preprocess(args.paths_config, args.experiment_config, force_extract=args.force_extract)
    elif args.mode == "sanity":
        result = run_sanity_check(args.paths_config, args.experiment_config)
    else:
        result = run_training(args.paths_config, args.experiment_config)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
