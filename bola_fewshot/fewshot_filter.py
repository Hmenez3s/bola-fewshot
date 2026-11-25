from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from bola_fewshot.detector import resolve_device
from bola_fewshot.metrics import precision_recall_f1

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
VALID_LAYERS = ("layer4", "layer3", "layer2")
VALID_MODES = ("positive_only", "pos_minus_neg")


def letterbox_crop(crop: Image.Image, canvas_wh: tuple[int, int] = (48, 96)) -> np.ndarray:
    rgb = crop.convert("RGB")
    target_w, target_h = canvas_wh
    scale = min(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
    new_w = max(1, int(round(rgb.width * scale)))
    new_h = max(1, int(round(rgb.height * scale)))
    resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            raw = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            raw = checkpoint["state_dict"]
        else:
            raw = checkpoint
    else:
        raise ValueError("FewShot-R18 checkpoint must be a mapping")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        name = str(key).removeprefix("module.")
        if name.startswith("fc."):
            continue
        cleaned[name] = value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
    return cleaned


class FewShotR18Encoder:
    def __init__(
        self,
        weights_path: Path,
        *,
        device: str = "auto",
        canvas_wh: tuple[int, int] = (48, 96),
        batch_size: int = 64,
    ) -> None:
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"FewShot-R18 weights not found: {self.weights_path}")
        self.device = torch.device(resolve_device(device))
        self.canvas_wh = (int(canvas_wh[0]), int(canvas_wh[1]))
        self.batch_size = max(1, int(batch_size))

        model = resnet18(weights=None)
        model.fc = nn.Identity()
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        model.eval()
        model.to(self.device)
        self.model = model
        self._mean = IMAGENET_MEAN.to(self.device)
        self._std = IMAGENET_STD.to(self.device)

    def _forward_layer(self, batch: torch.Tensor, layer: str) -> torch.Tensor:
        if layer not in VALID_LAYERS:
            raise ValueError(f"unsupported layer '{layer}', expected one of {VALID_LAYERS}")
        m = self.model
        x = m.conv1(batch)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        if layer == "layer2":
            return torch.flatten(m.avgpool(x), 1)
        x = m.layer3(x)
        if layer == "layer3":
            return torch.flatten(m.avgpool(x), 1)
        x = m.layer4(x)
        return torch.flatten(m.avgpool(x), 1)

    def embed_crops(
        self,
        crops: Sequence[Image.Image],
        *,
        layer: str = "layer4",
    ) -> np.ndarray:
        dim = {"layer2": 128, "layer3": 256, "layer4": 512}[layer]
        if not crops:
            return np.empty((0, dim), dtype=np.float32)
        arrays = [letterbox_crop(crop, self.canvas_wh) for crop in crops]
        stacked = np.stack(arrays, axis=0)
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(stacked), self.batch_size):
                chunk = stacked[start : start + self.batch_size]
                tensor = (
                    torch.from_numpy(chunk)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .to(self.device, dtype=torch.float32)
                    .div_(255.0)
                )
                tensor = (tensor - self._mean) / self._std
                feats = self._forward_layer(tensor, layer)
                feats = F.normalize(feats, dim=1)
                outputs.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0.0 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0.0)


@dataclass(frozen=True)
class PrototypeFilter:
    mode: str
    pos_prototype: np.ndarray
    neg_prototype: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        pos_embeddings: np.ndarray,
        neg_embeddings: np.ndarray | None = None,
        *,
        mode: str = "positive_only",
    ) -> "PrototypeFilter":
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported filter mode '{mode}', expected one of {VALID_MODES}")
        if len(pos_embeddings) == 0:
            raise ValueError("pos_embeddings must contain at least one reference")
        pos_proto = _l2_normalize_rows(_l2_normalize_rows(pos_embeddings).mean(axis=0))
        neg_proto = None
        if mode == "pos_minus_neg":
            if neg_embeddings is None or len(neg_embeddings) == 0:
                raise ValueError("neg_embeddings required for mode 'pos_minus_neg'")
            neg_proto = _l2_normalize_rows(_l2_normalize_rows(neg_embeddings).mean(axis=0))
        return cls(mode=mode, pos_prototype=pos_proto, neg_prototype=neg_proto)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.empty((0,), dtype=np.float64)
        normed = _l2_normalize_rows(embeddings).astype(np.float64)
        pos_sim = normed @ self.pos_prototype.astype(np.float64)
        if self.mode == "positive_only":
            return pos_sim
        assert self.neg_prototype is not None
        neg_sim = normed @ self.neg_prototype.astype(np.float64)
        return pos_sim - neg_sim


def calibrate_threshold(
    scores: Sequence[float] | np.ndarray,
    is_tp: Sequence[bool] | np.ndarray,
    total_gt: int,
) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    tp_flags = np.asarray(is_tp, dtype=bool)
    if len(s) != len(tp_flags):
        raise ValueError("scores and is_tp must have the same length")
    base_tp = int(tp_flags.sum())
    base_fp = int((~tp_flags).sum())
    base_p, base_r, base_f1 = precision_recall_f1(base_tp, base_fp, total_gt)

    if len(s) == 0:
        return {
            "threshold": -math.inf,
            "is_noop": True,
            "improved_on_calib": False,
            "calib_base_f1": base_f1,
            "calib_best_f1": base_f1,
            "calib_best_precision": base_p,
            "calib_best_recall": base_r,
            "n_candidates_checked": 1,
            "best_active_threshold": None,
            "best_active_f1": None,
        }

    unique_scores = np.unique(s)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0 if len(unique_scores) > 1 else np.empty((0,))
    candidates = [-math.inf, *midpoints.tolist(), float(unique_scores[-1]) + 1e-6]

    best_key = (base_f1, base_r, math.inf)
    best_thresh = -math.inf
    best_p, best_r, best_f1 = base_p, base_r, base_f1

    best_active_key = (-1.0, -1.0, -math.inf)
    best_active_thresh: float | None = None
    best_active_f1: float | None = None

    for thresh in candidates:
        kept = s >= thresh
        tp = int((kept & tp_flags).sum())
        fp = int((kept & (~tp_flags)).sum())
        p, r, f1 = precision_recall_f1(tp, fp, total_gt)
        key = (round(f1, 12), round(r, 12), -thresh)
        if key > best_key:
            best_key = key
            best_thresh = thresh
            best_p, best_r, best_f1 = p, r, f1
        if int((~kept).sum()) > 0:
            active_key = (round(f1, 12), round(r, 12), -thresh)
            if active_key > best_active_key:
                best_active_key = active_key
                best_active_thresh = thresh
                best_active_f1 = f1

    return {
        "threshold": float(best_thresh),
        "is_noop": math.isinf(best_thresh) and best_thresh < 0,
        "improved_on_calib": bool(best_f1 > base_f1 + 1e-12),
        "calib_base_f1": base_f1,
        "calib_best_f1": best_f1,
        "calib_best_precision": best_p,
        "calib_best_recall": best_r,
        "n_candidates_checked": len(candidates),
        "best_active_threshold": best_active_thresh,
        "best_active_f1": best_active_f1,
    }
