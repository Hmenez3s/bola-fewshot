from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from bola_fewshot.dataset import DatasetSplit, ImageRecord, crop_box
from bola_fewshot.detector import BallDetectorProtocol, Detection
from bola_fewshot.fewshot_filter import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PrototypeFilter,
    calibrate_threshold,
    letterbox_crop,
)
from bola_fewshot.metrics import box_iou_matrix, compute_pr_auc, compute_roc_auc, match_image_detections


@dataclass(frozen=True)
class BallTrainConfig:
    epochs: int = 8
    steps_per_epoch: int = 180
    references_per_class: int = 6
    queries_per_class: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    cosine_scale: float = 20.0
    seed: int = 25
    canvas_wh: tuple[int, int] = (48, 96)
    device: str = "auto"
    imagenet_weights_path: Path | None = None
    checkpoint_path: Path = Path("models/fewshot_r18_ball.pth")


@dataclass(frozen=True)
class MinedTrainingPool:
    pos_arrays: np.ndarray
    pos_image_ids: tuple[str, ...]
    neg_arrays: np.ndarray
    neg_image_ids: tuple[str, ...]
    stats: dict


def build_trainable_resnet18(
    device: torch.device,
    imagenet_weights_path: Path | None = None,
) -> nn.Module:
    if imagenet_weights_path is not None and Path(imagenet_weights_path).is_file():
        model = resnet18(weights=None)
        state = torch.load(imagenet_weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("ImageNet checkpoint must contain a state dict")
        cleaned = {str(k).removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=True)
    else:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(("layer3.", "layer4.")))
    model.to(device)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model


def _sample_background_boxes(
    width: int,
    height: int,
    gt_boxes: Sequence[tuple[float, float, float, float]],
    n_needed: int,
    rng: random.Random,
) -> list[tuple[float, float, float, float]]:
    gt_arr = np.asarray(gt_boxes, dtype=np.float64).reshape(len(gt_boxes), 4) if gt_boxes else np.empty((0, 4))
    results: list[tuple[float, float, float, float]] = []
    attempts = 0
    max_attempts = max(20, n_needed * 25)
    while len(results) < n_needed and attempts < max_attempts:
        attempts += 1
        side = rng.randint(12, max(14, min(36, min(width, height) // 4)))
        if width <= side or height <= side:
            break
        x1 = float(rng.randint(0, width - side))
        y1 = float(rng.randint(0, height - side))
        cand = (x1, y1, x1 + side, y1 + side)
        if len(gt_arr) > 0:
            ious = box_iou_matrix(np.asarray([cand]), gt_arr)
            if float(ious.max()) > 0.0:
                continue
        results.append(cand)
    return results


def mine_training_crop_pool(
    train_split: DatasetSplit,
    mining_detector: BallDetectorProtocol,
    *,
    eval_conf: float = 0.35,
    mining_conf: float = 0.05,
    eval_iou: float = 0.50,
    max_images: int = 1500,
    min_negatives: int = 200,
    max_pos_crops: int = 3000,
    max_neg_crops: int = 3000,
    seed: int = 25,
    canvas_wh: tuple[int, int] = (48, 96),
    context_scale: float = 1.0,
    square_crop: bool = False,
    cache_path: Path | None = None,
) -> MinedTrainingPool:
    ordered = sorted(train_split.records, key=lambda r: r.image_id)
    rng = random.Random(seed + 11)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    chosen_records = tuple(ordered[i] for i in sorted(indices[: min(max_images, len(ordered))]))
    dets = mining_detector.predict_records(chosen_records, cache_path=cache_path)

    pos_items: list[tuple[str, np.ndarray]] = []
    neg_high_items: list[tuple[str, np.ndarray]] = []
    neg_low_items: list[tuple[str, np.ndarray]] = []

    high_conf_tp = 0
    high_conf_fp = 0
    low_conf_tp = 0
    low_conf_fp = 0
    gt_ball_count = 0

    for record in chosen_records:
        gt_ball_count += len(record.gt_boxes)
        img_dets = dets.get(record.image_id, ())
        high_dets = [d for d in img_dets if d.conf >= eval_conf]
        if high_dets:
            high_match = match_image_detections(
                [d.box for d in high_dets],
                [d.conf for d in high_dets],
                record.gt_boxes,
                iou_threshold=eval_iou,
            )
            high_conf_tp += int(high_match.sum())
            high_conf_fp += int((~high_match).sum())

        all_match = (
            match_image_detections(
                [d.box for d in img_dets],
                [d.conf for d in img_dets],
                record.gt_boxes,
                iou_threshold=eval_iou,
            )
            if img_dets
            else np.zeros(0, dtype=bool)
        )
        tp_boxes: list[tuple[float, float, float, float]] = []
        fp_high_boxes: list[tuple[float, float, float, float]] = []
        fp_low_boxes: list[tuple[float, float, float, float]] = []
        for det, is_tp in zip(img_dets, all_match):
            if is_tp:
                tp_boxes.append(det.box)
                if det.conf < eval_conf:
                    low_conf_tp += 1
            else:
                if det.conf >= eval_conf:
                    fp_high_boxes.append(det.box)
                elif det.conf >= mining_conf:
                    fp_low_boxes.append(det.box)
                    low_conf_fp += 1

        if not tp_boxes and record.gt_boxes:
            tp_boxes.extend(record.gt_boxes)

        if not tp_boxes and not fp_high_boxes and not fp_low_boxes:
            continue

        with Image.open(record.image_path) as img:
            rgb = img.convert("RGB")
            for box in tp_boxes:
                crop = crop_box(rgb, box, context_scale=context_scale, square=square_crop)
                pos_items.append((record.image_id, letterbox_crop(crop, canvas_wh)))
            for box in fp_high_boxes:
                crop = crop_box(rgb, box, context_scale=context_scale, square=square_crop)
                neg_high_items.append((record.image_id, letterbox_crop(crop, canvas_wh)))
            for box in fp_low_boxes:
                crop = crop_box(rgb, box, context_scale=context_scale, square=square_crop)
                neg_low_items.append((record.image_id, letterbox_crop(crop, canvas_wh)))

    neg_items = list(neg_high_items) + list(neg_low_items)
    bg_supplement_count = 0
    if len(neg_items) < min_negatives:
        needed = min_negatives - len(neg_items)
        per_img = max(1, math.ceil(needed / max(1, len(chosen_records))))
        bg_rng = random.Random(seed + 19)
        for record in chosen_records:
            if len(neg_items) >= min_negatives:
                break
            boxes = _sample_background_boxes(
                record.width,
                record.height,
                record.gt_boxes,
                min(per_img, min_negatives - len(neg_items)),
                bg_rng,
            )
            if not boxes:
                continue
            with Image.open(record.image_path) as img:
                rgb = img.convert("RGB")
                for box in boxes:
                    crop = crop_box(rgb, box, context_scale=context_scale, square=square_crop)
                    neg_items.append((record.image_id, letterbox_crop(crop, canvas_wh)))
                    bg_supplement_count += 1

    cap_rng = random.Random(seed + 23)
    if len(pos_items) > max_pos_crops:
        cap_rng.shuffle(pos_items)
        pos_items = pos_items[:max_pos_crops]
    if len(neg_items) > max_neg_crops:
        cap_rng.shuffle(neg_items)
        neg_items = neg_items[:max_neg_crops]

    if not pos_items:
        raise ValueError("mined training pool has zero positive ball crops")
    if not neg_items:
        raise ValueError("mined training pool has zero negative crops")

    pos_arrays = np.stack([item[1] for item in pos_items], axis=0).astype(np.uint8)
    neg_arrays = np.stack([item[1] for item in neg_items], axis=0).astype(np.uint8)
    n_scanned = len(chosen_records)
    stats = {
        "train_images_scanned": n_scanned,
        "train_gt_balls_in_scanned": gt_ball_count,
        "eval_conf": float(eval_conf),
        "mining_conf": float(mining_conf),
        "train_tp_at_eval_conf": high_conf_tp,
        "train_fp_at_eval_conf": high_conf_fp,
        "train_fp_per_image_at_eval_conf": float(high_conf_fp / n_scanned) if n_scanned > 0 else 0.0,
        "train_precision_at_eval_conf": (
            float(high_conf_tp / (high_conf_tp + high_conf_fp)) if (high_conf_tp + high_conf_fp) > 0 else 0.0
        ),
        "train_fp_low_conf_mined": len(neg_low_items),
        "train_fp_high_conf_mined": len(neg_high_items),
        "train_tp_low_conf": low_conf_tp,
        "background_supplement_crops": bg_supplement_count,
        "pool_pos_crops": int(len(pos_arrays)),
        "pool_neg_crops": int(len(neg_arrays)),
    }
    return MinedTrainingPool(
        pos_arrays=pos_arrays,
        pos_image_ids=tuple(item[0] for item in pos_items),
        neg_arrays=neg_arrays,
        neg_image_ids=tuple(item[0] for item in neg_items),
        stats=stats,
    )


def _arrays_to_tensor(
    arrays: np.ndarray,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.ascontiguousarray(arrays))
        .permute(0, 3, 1, 2)
        .to(device, dtype=torch.float32)
        .div_(255.0)
    )
    return (tensor - mean) / std


def _embed_uint8_arrays(
    model: nn.Module,
    arrays: np.ndarray,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 64,
) -> np.ndarray:
    if len(arrays) == 0:
        return np.empty((0, 512), dtype=np.float32)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(arrays), batch_size):
            chunk = arrays[start : start + batch_size]
            feats = model(_arrays_to_tensor(chunk, device, mean, std))
            feats = F.normalize(feats, dim=1)
            outputs.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _sample_episode_indices(
    n_items: int,
    unit_ids: Sequence[str],
    k_support: int,
    n_query: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    all_idx = list(range(n_items))
    if n_items >= k_support + n_query:
        support = rng.sample(all_idx, k_support)
        support_units = {unit_ids[i] for i in support}
        disjoint_pool = [i for i in all_idx if i not in set(support) and unit_ids[i] not in support_units]
        if len(disjoint_pool) >= n_query:
            query = rng.sample(disjoint_pool, n_query)
        else:
            remaining = [i for i in all_idx if i not in set(support)]
            query = rng.sample(remaining, n_query)
        return support, query
    support = [rng.choice(all_idx) for _ in range(k_support)]
    query = [rng.choice(all_idx) for _ in range(n_query)]
    return support, query


def train_fewshot_ball(
    pool: MinedTrainingPool,
    *,
    val_calib_crops: Sequence[Image.Image],
    val_calib_is_tp: np.ndarray,
    val_calib_total_gt: int,
    fixed_pos_ref_crops: Sequence[Image.Image],
    fixed_neg_ref_crops: Sequence[Image.Image],
    config: BallTrainConfig,
    group_by: Mapping[str, str] | None = None,
) -> dict:
    from bola_fewshot.detector import resolve_device

    device = torch.device(resolve_device(config.device))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = build_trainable_resnet18(device, config.imagenet_weights_path)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * config.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    val_arrays = (
        np.stack([letterbox_crop(c, config.canvas_wh) for c in val_calib_crops], axis=0).astype(np.uint8)
        if val_calib_crops
        else np.empty((0, config.canvas_wh[1], config.canvas_wh[0], 3), dtype=np.uint8)
    )
    ref_pos_arrays = np.stack([letterbox_crop(c, config.canvas_wh) for c in fixed_pos_ref_crops], axis=0).astype(np.uint8)
    ref_neg_arrays = np.stack([letterbox_crop(c, config.canvas_wh) for c in fixed_neg_ref_crops], axis=0).astype(np.uint8)

    rng = random.Random(config.seed)
    k_sup = config.references_per_class
    n_qry = config.queries_per_class
    pos_units = (
        tuple(group_by.get(i, i) for i in pool.pos_image_ids)
        if group_by is not None
        else pool.pos_image_ids
    )
    neg_units = (
        tuple(group_by.get(i, i) for i in pool.neg_image_ids)
        if group_by is not None
        else pool.neg_image_ids
    )

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_key: tuple[float, float] | None = None
    best_epoch = 1
    first_episode_s = 0.0
    train_start = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        losses: list[float] = []
        ep_t0 = time.monotonic()
        for step in range(config.steps_per_epoch):
            step_t0 = time.monotonic()
            pos_sup, pos_qry = _sample_episode_indices(
                len(pool.pos_arrays), pos_units, k_sup, n_qry, rng
            )
            neg_sup, neg_qry = _sample_episode_indices(
                len(pool.neg_arrays), neg_units, k_sup, n_qry, rng
            )
            sup_arr = np.concatenate([pool.neg_arrays[neg_sup], pool.pos_arrays[pos_sup]], axis=0)
            qry_arr = np.concatenate([pool.neg_arrays[neg_qry], pool.pos_arrays[pos_qry]], axis=0)
            qry_labels = torch.tensor([0] * len(neg_qry) + [1] * len(pos_qry), dtype=torch.long, device=device)

            all_feats = model(
                _arrays_to_tensor(np.concatenate([sup_arr, qry_arr], axis=0), device, mean, std)
            )
            sup_feats = F.normalize(all_feats[: 2 * k_sup], dim=1)
            qry_feats = F.normalize(all_feats[2 * k_sup :], dim=1)
            proto_neg = sup_feats[:k_sup].mean(dim=0)
            proto_pos = sup_feats[k_sup:].mean(dim=0)
            prototypes = F.normalize(torch.stack([proto_neg, proto_pos], dim=0), dim=1)

            logits = (qry_feats @ prototypes.T) * config.cosine_scale
            loss = F.cross_entropy(logits, qry_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if epoch == 1 and step == 0:
                if device.type == "mps":
                    torch.mps.synchronize()
                first_episode_s = time.monotonic() - step_t0
                print(
                    f"train_fewshot_ball[{device}]: first episode={first_episode_s:.3f}s "
                    f"(est {total_steps} steps = {first_episode_s * total_steps:.1f}s)",
                    flush=True,
                )

        pos_ref_emb = _embed_uint8_arrays(model, ref_pos_arrays, device, mean, std)
        neg_ref_emb = _embed_uint8_arrays(model, ref_neg_arrays, device, mean, std)
        val_emb = _embed_uint8_arrays(model, val_arrays, device, mean, std)
        proto_filter = PrototypeFilter.fit(pos_ref_emb, neg_ref_emb, mode="pos_minus_neg")
        val_scores = proto_filter.score(val_emb)
        val_roc_auc = compute_roc_auc(val_scores, val_calib_is_tp)
        val_pr_auc = compute_pr_auc(val_scores, val_calib_is_tp)
        val_calib = calibrate_threshold(val_scores, val_calib_is_tp, val_calib_total_gt)

        mean_loss = float(np.mean(losses))
        ep_dt = time.monotonic() - ep_t0
        print(
            f"train_fewshot_ball epoch {epoch}/{config.epochs}: "
            f"loss={mean_loss:.4f} val_calib_auc={val_roc_auc if val_roc_auc is not None else -1:.4f} "
            f"val_calib_f1={val_calib['calib_best_f1']:.4f} ({ep_dt:.1f}s)",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "val_calib_roc_auc": val_roc_auc,
                "val_calib_pr_auc": val_pr_auc,
                "val_calib_best_f1": val_calib["calib_best_f1"],
                "elapsed_s": round(ep_dt, 2),
            }
        )
        score_key = (
            round(val_roc_auc if val_roc_auc is not None else -1.0, 12),
            round(val_calib["calib_best_f1"], 12),
        )
        if best_key is None or score_key > best_key:
            best_key = score_key
            best_epoch = epoch
            tmp_ckpt = config.checkpoint_path.with_suffix(".pth.tmp")
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "references_per_class": config.references_per_class,
                    "val_calib_roc_auc": val_roc_auc,
                    "val_calib_best_f1": val_calib["calib_best_f1"],
                    "seed": config.seed,
                },
                tmp_ckpt,
            )
            tmp_ckpt.replace(config.checkpoint_path)

    total_dt = time.monotonic() - train_start
    return {
        "device": str(device),
        "checkpoint_path": str(config.checkpoint_path),
        "seed": config.seed,
        "epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "references_per_class": config.references_per_class,
        "queries_per_class": config.queries_per_class,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "cosine_scale": config.cosine_scale,
        "canvas_wh": list(config.canvas_wh),
        "first_episode_s": round(first_episode_s, 4),
        "total_train_s": round(total_dt, 2),
        "best_epoch": best_epoch,
        "selection_criterion": "highest val_calib ROC-AUC (tie-broken by val_calib F1) with 6 pos + 6 neg train prototypes",
        "pool_stats": pool.stats,
        "history": history,
    }
