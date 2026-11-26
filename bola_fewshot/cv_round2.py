from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from bola_fewshot.dataset import (
    DatasetSplit,
    assert_disjoint_splits,
    audit_group_overlap,
    count_gt_boxes,
    crop_box,
    load_yolo_dataset,
    make_group_kfold,
)
from bola_fewshot.detector import BallDetectorProtocol, Detection, YoloBallDetector
from bola_fewshot.fewshot_filter import FewShotR18Encoder, PrototypeFilter, calibrate_threshold
from bola_fewshot.metrics import (
    compute_filter_cost,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_detections,
    match_image_detections,
    paired_bootstrap_diff,
    paired_cluster_bootstrap_diff,
)
from bola_fewshot.train_ball import (
    BallTrainConfig,
    build_pool_from_mined_jsonl,
    mine_training_boxes_incremental,
    train_fewshot_ball_epochs_only,
)


@dataclass(frozen=True)
class Round2Config:
    data_yaml: Path
    detector_weights: Path
    fewshot_weights: Path
    imagenet_weights: Path | None
    output_dir: Path
    cache_dir: Path
    group_map_path: Path | None = None
    path_remap: tuple[str, str] | None = None
    imgsz: int = 1920
    conf: float = 0.35
    iou: float = 0.70
    eval_iou: float = 0.50
    max_det: int = 300
    target_class: int = 0
    context_scales: tuple[float, ...] = (2.0, 3.0, 4.0)
    canvas_wh: tuple[int, int] = (96, 96)
    k_values: tuple[int, ...] = (1, 6)
    n_folds: int = 5
    seed: int = 42
    train_seed: int = 25
    train_epochs: int = 16
    train_steps_per_epoch: int = 360
    train_mining_conf: float = 0.05
    train_mining_imgsz: int = 1280
    train_max_pos_crops: int = 12000
    train_max_neg_crops: int = 12000
    bootstrap_resamples: int = 2000
    device: str = "auto"
    include_all_layers_variation: bool = True


def _sanitize_for_json(obj: object) -> object:
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "-inf" if obj < 0 else "inf"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _strip_per_image(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_image"}


def _preds_to_eval_format(
    detections_by_image: Mapping[str, Sequence[Detection]],
) -> dict[str, list[tuple[tuple[float, float, float, float], float]]]:
    return {
        image_id: [(det.box, det.conf) for det in dets]
        for image_id, dets in detections_by_image.items()
    }


def _load_group_map(group_map_path: Path | None, splits: Sequence[DatasetSplit]) -> dict[str, str] | None:
    if group_map_path is None:
        return None
    raw = json.loads(Path(group_map_path).read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for split in splits:
        for record in split.records:
            gid = raw.get(record.image_id) or raw.get(record.image_path.name) or raw.get(record.image_path.stem)
            if gid is not None:
                resolved[record.image_id] = str(gid)
    return resolved if resolved else None


def _extract_val_crops_by_context(
    val_split: DatasetSplit,
    dets_by_image: Mapping[str, Sequence[Detection]],
    *,
    context_scales: Sequence[float],
    eval_iou: float,
) -> tuple[list[Detection], np.ndarray, dict[float, list[Image.Image]]]:
    flat_dets: list[Detection] = []
    flat_is_tp: list[bool] = []
    crops_by_ctx: dict[float, list[Image.Image]] = {ctx: [] for ctx in context_scales}
    for record in val_split.records:
        dets = list(dets_by_image.get(record.image_id, ()))
        if not dets:
            continue
        matched = match_image_detections(
            [d.box for d in dets],
            [d.conf for d in dets],
            record.gt_boxes,
            iou_threshold=eval_iou,
        )
        flat_dets.extend(dets)
        flat_is_tp.extend(matched.tolist())
        with Image.open(record.image_path) as img:
            rgb = img.convert("RGB")
            for det in dets:
                for ctx in context_scales:
                    crops_by_ctx[ctx].append(crop_box(rgb, det.box, context_scale=ctx, square=True))
    return flat_dets, np.asarray(flat_is_tp, dtype=bool), crops_by_ctx


def _sample_train_pos_refs_by_context(
    train_split: DatasetSplit,
    max_k: int,
    seed: int,
    context_scales: Sequence[float],
) -> tuple[dict[float, list[Image.Image]], list[dict]]:
    candidates: list[tuple[str, Path, int, tuple[float, float, float, float]]] = []
    for record in sorted(train_split.records, key=lambda r: r.image_id):
        for box_idx, box in enumerate(record.gt_boxes):
            candidates.append((record.image_id, record.image_path, box_idx, box))
    rng = random.Random(seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    chosen = [candidates[i] for i in order[: min(max_k, len(candidates))]]
    crops_by_ctx: dict[float, list[Image.Image]] = {ctx: [] for ctx in context_scales}
    meta: list[dict] = []
    for iid, ipath, box_idx, box in chosen:
        with Image.open(ipath) as img:
            rgb = img.convert("RGB")
            for ctx in context_scales:
                crops_by_ctx[ctx].append(crop_box(rgb, box, context_scale=ctx, square=True))
        meta.append({"image_id": iid, "box_index": box_idx, "box": [round(v, 2) for v in box]})
    return crops_by_ctx, meta


def _sample_train_neg_refs_by_context(
    mined_jsonl: Path,
    max_k: int,
    seed: int,
    context_scales: Sequence[float],
    eval_conf: float,
    mining_conf: float,
) -> tuple[dict[float, list[Image.Image]], list[dict]]:
    high_fps: list[tuple[str, str, tuple[float, float, float, float], float]] = []
    low_fps: list[tuple[str, str, tuple[float, float, float, float], float]] = []
    with mined_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            for d in row["dets"]:
                if not d["is_tp"]:
                    conf = float(d["conf"])
                    entry = (str(row["image_id"]), str(row["image_path"]), tuple(float(v) for v in d["box"]), conf)
                    if conf >= eval_conf:
                        high_fps.append(entry)
                    elif conf >= mining_conf:
                        low_fps.append(entry)
    rng = random.Random(seed + 2)
    rng.shuffle(high_fps)
    rng.shuffle(low_fps)
    pool = high_fps + low_fps
    chosen = pool[: min(max_k, len(pool))]
    crops_by_ctx: dict[float, list[Image.Image]] = {ctx: [] for ctx in context_scales}
    meta: list[dict] = []
    for iid, ipath, box, conf in chosen:
        with Image.open(ipath) as img:
            rgb = img.convert("RGB")
            for ctx in context_scales:
                crops_by_ctx[ctx].append(crop_box(rgb, box, context_scale=ctx, square=True))
        meta.append({"image_id": iid, "box": [round(v, 2) for v in box], "conf": round(conf, 4)})
    return crops_by_ctx, meta


def _apply_mask_to_split(
    split: DatasetSplit,
    flat_dets: Sequence[Detection],
    kept_mask: np.ndarray,
) -> dict[str, list[Detection]]:
    filtered: dict[str, list[Detection]] = {r.image_id: [] for r in split.records}
    for det, keep in zip(flat_dets, kept_mask):
        if bool(keep):
            filtered[det.image_id].append(det)
    return filtered


def _evaluate_oof_candidate_scores(
    *,
    val_split: DatasetSplit,
    folds: Sequence[DatasetSplit],
    flat_dets: Sequence[Detection],
    flat_is_tp: np.ndarray,
    det_fold_idx: np.ndarray,
    scores_by_epoch: Mapping[int, np.ndarray],
    base_metrics: dict,
    eval_iou: float,
    bootstrap_resamples: int,
    seed: int,
    group_by: Mapping[str, str] | None,
) -> tuple[dict, dict[str, dict[str, int]], np.ndarray]:
    n_folds = len(folds)
    gt_by_image = val_split.gt_by_image()
    oof_kept = np.ones(len(flat_dets), dtype=bool)
    oof_scores = np.zeros(len(flat_dets), dtype=np.float64)
    fold_summaries: list[dict] = []

    for f_idx, fold in enumerate(folds):
        test_mask = det_fold_idx == f_idx
        calib_mask = ~test_mask
        calib_image_ids = [r.image_id for i, fl in enumerate(folds) if i != f_idx for r in fl.records]
        calib_total_gt = sum(len(gt_by_image[iid]) for iid in calib_image_ids)

        best_epoch = None
        best_calib = None
        best_key = None
        best_calib_auc = None
        for epoch, all_scores in sorted(scores_by_epoch.items()):
            c_scores = all_scores[calib_mask]
            c_tp = flat_is_tp[calib_mask]
            c_auc = compute_roc_auc(c_scores, c_tp)
            c_cal = calibrate_threshold(c_scores, c_tp, calib_total_gt)
            key = (
                round(c_cal["calib_best_f1"], 12),
                round(c_cal["calib_best_recall"], 12),
                round(c_auc if c_auc is not None else -1.0, 12),
                -epoch,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_calib = c_cal
                best_calib_auc = c_auc

        assert best_epoch is not None and best_calib is not None
        frozen_thresh = float(best_calib["threshold"])
        t_scores = scores_by_epoch[best_epoch][test_mask]
        t_tp = flat_is_tp[test_mask]
        t_kept = t_scores >= frozen_thresh
        oof_kept[test_mask] = t_kept
        oof_scores[test_mask] = t_scores

        fold_dets_list = [d for d, m in zip(flat_dets, test_mask) if m]
        fold_base_by_img = _apply_mask_to_split(fold, fold_dets_list, np.ones(len(fold_dets_list), dtype=bool))
        fold_filt_by_img = _apply_mask_to_split(fold, fold_dets_list, t_kept)
        fold_base_m = evaluate_detections(_preds_to_eval_format(fold_base_by_img), fold.gt_by_image(), fold.image_ids, iou_threshold=eval_iou)
        fold_filt_m = evaluate_detections(_preds_to_eval_format(fold_filt_by_img), fold.gt_by_image(), fold.image_ids, iou_threshold=eval_iou)
        fold_cost = compute_filter_cost(t_tp, t_kept)
        fold_summaries.append(
            {
                "fold": f_idx,
                "n_images": len(fold.records),
                "selected_epoch": int(best_epoch),
                "frozen_threshold": frozen_thresh,
                "calib_roc_auc": best_calib_auc,
                "calib_base_f1": best_calib["calib_base_f1"],
                "calib_best_f1": best_calib["calib_best_f1"],
                "fold_roc_auc": compute_roc_auc(t_scores, t_tp),
                "fold_base_f1": fold_base_m["f1"],
                "fold_filt_f1": fold_filt_m["f1"],
                "fold_delta_f1": float(fold_filt_m["f1"] - fold_base_m["f1"]),
                "fps_removed": fold_cost["fps_removed"],
                "tps_removed": fold_cost["tps_removed"],
            }
        )

    pooled_filt_by_img = _apply_mask_to_split(val_split, flat_dets, oof_kept)
    pooled_metrics = evaluate_detections(
        _preds_to_eval_format(pooled_filt_by_img),
        gt_by_image,
        val_split.image_ids,
        iou_threshold=eval_iou,
    )
    pooled_cost = compute_filter_cost(flat_is_tp, oof_kept)
    boot_img = paired_bootstrap_diff(
        base_metrics["per_image"],
        pooled_metrics["per_image"],
        val_split.image_ids,
        n_resamples=bootstrap_resamples,
        seed=seed + 1000,
    )
    boot_group = None
    if group_by is not None:
        boot_group = paired_cluster_bootstrap_diff(
            base_metrics["per_image"],
            pooled_metrics["per_image"],
            val_split.image_ids,
            group_by,
            n_resamples=bootstrap_resamples,
            seed=seed + 1500,
        )

    fold_deltas = [fs["fold_delta_f1"] for fs in fold_summaries]
    fold_aucs = [fs["fold_roc_auc"] for fs in fold_summaries if fs["fold_roc_auc"] is not None]
    summary = {
        "pooled_oof_metrics": _strip_per_image(pooled_metrics),
        "pooled_oof_cost": pooled_cost,
        "pooled_oof_roc_auc": compute_roc_auc(oof_scores, flat_is_tp),
        "pooled_oof_pr_auc": compute_pr_auc(oof_scores, flat_is_tp),
        "mean_fold_roc_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
        "sd_fold_roc_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "mean_fold_delta_f1": float(np.mean(fold_deltas)),
        "sd_fold_delta_f1": float(np.std(fold_deltas, ddof=1)) if len(fold_deltas) > 1 else 0.0,
        "delta_map50": float(pooled_metrics["map50"] - base_metrics["map50"]),
        "delta_map50_95": float(pooled_metrics["map50_95"] - base_metrics["map50_95"]),
        "bootstrap_image": boot_img,
        "bootstrap_group": boot_group,
        "folds": fold_summaries,
    }
    return summary, pooled_metrics["per_image"], oof_kept


def _evaluate_nested_arm_cv(
    *,
    arm_name: str,
    val_split: DatasetSplit,
    folds: Sequence[DatasetSplit],
    flat_dets: Sequence[Detection],
    flat_is_tp: np.ndarray,
    det_fold_idx: np.ndarray,
    candidates: Mapping[str, Mapping[int, np.ndarray]],
    base_metrics: dict,
    eval_iou: float,
    bootstrap_resamples: int,
    seed: int,
    group_by: Mapping[str, str] | None,
) -> tuple[dict, dict[str, dict[str, int]], np.ndarray]:
    gt_by_image = val_split.gt_by_image()
    oof_kept = np.ones(len(flat_dets), dtype=bool)
    oof_scores = np.zeros(len(flat_dets), dtype=np.float64)
    fold_choices: list[dict] = []

    for f_idx, fold in enumerate(folds):
        test_mask = det_fold_idx == f_idx
        calib_mask = ~test_mask
        calib_image_ids = [r.image_id for i, fl in enumerate(folds) if i != f_idx for r in fl.records]
        calib_total_gt = sum(len(gt_by_image[iid]) for iid in calib_image_ids)

        best_choice = None
        best_key = None
        for cfg_id, scores_by_epoch in sorted(candidates.items()):
            for epoch, all_scores in sorted(scores_by_epoch.items()):
                c_scores = all_scores[calib_mask]
                c_tp = flat_is_tp[calib_mask]
                c_auc = compute_roc_auc(c_scores, c_tp)
                c_cal = calibrate_threshold(c_scores, c_tp, calib_total_gt)
                key = (
                    round(c_cal["calib_best_f1"], 12),
                    round(c_cal["calib_best_recall"], 12),
                    round(c_auc if c_auc is not None else -1.0, 12),
                    -epoch,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_choice = (cfg_id, epoch, c_cal, c_auc)

        assert best_choice is not None
        cfg_id, epoch, c_cal, c_auc = best_choice
        frozen_thresh = float(c_cal["threshold"])
        t_scores = candidates[cfg_id][epoch][test_mask]
        t_tp = flat_is_tp[test_mask]
        t_kept = t_scores >= frozen_thresh
        oof_kept[test_mask] = t_kept
        oof_scores[test_mask] = t_scores

        fold_dets_list = [d for d, m in zip(flat_dets, test_mask) if m]
        fold_base_by_img = _apply_mask_to_split(fold, fold_dets_list, np.ones(len(fold_dets_list), dtype=bool))
        fold_filt_by_img = _apply_mask_to_split(fold, fold_dets_list, t_kept)
        fold_base_m = evaluate_detections(_preds_to_eval_format(fold_base_by_img), fold.gt_by_image(), fold.image_ids, iou_threshold=eval_iou)
        fold_filt_m = evaluate_detections(_preds_to_eval_format(fold_filt_by_img), fold.gt_by_image(), fold.image_ids, iou_threshold=eval_iou)
        fold_cost = compute_filter_cost(t_tp, t_kept)
        fold_choices.append(
            {
                "fold": f_idx,
                "n_images": len(fold.records),
                "chosen_config_id": cfg_id,
                "chosen_epoch": int(epoch),
                "frozen_threshold": frozen_thresh,
                "calib_roc_auc": c_auc,
                "calib_base_f1": c_cal["calib_base_f1"],
                "calib_best_f1": c_cal["calib_best_f1"],
                "fold_roc_auc": compute_roc_auc(t_scores, t_tp),
                "fold_base_f1": fold_base_m["f1"],
                "fold_filt_f1": fold_filt_m["f1"],
                "fold_delta_f1": float(fold_filt_m["f1"] - fold_base_m["f1"]),
                "fps_removed": fold_cost["fps_removed"],
                "tps_removed": fold_cost["tps_removed"],
            }
        )

    pooled_filt_by_img = _apply_mask_to_split(val_split, flat_dets, oof_kept)
    pooled_metrics = evaluate_detections(
        _preds_to_eval_format(pooled_filt_by_img),
        gt_by_image,
        val_split.image_ids,
        iou_threshold=eval_iou,
    )
    pooled_cost = compute_filter_cost(flat_is_tp, oof_kept)
    boot_img = paired_bootstrap_diff(
        base_metrics["per_image"],
        pooled_metrics["per_image"],
        val_split.image_ids,
        n_resamples=bootstrap_resamples,
        seed=seed + 2000,
    )
    boot_group = (
        paired_cluster_bootstrap_diff(
            base_metrics["per_image"],
            pooled_metrics["per_image"],
            val_split.image_ids,
            group_by,
            n_resamples=bootstrap_resamples,
            seed=seed + 2500,
        )
        if group_by is not None
        else None
    )

    v14_breakdown = None
    if group_by is not None:
        v14_ids = [iid for iid in val_split.image_ids if group_by.get(iid) == "v14"]
        non_v14_ids = [iid for iid in val_split.image_ids if group_by.get(iid) != "v14"]
        if v14_ids and non_v14_ids:
            base_by_img = _apply_mask_to_split(val_split, flat_dets, np.ones(len(flat_dets), dtype=bool))
            v14_base = evaluate_detections(_preds_to_eval_format(base_by_img), gt_by_image, v14_ids, iou_threshold=eval_iou)
            v14_filt = evaluate_detections(_preds_to_eval_format(pooled_filt_by_img), gt_by_image, v14_ids, iou_threshold=eval_iou)
            nv_base = evaluate_detections(_preds_to_eval_format(base_by_img), gt_by_image, non_v14_ids, iou_threshold=eval_iou)
            nv_filt = evaluate_detections(_preds_to_eval_format(pooled_filt_by_img), gt_by_image, non_v14_ids, iou_threshold=eval_iou)
            v14_breakdown = {
                "v14_only": {
                    "n_images": len(v14_ids),
                    "base_f1": v14_base["f1"],
                    "filt_f1": v14_filt["f1"],
                    "delta_f1": float(v14_filt["f1"] - v14_base["f1"]),
                },
                "non_v14_792_games": {
                    "n_images": len(non_v14_ids),
                    "base_f1": nv_base["f1"],
                    "filt_f1": nv_filt["f1"],
                    "delta_f1": float(nv_filt["f1"] - nv_base["f1"]),
                },
            }

    fold_deltas = [fc["fold_delta_f1"] for fc in fold_choices]
    summary = {
        "arm": arm_name,
        "pooled_oof_metrics": _strip_per_image(pooled_metrics),
        "pooled_oof_cost": pooled_cost,
        "pooled_oof_roc_auc": compute_roc_auc(oof_scores, flat_is_tp),
        "pooled_oof_pr_auc": compute_pr_auc(oof_scores, flat_is_tp),
        "mean_fold_delta_f1": float(np.mean(fold_deltas)),
        "sd_fold_delta_f1": float(np.std(fold_deltas, ddof=1)) if len(fold_deltas) > 1 else 0.0,
        "delta_map50": float(pooled_metrics["map50"] - base_metrics["map50"]),
        "delta_map50_95": float(pooled_metrics["map50_95"] - base_metrics["map50_95"]),
        "bootstrap_image": boot_img,
        "bootstrap_group": boot_group,
        "v14_breakdown": v14_breakdown,
        "folds": fold_choices,
    }
    return summary, pooled_metrics["per_image"], oof_kept


def run_round2_cv(
    config: Round2Config,
    *,
    val_detector: BallDetectorProtocol | None = None,
    mining_detector: BallDetectorProtocol | None = None,
) -> dict:
    splits = load_yolo_dataset(
        config.data_yaml,
        path_remap=config.path_remap,
        target_class=config.target_class,
        require_train=True,
    )
    train_split = splits["train"]
    val_split = splits["val"]
    assert_disjoint_splits(train_split, val_split)

    group_map = _load_group_map(config.group_map_path, (train_split, val_split))
    group_audit = (
        audit_group_overlap(train_split, val_split, group_map)
        if group_map is not None
        else None
    )
    folds = make_group_kfold(val_split, n_folds=config.n_folds, seed=config.seed, group_by=group_map)
    fold_by_image: dict[str, int] = {}
    for f_idx, fold in enumerate(folds):
        for r in fold.records:
            fold_by_image[r.image_id] = f_idx

    val_det = val_detector or YoloBallDetector(
        config.detector_weights,
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        max_det=config.max_det,
        target_class=config.target_class,
        device=config.device,
        batch_size=1,
    )
    val_cache = config.cache_dir / "dets_val_all.json"
    val_dets = val_det.predict_records(val_split.records, cache_path=val_cache)
    val_base_metrics = evaluate_detections(
        _preds_to_eval_format(val_dets),
        val_split.gt_by_image(),
        val_split.image_ids,
        iou_threshold=config.eval_iou,
    )

    flat_dets, flat_is_tp, val_crops_by_ctx = _extract_val_crops_by_context(
        val_split,
        val_dets,
        context_scales=config.context_scales,
        eval_iou=config.eval_iou,
    )
    det_fold_idx = np.array([fold_by_image[d.image_id] for d in flat_dets], dtype=np.int64)
    detector_conf_sep = {
        "roc_auc": compute_roc_auc([d.conf for d in flat_dets], flat_is_tp),
        "pr_auc": compute_pr_auc([d.conf for d in flat_dets], flat_is_tp),
    }

    mine_det = mining_detector or YoloBallDetector(
        config.detector_weights,
        imgsz=config.train_mining_imgsz,
        conf=config.train_mining_conf,
        iou=config.iou,
        max_det=config.max_det,
        target_class=config.target_class,
        device=config.device,
        batch_size=1,
    )
    mined_jsonl = config.cache_dir / f"train_mined_boxes_imgsz{config.train_mining_imgsz}.jsonl"
    mining_stats = mine_training_boxes_incremental(
        train_split.records,
        mine_det,
        mined_jsonl,
        eval_conf=config.conf,
        mining_conf=config.train_mining_conf,
        eval_iou=config.eval_iou,
    )

    max_k = max(config.k_values)
    pos_refs_by_ctx, pos_ref_meta = _sample_train_pos_refs_by_context(
        train_split, max_k=max_k, seed=config.seed, context_scales=config.context_scales
    )
    neg_refs_by_ctx, neg_ref_meta = _sample_train_neg_refs_by_context(
        mined_jsonl,
        max_k=max_k,
        seed=config.seed,
        context_scales=config.context_scales,
        eval_conf=config.conf,
        mining_conf=config.train_mining_conf,
    )

    arm_candidates: dict[str, dict[str, dict[int, np.ndarray]]] = {
        "frozen_players": {},
        "trained_ball_layer34": {},
    }
    if config.include_all_layers_variation:
        arm_candidates["trained_ball_all_layers"] = {}

    frozen_enc = FewShotR18Encoder(
        config.fewshot_weights,
        device=config.device,
        canvas_wh=config.canvas_wh,
    )
    for ctx in config.context_scales:
        p_emb = frozen_enc.embed_crops(pos_refs_by_ctx[ctx], layer="layer4")
        n_emb = frozen_enc.embed_crops(neg_refs_by_ctx[ctx], layer="layer4")
        v_emb = frozen_enc.embed_crops(val_crops_by_ctx[ctx], layer="layer4")
        for mode in ("positive_only", "pos_minus_neg"):
            for k in config.k_values:
                proto = PrototypeFilter.fit(
                    p_emb[: min(k, len(p_emb))],
                    n_emb[: min(k, len(n_emb))] if mode == "pos_minus_neg" else None,
                    mode=mode,
                )
                cfg_id = f"frozen_ctx{int(ctx)}_{mode}_k{k}"
                arm_candidates["frozen_players"][cfg_id] = {0: proto.score(v_emb)}

    training_runs: list[dict] = []
    train_modes = [("trained_ball_layer34", "layer3_layer4")]
    if config.include_all_layers_variation:
        train_modes.append(("trained_ball_all_layers", "all"))

    for ctx in config.context_scales:
        pool = build_pool_from_mined_jsonl(
            mined_jsonl,
            context_scale=ctx,
            square_crop=True,
            canvas_wh=config.canvas_wh,
            eval_conf=config.conf,
            mining_conf=config.train_mining_conf,
            max_pos_crops=config.train_max_pos_crops,
            max_neg_crops=config.train_max_neg_crops,
            seed=config.train_seed,
        )
        for arm_name, trainable_layers in train_modes:
            ckpt_dir = config.cache_dir / f"ckpts_{arm_name}_ctx{int(ctx)}"
            train_info = train_fewshot_ball_epochs_only(
                pool,
                config=BallTrainConfig(
                    epochs=config.train_epochs,
                    steps_per_epoch=config.train_steps_per_epoch,
                    references_per_class=min(6, max_k),
                    queries_per_class=8,
                    seed=config.train_seed,
                    canvas_wh=config.canvas_wh,
                    device=config.device,
                    trainable_layers=trainable_layers,
                    imagenet_weights_path=config.imagenet_weights,
                    epoch_checkpoints_dir=ckpt_dir,
                ),
                group_by=group_map,
            )
            train_info["arm"] = arm_name
            train_info["context_scale"] = float(ctx)
            training_runs.append(train_info)

            for ep in range(1, config.train_epochs + 1):
                ep_enc = FewShotR18Encoder(
                    ckpt_dir / f"epoch_{ep:02d}.pth",
                    device=config.device,
                    canvas_wh=config.canvas_wh,
                )
                p_emb = ep_enc.embed_crops(pos_refs_by_ctx[ctx], layer="layer4")
                n_emb = ep_enc.embed_crops(neg_refs_by_ctx[ctx], layer="layer4")
                v_emb = ep_enc.embed_crops(val_crops_by_ctx[ctx], layer="layer4")
                for mode in ("pos_minus_neg", "positive_only"):
                    for k in config.k_values:
                        proto = PrototypeFilter.fit(
                            p_emb[: min(k, len(p_emb))],
                            n_emb[: min(k, len(n_emb))] if mode == "pos_minus_neg" else None,
                            mode=mode,
                        )
                        cfg_id = f"{arm_name}_ctx{int(ctx)}_{mode}_k{k}"
                        arm_candidates[arm_name].setdefault(cfg_id, {})[ep] = proto.score(v_emb)

    fixed_config_results: list[dict] = []
    for arm_name, cand_map in arm_candidates.items():
        for cfg_id, scores_by_ep in sorted(cand_map.items()):
            res, _, _ = _evaluate_oof_candidate_scores(
                val_split=val_split,
                folds=folds,
                flat_dets=flat_dets,
                flat_is_tp=flat_is_tp,
                det_fold_idx=det_fold_idx,
                scores_by_epoch=scores_by_ep,
                base_metrics=val_base_metrics,
                eval_iou=config.eval_iou,
                bootstrap_resamples=config.bootstrap_resamples,
                seed=config.seed,
                group_by=group_map,
            )
            res["arm"] = arm_name
            res["config_id"] = cfg_id
            fixed_config_results.append(res)

    nested_arm_results: dict[str, dict] = {}
    nested_per_image: dict[str, dict[str, dict[str, int]]] = {}
    for arm_name, cand_map in arm_candidates.items():
        summary, per_img, _ = _evaluate_nested_arm_cv(
            arm_name=arm_name,
            val_split=val_split,
            folds=folds,
            flat_dets=flat_dets,
            flat_is_tp=flat_is_tp,
            det_fold_idx=det_fold_idx,
            candidates=cand_map,
            base_metrics=val_base_metrics,
            eval_iou=config.eval_iou,
            bootstrap_resamples=config.bootstrap_resamples,
            seed=config.seed,
            group_by=group_map,
        )
        nested_arm_results[arm_name] = summary
        nested_per_image[arm_name] = per_img

    arm_comparisons: dict[str, dict] = {}
    for trained_arm in ("trained_ball_layer34", "trained_ball_all_layers"):
        if trained_arm not in nested_per_image:
            continue
        comp_img = paired_bootstrap_diff(
            nested_per_image["frozen_players"],
            nested_per_image[trained_arm],
            val_split.image_ids,
            n_resamples=config.bootstrap_resamples,
            seed=config.seed + 3000,
        )
        comp_grp = (
            paired_cluster_bootstrap_diff(
                nested_per_image["frozen_players"],
                nested_per_image[trained_arm],
                val_split.image_ids,
                group_map,
                n_resamples=config.bootstrap_resamples,
                seed=config.seed + 3500,
            )
            if group_map is not None
            else None
        )
        arm_comparisons[f"{trained_arm}_minus_frozen_players"] = {
            "bootstrap_image": comp_img,
            "bootstrap_group": comp_grp,
        }

    fold_info = [
        {
            "fold": i,
            "n_images": len(fl.records),
            "n_gt_balls": count_gt_boxes(fl.records),
            "n_groups": len({group_map[r.image_id] for r in fl.records}) if group_map else len(fl.records),
            "has_v14": any(group_map.get(r.image_id) == "v14" for r in fl.records) if group_map else False,
        }
        for i, fl in enumerate(folds)
    ]

    payload = {
        "round": 2,
        "protocol": {
            "data_yaml": str(config.data_yaml),
            "detector_weights": config.detector_weights.name,
            "fewshot_weights": config.fewshot_weights.name,
            "imgsz": config.imgsz,
            "conf": config.conf,
            "nms_iou": config.iou,
            "eval_iou": config.eval_iou,
            "context_scales": list(config.context_scales),
            "square_crop": True,
            "canvas_wh": list(config.canvas_wh),
            "k_values": list(config.k_values),
            "n_folds": config.n_folds,
            "seed": config.seed,
            "train_seed": config.train_seed,
            "train_epochs": config.train_epochs,
            "train_steps_per_epoch": config.train_steps_per_epoch,
            "train_total_episodes": config.train_epochs * config.train_steps_per_epoch,
            "train_mining_conf": config.train_mining_conf,
            "train_mining_imgsz": config.train_mining_imgsz,
            "bootstrap_resamples": config.bootstrap_resamples,
            "include_all_layers_variation": config.include_all_layers_variation,
            "total_fixed_configs_tested": len(fixed_config_results),
            "total_nested_arm_selectors": len(nested_arm_results),
        },
        "splits": {
            "train_images": len(train_split.records),
            "val_images": len(val_split.records),
            "val_gt_balls": count_gt_boxes(val_split.records),
            "train_val_group_audit": group_audit,
            "folds": fold_info,
        },
        "mining_stats": mining_stats,
        "references": {
            "pos_refs_sampled": len(pos_ref_meta),
            "neg_refs_sampled": len(neg_ref_meta),
        },
        "baseline_val_all": _strip_per_image(val_base_metrics),
        "detector_conf_separability": detector_conf_sep,
        "training_runs": training_runs,
        "nested_arms": nested_arm_results,
        "arm_comparisons": arm_comparisons,
        "fixed_configs": fixed_config_results,
    }
    sanitized = _sanitize_for_json(payload)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = config.output_dir / "rodada2_experiment_results.json"
    tmp_json = out_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_json.replace(out_json)
    return sanitized
