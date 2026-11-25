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
    ImageRecord,
    assert_disjoint_splits,
    count_gt_boxes,
    crop_box,
    load_image_list_split,
    load_yolo_dataset,
    partition_split,
    stratified_sample_split,
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
)


@dataclass(frozen=True)
class FilterVariantSpec:
    name: str
    layer: str
    mode: str
    stage: str


@dataclass(frozen=True)
class ExperimentConfig:
    data_yaml: Path
    detector_weights: Path
    fewshot_weights: Path
    output_dir: Path
    report_path: Path
    images_root: Path | None = None
    labels_root: Path | None = None
    path_remap: tuple[str, str] | None = None
    val_list_override: Path | None = None
    max_val_images: int | None = None
    imgsz: int = 1920
    conf: float = 0.35
    iou: float = 0.70
    eval_iou: float = 0.50
    max_det: int = 300
    target_class: int = 0
    k_values: tuple[int, ...] = (1, 6)
    seed: int = 42
    calib_source: str = "val_split"
    calib_fraction: float = 0.30
    bootstrap_resamples: int = 2000
    canvas_wh: tuple[int, int] = (48, 96)
    context_scale: float = 1.0
    square_crop: bool = False
    variations: str = "auto"
    auc_variation_trigger: float = 0.70
    max_train_neg_images: int = 300
    device: str = "auto"
    batch_size: int = 1
    cache_dir: Path | None = None
    extra_datasets: tuple[tuple[str, Path], ...] = ()
    extra_val_lists: tuple[tuple[str, Path], ...] = ()


PRIMARY_VARIANT = FilterVariantSpec(
    name="layer4_positive_only",
    layer="layer4",
    mode="positive_only",
    stage="primary",
)

CONTROLLED_VARIATIONS = (
    FilterVariantSpec(
        name="layer4_pos_minus_neg",
        layer="layer4",
        mode="pos_minus_neg",
        stage="variation",
    ),
    FilterVariantSpec(
        name="layer3_positive_only",
        layer="layer3",
        mode="positive_only",
        stage="variation",
    ),
    FilterVariantSpec(
        name="layer2_positive_only",
        layer="layer2",
        mode="positive_only",
        stage="variation",
    ),
)


def _preds_to_eval_format(
    detections_by_image: Mapping[str, Sequence[Detection]],
) -> dict[str, list[tuple[tuple[float, float, float, float], float]]]:
    return {
        image_id: [(det.box, det.conf) for det in dets]
        for image_id, dets in detections_by_image.items()
    }


def _extract_detection_crops(
    split: DatasetSplit,
    detections_by_image: Mapping[str, Sequence[Detection]],
    *,
    eval_iou: float,
    context_scale: float,
    square_crop: bool,
) -> tuple[list[Detection], np.ndarray, list[Image.Image]]:
    flat_dets: list[Detection] = []
    flat_is_tp: list[bool] = []
    crops: list[Image.Image] = []
    for record in split.records:
        dets = list(detections_by_image.get(record.image_id, ()))
        if not dets:
            continue
        boxes = [d.box for d in dets]
        confs = [d.conf for d in dets]
        matched = match_image_detections(boxes, confs, record.gt_boxes, iou_threshold=eval_iou)
        flat_dets.extend(dets)
        flat_is_tp.extend(matched.tolist())
        with Image.open(record.image_path) as img:
            rgb = img.convert("RGB")
            for det in dets:
                crops.append(
                    crop_box(
                        rgb,
                        det.box,
                        context_scale=context_scale,
                        square=square_crop,
                    )
                )
    return flat_dets, np.asarray(flat_is_tp, dtype=bool), crops


def _sample_train_positive_references(
    train_ref_split: DatasetSplit,
    max_k: int,
    seed: int,
    *,
    context_scale: float,
    square_crop: bool,
) -> tuple[list[Image.Image], list[dict]]:
    candidates: list[tuple[ImageRecord, int, tuple[float, float, float, float]]] = []
    for record in sorted(train_ref_split.records, key=lambda r: r.image_id):
        for box_idx, box in enumerate(record.gt_boxes):
            candidates.append((record, box_idx, box))
    if not candidates:
        raise ValueError("train reference pool has zero ground-truth ball boxes")
    rng = random.Random(seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    chosen = [candidates[i] for i in order[: min(max_k, len(candidates))]]

    crops: list[Image.Image] = []
    metadata: list[dict] = []
    by_image: dict[Path, Image.Image] = {}
    try:
        for record, box_idx, box in chosen:
            if record.image_path not in by_image:
                with Image.open(record.image_path) as img:
                    by_image[record.image_path] = img.convert("RGB")
            crops.append(
                crop_box(
                    by_image[record.image_path],
                    box,
                    context_scale=context_scale,
                    square=square_crop,
                )
            )
            metadata.append(
                {
                    "image_id": record.image_id,
                    "box_index": box_idx,
                    "box": [round(v, 2) for v in box],
                }
            )
    finally:
        for img in by_image.values():
            img.close()
    return crops, metadata


def _sample_train_negative_references(
    train_ref_split: DatasetSplit,
    detector: BallDetectorProtocol,
    max_k: int,
    seed: int,
    *,
    eval_iou: float,
    max_images: int,
    context_scale: float,
    square_crop: bool,
    cache_path: Path | None = None,
) -> tuple[list[Image.Image], list[dict]]:
    ordered = sorted(train_ref_split.records, key=lambda r: r.image_id)
    rng = random.Random(seed + 1)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    subset_records = tuple(ordered[i] for i in sorted(indices[: min(max_images, len(ordered))]))
    subset_split = DatasetSplit(name="train_neg_sample", records=subset_records)
    dets = detector.predict_records(subset_split.records, cache_path=cache_path)

    fp_candidates: list[tuple[ImageRecord, Detection]] = []
    for record in subset_split.records:
        img_dets = dets.get(record.image_id, ())
        if not img_dets:
            continue
        matched = match_image_detections(
            [d.box for d in img_dets],
            [d.conf for d in img_dets],
            record.gt_boxes,
            iou_threshold=eval_iou,
        )
        for det, is_tp in zip(img_dets, matched):
            if not is_tp:
                fp_candidates.append((record, det))

    if not fp_candidates:
        return [], []

    rng_fp = random.Random(seed + 2)
    fp_order = list(range(len(fp_candidates)))
    rng_fp.shuffle(fp_order)
    chosen = [fp_candidates[i] for i in fp_order[: min(max_k, len(fp_candidates))]]

    crops: list[Image.Image] = []
    metadata: list[dict] = []
    by_image: dict[Path, Image.Image] = {}
    try:
        for record, det in chosen:
            if record.image_path not in by_image:
                with Image.open(record.image_path) as img:
                    by_image[record.image_path] = img.convert("RGB")
            crops.append(
                crop_box(
                    by_image[record.image_path],
                    det.box,
                    context_scale=context_scale,
                    square=square_crop,
                )
            )
            metadata.append(
                {
                    "image_id": record.image_id,
                    "box": [round(v, 2) for v in det.box],
                    "conf": round(det.conf, 4),
                }
            )
    finally:
        for img in by_image.values():
            img.close()
    return crops, metadata


def _apply_mask_to_detections(
    split: DatasetSplit,
    flat_dets: Sequence[Detection],
    kept_mask: np.ndarray,
) -> dict[str, list[Detection]]:
    filtered: dict[str, list[Detection]] = {record.image_id: [] for record in split.records}
    for det, keep in zip(flat_dets, kept_mask):
        if bool(keep):
            filtered[det.image_id].append(det)
    return filtered


def _strip_per_image(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_image"}


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


def _evaluate_configuration(
    *,
    variant: FilterVariantSpec,
    k: int,
    pos_emb_pool: np.ndarray,
    neg_emb_pool: np.ndarray | None,
    calib_split: DatasetSplit,
    calib_emb: np.ndarray,
    calib_is_tp: np.ndarray,
    eval_split: DatasetSplit,
    eval_flat_dets: Sequence[Detection],
    eval_emb: np.ndarray,
    eval_is_tp: np.ndarray,
    eval_base_metrics: dict,
    eval_iou: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[dict, PrototypeFilter, float]:
    actual_k_pos = min(k, len(pos_emb_pool))
    pos_subset = pos_emb_pool[:actual_k_pos]
    neg_subset = None
    actual_k_neg = 0
    if variant.mode == "pos_minus_neg":
        if neg_emb_pool is None or len(neg_emb_pool) == 0:
            raise ValueError("negative embeddings unavailable for pos_minus_neg")
        actual_k_neg = min(k, len(neg_emb_pool))
        neg_subset = neg_emb_pool[:actual_k_neg]

    proto_filter = PrototypeFilter.fit(pos_subset, neg_subset, mode=variant.mode)
    calib_scores = proto_filter.score(calib_emb)
    calib_roc_auc = compute_roc_auc(calib_scores, calib_is_tp)
    calib_pr_auc = compute_pr_auc(calib_scores, calib_is_tp)

    calib_total_gt = count_gt_boxes(calib_split.records)
    calib_result = calibrate_threshold(calib_scores, calib_is_tp, calib_total_gt)
    frozen_threshold = float(calib_result["threshold"])

    eval_scores = proto_filter.score(eval_emb)
    eval_roc_auc = compute_roc_auc(eval_scores, eval_is_tp)
    eval_pr_auc = compute_pr_auc(eval_scores, eval_is_tp)

    kept_mask = eval_scores >= frozen_threshold
    filtered_by_image = _apply_mask_to_detections(eval_split, eval_flat_dets, kept_mask)
    eval_filt_metrics = evaluate_detections(
        _preds_to_eval_format(filtered_by_image),
        eval_split.gt_by_image(),
        eval_split.image_ids,
        iou_threshold=eval_iou,
    )
    cost = compute_filter_cost(eval_is_tp, kept_mask)
    bootstrap = paired_bootstrap_diff(
        eval_base_metrics["per_image"],
        eval_filt_metrics["per_image"],
        eval_split.image_ids,
        n_resamples=bootstrap_resamples,
        seed=seed + 1000,
    )

    config_id = f"{variant.name}_k{k}"
    entry = {
        "config_id": config_id,
        "variant": variant.name,
        "stage": variant.stage,
        "layer": variant.layer,
        "mode": variant.mode,
        "k_requested": int(k),
        "k_pos_used": int(actual_k_pos),
        "k_neg_used": int(actual_k_neg),
        "separability": {
            "calib_roc_auc": calib_roc_auc,
            "calib_pr_auc": calib_pr_auc,
            "eval_roc_auc": eval_roc_auc,
            "eval_pr_auc": eval_pr_auc,
            "calib_n_tp": int(calib_is_tp.sum()),
            "calib_n_fp": int((~calib_is_tp).sum()),
            "eval_n_tp": int(eval_is_tp.sum()),
            "eval_n_fp": int((~eval_is_tp).sum()),
        },
        "calibration": calib_result,
        "eval_metrics": _strip_per_image(eval_filt_metrics),
        "filter_cost": cost,
        "delta_map50": float(eval_filt_metrics["map50"] - eval_base_metrics["map50"]),
        "delta_map50_95": float(eval_filt_metrics["map50_95"] - eval_base_metrics["map50_95"]),
        "bootstrap": bootstrap,
    }
    return entry, proto_filter, frozen_threshold


def _evaluate_extra_split(
    *,
    dataset_name: str,
    source_desc: str,
    split: DatasetSplit,
    detector: BallDetectorProtocol,
    encoder: FewShotR18Encoder,
    chosen_entry: dict,
    chosen_filter: PrototypeFilter,
    frozen_threshold: float,
    config: ExperimentConfig,
) -> dict:
    cache_path = config.cache_dir / f"dets_{dataset_name}.json" if config.cache_dir else None
    dets_by_image = detector.predict_records(split.records, cache_path=cache_path)
    base_metrics = evaluate_detections(
        _preds_to_eval_format(dets_by_image),
        split.gt_by_image(),
        split.image_ids,
        iou_threshold=config.eval_iou,
    )
    flat_dets, is_tp, crops = _extract_detection_crops(
        split,
        dets_by_image,
        eval_iou=config.eval_iou,
        context_scale=config.context_scale,
        square_crop=config.square_crop,
    )
    emb = encoder.embed_crops(crops, layer=chosen_entry["layer"])
    scores = chosen_filter.score(emb)
    roc_auc = compute_roc_auc(scores, is_tp)
    pr_auc = compute_pr_auc(scores, is_tp)
    kept_mask = scores >= frozen_threshold
    filtered_by_image = _apply_mask_to_detections(split, flat_dets, kept_mask)
    filt_metrics = evaluate_detections(
        _preds_to_eval_format(filtered_by_image),
        split.gt_by_image(),
        split.image_ids,
        iou_threshold=config.eval_iou,
    )
    cost = compute_filter_cost(is_tp, kept_mask)
    bootstrap = paired_bootstrap_diff(
        base_metrics["per_image"],
        filt_metrics["per_image"],
        split.image_ids,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed + 2000,
    )
    return {
        "name": dataset_name,
        "status": "measured",
        "data_yaml": source_desc,
        "applied_config_id": chosen_entry["config_id"],
        "frozen_threshold": frozen_threshold,
        "baseline": _strip_per_image(base_metrics),
        "filtered": _strip_per_image(filt_metrics),
        "separability": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "n_tp": int(is_tp.sum()),
            "n_fp": int((~is_tp).sum()),
        },
        "filter_cost": cost,
        "delta_map50": float(filt_metrics["map50"] - base_metrics["map50"]),
        "delta_map50_95": float(filt_metrics["map50_95"] - base_metrics["map50_95"]),
        "bootstrap": bootstrap,
    }


def _evaluate_extra_dataset(
    *,
    dataset_name: str,
    data_yaml: Path,
    detector: BallDetectorProtocol,
    encoder: FewShotR18Encoder,
    chosen_entry: dict,
    chosen_filter: PrototypeFilter,
    frozen_threshold: float,
    config: ExperimentConfig,
) -> dict:
    extra_splits = load_yolo_dataset(
        data_yaml,
        images_root=config.images_root,
        labels_root=config.labels_root,
        path_remap=config.path_remap,
        target_class=config.target_class,
        require_train=False,
    )
    split = extra_splits.get("val") or extra_splits["test"]
    return _evaluate_extra_split(
        dataset_name=dataset_name,
        source_desc=str(data_yaml),
        split=split,
        detector=detector,
        encoder=encoder,
        chosen_entry=chosen_entry,
        chosen_filter=chosen_filter,
        frozen_threshold=frozen_threshold,
        config=config,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    detector: BallDetectorProtocol | None = None,
    encoder: FewShotR18Encoder | None = None,
) -> dict:
    if config.calib_source not in ("val_split", "train_holdout"):
        raise ValueError(f"unsupported calib_source: {config.calib_source}")
    if config.variations not in ("auto", "always", "never"):
        raise ValueError(f"unsupported variations policy: {config.variations}")
    if not config.k_values or any(k < 1 for k in config.k_values):
        raise ValueError("k_values must contain positive integers")

    splits = load_yolo_dataset(
        config.data_yaml,
        images_root=config.images_root,
        labels_root=config.labels_root,
        path_remap=config.path_remap,
        target_class=config.target_class,
        require_train=True,
    )
    train_full = splits["train"]
    if config.val_list_override is not None:
        val_full = load_image_list_split(
            "val",
            config.val_list_override,
            images_root=config.images_root,
            labels_root=config.labels_root,
            path_remap=config.path_remap,
            target_class=config.target_class,
        )
    else:
        val_full = splits["val"]
    assert_disjoint_splits(train_full, val_full)

    val_full, val_sampling = stratified_sample_split(
        val_full,
        max_images=config.max_val_images or 0,
        seed=config.seed + 5,
    )

    if config.calib_source == "val_split":
        train_ref = train_full
        calib_split, eval_split = partition_split(
            val_full,
            first_fraction=config.calib_fraction,
            seed=config.seed + 10,
            first_name="val_calib",
            second_name="val_eval",
        )
    else:
        calib_split, train_ref = partition_split(
            train_full,
            first_fraction=config.calib_fraction,
            seed=config.seed + 10,
            first_name="train_calib",
            second_name="train_ref",
        )
        eval_split = val_full

    assert_disjoint_splits(train_ref, calib_split)
    assert_disjoint_splits(train_ref, eval_split)
    assert_disjoint_splits(calib_split, eval_split)

    det_runner = detector or YoloBallDetector(
        config.detector_weights,
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        max_det=config.max_det,
        target_class=config.target_class,
        device=config.device,
        batch_size=config.batch_size,
    )
    enc = encoder or FewShotR18Encoder(
        config.fewshot_weights,
        device=config.device,
        canvas_wh=config.canvas_wh,
    )

    calib_cache = config.cache_dir / f"dets_{calib_split.name}.json" if config.cache_dir else None
    eval_cache = config.cache_dir / f"dets_{eval_split.name}.json" if config.cache_dir else None
    calib_dets = det_runner.predict_records(calib_split.records, cache_path=calib_cache)
    eval_dets = det_runner.predict_records(eval_split.records, cache_path=eval_cache)

    calib_base_metrics = evaluate_detections(
        _preds_to_eval_format(calib_dets),
        calib_split.gt_by_image(),
        calib_split.image_ids,
        iou_threshold=config.eval_iou,
    )
    eval_base_metrics = evaluate_detections(
        _preds_to_eval_format(eval_dets),
        eval_split.gt_by_image(),
        eval_split.image_ids,
        iou_threshold=config.eval_iou,
    )

    _, calib_is_tp, calib_crops = _extract_detection_crops(
        calib_split,
        calib_dets,
        eval_iou=config.eval_iou,
        context_scale=config.context_scale,
        square_crop=config.square_crop,
    )
    eval_flat_dets, eval_is_tp, eval_crops = _extract_detection_crops(
        eval_split,
        eval_dets,
        eval_iou=config.eval_iou,
        context_scale=config.context_scale,
        square_crop=config.square_crop,
    )

    max_k = max(config.k_values)
    pos_ref_crops, pos_ref_meta = _sample_train_positive_references(
        train_ref,
        max_k=max_k,
        seed=config.seed,
        context_scale=config.context_scale,
        square_crop=config.square_crop,
    )

    emb_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get_layer_embeddings(layer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if layer not in emb_cache:
            emb_cache[layer] = (
                enc.embed_crops(pos_ref_crops, layer=layer),
                enc.embed_crops(calib_crops, layer=layer),
                enc.embed_crops(eval_crops, layer=layer),
            )
        return emb_cache[layer]

    results_list: list[dict] = []
    fitted_models: dict[str, tuple[PrototypeFilter, float]] = {}
    skipped_variants: list[dict] = []

    pos_emb_l4, calib_emb_l4, eval_emb_l4 = get_layer_embeddings(PRIMARY_VARIANT.layer)
    for k in config.k_values:
        entry, proto, thresh = _evaluate_configuration(
            variant=PRIMARY_VARIANT,
            k=k,
            pos_emb_pool=pos_emb_l4,
            neg_emb_pool=None,
            calib_split=calib_split,
            calib_emb=calib_emb_l4,
            calib_is_tp=calib_is_tp,
            eval_split=eval_split,
            eval_flat_dets=eval_flat_dets,
            eval_emb=eval_emb_l4,
            eval_is_tp=eval_is_tp,
            eval_base_metrics=eval_base_metrics,
            eval_iou=config.eval_iou,
            bootstrap_resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        results_list.append(entry)
        fitted_models[entry["config_id"]] = (proto, thresh)

    primary_aucs = [
        r["separability"]["calib_roc_auc"]
        for r in results_list
        if r["separability"]["calib_roc_auc"] is not None
    ]
    best_primary_calib_auc = max(primary_aucs) if primary_aucs else None
    should_run_variations = (
        config.variations == "always"
        or (
            config.variations == "auto"
            and (best_primary_calib_auc is None or best_primary_calib_auc < config.auc_variation_trigger)
        )
    )

    neg_ref_meta: list[dict] = []
    neg_emb_by_layer: dict[str, np.ndarray] = {}
    if should_run_variations:
        neg_cache = config.cache_dir / "dets_train_neg_sample.json" if config.cache_dir else None
        neg_ref_crops, neg_ref_meta = _sample_train_negative_references(
            train_ref,
            det_runner,
            max_k=max_k,
            seed=config.seed,
            eval_iou=config.eval_iou,
            max_images=config.max_train_neg_images,
            context_scale=config.context_scale,
            square_crop=config.square_crop,
            cache_path=neg_cache,
        )
        for variant in CONTROLLED_VARIATIONS:
            pos_emb, calib_emb, eval_emb = get_layer_embeddings(variant.layer)
            neg_emb = None
            if variant.mode == "pos_minus_neg":
                if not neg_ref_crops:
                    skipped_variants.append(
                        {
                            "variant": variant.name,
                            "reason": "detector produced zero FPs in train reference sample",
                        }
                    )
                    continue
                if variant.layer not in neg_emb_by_layer:
                    neg_emb_by_layer[variant.layer] = enc.embed_crops(neg_ref_crops, layer=variant.layer)
                neg_emb = neg_emb_by_layer[variant.layer]
            for k in config.k_values:
                entry, proto, thresh = _evaluate_configuration(
                    variant=variant,
                    k=k,
                    pos_emb_pool=pos_emb,
                    neg_emb_pool=neg_emb,
                    calib_split=calib_split,
                    calib_emb=calib_emb,
                    calib_is_tp=calib_is_tp,
                    eval_split=eval_split,
                    eval_flat_dets=eval_flat_dets,
                    eval_emb=eval_emb,
                    eval_is_tp=eval_is_tp,
                    eval_base_metrics=eval_base_metrics,
                    eval_iou=config.eval_iou,
                    bootstrap_resamples=config.bootstrap_resamples,
                    seed=config.seed,
                )
                results_list.append(entry)
                fitted_models[entry["config_id"]] = (proto, thresh)

    chosen_entry = max(
        results_list,
        key=lambda r: (
            round(r["calibration"]["calib_best_f1"], 12),
            round(r["calibration"]["calib_best_recall"], 12),
            r["separability"]["calib_roc_auc"] if r["separability"]["calib_roc_auc"] is not None else -1.0,
            1 if r["stage"] == "primary" else 0,
            -r["k_requested"],
        ),
    )
    chosen_filter, chosen_threshold = fitted_models[chosen_entry["config_id"]]

    extra_results: list[dict] = []
    for ds_name, ds_yaml in config.extra_datasets:
        extra_results.append(
            _evaluate_extra_dataset(
                dataset_name=ds_name,
                data_yaml=ds_yaml,
                detector=det_runner,
                encoder=enc,
                chosen_entry=chosen_entry,
                chosen_filter=chosen_filter,
                frozen_threshold=chosen_threshold,
                config=config,
            )
        )
    for list_name, list_path in config.extra_val_lists:
        extra_split = load_image_list_split(
            list_name,
            list_path,
            images_root=config.images_root,
            labels_root=config.labels_root,
            path_remap=config.path_remap,
            target_class=config.target_class,
        )
        assert_disjoint_splits(train_ref, extra_split)
        extra_results.append(
            _evaluate_extra_split(
                dataset_name=list_name,
                source_desc=str(list_path),
                split=extra_split,
                detector=det_runner,
                encoder=enc,
                chosen_entry=chosen_entry,
                chosen_filter=chosen_filter,
                frozen_threshold=chosen_threshold,
                config=config,
            )
        )

    warnings: list[str] = []
    if calib_base_metrics["fp"] < 10:
        warnings.append(
            f"calibration split '{calib_split.name}' has only {calib_base_metrics['fp']} detector FPs; "
            "threshold selection and calibration AUC have high variance"
        )
    if eval_base_metrics["fp"] < 10:
        warnings.append(
            f"evaluation split '{eval_split.name}' has only {eval_base_metrics['fp']} detector FPs; "
            "precision headroom is narrow"
        )

    payload = {
        "protocol": {
            "data_yaml": str(config.data_yaml),
            "detector_weights": config.detector_weights.name,
            "fewshot_weights": config.fewshot_weights.name,
            "imgsz": config.imgsz,
            "conf": config.conf,
            "nms_iou": config.iou,
            "eval_iou": config.eval_iou,
            "k_values": list(config.k_values),
            "seed": config.seed,
            "calib_source": config.calib_source,
            "calib_fraction": config.calib_fraction,
            "bootstrap_resamples": config.bootstrap_resamples,
            "canvas_wh": list(config.canvas_wh),
            "context_scale": config.context_scale,
            "square_crop": config.square_crop,
            "variations_policy": config.variations,
            "auc_variation_trigger": config.auc_variation_trigger,
            "variations_executed": should_run_variations,
            "total_configs_tested": len(results_list),
            "skipped_variants": skipped_variants,
        },
        "splits": {
            "train_total_images": len(train_full.records),
            "train_ref_images": len(train_ref.records),
            "train_ref_gt_balls": count_gt_boxes(train_ref.records),
            "val_total_images": len(val_full.records),
            "val_list_source": str(config.val_list_override) if config.val_list_override else "data.yaml:val",
            "val_sampling": val_sampling,
            "calib_split_name": calib_split.name,
            "calib_images": len(calib_split.records),
            "calib_gt_balls": count_gt_boxes(calib_split.records),
            "eval_split_name": eval_split.name,
            "eval_images": len(eval_split.records),
            "eval_gt_balls": count_gt_boxes(eval_split.records),
            "train_val_disjoint_verified": True,
        },
        "references": {
            "positive_references_sampled": len(pos_ref_meta),
            "negative_references_sampled": len(neg_ref_meta),
            "positive_metadata": pos_ref_meta,
            "negative_metadata": neg_ref_meta,
        },
        "baseline_calib": _strip_per_image(calib_base_metrics),
        "baseline_eval": _strip_per_image(eval_base_metrics),
        "best_primary_calib_roc_auc": best_primary_calib_auc,
        "selected_config_id": chosen_entry["config_id"],
        "selection_rationale": (
            f"selected strictly on '{calib_split.name}' by highest calibration F1 "
            f"({chosen_entry['calibration']['calib_best_f1']:.4f} vs baseline "
            f"{chosen_entry['calibration']['calib_base_f1']:.4f}), tie-broken by recall and calib ROC-AUC; "
            f"threshold frozen before opening '{eval_split.name}'"
        ),
        "configurations": results_list,
        "extra_datasets": extra_results,
        "warnings": warnings,
    }

    sanitized = _sanitize_for_json(payload)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.output_dir / "experiment_results.json"
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_json.replace(json_path)

    from bola_fewshot.report import render_report

    report_text = render_report(sanitized)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = config.report_path.with_suffix(".md.tmp")
    tmp_report.write_text(report_text, encoding="utf-8")
    tmp_report.replace(config.report_path)
    return sanitized
