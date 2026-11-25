from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

MAP_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def match_image_detections(
    boxes: Sequence[Sequence[float]],
    confidences: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float = 0.50,
) -> np.ndarray:
    n_pred = len(boxes)
    is_tp = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or len(gt_boxes) == 0:
        return is_tp
    pred_arr = np.asarray(boxes, dtype=np.float64).reshape(n_pred, 4)
    gt_arr = np.asarray(gt_boxes, dtype=np.float64).reshape(len(gt_boxes), 4)
    conf_arr = np.asarray(confidences, dtype=np.float64)
    order = np.argsort(-conf_arr, kind="mergesort")
    iou = box_iou_matrix(pred_arr, gt_arr)
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    for pred_idx in order:
        candidates = np.where(~matched_gt, iou[pred_idx], -1.0)
        best_gt = int(np.argmax(candidates))
        if candidates[best_gt] >= iou_threshold:
            matched_gt[best_gt] = True
            is_tp[pred_idx] = True
    return is_tp


def compute_ap(
    confidences: Sequence[float] | np.ndarray,
    is_tp: Sequence[bool] | np.ndarray,
    total_gt: int,
) -> float:
    if total_gt <= 0 or len(confidences) == 0:
        return 0.0
    conf_arr = np.asarray(confidences, dtype=np.float64)
    tp_arr = np.asarray(is_tp, dtype=bool)
    order = np.argsort(-conf_arr, kind="mergesort")
    tp_sorted = tp_arr[order].astype(np.float64)
    fp_sorted = (~tp_arr[order]).astype(np.float64)
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recall = tp_cum / float(total_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    changes = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[changes] - mrec[changes - 1]) * mpre[changes]))


def precision_recall_f1(tp: int, fp: int, total_gt: int) -> tuple[float, float, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / total_gt) if total_gt > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return precision, recall, f1


def evaluate_detections(
    predictions_by_image: Mapping[str, Sequence[tuple[Sequence[float], float]]],
    gt_by_image: Mapping[str, Sequence[Sequence[float]]],
    image_ids: Sequence[str],
    iou_threshold: float = 0.50,
    map_iou_thresholds: Sequence[float] = MAP_IOU_THRESHOLDS,
) -> dict:
    total_gt = sum(len(gt_by_image.get(image_id, ())) for image_id in image_ids)
    total_tp = 0
    total_fp = 0
    per_image: dict[str, dict[str, int]] = {}
    all_confs: list[float] = []
    all_tp_by_thresh: dict[float, list[bool]] = {t: [] for t in map_iou_thresholds}
    primary_tp_list: list[bool] = []

    for image_id in image_ids:
        preds = predictions_by_image.get(image_id, ())
        gts = gt_by_image.get(image_id, ())
        boxes = [item[0] for item in preds]
        confs = [float(item[1]) for item in preds]
        all_confs.extend(confs)

        primary_match = match_image_detections(boxes, confs, gts, iou_threshold=iou_threshold)
        primary_tp_list.extend(primary_match.tolist())
        img_tp = int(primary_match.sum())
        img_fp = int(len(preds) - img_tp)
        total_tp += img_tp
        total_fp += img_fp
        per_image[image_id] = {"tp": img_tp, "fp": img_fp, "gt": len(gts)}

        for thresh in map_iou_thresholds:
            if abs(thresh - iou_threshold) < 1e-9:
                all_tp_by_thresh[thresh].extend(primary_match.tolist())
            else:
                matched = match_image_detections(boxes, confs, gts, iou_threshold=thresh)
                all_tp_by_thresh[thresh].extend(matched.tolist())

    fn = total_gt - total_tp
    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_gt)
    ap_by_iou = {
        f"{thresh:.2f}": compute_ap(all_confs, all_tp_by_thresh[thresh], total_gt)
        for thresh in map_iou_thresholds
    }
    ap50 = ap_by_iou.get("0.50", compute_ap(all_confs, primary_tp_list, total_gt))
    map50_95 = float(np.mean(list(ap_by_iou.values()))) if ap_by_iou else 0.0

    max_precision = 1.0 if total_tp > 0 else 0.0
    precision_headroom = max_precision - precision
    max_f1 = float((2.0 * recall) / (1.0 + recall)) if recall > 0.0 else 0.0
    f1_headroom = max_f1 - f1
    n_images = len(image_ids)

    return {
        "n_images": n_images,
        "total_gt": int(total_gt),
        "total_pred": int(total_tp + total_fp),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": ap50,
        "map50_95": map50_95,
        "fp_per_image": float(total_fp / n_images) if n_images > 0 else 0.0,
        "headroom": {
            "max_precision_perfect_filter": max_precision,
            "precision_headroom": precision_headroom,
            "max_f1_perfect_filter": max_f1,
            "f1_headroom": f1_headroom,
        },
        "per_image": per_image,
    }


def compute_filter_cost(base_is_tp: Sequence[bool] | np.ndarray, kept_mask: Sequence[bool] | np.ndarray) -> dict:
    tp_arr = np.asarray(base_is_tp, dtype=bool)
    kept_arr = np.asarray(kept_mask, dtype=bool)
    if len(tp_arr) != len(kept_arr):
        raise ValueError("base_is_tp and kept_mask must have the same length")
    removed = ~kept_arr
    base_fp = int((~tp_arr).sum())
    base_tp = int(tp_arr.sum())
    fps_removed = int((removed & (~tp_arr)).sum())
    tps_removed = int((removed & tp_arr).sum())
    return {
        "base_tp": base_tp,
        "base_fp": base_fp,
        "tps_kept": int((kept_arr & tp_arr).sum()),
        "fps_kept": int((kept_arr & (~tp_arr)).sum()),
        "tps_removed": tps_removed,
        "fps_removed": fps_removed,
        "fp_removal_rate": float(fps_removed / base_fp) if base_fp > 0 else 0.0,
        "tp_loss_rate": float(tps_removed / base_tp) if base_tp > 0 else 0.0,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    n = len(values)
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def compute_roc_auc(scores: Sequence[float] | np.ndarray, labels: Sequence[bool | int] | np.ndarray) -> float | None:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(s)
    pos_rank_sum = float(ranks[y].sum())
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_pr_auc(scores: Sequence[float] | np.ndarray, labels: Sequence[bool | int] | np.ndarray) -> float | None:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    distinct_ends = np.r_[np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[distinct_ends].astype(np.float64)
    fps = (distinct_ends + 1 - tps).astype(np.float64)
    recall = tps / float(n_pos)
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def paired_bootstrap_diff(
    base_per_image: Mapping[str, Mapping[str, int]],
    filt_per_image: Mapping[str, Mapping[str, int]],
    image_ids: Sequence[str],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    n = len(image_ids)
    if n == 0:
        raise ValueError("image_ids must be non-empty for bootstrap")

    base_tp = np.array([base_per_image[i]["tp"] for i in image_ids], dtype=np.float64)
    base_fp = np.array([base_per_image[i]["fp"] for i in image_ids], dtype=np.float64)
    base_gt = np.array([base_per_image[i]["gt"] for i in image_ids], dtype=np.float64)

    filt_tp = np.array([filt_per_image[i]["tp"] for i in image_ids], dtype=np.float64)
    filt_fp = np.array([filt_per_image[i]["fp"] for i in image_ids], dtype=np.float64)

    bp0, br0, bf10 = precision_recall_f1(int(base_tp.sum()), int(base_fp.sum()), int(base_gt.sum()))
    fp0, fr0, ff10 = precision_recall_f1(int(filt_tp.sum()), int(filt_fp.sum()), int(base_gt.sum()))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))

    b_tp = base_tp[idx].sum(axis=1)
    b_fp = base_fp[idx].sum(axis=1)
    gt = base_gt[idx].sum(axis=1)

    f_tp = filt_tp[idx].sum(axis=1)
    f_fp = filt_fp[idx].sum(axis=1)

    b_p = np.divide(b_tp, b_tp + b_fp, out=np.zeros_like(b_tp), where=(b_tp + b_fp) > 0)
    b_r = np.divide(b_tp, gt, out=np.zeros_like(b_tp), where=gt > 0)
    b_f1 = np.divide(2.0 * b_p * b_r, b_p + b_r, out=np.zeros_like(b_p), where=(b_p + b_r) > 0)

    f_p = np.divide(f_tp, f_tp + f_fp, out=np.zeros_like(f_tp), where=(f_tp + f_fp) > 0)
    f_r = np.divide(f_tp, gt, out=np.zeros_like(f_tp), where=gt > 0)
    f_f1 = np.divide(2.0 * f_p * f_r, f_p + f_r, out=np.zeros_like(f_p), where=(f_p + f_r) > 0)

    def summarize(point: float, diffs: np.ndarray) -> dict:
        low, high = np.percentile(diffs, [2.5, 97.5])
        low_f, high_f = float(low), float(high)
        distinguishable = bool(low_f > 0.0 or high_f < 0.0)
        if low_f > 0.0:
            direction = "improvement"
        elif high_f < 0.0:
            direction = "degradation"
        else:
            direction = "indistinguishable"
        return {
            "point_estimate": float(point),
            "bootstrap_mean": float(diffs.mean()),
            "ci_95_low": low_f,
            "ci_95_high": high_f,
            "distinguishable_from_zero": distinguishable,
            "direction": direction,
        }

    return {
        "n_images": n,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "delta_f1": summarize(ff10 - bf10, f_f1 - b_f1),
        "delta_precision": summarize(fp0 - bp0, f_p - b_p),
        "delta_recall": summarize(fr0 - br0, f_r - b_r),
    }
