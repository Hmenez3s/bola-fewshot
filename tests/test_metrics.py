import numpy as np
import pytest

from bola_fewshot.metrics import (
    box_iou_matrix,
    compute_ap,
    compute_filter_cost,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_detections,
    match_image_detections,
    paired_bootstrap_diff,
)


def test_iou_and_greedy_matching() -> None:
    gt = [(10.0, 10.0, 30.0, 30.0)]
    preds = [
        (10.0, 10.0, 30.0, 30.0),
        (11.0, 11.0, 31.0, 31.0),
        (100.0, 100.0, 120.0, 120.0),
    ]
    confs = [0.6, 0.9, 0.8]
    iou = box_iou_matrix(np.array(preds), np.array(gt))
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[2, 0] == pytest.approx(0.0)

    matched = match_image_detections(preds, confs, gt, iou_threshold=0.50)
    assert matched.tolist() == [False, True, False]


def test_compute_ap_exact_cases() -> None:
    assert compute_ap([0.9], [True], total_gt=1) == pytest.approx(1.0)
    assert compute_ap([0.9, 0.8], [False, True], total_gt=1) == pytest.approx(0.5)
    assert compute_ap([], [], total_gt=2) == pytest.approx(0.0)


def test_evaluate_detections_and_headroom() -> None:
    gt = {
        "img1": [(0.0, 0.0, 20.0, 20.0)],
        "img2": [(50.0, 50.0, 70.0, 70.0)],
    }
    preds = {
        "img1": [((0.0, 0.0, 20.0, 20.0), 0.9), ((100.0, 100.0, 120.0, 120.0), 0.7)],
        "img2": [((100.0, 100.0, 120.0, 120.0), 0.6)],
    }
    result = evaluate_detections(preds, gt, ["img1", "img2"])
    assert result["tp"] == 1
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["precision"] == pytest.approx(1.0 / 3.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.4)
    assert result["map50"] == pytest.approx(0.5)
    assert result["map50_95"] == pytest.approx(0.5)
    assert result["headroom"]["precision_headroom"] == pytest.approx(2.0 / 3.0)
    assert result["headroom"]["max_f1_perfect_filter"] == pytest.approx(2.0 / 3.0)
    assert result["headroom"]["f1_headroom"] == pytest.approx(2.0 / 3.0 - 0.4)


def test_roc_auc_and_pr_auc_including_ties() -> None:
    assert compute_roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert compute_roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert compute_roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)
    assert compute_roc_auc([0.9, 0.8], [1, 1]) is None
    assert compute_pr_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert compute_pr_auc([0.5, 0.5], [0, 0]) is None


def test_filter_cost_and_paired_bootstrap() -> None:
    cost = compute_filter_cost([True, False, False, True], [True, False, True, False])
    assert cost["fps_removed"] == 1
    assert cost["tps_removed"] == 1
    assert cost["fp_removal_rate"] == pytest.approx(0.5)
    assert cost["tp_loss_rate"] == pytest.approx(0.5)

    ids = [f"img_{i}" for i in range(20)]
    base = {i: {"tp": 1, "fp": 1, "gt": 1} for i in ids}
    filt = {i: {"tp": 1, "fp": 0, "gt": 1} for i in ids}
    boot = paired_bootstrap_diff(base, filt, ids, n_resamples=500, seed=7)
    assert boot["delta_f1"]["point_estimate"] == pytest.approx(1.0 - 2.0 / 3.0)
    assert boot["delta_f1"]["ci_95_low"] > 0.0
    assert boot["delta_f1"]["distinguishable_from_zero"] is True
    assert boot["delta_f1"]["direction"] == "improvement"

    same = paired_bootstrap_diff(base, base, ids, n_resamples=200, seed=7)
    assert same["delta_f1"]["distinguishable_from_zero"] is False
    assert same["delta_f1"]["direction"] == "indistinguishable"
