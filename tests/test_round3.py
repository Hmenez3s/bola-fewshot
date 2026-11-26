import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from bola_fewshot.dataset import ImageRecord, load_yolo_dataset
from bola_fewshot.detector import Detection
from bola_fewshot.round3_per_game import (
    Round3Config,
    apply_logistic_fusion,
    apply_post_nms,
    apply_top1_per_frame,
    fit_doubt_zone_filter,
    fit_logistic_fusion,
    run_round3_experiment,
    select_game_auto_references,
)
from tests.test_experiment_e2e import _draw_synthetic_pitch_frame, _write_yolo_label

REAL_FEWSHOT_WEIGHTS = Path("/Users/j.henrique/four-reference-team-classifier/models/six-references.pth")
REAL_IMAGENET_WEIGHTS = Path("/Users/j.henrique/four-reference-team-classifier/data/inputs/resnet18.pth")


def test_post_nms_and_top1() -> None:
    d1 = Detection("img1", (10.0, 10.0, 30.0, 30.0), 0.90)
    d2 = Detection("img1", (12.0, 12.0, 32.0, 32.0), 0.70)
    d3 = Detection("img1", (100.0, 100.0, 120.0, 120.0), 0.50)
    nms_res = apply_post_nms({"img1": [d1, d2, d3]}, iou_threshold=0.50)
    assert len(nms_res["img1"]) == 2
    assert nms_res["img1"][0].conf == 0.90
    assert nms_res["img1"][1].conf == 0.50

    top1 = apply_top1_per_frame({"img1": [d1, d2, d3]})
    assert len(top1["img1"]) == 1
    assert top1["img1"][0].conf == 0.90


def test_logistic_fusion_and_doubt_zone() -> None:
    confs = np.array([0.85, 0.80, 0.25, 0.20, 0.22, 0.18])
    scores = np.array([0.90, 0.85, 0.80, -0.50, -0.60, -0.70])
    is_tp = np.array([True, True, True, False, False, False])

    lf = fit_logistic_fusion(confs, scores, is_tp, total_gt=3)
    fused = apply_logistic_fusion(lf, confs, scores)
    assert fused[:3].min() > fused[3:].max()
    assert lf["calibration"]["calib_best_f1"] == pytest.approx(1.0)

    dz = fit_doubt_zone_filter(confs, scores, is_tp, total_gt=3)
    assert dz["calib_best_f1"] == pytest.approx(1.0)


class _MultiGameSynthDetector:
    def __init__(self, boxes_by_name: dict[str, list[tuple[tuple[float, float, float, float], float]]]) -> None:
        self._boxes = boxes_by_name

    def predict_records(
        self,
        records: Sequence[ImageRecord],
        cache_path: Path | None = None,
    ) -> dict[str, list[Detection]]:
        return {
            r.image_id: [
                Detection(image_id=r.image_id, box=box, conf=conf)
                for box, conf in self._boxes.get(r.image_path.name, [])
            ]
            for r in records
        }


@pytest.mark.skipif(
    not (REAL_FEWSHOT_WEIGHTS.is_file() and REAL_IMAGENET_WEIGHTS.is_file()),
    reason="local FewShot-R18 and ImageNet weights not present",
)
def test_round3_per_game_pipeline_synthetic(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    boxes_by_name: dict[str, list[tuple[tuple[float, float, float, float], float]]] = {}
    gmap: dict[str, str] = {}
    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for g in range(3):
        for i in range(10):
            bx = 40 + (i * 15) % 160
            by = 40 + (i * 13) % 120
            ball = (bx, by, bx + 24, by + 24)
            fp = (230, 140, 254, 174)
            img = root / "images" / f"tr_g{g}_{i:02d}.png"
            lbl = root / "labels" / f"tr_g{g}_{i:02d}.txt"
            _draw_synthetic_pitch_frame(img, ball, [fp], seed=g * 100 + i)
            _write_yolo_label(lbl, [ball])
            train_paths.append(img)
            gmap[img.name] = f"tr_game_{g}"
            boxes_by_name[img.name] = [
                (tuple(float(v) for v in ball), 0.85),
                (tuple(float(v) for v in fp), 0.14),
            ]

    for g in range(5):
        n_frames = 10 if g < 4 else 2
        for i in range(n_frames):
            bx = 45 + (i * 17) % 150
            by = 45 + (i * 11) % 120
            ball = (bx, by, bx + 24, by + 24)
            fp = (225, 135, 249, 169)
            img = root / "images" / f"va_g{g}_{i:02d}.png"
            lbl = root / "labels" / f"va_g{g}_{i:02d}.txt"
            _draw_synthetic_pitch_frame(img, ball, [fp], seed=500 + g * 100 + i)
            _write_yolo_label(lbl, [ball])
            val_paths.append(img)
            gmap[img.name] = f"va_game_{g}"
            ball_conf = 0.80 if i < 7 else 0.25
            boxes_by_name[img.name] = [
                (tuple(float(v) for v in ball), ball_conf),
                (tuple(float(v) for v in fp), 0.40),
                ((float(ball[0] + 2), float(ball[1] + 2), float(ball[2] + 2), float(ball[3] + 2)), 0.36),
            ]

    (root / "train.txt").write_text("\n".join(str(p) for p in train_paths) + "\n", encoding="utf-8")
    (root / "val.txt").write_text("\n".join(str(p) for p in val_paths) + "\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("train: train.txt\nval: val.txt\n", encoding="utf-8")
    gmap_path = root / "gmap.json"
    gmap_path.write_text(json.dumps(gmap), encoding="utf-8")

    det = _MultiGameSynthDetector(boxes_by_name)
    splits = load_yolo_dataset(data_yaml)
    val_gmap = {r.image_id: gmap[r.image_path.name] for r in splits["val"].records}
    auto = select_game_auto_references(
        splits["val"], det.predict_records(splits["val"].records), val_gmap, k_refs=6, ref_min_conf=0.60
    )
    assert auto["coverage"]["eligible_games"] == 4
    assert auto["coverage"]["eligible_ref_images_excluded"] == 24
    assert set(auto["eligible_query_ids"]).isdisjoint(auto["ref_image_ids"])
    assert auto["coverage"]["auto_ref_precision_iou50"] == pytest.approx(1.0)

    cfg = Round3Config(
        data_yaml=data_yaml,
        detector_weights=tmp_path / "unused.pt",
        fewshot_weights=REAL_FEWSHOT_WEIGHTS,
        imagenet_weights=REAL_IMAGENET_WEIGHTS,
        output_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        group_map_path=gmap_path,
        operating_confs=(0.35, 0.15),
        context_scales=(2.0,),
        canvas_wh=(64, 64),
        n_refs_per_game=6,
        ref_min_conf=0.60,
        n_folds=5,
        seed=42,
        train_seed=25,
        train_epochs=2,
        train_steps_per_epoch=5,
        bootstrap_resamples=100,
        device="cpu",
    )
    payload = run_round3_experiment(cfg, val_detector=det, mining_detector=det)
    assert (tmp_path / "results" / "rodada3_experiment_results.json").is_file()
    assert payload["auto_reference_audit"]["eligible_games"] == 4
    op35 = payload["operating_points"]["conf_0.35"]
    assert op35["cheap_baselines"]["top1_per_frame"]["all_val"]["filtered"]["f1"] > op35["baseline_all_val"]["f1"]
    assert op35["methods"]["trained_per_game_logistic_fusion"]["eligible_queries"]["filtered"]["f1"] >= op35["baseline_eligible_queries"]["f1"]
