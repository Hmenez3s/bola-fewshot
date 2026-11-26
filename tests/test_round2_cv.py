import json
from pathlib import Path

import pytest

from bola_fewshot.cv_round2 import Round2Config, run_round2_cv
from bola_fewshot.dataset import load_yolo_dataset, make_group_kfold
from bola_fewshot.fp_audit import build_fp_sample_grid, estimate_noise_adjusted_metrics
from tests.test_experiment_e2e import _build_synthetic_dataset

REAL_FEWSHOT_WEIGHTS = Path("/Users/j.henrique/four-reference-team-classifier/models/six-references.pth")
REAL_IMAGENET_WEIGHTS = Path("/Users/j.henrique/four-reference-team-classifier/data/inputs/resnet18.pth")


def test_make_group_kfold_disjoint_and_complete(tmp_path: Path) -> None:
    data_yaml, _, _ = _build_synthetic_dataset(tmp_path / "ds")
    splits = load_yolo_dataset(data_yaml)
    val = splits["val"]
    group_by = {r.image_id: f"g_{idx // 2}" for idx, r in enumerate(val.records)}
    folds = make_group_kfold(val, n_folds=5, seed=42, group_by=group_by)
    assert len(folds) == 5
    seen_imgs = set()
    seen_groups = set()
    for fl in folds:
        fg = {group_by[r.image_id] for r in fl.records}
        assert seen_groups.isdisjoint(fg)
        seen_groups.update(fg)
        seen_imgs.update(fl.image_ids)
    assert seen_imgs == set(val.image_ids)


def test_fp_audit_grid_and_noise_adjustment(tmp_path: Path) -> None:
    data_yaml, _, synth_detector = _build_synthetic_dataset(tmp_path / "ds")
    splits = load_yolo_dataset(data_yaml)
    val = splits["val"]
    dets = synth_detector.predict_records(val.records)
    out_img = tmp_path / "amostra_fp.png"
    out_meta = tmp_path / "amostra_fp.json"
    meta = build_fp_sample_grid(
        val,
        dets,
        output_image=out_img,
        output_metadata=out_meta,
        min_conf=0.50,
        sample_size=10,
        seed=42,
    )
    assert out_img.is_file()
    assert out_meta.is_file()
    assert meta["sample_size"] == 10

    adj = estimate_noise_adjusted_metrics(
        base_metrics={"tp": 100, "fp": 40, "total_gt": 120},
        filtered_metrics={"tp": 95, "fp": 20, "total_gt": 120},
        n_audited=40,
        n_unannotated_real_balls=4,
        n_ambiguous=2,
        n_true_fps=34,
        high_conf_fp_count=30,
        total_fp_count=40,
    )
    assert adj["is_estimate"] is True
    assert adj["original_labels_modified"] is False
    assert adj["unannotated_ball_rate_strict"] == pytest.approx(0.10)


@pytest.mark.skipif(
    not (REAL_FEWSHOT_WEIGHTS.is_file() and REAL_IMAGENET_WEIGHTS.is_file()),
    reason="local FewShot-R18 and ImageNet weights not present",
)
def test_round2_nested_cv_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    data_yaml, _, synth_detector = _build_synthetic_dataset(tmp_path / "ds")
    splits = load_yolo_dataset(data_yaml)
    gmap = {}
    for idx, r in enumerate(splits["train"].records):
        gmap[r.image_path.name] = f"tr_g{idx // 2}"
    for idx, r in enumerate(splits["val"].records):
        gmap[r.image_path.name] = f"va_g{idx // 2}"
    gmap_path = tmp_path / "gmap.json"
    gmap_path.write_text(json.dumps(gmap), encoding="utf-8")

    cfg = Round2Config(
        data_yaml=data_yaml,
        detector_weights=tmp_path / "unused.pt",
        fewshot_weights=REAL_FEWSHOT_WEIGHTS,
        imagenet_weights=REAL_IMAGENET_WEIGHTS,
        output_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        group_map_path=gmap_path,
        context_scales=(2.0, 3.0),
        canvas_wh=(64, 64),
        k_values=(1, 6),
        n_folds=5,
        seed=42,
        train_seed=25,
        train_epochs=2,
        train_steps_per_epoch=5,
        bootstrap_resamples=100,
        device="cpu",
        include_all_layers_variation=True,
    )
    payload = run_round2_cv(cfg, val_detector=synth_detector, mining_detector=synth_detector)
    assert (tmp_path / "results" / "rodada2_experiment_results.json").is_file()
    assert payload["splits"]["train_val_group_audit"]["shared_groups"] == 0
    assert set(payload["nested_arms"].keys()) == {
        "frozen_players",
        "trained_ball_layer34",
        "trained_ball_all_layers",
    }
    assert payload["nested_arms"]["trained_ball_layer34"]["pooled_oof_roc_auc"] > 0.90
    assert payload["nested_arms"]["trained_ball_layer34"]["bootstrap_group"] is not None
