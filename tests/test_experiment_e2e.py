import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import pytest

from bola_fewshot.dataset import ImageRecord
from bola_fewshot.detector import Detection, YoloBallDetector
from bola_fewshot.experiment import ExperimentConfig, run_experiment
from bola_fewshot.fewshot_filter import FewShotR18Encoder

REAL_FEWSHOT_WEIGHTS = Path("/Users/j.henrique/four-reference-team-classifier/models/six-references.pth")
REAL_IMAGENET_WEIGHTS = Path("/Users/j.henrique/four-reference-team-classifier/data/inputs/resnet18.pth")
REAL_YOLO_WEIGHTS = Path("models/t1b_v5_etapa1_rodada1.pt")


def _draw_synthetic_pitch_frame(
    path: Path,
    ball_box: tuple[int, int, int, int] | None,
    fp_boxes: Sequence[tuple[int, int, int, int]],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    base = np.zeros((240, 320, 3), dtype=np.uint8)
    base[:, :, 0] = rng.integers(25, 40, size=(240, 320), dtype=np.uint8)
    base[:, :, 1] = rng.integers(115, 135, size=(240, 320), dtype=np.uint8)
    base[:, :, 2] = rng.integers(25, 40, size=(240, 320), dtype=np.uint8)
    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)
    if ball_box is not None:
        draw.ellipse(ball_box, fill=(248, 248, 248), outline=(25, 25, 25))
    for fp in fp_boxes:
        draw.rectangle(fp, fill=(210, 35, 35), outline=(90, 10, 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_yolo_label(path: Path, boxes: Sequence[tuple[int, int, int, int]], w: int = 320, h: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class _SyntheticDetector:
    def __init__(self, boxes_by_name: dict[str, list[tuple[tuple[float, float, float, float], float]]]) -> None:
        self._boxes = boxes_by_name

    def predict_records(
        self,
        records: Sequence[ImageRecord],
        cache_path: Path | None = None,
    ) -> dict[str, list[Detection]]:
        out: dict[str, list[Detection]] = {}
        for record in records:
            items = self._boxes.get(record.image_path.name, [])
            out[record.image_id] = [
                Detection(image_id=record.image_id, box=box, conf=conf)
                for box, conf in items
            ]
        return out


def _build_synthetic_dataset(root: Path) -> tuple[Path, Path, _SyntheticDetector]:
    boxes_by_name: dict[str, list[tuple[tuple[float, float, float, float], float]]] = {}
    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for i in range(12):
        bx = 40 + (i * 17) % 180
        by = 40 + (i * 19) % 140
        ball = (bx, by, bx + 26, by + 26)
        fp = (240 - (i * 11) % 80, 150 - (i * 13) % 70, 266 - (i * 11) % 80, 186 - (i * 13) % 70)
        img_path = root / "r1" / "images" / f"train_{i:02d}.png"
        lbl_path = root / "r1" / "labels" / f"train_{i:02d}.txt"
        _draw_synthetic_pitch_frame(img_path, ball, [fp], seed=i)
        _write_yolo_label(lbl_path, [ball])
        train_paths.append(img_path)
        boxes_by_name[img_path.name] = [((float(ball[0]), float(ball[1]), float(ball[2]), float(ball[3])), 0.85), ((float(fp[0]), float(fp[1]), float(fp[2]), float(fp[3])), 0.55)]

    for i in range(20):
        bx = 35 + (i * 23) % 190
        by = 35 + (i * 29) % 150
        ball = (bx, by, bx + 26, by + 26)
        fp = (220 - (i * 9) % 90, 140 - (i * 7) % 80, 246 - (i * 9) % 90, 176 - (i * 7) % 80)
        img_path = root / "r1" / "images" / f"val_{i:02d}.png"
        lbl_path = root / "r1" / "labels" / f"val_{i:02d}.txt"
        _draw_synthetic_pitch_frame(img_path, ball, [fp], seed=100 + i)
        _write_yolo_label(lbl_path, [ball])
        val_paths.append(img_path)
        boxes_by_name[img_path.name] = [
            ((float(ball[0]), float(ball[1]), float(ball[2]), float(ball[3])), 0.82),
            ((float(fp[0]), float(fp[1]), float(fp[2]), float(fp[3])), 0.60),
        ]

    split_dir = root / "r5_9010"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.txt").write_text("\n".join(str(p) for p in train_paths) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(str(p) for p in val_paths) + "\n", encoding="utf-8")
    data_yaml = split_dir / "data.yaml"
    data_yaml.write_text("train: train.txt\nval: val.txt\nnames:\n  0: ball\n", encoding="utf-8")

    b2b_dir = root / "box2boxcam"
    for i in range(8):
        bx = 50 + (i * 21) % 160
        by = 50 + (i * 15) % 120
        ball = (bx, by, bx + 24, by + 24)
        fp1 = (230, 30 + i * 15, 254, 62 + i * 15)
        fp2 = (20, 150, 44, 182)
        img_path = b2b_dir / "images" / f"b2b_{i:02d}.png"
        lbl_path = b2b_dir / "labels" / f"b2b_{i:02d}.txt"
        _draw_synthetic_pitch_frame(img_path, ball, [fp1, fp2], seed=500 + i)
        _write_yolo_label(lbl_path, [ball])
        boxes_by_name[img_path.name] = [
            ((float(ball[0]), float(ball[1]), float(ball[2]), float(ball[3])), 0.70),
            ((float(fp1[0]), float(fp1[1]), float(fp1[2]), float(fp1[3])), 0.50),
            ((float(fp2[0]), float(fp2[1]), float(fp2[2]), float(fp2[3])), 0.45),
        ]
    b2b_yaml = b2b_dir / "data.yaml"
    b2b_yaml.write_text("path: .\nval: images\nnames:\n  0: ball\n", encoding="utf-8")

    return data_yaml, b2b_yaml, _SyntheticDetector(boxes_by_name)


@pytest.mark.skipif(not REAL_FEWSHOT_WEIGHTS.is_file(), reason="local FewShot-R18 weights not present")
def test_end_to_end_pipeline_with_real_fewshot_r18_on_synthetic_data(tmp_path: Path) -> None:
    data_yaml, b2b_yaml, synth_detector = _build_synthetic_dataset(tmp_path / "ds")
    out_dir = tmp_path / "results"
    report_path = tmp_path / "RELATORIO.md"

    config = ExperimentConfig(
        data_yaml=data_yaml,
        detector_weights=tmp_path / "unused_detector.pt",
        fewshot_weights=REAL_FEWSHOT_WEIGHTS,
        output_dir=out_dir,
        report_path=report_path,
        k_values=(1, 6),
        seed=42,
        calib_source="val_split",
        calib_fraction=0.30,
        bootstrap_resamples=400,
        variations="always",
        device="cpu",
        extra_datasets=(("box2boxcam", b2b_yaml),),
    )
    encoder = FewShotR18Encoder(REAL_FEWSHOT_WEIGHTS, device="cpu")
    payload = run_experiment(config, detector=synth_detector, encoder=encoder)

    assert (out_dir / "experiment_results.json").is_file()
    assert report_path.is_file()
    loaded = json.loads((out_dir / "experiment_results.json").read_text(encoding="utf-8"))

    assert loaded["splits"]["train_val_disjoint_verified"] is True
    assert loaded["splits"]["calib_images"] == 6
    assert loaded["splits"]["eval_images"] == 14
    assert loaded["protocol"]["total_configs_tested"] == 8

    base = loaded["baseline_eval"]
    assert base["tp"] == 14
    assert base["fp"] == 14
    assert base["precision"] == pytest.approx(0.5)
    assert base["recall"] == pytest.approx(1.0)

    selected = next(c for c in loaded["configurations"] if c["config_id"] == loaded["selected_config_id"])
    assert selected["separability"]["eval_roc_auc"] > 0.90
    assert selected["eval_metrics"]["f1"] > base["f1"]
    assert selected["filter_cost"]["fps_removed"] > 0
    assert selected["bootstrap"]["delta_f1"]["distinguishable_from_zero"] is True
    assert selected["bootstrap"]["delta_f1"]["direction"] == "improvement"

    assert len(loaded["extra_datasets"]) == 1
    assert loaded["extra_datasets"][0]["name"] == "box2boxcam"
    assert loaded["extra_datasets"][0]["separability"]["roc_auc"] > 0.90

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Resumo (8 linhas)" in report_text
    assert "## O FewShot-R18 separa bola de falso positivo?" in report_text
    assert "## Teto de ganho possível (headroom)" in report_text
    assert "### box2boxcam" in report_text
    summary_section = report_text.split("## Resumo (8 linhas)\n\n", 1)[1].split("\n\n## ", 1)[0]
    summary_lines = [line for line in summary_section.splitlines() if line.strip()]
    assert len(summary_lines) == 8


@pytest.mark.skipif(not REAL_FEWSHOT_WEIGHTS.is_file(), reason="local FewShot-R18 weights not present")
def test_train_holdout_calibration_and_missing_extra_dataset_report(tmp_path: Path) -> None:
    data_yaml, _, synth_detector = _build_synthetic_dataset(tmp_path / "ds")
    out_dir = tmp_path / "results"
    report_path = tmp_path / "RELATORIO.md"
    config = ExperimentConfig(
        data_yaml=data_yaml,
        detector_weights=tmp_path / "unused_detector.pt",
        fewshot_weights=REAL_FEWSHOT_WEIGHTS,
        output_dir=out_dir,
        report_path=report_path,
        k_values=(1, 6),
        seed=7,
        calib_source="train_holdout",
        calib_fraction=0.30,
        bootstrap_resamples=200,
        variations="never",
        device="cpu",
    )
    encoder = FewShotR18Encoder(REAL_FEWSHOT_WEIGHTS, device="cpu")
    payload = run_experiment(config, detector=synth_detector, encoder=encoder)
    assert payload["splits"]["eval_split_name"] == "val"
    assert payload["splits"]["eval_images"] == 20
    assert payload["protocol"]["total_configs_tested"] == 2
    report_text = report_path.read_text(encoding="utf-8")
    assert "Não medido:" in report_text


@pytest.mark.skipif(not REAL_YOLO_WEIGHTS.is_file(), reason="local YOLO v5 weights not downloaded")
def test_real_yolo_ball_detector_smoke_and_cache(tmp_path: Path) -> None:
    img_path = tmp_path / "images" / "frame_0.png"
    _draw_synthetic_pitch_frame(img_path, (100, 100, 124, 124), [], seed=1)
    record = ImageRecord(
        image_id="frame_0.png",
        image_path=img_path,
        label_path=None,
        width=320,
        height=240,
        gt_boxes=(),
    )
    detector = YoloBallDetector(REAL_YOLO_WEIGHTS, imgsz=640, conf=0.35, device="cpu")
    cache_path = tmp_path / "cache" / "dets.json"
    first = detector.predict_records([record], cache_path=cache_path)
    assert "frame_0.png" in first
    assert cache_path.is_file()
    second = detector.predict_records([record], cache_path=cache_path)
    assert second == first


@pytest.mark.skipif(
    not (REAL_FEWSHOT_WEIGHTS.is_file() and REAL_IMAGENET_WEIGHTS.is_file()),
    reason="local FewShot-R18 and ImageNet weights not present",
)
def test_three_arm_pipeline_with_trained_ball_on_synthetic_data(tmp_path: Path) -> None:
    data_yaml, b2b_yaml, synth_detector = _build_synthetic_dataset(tmp_path / "ds")
    out_dir = tmp_path / "results"
    report_path = tmp_path / "RELATORIO.md"
    ckpt_path = tmp_path / "trained_ball.pth"

    config = ExperimentConfig(
        data_yaml=data_yaml,
        detector_weights=tmp_path / "unused.pt",
        fewshot_weights=REAL_FEWSHOT_WEIGHTS,
        output_dir=out_dir,
        report_path=report_path,
        k_values=(1, 6),
        seed=42,
        calib_source="val_split",
        calib_fraction=0.50,
        bootstrap_resamples=200,
        variations="never",
        device="cpu",
        extra_datasets=(("box2boxcam", b2b_yaml),),
        train_ball_arm=True,
        train_epochs=2,
        train_steps_per_epoch=8,
        train_seed=25,
        train_mining_max_images=12,
        train_min_negatives=12,
        imagenet_weights_path=REAL_IMAGENET_WEIGHTS,
        trained_checkpoint_path=ckpt_path,
    )
    encoder = FewShotR18Encoder(REAL_FEWSHOT_WEIGHTS, device="cpu")
    payload = run_experiment(config, detector=synth_detector, encoder=encoder)
    assert ckpt_path.is_file()
    assert payload["selected_trained_config_id"] is not None
    assert payload["arm3_vs_arm2_bootstrap"] is not None
    assert payload["trained_ball_summary"]["best_epoch"] in (1, 2)
    assert len(payload["extra_datasets"][0]["arms"]) == 2
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Treino episódico do FewShot-R18-bola (receita do paper)" in report_text
    summary_section = report_text.split("## Resumo (8 linhas)\n\n", 1)[1].split("\n\n## ", 1)[0]
    summary_lines = [line for line in summary_section.splitlines() if line.strip()]
    assert len(summary_lines) == 8
