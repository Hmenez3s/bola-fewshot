from pathlib import Path

from PIL import Image
import pytest
import torch

from bola_fewshot.dataset import load_yolo_dataset
from bola_fewshot.fewshot_filter import FewShotR18Encoder
from bola_fewshot.metrics import compute_roc_auc
from bola_fewshot.train_ball import (
    BallTrainConfig,
    mine_training_crop_pool,
    train_fewshot_ball,
)
from tests.test_experiment_e2e import _build_synthetic_dataset

IMAGENET_RESNET18 = Path("/Users/j.henrique/four-reference-team-classifier/data/inputs/resnet18.pth")


@pytest.mark.skipif(not IMAGENET_RESNET18.is_file(), reason="local ImageNet resnet18.pth not present")
def test_mine_and_train_fewshot_ball_on_synthetic_data(tmp_path: Path) -> None:
    data_yaml, _, synth_detector = _build_synthetic_dataset(tmp_path / "ds")
    splits = load_yolo_dataset(data_yaml)
    train_split = splits["train"]
    val_split = splits["val"]

    pool = mine_training_crop_pool(
        train_split,
        synth_detector,
        eval_conf=0.35,
        mining_conf=0.05,
        max_images=12,
        min_negatives=12,
        seed=25,
    )
    assert pool.stats["pool_pos_crops"] >= 12
    assert pool.stats["pool_neg_crops"] >= 12
    assert pool.stats["train_fp_high_conf_mined"] == 12
    assert pool.stats["background_supplement_crops"] == 0

    val_dets = synth_detector.predict_records(val_split.records)
    val_crops = []
    val_is_tp = []
    for record in val_split.records:
        with Image.open(record.image_path) as img:
            rgb = img.convert("RGB")
            for idx, det in enumerate(val_dets[record.image_id]):
                x1, y1, x2, y2 = (int(v) for v in det.box)
                val_crops.append(rgb.crop((x1, y1, x2, y2)))
                val_is_tp.append(idx == 0)

    ref_pos = [Image.fromarray(pool.pos_arrays[i]) for i in range(6)]
    ref_neg = [Image.fromarray(pool.neg_arrays[i]) for i in range(6)]
    ckpt_path = tmp_path / "fewshot_ball.pth"

    summary = train_fewshot_ball(
        pool,
        val_calib_crops=val_crops,
        val_calib_is_tp=torch.tensor(val_is_tp, dtype=torch.bool).numpy(),
        val_calib_total_gt=len(val_split.records),
        fixed_pos_ref_crops=ref_pos,
        fixed_neg_ref_crops=ref_neg,
        config=BallTrainConfig(
            epochs=2,
            steps_per_epoch=10,
            references_per_class=6,
            queries_per_class=4,
            seed=25,
            device="cpu",
            imagenet_weights_path=IMAGENET_RESNET18,
            checkpoint_path=ckpt_path,
        ),
    )
    assert ckpt_path.is_file()
    assert summary["best_epoch"] in (1, 2)
    assert len(summary["history"]) == 2
    assert summary["history"][-1]["val_calib_roc_auc"] > 0.90

    loaded_encoder = FewShotR18Encoder(ckpt_path, device="cpu")
    emb = loaded_encoder.embed_crops(val_crops[:4], layer="layer4")
    assert emb.shape == (4, 512)
