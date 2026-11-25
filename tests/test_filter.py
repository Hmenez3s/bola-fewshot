import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

from bola_fewshot.fewshot_filter import (
    FewShotR18Encoder,
    PrototypeFilter,
    calibrate_threshold,
    letterbox_crop,
)
from bola_fewshot.metrics import compute_roc_auc

REAL_FEWSHOT_WEIGHTS = Path("/Users/j.henrique/four-reference-team-classifier/models/six-references.pth")


def _make_ball_crop(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (40, 40), (35, 130, 40))
    draw = ImageDraw.Draw(img)
    r = int(rng.integers(10, 14))
    cx = int(rng.integers(18, 22))
    cy = int(rng.integers(18, 22))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(245, 245, 245), outline=(20, 20, 20))
    return img


def _make_fp_crop(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (40, 40), (30, 110, 35))
    draw = ImageDraw.Draw(img)
    x0 = int(rng.integers(4, 12))
    draw.rectangle((x0, 4, x0 + 18, 36), fill=(200, 30, 30))
    return img


def test_letterbox_crop_preserves_aspect_ratio() -> None:
    crop = Image.new("RGB", (30, 30), (255, 255, 255))
    canvas = letterbox_crop(crop, canvas_wh=(48, 96))
    assert canvas.shape == (96, 48, 3)
    assert canvas.dtype == np.uint8
    assert canvas[0, 0].tolist() == [0, 0, 0]
    assert canvas[48, 24].tolist() == [255, 255, 255]


@pytest.mark.skipif(not REAL_FEWSHOT_WEIGHTS.is_file(), reason="local FewShot-R18 weights not present")
def test_real_fewshot_r18_encoder_layers_and_prototype_separability() -> None:
    encoder = FewShotR18Encoder(REAL_FEWSHOT_WEIGHTS, device="cpu", canvas_wh=(48, 96))
    ref_balls = [_make_ball_crop(i) for i in range(6)]
    ref_fps = [_make_fp_crop(100 + i) for i in range(6)]
    query_balls = [_make_ball_crop(200 + i) for i in range(8)]
    query_fps = [_make_fp_crop(300 + i) for i in range(8)]
    queries = query_balls + query_fps
    labels = [True] * len(query_balls) + [False] * len(query_fps)

    for layer, expected_dim in (("layer4", 512), ("layer3", 256), ("layer2", 128)):
        pos_emb = encoder.embed_crops(ref_balls, layer=layer)
        neg_emb = encoder.embed_crops(ref_fps, layer=layer)
        q_emb = encoder.embed_crops(queries, layer=layer)
        assert pos_emb.shape == (6, expected_dim)
        assert np.allclose(np.linalg.norm(pos_emb, axis=1), 1.0, atol=1e-5)

        pos_filter = PrototypeFilter.fit(pos_emb, mode="positive_only")
        scores_pos = pos_filter.score(q_emb)
        auc_pos = compute_roc_auc(scores_pos, labels)
        assert auc_pos is not None and auc_pos > 0.85

        margin_filter = PrototypeFilter.fit(pos_emb, neg_emb, mode="pos_minus_neg")
        scores_margin = margin_filter.score(q_emb)
        auc_margin = compute_roc_auc(scores_margin, labels)
        assert auc_margin is not None and auc_margin > 0.85


def test_calibrate_threshold_selects_optimal_cut_and_handles_noop() -> None:
    res = calibrate_threshold(
        scores=[0.9, 0.85, 0.2, 0.1],
        is_tp=[True, True, False, False],
        total_gt=2,
    )
    assert res["improved_on_calib"] is True
    assert res["is_noop"] is False
    assert 0.2 < res["threshold"] < 0.85
    assert res["calib_best_f1"] == pytest.approx(1.0)

    noop = calibrate_threshold(
        scores=[0.1, 0.2, 0.8, 0.9],
        is_tp=[True, True, False, False],
        total_gt=2,
    )
    assert noop["is_noop"] is True
    assert math.isinf(noop["threshold"]) and noop["threshold"] < 0
    assert noop["improved_on_calib"] is False
