from pathlib import Path

from PIL import Image
import pytest

from bola_fewshot.dataset import (
    crop_box,
    load_yolo_dataset,
    partition_split,
    stratified_sample_split,
)


def _create_image(path: Path, size: tuple[int, int] = (200, 100), color: tuple[int, int, int] = (30, 120, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_yolo_dataset_from_txt_lists_and_external_images(tmp_path: Path) -> None:
    r1 = tmp_path / "dataset_r1"
    r5 = tmp_path / "dataset_r5"
    r5.mkdir()

    train_imgs = []
    for i in range(3):
        img = r1 / "images" / f"train_{i}.jpg"
        _create_image(img, (200, 100))
        _write_label(r1 / "labels" / f"train_{i}.txt", ["0 0.5 0.5 0.1 0.2"])
        train_imgs.append(img)

    val_imgs = []
    for i in range(2):
        img = r1 / "images" / f"val_{i}.jpg"
        _create_image(img, (200, 100))
        if i == 0:
            _write_label(r1 / "labels" / f"val_{i}.txt", ["0 0.25 0.5 0.1 0.2", "1 0.5 0.5 0.2 0.2"])
        val_imgs.append(img)

    (r5 / "train.txt").write_text("\n".join(str(p) for p in train_imgs) + "\n", encoding="utf-8")
    (r5 / "val.txt").write_text("\n".join(str(p) for p in val_imgs) + "\n", encoding="utf-8")
    data_yaml = r5 / "data.yaml"
    data_yaml.write_text("train: train.txt\nval: val.txt\nnames:\n  0: ball\n", encoding="utf-8")

    splits = load_yolo_dataset(data_yaml)
    assert len(splits["train"].records) == 3
    assert len(splits["val"].records) == 2

    first_val = splits["val"].records[0]
    assert len(first_val.gt_boxes) == 1
    assert first_val.gt_boxes[0] == pytest.approx((40.0, 40.0, 60.0, 60.0))
    assert splits["val"].records[1].gt_boxes == ()


def test_load_yolo_dataset_rejects_train_val_overlap(tmp_path: Path) -> None:
    img = tmp_path / "images" / "shared.jpg"
    _create_image(img)
    (tmp_path / "train.txt").write_text(f"{img}\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text(f"{img}\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: train.txt\nval: val.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data leakage detected"):
        load_yolo_dataset(data_yaml)


def test_partition_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    for i in range(10):
        _create_image(tmp_path / "images" / "val" / f"img_{i:02d}.jpg")
        _create_image(tmp_path / "images" / "train" / f"tr_{i:02d}.jpg")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    splits = load_yolo_dataset(data_yaml)

    a1, b1 = partition_split(splits["val"], 0.3, seed=42, first_name="calib", second_name="test")
    a2, b2 = partition_split(splits["val"], 0.3, seed=42, first_name="calib", second_name="test")
    assert a1.image_ids == a2.image_ids
    assert b1.image_ids == b2.image_ids
    assert len(a1.records) == 3
    assert len(b1.records) == 7
    assert set(a1.image_ids).isdisjoint(b1.image_ids)


def test_crop_box_square_context_and_clamping() -> None:
    img = Image.new("RGB", (100, 80), (10, 20, 30))
    crop = crop_box(img, (0.0, 0.0, 10.0, 20.0), context_scale=1.5, square=True)
    assert crop.mode == "RGB"
    assert crop.size[0] >= 1 and crop.size[1] >= 1
    assert crop.size[0] <= 100 and crop.size[1] <= 80


def test_windows_absolute_prefix_remapping_and_stratified_sampling(tmp_path: Path) -> None:
    r1 = tmp_path / "item488_t1b_v5_dataset_r1"
    r5 = tmp_path / "item488_t1b_v5_dataset_r5_9010"
    r5.mkdir()

    train_lines = []
    for i in range(4):
        name = f"camA_tr_{i}.jpg"
        _create_image(r1 / "images" / "train" / name, (200, 100))
        _write_label(r1 / "labels" / "train" / f"camA_tr_{i}.txt", ["0 0.5 0.5 0.1 0.2"])
        train_lines.append(f"C:/Footlabs_Miguel_cache/item488_t1b_v5_dataset_r1/images/train/{name}")

    val_lines = []
    for i in range(10):
        prefix = "camA" if i < 5 else "camB"
        name = f"{prefix}_val_{i}.jpg"
        _create_image(r1 / "images" / "train" / name, (200, 100))
        if i % 2 == 0:
            _write_label(r1 / "labels" / "train" / f"{prefix}_val_{i}.txt", ["0 0.5 0.5 0.1 0.2"])
        val_lines.append(f"C:/Footlabs_Miguel_cache/item488_t1b_v5_dataset_r1/images/train/{name}")

    (r5 / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (r5 / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    data_yaml = r5 / "data.yaml"
    data_yaml.write_text(
        "path: C:/Footlabs_Miguel_cache/item488_t1b_v5_dataset_r5_9010\ntrain: train.txt\nval: val.txt\n",
        encoding="utf-8",
    )

    splits_remap = load_yolo_dataset(
        data_yaml,
        path_remap=("C:/Footlabs_Miguel_cache", str(tmp_path)),
    )
    assert len(splits_remap["train"].records) == 4
    assert len(splits_remap["val"].records) == 10
    assert sum(1 for r in splits_remap["val"].records if r.gt_boxes) == 5

    splits_root = load_yolo_dataset(
        data_yaml,
        images_root=r1 / "images",
        labels_root=r1 / "labels",
    )
    assert len(splits_root["val"].records) == 10

    sampled, meta = stratified_sample_split(splits_remap["val"], max_images=4, seed=42)
    assert meta["subsampled"] is True
    assert len(sampled.records) == 4
    assert meta["sampled_pos_images"] == 2

