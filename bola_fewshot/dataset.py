from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import random
import re
from typing import Iterable, Mapping, Sequence

from PIL import Image
import yaml

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    label_path: Path | None
    width: int
    height: int
    gt_boxes: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    records: tuple[ImageRecord, ...]

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(record.image_id for record in self.records)

    def gt_by_image(self) -> dict[str, list[tuple[float, float, float, float]]]:
        return {record.image_id: list(record.gt_boxes) for record in self.records}


def _apply_path_remap(raw: str, path_remap: tuple[str, str] | None) -> str:
    text = raw.strip()
    if path_remap is None:
        return text
    old_prefix, new_prefix = path_remap
    norm_text = text.replace("\\", "/")
    norm_old = old_prefix.replace("\\", "/").rstrip("/")
    norm_new = new_prefix.replace("\\", "/").rstrip("/")
    if norm_text.lower() == norm_old.lower():
        return norm_new
    if norm_text.lower().startswith(norm_old.lower() + "/"):
        return norm_new + norm_text[len(norm_old) :]
    return text


def _normalize_path_text(raw: str) -> Path:
    text = raw.strip()
    if "\\" in text and "/" not in text:
        win = PureWindowsPath(text)
        if win.drive:
            return Path(text)
        return Path(*win.parts)
    return Path(text)


def _match_images_root(raw: str, images_root: Path | None) -> Path | None:
    if images_root is None:
        return None
    norm = raw.strip().replace("\\", "/")
    lower = norm.lower()
    marker = "/images/"
    idx = lower.rfind(marker)
    if idx != -1:
        rel_after_images = norm[idx + len(marker) :]
        candidate = images_root / rel_after_images
        if candidate.exists():
            return candidate
        parts = rel_after_images.split("/")
        if len(parts) > 1 and images_root.name.lower() == parts[0].lower():
            candidate_trimmed = images_root / "/".join(parts[1:])
            if candidate_trimmed.exists():
                return candidate_trimmed
    filename = norm.rsplit("/", 1)[-1]
    direct = images_root / filename
    if direct.exists():
        return direct
    in_train = images_root / "train" / filename
    if in_train.exists():
        return in_train
    return None


def _resolve_entry_path(
    raw: str,
    base_dirs: Sequence[Path],
    *,
    images_root: Path | None = None,
    path_remap: tuple[str, str] | None = None,
) -> Path:
    remapped = _apply_path_remap(raw, path_remap)
    candidate = _normalize_path_text(remapped)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if not WINDOWS_DRIVE_RE.match(remapped):
        if candidate.is_absolute():
            return candidate
        for base in base_dirs:
            joined = base / candidate
            if joined.exists():
                return joined
        if candidate.exists():
            return candidate.resolve()
    matched = _match_images_root(remapped, images_root)
    if matched is not None:
        return matched
    if WINDOWS_DRIVE_RE.match(remapped):
        return Path(remapped)
    return base_dirs[0] / candidate


def _collect_image_paths(
    entry: str | Sequence[str],
    base_dirs: Sequence[Path],
    *,
    images_root: Path | None = None,
    path_remap: tuple[str, str] | None = None,
) -> list[Path]:
    items = [entry] if isinstance(entry, str) else list(entry)
    collected: list[Path] = []
    search_bases = list(base_dirs)
    if images_root is not None:
        search_bases.insert(0, images_root)
    for item in items:
        resolved = _resolve_entry_path(
            str(item),
            search_bases,
            images_root=images_root,
            path_remap=path_remap,
        )
        if resolved.is_file() and resolved.suffix.lower() == ".txt":
            txt_bases = [resolved.parent, *search_bases]
            for line in resolved.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                img_path = _resolve_entry_path(
                    stripped,
                    txt_bases,
                    images_root=images_root,
                    path_remap=path_remap,
                )
                collected.append(img_path)
        elif resolved.is_dir():
            for path in sorted(resolved.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    collected.append(path)
        elif resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            collected.append(resolved)
        else:
            raise FileNotFoundError(f"dataset split entry not found: {item} (resolved: {resolved})")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in collected:
        canonical = path.resolve()
        if canonical not in seen:
            seen.add(canonical)
            unique.append(canonical)
    return unique


def resolve_label_path(
    image_path: Path,
    images_root: Path | None = None,
    labels_root: Path | None = None,
) -> Path | None:
    if labels_root is not None:
        if images_root is not None:
            try:
                rel = image_path.resolve().relative_to(images_root.resolve())
                candidate = (labels_root / rel).with_suffix(".txt")
                if candidate.is_file():
                    return candidate
            except ValueError:
                pass
        direct = labels_root / f"{image_path.stem}.txt"
        if direct.is_file():
            return direct
        in_parent = labels_root / image_path.parent.name / f"{image_path.stem}.txt"
        if in_parent.is_file():
            return in_parent
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            replaced = list(parts)
            replaced[idx] = "labels"
            candidate = Path(*replaced).with_suffix(".txt")
            if candidate.is_file():
                return candidate
            break
    sibling = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if sibling.is_file():
        return sibling
    beside = image_path.with_suffix(".txt")
    if beside.is_file():
        return beside
    return None


def parse_yolo_labels(
    label_path: Path | None,
    width: int,
    height: int,
    target_class: int = 0,
) -> tuple[tuple[float, float, float, float], ...]:
    if label_path is None or not label_path.is_file():
        return ()
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 5:
            continue
        cls_id = int(float(tokens[0]))
        if cls_id != target_class:
            continue
        cx, cy, bw, bh = (float(tokens[i]) for i in range(1, 5))
        x1 = max(0.0, min(float(width), (cx - bw / 2.0) * width))
        y1 = max(0.0, min(float(height), (cy - bh / 2.0) * height))
        x2 = max(0.0, min(float(width), (cx + bw / 2.0) * width))
        y2 = max(0.0, min(float(height), (cy + bh / 2.0) * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return tuple(boxes)


def _make_image_id(image_path: Path, anchor_root: Path | None) -> str:
    if anchor_root is not None:
        try:
            return image_path.resolve().relative_to(anchor_root.resolve()).as_posix()
        except ValueError:
            pass
    return f"{image_path.parent.name}/{image_path.name}"


def _build_split(
    name: str,
    image_paths: Sequence[Path],
    anchor_root: Path | None,
    images_root: Path | None,
    labels_root: Path | None,
    target_class: int,
) -> DatasetSplit:
    records: list[ImageRecord] = []
    seen_ids: set[str] = set()
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        with Image.open(path) as img:
            width, height = img.size
        label_path = resolve_label_path(path, images_root=images_root, labels_root=labels_root)
        gt_boxes = parse_yolo_labels(label_path, width, height, target_class=target_class)
        image_id = _make_image_id(path, anchor_root)
        if image_id in seen_ids:
            image_id = path.resolve().as_posix()
        seen_ids.add(image_id)
        records.append(
            ImageRecord(
                image_id=image_id,
                image_path=path,
                label_path=label_path,
                width=width,
                height=height,
                gt_boxes=gt_boxes,
            )
        )
    return DatasetSplit(name=name, records=tuple(records))


def load_image_list_split(
    name: str,
    list_path: Path,
    *,
    images_root: Path | None = None,
    labels_root: Path | None = None,
    path_remap: tuple[str, str] | None = None,
    target_class: int = 0,
) -> DatasetSplit:
    resolved_list = Path(list_path).resolve()
    if not resolved_list.is_file():
        raise FileNotFoundError(f"image list not found: {resolved_list}")
    paths = _collect_image_paths(
        str(resolved_list),
        [resolved_list.parent],
        images_root=images_root,
        path_remap=path_remap,
    )
    anchor_root = images_root.resolve() if images_root is not None else resolved_list.parent
    return _build_split(
        name,
        paths,
        anchor_root=anchor_root,
        images_root=images_root,
        labels_root=labels_root,
        target_class=target_class,
    )


def load_yolo_dataset(
    data_yaml: Path,
    *,
    images_root: Path | None = None,
    labels_root: Path | None = None,
    path_remap: tuple[str, str] | None = None,
    target_class: int = 0,
    require_train: bool = True,
) -> dict[str, DatasetSplit]:
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid data.yaml mapping: {yaml_path}")
    base_dirs: list[Path] = []
    raw_root = config.get("path")
    if raw_root:
        remapped_root = _apply_path_remap(str(raw_root), path_remap)
        root_path = _normalize_path_text(remapped_root)
        if not root_path.is_absolute():
            root_path = (yaml_path.parent / root_path).resolve()
        if root_path.exists():
            base_dirs.append(root_path)
    base_dirs.append(yaml_path.parent)
    anchor_root = images_root.resolve() if images_root is not None else base_dirs[0]

    splits: dict[str, DatasetSplit] = {}
    for split_name in ("train", "val", "test"):
        entry = config.get(split_name)
        if entry is None:
            continue
        paths = _collect_image_paths(
            entry,
            base_dirs,
            images_root=images_root,
            path_remap=path_remap,
        )
        splits[split_name] = _build_split(
            split_name,
            paths,
            anchor_root=anchor_root,
            images_root=images_root,
            labels_root=labels_root,
            target_class=target_class,
        )

    if require_train and "train" not in splits:
        raise ValueError(f"data.yaml missing required 'train' split: {yaml_path}")
    if "val" not in splits and "test" not in splits:
        raise ValueError(f"data.yaml missing 'val' or 'test' split: {yaml_path}")

    if "train" in splits and "val" in splits:
        assert_disjoint_splits(splits["train"], splits["val"])
    if "train" in splits and "test" in splits:
        assert_disjoint_splits(splits["train"], splits["test"])
    if "val" in splits and "test" in splits:
        assert_disjoint_splits(splits["val"], splits["test"])
    return splits


def load_image_folder_split(
    name: str,
    images_dir: Path,
    labels_dir: Path | None = None,
    target_class: int = 0,
) -> DatasetSplit:
    resolved_dir = Path(images_dir).resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {resolved_dir}")
    paths = [
        path.resolve()
        for path in sorted(resolved_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return _build_split(
        name,
        paths,
        anchor_root=resolved_dir,
        images_root=resolved_dir,
        labels_root=labels_dir.resolve() if labels_dir is not None else None,
        target_class=target_class,
    )


def assert_disjoint_splits(split_a: DatasetSplit, split_b: DatasetSplit) -> None:
    paths_a = {record.image_path.resolve() for record in split_a.records}
    paths_b = {record.image_path.resolve() for record in split_b.records}
    overlap = paths_a & paths_b
    if overlap:
        sample = sorted(str(p) for p in overlap)[:3]
        raise ValueError(
            f"data leakage detected: {len(overlap)} overlapping images between "
            f"'{split_a.name}' and '{split_b.name}': {sample}"
        )


def audit_group_overlap(
    split_a: DatasetSplit,
    split_b: DatasetSplit,
    group_by: Mapping[str, str],
) -> dict:
    groups_a = {group_by.get(r.image_id, r.image_id) for r in split_a.records}
    groups_b = {group_by.get(r.image_id, r.image_id) for r in split_b.records}
    shared = groups_a & groups_b
    return {
        "split_a": split_a.name,
        "split_b": split_b.name,
        "groups_a": len(groups_a),
        "groups_b": len(groups_b),
        "shared_groups": len(shared),
        "disjoint_by_group": len(shared) == 0,
    }


def partition_split(
    split: DatasetSplit,
    first_fraction: float,
    seed: int,
    first_name: str,
    second_name: str,
    group_by: Mapping[str, str] | None = None,
) -> tuple[DatasetSplit, DatasetSplit]:
    if not 0.0 < first_fraction < 1.0:
        raise ValueError("first_fraction must be strictly between 0 and 1")
    ordered = sorted(split.records, key=lambda r: r.image_id)
    if len(ordered) < 2:
        raise ValueError(f"split '{split.name}' needs at least 2 images to partition")
    rng = random.Random(seed)
    if group_by is None:
        indices = list(range(len(ordered)))
        rng.shuffle(indices)
        cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * first_fraction))))
        first_records = tuple(ordered[i] for i in sorted(indices[:cut]))
        second_records = tuple(ordered[i] for i in sorted(indices[cut:]))
    else:
        grouped: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in ordered:
            grouped[group_by.get(record.image_id, record.image_id)].append(record)
        group_ids = sorted(grouped.keys())
        if len(group_ids) < 2:
            raise ValueError(f"split '{split.name}' needs at least 2 groups to partition by group")
        rng.shuffle(group_ids)
        target_count = len(ordered) * first_fraction
        first_list: list[ImageRecord] = []
        second_list: list[ImageRecord] = []
        for idx, gid in enumerate(group_ids):
            members = grouped[gid]
            remaining_groups = len(group_ids) - idx - 1
            if not first_list:
                first_list.extend(members)
            elif remaining_groups == 0 and not second_list:
                second_list.extend(members)
            else:
                after_add = len(first_list) + len(members)
                if abs(after_add - target_count) <= abs(len(first_list) - target_count):
                    first_list.extend(members)
                else:
                    second_list.extend(members)
        if not second_list:
            moved_gid = group_ids[-1]
            moved_ids = {r.image_id for r in grouped[moved_gid]}
            first_list = [r for r in first_list if r.image_id not in moved_ids]
            second_list = list(grouped[moved_gid])
        first_list.sort(key=lambda r: r.image_id)
        second_list.sort(key=lambda r: r.image_id)
        first_records = tuple(first_list)
        second_records = tuple(second_list)
    first = DatasetSplit(name=first_name, records=first_records)
    second = DatasetSplit(name=second_name, records=second_records)
    assert_disjoint_splits(first, second)
    if group_by is not None:
        overlap = audit_group_overlap(first, second, group_by)
        if not overlap["disjoint_by_group"]:
            raise ValueError(f"group leakage detected between '{first_name}' and '{second_name}'")
    return first, second


def make_group_kfold(
    split: DatasetSplit,
    n_folds: int = 5,
    seed: int = 42,
    group_by: Mapping[str, str] | None = None,
) -> list[DatasetSplit]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    ordered = sorted(split.records, key=lambda r: r.image_id)
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in ordered:
        gid = group_by.get(record.image_id, record.image_id) if group_by is not None else record.image_id
        grouped[gid].append(record)
    group_ids = sorted(grouped.keys())
    if len(group_ids) < n_folds:
        raise ValueError(f"need at least {n_folds} groups for {n_folds}-fold CV, got {len(group_ids)}")
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    rank = {gid: idx for idx, gid in enumerate(group_ids)}
    ordered_groups = sorted(group_ids, key=lambda gid: (-len(grouped[gid]), rank[gid]))

    fold_lists: list[list[ImageRecord]] = [[] for _ in range(n_folds)]
    fold_counts = [0] * n_folds
    for gid in ordered_groups:
        target = min(range(n_folds), key=lambda f: (fold_counts[f], f))
        members = grouped[gid]
        fold_lists[target].extend(members)
        fold_counts[target] += len(members)

    folds = [
        DatasetSplit(name=f"{split.name}_fold{i}", records=tuple(sorted(items, key=lambda r: r.image_id)))
        for i, items in enumerate(fold_lists)
    ]
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            assert_disjoint_splits(folds[i], folds[j])
            if group_by is not None:
                overlap = audit_group_overlap(folds[i], folds[j], group_by)
                if not overlap["disjoint_by_group"]:
                    raise ValueError(f"group leakage between fold {i} and fold {j}")
    return folds


def stratified_sample_split(
    split: DatasetSplit,
    max_images: int,
    seed: int,
) -> tuple[DatasetSplit, dict]:
    total = len(split.records)
    if max_images <= 0 or max_images >= total:
        return split, {
            "subsampled": False,
            "original_images": total,
            "sampled_images": total,
            "seed": seed,
        }
    strata: dict[tuple[str, bool], list[ImageRecord]] = defaultdict(list)
    for record in sorted(split.records, key=lambda r: r.image_id):
        stem = record.image_path.stem
        prefix = stem.split("_", 1)[0] if "_" in stem else "default"
        strata[(prefix, len(record.gt_boxes) > 0)].append(record)

    rng = random.Random(seed)
    keys = sorted(strata.keys())
    raw_alloc = {k: len(strata[k]) * max_images / total for k in keys}
    alloc = {k: min(len(strata[k]), int(raw_alloc[k])) for k in keys}
    remainder = max_images - sum(alloc.values())
    order_rem = sorted(keys, key=lambda k: (-(raw_alloc[k] - alloc[k]), k))
    for k in order_rem:
        if remainder <= 0:
            break
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            remainder -= 1

    chosen: list[ImageRecord] = []
    for k in keys:
        group = list(strata[k])
        rng.shuffle(group)
        chosen.extend(group[: alloc[k]])
    chosen.sort(key=lambda r: r.image_id)
    sampled = DatasetSplit(name=split.name, records=tuple(chosen))
    return sampled, {
        "subsampled": True,
        "original_images": total,
        "sampled_images": len(chosen),
        "n_strata": len(keys),
        "seed": seed,
        "original_pos_images": sum(1 for r in split.records if r.gt_boxes),
        "sampled_pos_images": sum(1 for r in chosen if r.gt_boxes),
    }


def crop_box(
    image: Image.Image,
    box_xyxy: Sequence[float],
    *,
    context_scale: float = 1.0,
    square: bool = False,
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1) * context_scale
    bh = max(1.0, y2 - y1) * context_scale
    if square:
        side = max(bw, bh)
        bw, bh = side, side
    left = max(0, min(width - 1, int(round(cx - bw / 2.0))))
    top = max(0, min(height - 1, int(round(cy - bh / 2.0))))
    right = max(left + 1, min(width, int(round(cx + bw / 2.0))))
    bottom = max(top + 1, min(height, int(round(cy + bh / 2.0))))
    return image.crop((left, top, right, bottom)).convert("RGB")


def count_gt_boxes(records: Iterable[ImageRecord]) -> int:
    return sum(len(record.gt_boxes) for record in records)
