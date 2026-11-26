from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from PIL import Image, ImageDraw

from bola_fewshot.dataset import DatasetSplit, crop_box
from bola_fewshot.detector import Detection
from bola_fewshot.metrics import match_image_detections, precision_recall_f1


def build_fp_sample_grid(
    val_split: DatasetSplit,
    dets_by_image: Mapping[str, Sequence[Detection]],
    *,
    output_image: Path,
    output_metadata: Path,
    min_conf: float = 0.50,
    eval_iou: float = 0.50,
    sample_size: int = 40,
    context_scale: float = 3.0,
    tile_size: int = 160,
    cols: int = 8,
    seed: int = 42,
) -> dict:
    fp_records: list[dict] = []
    for record in sorted(val_split.records, key=lambda r: r.image_id):
        dets = list(dets_by_image.get(record.image_id, ()))
        if not dets:
            continue
        matched = match_image_detections(
            [d.box for d in dets],
            [d.conf for d in dets],
            record.gt_boxes,
            iou_threshold=eval_iou,
        )
        for det, is_tp in zip(dets, matched):
            if not is_tp and det.conf >= min_conf:
                fp_records.append(
                    {
                        "image_id": record.image_id,
                        "image_path": str(record.image_path),
                        "box": [round(float(v), 2) for v in det.box],
                        "conf": round(float(det.conf), 4),
                        "n_gt_in_image": len(record.gt_boxes),
                    }
                )

    rng = random.Random(seed)
    indices = list(range(len(fp_records)))
    rng.shuffle(indices)
    chosen = [fp_records[i] for i in indices[: min(sample_size, len(fp_records))]]

    tiles: list[Image.Image] = []
    for idx, item in enumerate(chosen, start=1):
        item["sample_index"] = idx
        with Image.open(item["image_path"]) as img:
            rgb = img.convert("RGB")
            x1, y1, x2, y2 = item["box"]
            w_img, h_img = rgb.size
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            side = max(bw, bh) * context_scale
            left = max(0, min(w_img - 1, int(round(cx - side / 2.0))))
            top = max(0, min(h_img - 1, int(round(cy - side / 2.0))))
            right = max(left + 1, min(w_img, int(round(cx + side / 2.0))))
            bottom = max(top + 1, min(h_img, int(round(cy + side / 2.0))))
            crop = rgb.crop((left, top, right, bottom))
            sx = tile_size / max(1, right - left)
            sy = tile_size / max(1, bottom - top)
            tile = crop.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
            draw = ImageDraw.Draw(tile)
            rx1 = max(0, min(tile_size - 1, int(round((x1 - left) * sx))))
            ry1 = max(0, min(tile_size - 1, int(round((y1 - top) * sy))))
            rx2 = max(rx1 + 1, min(tile_size - 1, int(round((x2 - left) * sx))))
            ry2 = max(ry1 + 1, min(tile_size - 1, int(round((y2 - top) * sy))))
            draw.rectangle((rx1, ry1, rx2, ry2), outline=(255, 60, 60), width=2)
            draw.rectangle((0, 0, tile_size, 18), fill=(0, 0, 0))
            draw.text(
                (4, 3),
                f"#{idx:02d} c={item['conf']:.2f} gt={item['n_gt_in_image']}",
                fill=(255, 255, 120),
            )
            tiles.append(tile)

    rows = max(1, math.ceil(len(tiles) / cols))
    grid = Image.new("RGB", (cols * tile_size, rows * tile_size), (24, 24, 24))
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % cols) * tile_size, (i // cols) * tile_size))

    output_image.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_image)

    meta = {
        "total_fps_at_min_conf": len(fp_records),
        "min_conf": float(min_conf),
        "sample_size": len(chosen),
        "context_scale": float(context_scale),
        "seed": int(seed),
        "image_path": str(output_image),
        "samples": chosen,
    }
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def wilson_ci_95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def estimate_noise_adjusted_metrics(
    *,
    base_metrics: dict,
    filtered_metrics: dict,
    n_audited: int,
    n_unannotated_real_balls: int,
    n_ambiguous: int,
    n_true_fps: int,
    high_conf_fp_count: int,
    total_fp_count: int,
) -> dict:
    rate_strict = n_unannotated_real_balls / max(1, n_audited)
    rate_upper = (n_unannotated_real_balls + n_ambiguous) / max(1, n_audited)
    ci_low, ci_high = wilson_ci_95(n_unannotated_real_balls, n_audited)

    def adjust_arm(metrics: dict, est_rate: float) -> dict:
        fp_obs = int(metrics["fp"])
        tp_obs = int(metrics["tp"])
        gt_obs = int(metrics["total_gt"])
        est_unannotated = int(round(fp_obs * est_rate))
        tp_adj = tp_obs + est_unannotated
        fp_adj = max(0, fp_obs - est_unannotated)
        gt_adj = gt_obs + int(round(total_fp_count * est_rate))
        p_adj, r_adj, f1_adj = precision_recall_f1(tp_adj, fp_adj, gt_adj)
        return {
            "estimated_unannotated_balls_in_fp": est_unannotated,
            "tp_adjusted": tp_adj,
            "fp_adjusted": fp_adj,
            "gt_adjusted": gt_adj,
            "precision_adjusted": p_adj,
            "recall_adjusted": r_adj,
            "f1_adjusted": f1_adj,
        }

    base_strict = adjust_arm(base_metrics, rate_strict)
    filt_strict = adjust_arm(filtered_metrics, rate_strict)
    base_high_only = adjust_arm(
        base_metrics,
        rate_strict * (high_conf_fp_count / max(1, total_fp_count)),
    )
    filt_high_only = adjust_arm(
        filtered_metrics,
        rate_strict * (high_conf_fp_count / max(1, total_fp_count)),
    )
    return {
        "is_estimate": True,
        "original_labels_modified": False,
        "n_audited": int(n_audited),
        "n_unannotated_real_balls": int(n_unannotated_real_balls),
        "n_ambiguous": int(n_ambiguous),
        "n_true_fps": int(n_true_fps),
        "unannotated_ball_rate_strict": float(rate_strict),
        "unannotated_ball_rate_wilson_95": [float(ci_low), float(ci_high)],
        "unannotated_ball_rate_including_ambiguous": float(rate_upper),
        "high_conf_fp_count": int(high_conf_fp_count),
        "total_fp_count": int(total_fp_count),
        "adjusted_applying_rate_to_all_fps": {
            "baseline": base_strict,
            "filtered": filt_strict,
            "delta_f1_adjusted": float(filt_strict["f1_adjusted"] - base_strict["f1_adjusted"]),
        },
        "adjusted_applying_rate_only_to_high_conf_fps": {
            "baseline": base_high_only,
            "filtered": filt_high_only,
            "delta_f1_adjusted": float(filt_high_only["f1_adjusted"] - base_high_only["f1_adjusted"]),
        },
    }
