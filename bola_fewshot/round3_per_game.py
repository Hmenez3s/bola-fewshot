from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from bola_fewshot.cv_round2 import (
    _apply_mask_to_split,
    _load_group_map,
    _preds_to_eval_format,
    _sanitize_for_json,
    _strip_per_image,
)
from bola_fewshot.dataset import (
    DatasetSplit,
    ImageRecord,
    assert_disjoint_splits,
    audit_group_overlap,
    count_gt_boxes,
    crop_box,
    load_yolo_dataset,
    make_group_kfold,
)
from bola_fewshot.detector import BallDetectorProtocol, Detection, YoloBallDetector, resolve_device
from bola_fewshot.fewshot_filter import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FewShotR18Encoder,
    PrototypeFilter,
    calibrate_threshold,
    letterbox_crop,
)
from bola_fewshot.metrics import (
    box_iou_matrix,
    compute_filter_cost,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_detections,
    match_image_detections,
    paired_bootstrap_diff,
    paired_cluster_bootstrap_diff,
    precision_recall_f1,
)
from bola_fewshot.train_ball import (
    BallTrainConfig,
    build_trainable_resnet18,
    mine_training_boxes_incremental,
)


@dataclass(frozen=True)
class Round3Config:
    data_yaml: Path
    detector_weights: Path
    fewshot_weights: Path
    imagenet_weights: Path | None
    output_dir: Path
    cache_dir: Path
    group_map_path: Path | None = None
    path_remap: tuple[str, str] | None = None
    imgsz: int = 1920
    val_mining_conf: float = 0.05
    operating_confs: tuple[float, ...] = (0.35, 0.15, 0.10)
    iou: float = 0.70
    eval_iou: float = 0.50
    max_det: int = 300
    target_class: int = 0
    context_scales: tuple[float, ...] = (2.0, 3.0)
    canvas_wh: tuple[int, int] = (96, 96)
    n_refs_per_game: int = 6
    ref_min_conf: float = 0.60
    same_game_neg_max_conf: float = 0.20
    n_folds: int = 5
    seed: int = 42
    train_seed: int = 25
    train_epochs: int = 16
    train_steps_per_epoch: int = 360
    train_mining_conf: float = 0.05
    train_mining_imgsz: int = 1280
    train_cap_per_game: int = 80
    bootstrap_resamples: int = 2000
    device: str = "auto"


def filter_dets_by_conf(
    dets_by_image: Mapping[str, Sequence[Detection]],
    min_conf: float,
) -> dict[str, list[Detection]]:
    return {
        iid: [d for d in dets if d.conf >= min_conf]
        for iid, dets in dets_by_image.items()
    }


def apply_post_nms(
    dets_by_image: Mapping[str, Sequence[Detection]],
    iou_threshold: float,
) -> dict[str, list[Detection]]:
    out: dict[str, list[Detection]] = {}
    for iid, dets in dets_by_image.items():
        if len(dets) <= 1:
            out[iid] = list(dets)
            continue
        ordered = sorted(dets, key=lambda d: -d.conf)
        boxes = np.asarray([d.box for d in ordered], dtype=np.float64)
        iou_mat = box_iou_matrix(boxes, boxes)
        kept: list[Detection] = []
        suppressed = np.zeros(len(ordered), dtype=bool)
        for i, det in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(det)
            for j in range(i + 1, len(ordered)):
                if iou_mat[i, j] >= iou_threshold:
                    suppressed[j] = True
        out[iid] = kept
    return out


def apply_top1_per_frame(
    dets_by_image: Mapping[str, Sequence[Detection]],
) -> dict[str, list[Detection]]:
    out: dict[str, list[Detection]] = {}
    for iid, dets in dets_by_image.items():
        if not dets:
            out[iid] = []
        else:
            best = max(dets, key=lambda d: d.conf)
            out[iid] = [best]
    return out


def select_game_auto_references(
    val_split: DatasetSplit,
    dets_low_conf: Mapping[str, Sequence[Detection]],
    group_by: Mapping[str, str],
    *,
    k_refs: int = 6,
    ref_min_conf: float = 0.60,
    neg_conf_max: float = 0.20,
    eval_iou: float = 0.50,
) -> dict:
    records_by_id = {r.image_id: r for r in val_split.records}
    games: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in sorted(val_split.records, key=lambda r: r.image_id):
        games[group_by.get(record.image_id, record.image_id)].append(record)

    eligible_games: list[str] = []
    pos_refs_by_game: dict[str, list[tuple[ImageRecord, Detection]]] = {}
    neg_refs_by_game: dict[str, list[tuple[ImageRecord, Detection]]] = {}
    ref_image_ids: set[str] = set()
    query_image_ids_by_game: dict[str, list[str]] = {}

    n_ref_tp_50 = 0
    n_ref_tp_30 = 0
    n_ref_total = 0
    n_same_game_neg_total = 0
    n_same_game_neg_clean = 0

    for gid in sorted(games.keys()):
        g_records = games[gid]
        if len(g_records) < k_refs + 1:
            continue
        frame_best: list[tuple[int, ImageRecord, Detection]] = []
        for idx, rec in enumerate(g_records):
            dets = dets_low_conf.get(rec.image_id, ())
            high = [d for d in dets if d.conf >= ref_min_conf]
            if high:
                best = max(high, key=lambda d: d.conf)
                frame_best.append((idx, rec, best))
        if len(frame_best) < k_refs:
            continue

        n_frames = len(g_records)
        chosen: list[tuple[ImageRecord, Detection]] = []
        used_idx: set[int] = set()
        for b in range(k_refs):
            lo = int(b * n_frames / k_refs)
            hi = int((b + 1) * n_frames / k_refs)
            in_bin = [item for item in frame_best if lo <= item[0] < hi and item[0] not in used_idx]
            if in_bin:
                pick = max(in_bin, key=lambda x: (x[2].conf, -x[0]))
                chosen.append((pick[1], pick[2]))
                used_idx.add(pick[0])
        if len(chosen) < k_refs:
            remaining = [item for item in frame_best if item[0] not in used_idx]
            remaining.sort(key=lambda x: (-x[2].conf, x[0]))
            for pick in remaining[: k_refs - len(chosen)]:
                chosen.append((pick[1], pick[2]))
                used_idx.add(pick[0])

        chosen_ids = {rec.image_id for rec, _ in chosen}
        query_ids = [rec.image_id for rec in g_records if rec.image_id not in chosen_ids]
        if not query_ids:
            continue

        eligible_games.append(gid)
        pos_refs_by_game[gid] = chosen
        ref_image_ids.update(chosen_ids)
        query_image_ids_by_game[gid] = query_ids

        game_negs: list[tuple[ImageRecord, Detection]] = []
        for rec, pos_det in chosen:
            dets = dets_low_conf.get(rec.image_id, ())
            for d in dets:
                if d.conf >= neg_conf_max:
                    continue
                iou_pos = float(
                    box_iou_matrix(np.asarray([d.box]), np.asarray([pos_det.box]))[0, 0]
                )
                if iou_pos <= 0.10:
                    game_negs.append((rec, d))
        game_negs.sort(key=lambda x: (-x[1].conf, x[0].image_id))
        neg_refs_by_game[gid] = game_negs[:k_refs]

        for rec, pos_det in chosen:
            n_ref_total += 1
            if rec.gt_boxes:
                max_iou = float(
                    box_iou_matrix(np.asarray([pos_det.box]), np.asarray(rec.gt_boxes)).max()
                )
                if max_iou >= eval_iou:
                    n_ref_tp_50 += 1
                if max_iou >= 0.30:
                    n_ref_tp_30 += 1

        for rec, neg_det in neg_refs_by_game[gid]:
            n_same_game_neg_total += 1
            if not rec.gt_boxes:
                n_same_game_neg_clean += 1
            else:
                max_iou = float(
                    box_iou_matrix(np.asarray([neg_det.box]), np.asarray(rec.gt_boxes)).max()
                )
                if max_iou < 0.10:
                    n_same_game_neg_clean += 1

    eligible_set = set(eligible_games)
    eligible_query_ids = [
        iid for gid in eligible_games for iid in query_image_ids_by_game[gid]
    ]
    eligible_no_v14_games = [gid for gid in eligible_games if gid != "v14"]
    eligible_no_v14_query_ids = [
        iid for gid in eligible_no_v14_games for iid in query_image_ids_by_game[gid]
    ]
    non_eligible_image_ids = [
        r.image_id for r in val_split.records if group_by.get(r.image_id, r.image_id) not in eligible_set
    ]

    coverage = {
        "total_games": len(games),
        "total_images": len(val_split.records),
        "k_refs_required": int(k_refs),
        "ref_min_conf": float(ref_min_conf),
        "eligible_games": len(eligible_games),
        "eligible_games_pct": float(len(eligible_games) / max(1, len(games))),
        "eligible_total_images": len(eligible_query_ids) + len(ref_image_ids),
        "eligible_ref_images_excluded": len(ref_image_ids),
        "eligible_query_images": len(eligible_query_ids),
        "eligible_no_v14_games": len(eligible_no_v14_games),
        "eligible_no_v14_query_images": len(eligible_no_v14_query_ids),
        "non_eligible_images": len(non_eligible_image_ids),
        "auto_ref_total": n_ref_total,
        "auto_ref_precision_iou50": float(n_ref_tp_50 / max(1, n_ref_total)),
        "auto_ref_precision_iou30": float(n_ref_tp_30 / max(1, n_ref_total)),
        "auto_ref_tp_iou50": n_ref_tp_50,
        "auto_ref_tp_iou30": n_ref_tp_30,
        "same_game_neg_refs_found": n_same_game_neg_total,
        "same_game_neg_clean_rate": float(n_same_game_neg_clean / max(1, n_same_game_neg_total)),
    }
    return {
        "eligible_games": eligible_games,
        "eligible_no_v14_games": eligible_no_v14_games,
        "pos_refs_by_game": pos_refs_by_game,
        "neg_refs_by_game": neg_refs_by_game,
        "ref_image_ids": ref_image_ids,
        "eligible_query_ids": eligible_query_ids,
        "eligible_no_v14_query_ids": eligible_no_v14_query_ids,
        "non_eligible_image_ids": non_eligible_image_ids,
        "coverage": coverage,
        "records_by_id": records_by_id,
    }


def _select_capped_train_records(
    train_split: DatasetSplit,
    group_by: Mapping[str, str],
    cap_per_game: int,
    seed: int,
) -> list[ImageRecord]:
    by_game: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in sorted(train_split.records, key=lambda x: x.image_id):
        by_game[group_by.get(r.image_id, r.image_id)].append(r)
    rng = random.Random(seed + 7)
    chosen: list[ImageRecord] = []
    for gid in sorted(by_game.keys()):
        recs = list(by_game[gid])
        if cap_per_game > 0 and len(recs) > cap_per_game:
            rng.shuffle(recs)
            recs = sorted(recs[:cap_per_game], key=lambda x: x.image_id)
        chosen.extend(recs)
    chosen.sort(key=lambda x: x.image_id)
    return chosen


def build_per_game_train_pools(
    jsonl_path: Path,
    group_by: Mapping[str, str],
    *,
    context_scale: float,
    canvas_wh: tuple[int, int] = (96, 96),
    eval_conf: float = 0.35,
    mining_conf: float = 0.05,
    min_pos_per_game: int = 8,
    max_pos_per_game: int = 60,
    max_neg_per_game: int = 60,
    seed: int = 25,
) -> dict:
    pos_by_game: dict[str, list[tuple[str, str, tuple[float, float, float, float]]]] = defaultdict(list)
    neg_by_game: dict[str, list[tuple[str, str, tuple[float, float, float, float], float]]] = defaultdict(list)

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row["image_id"])
            ipath = str(row["image_path"])
            gid = group_by.get(iid) or group_by.get(Path(ipath).name) or iid
            gt_boxes = [tuple(float(v) for v in b) for b in row["gt_boxes"]]
            gt_arr = np.asarray(gt_boxes, dtype=np.float64).reshape(len(gt_boxes), 4) if gt_boxes else np.empty((0, 4))
            img_pos: list[tuple[float, float, float, float]] = []
            for d in row["dets"]:
                box = tuple(float(v) for v in d["box"])
                conf = float(d["conf"])
                if d["is_tp"]:
                    img_pos.append(box)
                else:
                    if len(gt_arr) > 0 and float(box_iou_matrix(np.asarray([box]), gt_arr).max()) > 0.0:
                        continue
                    if conf >= mining_conf:
                        neg_by_game[gid].append((iid, ipath, box, conf))
            if not img_pos and gt_boxes:
                img_pos.extend(gt_boxes)
            for box in img_pos:
                pos_by_game[gid].append((iid, ipath, box))

    rng = random.Random(seed + 31)
    eligible_gids = [gid for gid, items in sorted(pos_by_game.items()) if len({x[0] for x in items}) >= min_pos_per_game]

    needed_by_path: dict[str, list[tuple[str, str, str, tuple[float, float, float, float]]]] = defaultdict(list)
    for gid in eligible_gids:
        p_items = list(pos_by_game[gid])
        if len(p_items) > max_pos_per_game:
            rng.shuffle(p_items)
            p_items = p_items[:max_pos_per_game]
        for iid, ipath, box in p_items:
            needed_by_path[ipath].append(("pos", gid, iid, box))

        n_items = list(neg_by_game.get(gid, []))
        n_items.sort(key=lambda x: -x[3])
        high_n = [x for x in n_items if x[3] >= eval_conf]
        low_n = [x for x in n_items if x[3] < eval_conf]
        rng.shuffle(high_n)
        rng.shuffle(low_n)
        picked_n = (high_n + low_n)[:max_neg_per_game]
        for iid, ipath, box, _ in picked_n:
            needed_by_path[ipath].append(("neg", gid, iid, box))

    all_global_negs = [item for gid, items in sorted(neg_by_game.items()) for item in items]
    rng.shuffle(all_global_negs)
    for iid, ipath, box, _ in all_global_negs[:2000]:
        needed_by_path[ipath].append(("gneg", "global", iid, box))

    pos_arrays_by_game: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    neg_arrays_by_game: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    global_neg_arrays: list[np.ndarray] = []

    for ipath, tasks in needed_by_path.items():
        with Image.open(ipath) as img:
            rgb = img.convert("RGB")
            for kind, gid, iid, box in tasks:
                arr = letterbox_crop(crop_box(rgb, box, context_scale=context_scale, square=True), canvas_wh)
                if kind == "pos":
                    pos_arrays_by_game[gid].append((iid, arr))
                elif kind == "neg":
                    neg_arrays_by_game[gid].append((iid, arr))
                else:
                    global_neg_arrays.append(arr)

    final_games: dict[str, dict] = {}
    for gid in eligible_gids:
        p_list = pos_arrays_by_game.get(gid, [])
        if len({x[0] for x in p_list}) < min_pos_per_game:
            continue
        n_list = neg_arrays_by_game.get(gid, [])
        final_games[gid] = {
            "pos_arr": np.stack([x[1] for x in p_list], axis=0).astype(np.uint8),
            "pos_ids": tuple(x[0] for x in p_list),
            "neg_arr": (
                np.stack([x[1] for x in n_list], axis=0).astype(np.uint8)
                if n_list
                else np.empty((0, canvas_wh[1], canvas_wh[0], 3), dtype=np.uint8)
            ),
            "neg_ids": tuple(x[0] for x in n_list),
        }

    gneg_arr = np.stack(global_neg_arrays, axis=0).astype(np.uint8)
    return {
        "games": final_games,
        "global_neg_arr": gneg_arr,
        "stats": {
            "eligible_train_games": len(final_games),
            "total_pos_crops": sum(len(v["pos_arr"]) for v in final_games.values()),
            "total_same_game_neg_crops": sum(len(v["neg_arr"]) for v in final_games.values()),
            "global_neg_fallback_crops": int(len(gneg_arr)),
            "context_scale": float(context_scale),
        },
    }


def _arrays_to_tensor(
    arrays: np.ndarray,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.ascontiguousarray(arrays))
        .permute(0, 3, 1, 2)
        .to(device, dtype=torch.float32)
        .div_(255.0)
    )
    return (tensor - mean) / std


def train_fewshot_ball_per_game(
    game_pools: dict,
    *,
    config: BallTrainConfig,
) -> dict:
    if config.epoch_checkpoints_dir is None:
        raise ValueError("epoch_checkpoints_dir required")
    device = torch.device(resolve_device(config.device))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = build_trainable_resnet18(
        device,
        config.imagenet_weights_path,
        trainable_layers=config.trainable_layers,
    )
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * config.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    games = game_pools["games"]
    game_ids = sorted(games.keys())
    if not game_ids:
        raise ValueError("no eligible games for per-game episodic training")
    gneg = game_pools["global_neg_arr"]

    rng = random.Random(config.seed)
    k_sup = config.references_per_class
    n_qry = config.queries_per_class
    config.epoch_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    same_game_neg_used = 0
    global_neg_used = 0
    t_start = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        losses: list[float] = []
        ep_t0 = time.monotonic()
        for _ in range(config.steps_per_epoch):
            gid = rng.choice(game_ids)
            g = games[gid]
            p_arr = g["pos_arr"]
            p_ids = g["pos_ids"]
            unique_frames = sorted(set(p_ids))
            rng.shuffle(unique_frames)
            sup_frames = set(unique_frames[:k_sup])
            sup_pos_idx = []
            for fid in unique_frames[:k_sup]:
                cands = [i for i, x in enumerate(p_ids) if x == fid]
                sup_pos_idx.append(rng.choice(cands))
            qry_pos_pool = [i for i, x in enumerate(p_ids) if x not in sup_frames]
            if len(qry_pos_pool) >= n_qry:
                qry_pos_idx = rng.sample(qry_pos_pool, n_qry)
            else:
                qry_pos_idx = [rng.choice(qry_pos_pool) for _ in range(n_qry)]

            n_arr = g["neg_arr"]
            n_ids = g["neg_ids"]
            same_sup_cands = [i for i, x in enumerate(n_ids) if x in sup_frames]
            same_other_cands = [i for i, x in enumerate(n_ids) if x not in sup_frames]
            sup_neg_list: list[np.ndarray] = []
            qry_neg_list: list[np.ndarray] = []

            all_same = list(range(len(n_arr)))
            rng.shuffle(all_same)
            used_same: set[int] = set()
            for idx in same_sup_cands + all_same:
                if len(sup_neg_list) >= k_sup:
                    break
                if idx not in used_same:
                    sup_neg_list.append(n_arr[idx])
                    used_same.add(idx)
                    same_game_neg_used += 1
            while len(sup_neg_list) < k_sup:
                sup_neg_list.append(gneg[rng.randrange(len(gneg))])
                global_neg_used += 1

            for idx in same_other_cands + all_same:
                if len(qry_neg_list) >= n_qry:
                    break
                if idx not in used_same:
                    qry_neg_list.append(n_arr[idx])
                    used_same.add(idx)
                    same_game_neg_used += 1
            while len(qry_neg_list) < n_qry:
                qry_neg_list.append(gneg[rng.randrange(len(gneg))])
                global_neg_used += 1

            sup_arr = np.concatenate([np.stack(sup_neg_list, axis=0), p_arr[sup_pos_idx]], axis=0)
            qry_arr = np.concatenate([np.stack(qry_neg_list, axis=0), p_arr[qry_pos_idx]], axis=0)
            qry_labels = torch.tensor([0] * n_qry + [1] * n_qry, dtype=torch.long, device=device)

            all_feats = model(
                _arrays_to_tensor(np.concatenate([sup_arr, qry_arr], axis=0), device, mean, std)
            )
            sup_feats = F.normalize(all_feats[: 2 * k_sup], dim=1)
            qry_feats = F.normalize(all_feats[2 * k_sup :], dim=1)
            proto_neg = sup_feats[:k_sup].mean(dim=0)
            proto_pos = sup_feats[k_sup:].mean(dim=0)
            prototypes = F.normalize(torch.stack([proto_neg, proto_pos], dim=0), dim=1)

            logits = (qry_feats @ prototypes.T) * config.cosine_scale
            loss = F.cross_entropy(logits, qry_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses))
        ep_dt = time.monotonic() - ep_t0
        ep_path = config.epoch_checkpoints_dir / f"epoch_{epoch:02d}.pth"
        ep_tmp = ep_path.with_suffix(".pth.tmp")
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "references_per_class": config.references_per_class,
                "seed": config.seed,
                "trainable_layers": config.trainable_layers,
            },
            ep_tmp,
        )
        ep_tmp.replace(ep_path)
        print(
            f"train_per_game[ctx={game_pools['stats']['context_scale']:.0f}] ep {epoch}/{config.epochs}: "
            f"loss={mean_loss:.4f} ({ep_dt:.1f}s)",
            flush=True,
        )
        history.append({"epoch": epoch, "loss": mean_loss, "elapsed_s": round(ep_dt, 2)})

    total_neg_slots = same_game_neg_used + global_neg_used
    return {
        "device": str(device),
        "trainable_layers": config.trainable_layers,
        "seed": config.seed,
        "epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "total_episodes": total_steps,
        "same_game_neg_fraction": float(same_game_neg_used / max(1, total_neg_slots)),
        "total_train_s": round(time.monotonic() - t_start, 2),
        "pool_stats": game_pools["stats"],
        "history": history,
    }


def fit_logistic_fusion(
    confs: np.ndarray,
    fs_scores: np.ndarray,
    is_tp: np.ndarray,
    total_gt: int,
    l2_reg: float = 1.0,
) -> dict:
    c_clip = np.clip(np.asarray(confs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    c_logit = np.log(c_clip / (1.0 - c_clip))
    fs = np.asarray(fs_scores, dtype=np.float64)
    y = np.asarray(is_tp, dtype=np.float64)

    mean_c, std_c = float(c_logit.mean()), float(max(1e-6, c_logit.std()))
    mean_f, std_f = float(fs.mean()), float(max(1e-6, fs.std()))
    x = np.column_stack(
        [
            np.ones(len(y), dtype=np.float64),
            (c_logit - mean_c) / std_c,
            (fs - mean_f) / std_f,
        ]
    )
    w = np.zeros(3, dtype=np.float64)
    reg = np.diag([0.0, l2_reg, l2_reg])
    for _ in range(25):
        z = np.clip(x @ w, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = x.T @ (p - y) + reg @ w
        weights_diag = np.maximum(1e-6, p * (1.0 - p))
        hess = (x.T * weights_diag) @ x + reg + 1e-6 * np.eye(3)
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.linalg.norm(step)) < 1e-6:
            break

    fused_scores = x @ w
    calib = calibrate_threshold(fused_scores, is_tp, total_gt)
    return {
        "weights": w.tolist(),
        "mean_c": mean_c,
        "std_c": std_c,
        "mean_f": mean_f,
        "std_f": std_f,
        "calibration": calib,
    }


def apply_logistic_fusion(model_params: dict, confs: np.ndarray, fs_scores: np.ndarray) -> np.ndarray:
    c_clip = np.clip(np.asarray(confs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    c_logit = np.log(c_clip / (1.0 - c_clip))
    fs = np.asarray(fs_scores, dtype=np.float64)
    w = np.asarray(model_params["weights"], dtype=np.float64)
    x = np.column_stack(
        [
            np.ones(len(fs), dtype=np.float64),
            (c_logit - model_params["mean_c"]) / model_params["std_c"],
            (fs - model_params["mean_f"]) / model_params["std_f"],
        ]
    )
    return x @ w


def fit_doubt_zone_filter(
    confs: np.ndarray,
    fs_scores: np.ndarray,
    is_tp: np.ndarray,
    total_gt: int,
    conf_high_grid: Sequence[float] = (0.45, 0.55, 0.65, 0.75, 1.01),
) -> dict:
    c = np.asarray(confs, dtype=np.float64)
    s = np.asarray(fs_scores, dtype=np.float64)
    tp = np.asarray(is_tp, dtype=bool)
    base_p, base_r, base_f1 = precision_recall_f1(int(tp.sum()), int((~tp).sum()), total_gt)

    best_key = (round(base_f1, 12), round(base_r, 12), math.inf)
    best_conf_high = 0.0
    best_thresh = -math.inf
    best_f1 = base_f1
    best_r = base_r

    for ch in conf_high_grid:
        in_doubt = c < ch
        if not np.any(in_doubt):
            continue
        doubt_scores = np.unique(s[in_doubt])
        if len(doubt_scores) > 80:
            qs = np.linspace(0.0, 1.0, 80)
            cands = np.unique(np.quantile(doubt_scores, qs)).tolist()
        else:
            mids = (doubt_scores[:-1] + doubt_scores[1:]) / 2.0 if len(doubt_scores) > 1 else np.empty(0)
            cands = mids.tolist() + [float(doubt_scores[-1]) + 1e-6]
        for th in cands:
            kept = (~in_doubt) | (s >= th)
            n_tp = int((kept & tp).sum())
            n_fp = int((kept & (~tp)).sum())
            _, r, f1 = precision_recall_f1(n_tp, n_fp, total_gt)
            key = (round(f1, 12), round(r, 12), -th)
            if key > best_key:
                best_key = key
                best_conf_high = float(ch)
                best_thresh = float(th)
                best_f1 = float(f1)
                best_r = float(r)

    return {
        "conf_high": best_conf_high,
        "threshold": best_thresh,
        "calib_base_f1": base_f1,
        "calib_best_f1": best_f1,
        "calib_best_recall": best_r,
    }


def _compute_subset_summary(
    *,
    subset_ids: Sequence[str],
    gt_by_image: Mapping[str, Sequence[tuple[float, float, float, float]]],
    base_by_img: Mapping[str, Sequence[Detection]],
    filt_by_img: Mapping[str, Sequence[Detection]],
    group_by: Mapping[str, str],
    eval_iou: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict:
    id_list = list(subset_ids)
    base_m = evaluate_detections(
        _preds_to_eval_format(base_by_img), gt_by_image, id_list, iou_threshold=eval_iou
    )
    filt_m = evaluate_detections(
        _preds_to_eval_format(filt_by_img), gt_by_image, id_list, iou_threshold=eval_iou
    )
    boot_img = paired_bootstrap_diff(
        base_m["per_image"], filt_m["per_image"], id_list, n_resamples=bootstrap_resamples, seed=seed
    )
    boot_grp = paired_cluster_bootstrap_diff(
        base_m["per_image"],
        filt_m["per_image"],
        id_list,
        group_by,
        n_resamples=bootstrap_resamples,
        seed=seed + 500,
    )
    tps_removed = max(0, base_m["tp"] - filt_m["tp"])
    fps_removed = max(0, base_m["fp"] - filt_m["fp"])
    return {
        "baseline": _strip_per_image(base_m),
        "filtered": _strip_per_image(filt_m),
        "fps_removed": int(fps_removed),
        "tps_removed": int(tps_removed),
        "delta_map50": float(filt_m["map50"] - base_m["map50"]),
        "bootstrap_image": boot_img,
        "bootstrap_group": boot_grp,
    }


def _score_per_game_and_global(
    *,
    encoder: FewShotR18Encoder,
    val_split: DatasetSplit,
    flat_dets: Sequence[Detection],
    val_crops: Sequence[Image.Image],
    auto_refs: dict,
    group_by: Mapping[str, str],
    global_pos_crops: Sequence[Image.Image],
    global_neg_crops: Sequence[Image.Image],
    context_scale: float,
) -> dict[str, dict[str, np.ndarray]]:
    val_emb = encoder.embed_crops(val_crops, layer="layer4")
    g_pos_emb = encoder.embed_crops(global_pos_crops, layer="layer4")
    g_neg_emb = encoder.embed_crops(global_neg_crops, layer="layer4")
    g_pos_proto = PrototypeFilter.fit(g_pos_emb, mode="positive_only")
    g_pn_proto = PrototypeFilter.fit(g_pos_emb, g_neg_emb, mode="pos_minus_neg")

    global_pos_scores = g_pos_proto.score(val_emb)
    global_pn_scores = g_pn_proto.score(val_emb)

    pos_crops_all: list[Image.Image] = []
    pos_slices: dict[str, tuple[int, int]] = {}
    neg_crops_all: list[Image.Image] = []
    neg_slices: dict[str, tuple[int, int]] = {}

    for gid in auto_refs["eligible_games"]:
        s0 = len(pos_crops_all)
        for rec, det in auto_refs["pos_refs_by_game"][gid]:
            with Image.open(rec.image_path) as img:
                pos_crops_all.append(crop_box(img.convert("RGB"), det.box, context_scale=context_scale, square=True))
        pos_slices[gid] = (s0, len(pos_crops_all))

        n0 = len(neg_crops_all)
        for rec, det in auto_refs["neg_refs_by_game"][gid]:
            with Image.open(rec.image_path) as img:
                neg_crops_all.append(crop_box(img.convert("RGB"), det.box, context_scale=context_scale, square=True))
        neg_slices[gid] = (n0, len(neg_crops_all))

    all_ref_pos_emb = encoder.embed_crops(pos_crops_all, layer="layer4") if pos_crops_all else np.empty((0, 512), dtype=np.float32)
    all_ref_neg_emb = encoder.embed_crops(neg_crops_all, layer="layer4") if neg_crops_all else np.empty((0, 512), dtype=np.float32)

    proto_pos_by_game: dict[str, PrototypeFilter] = {}
    proto_pn_same_by_game: dict[str, PrototypeFilter] = {}
    proto_pn_global_by_game: dict[str, PrototypeFilter] = {}

    for gid in auto_refs["eligible_games"]:
        a, b = pos_slices[gid]
        p_e = all_ref_pos_emb[a:b]
        proto_pos_by_game[gid] = PrototypeFilter.fit(p_e, mode="positive_only")
        proto_pn_global_by_game[gid] = PrototypeFilter.fit(p_e, g_neg_emb, mode="pos_minus_neg")
        na, nb = neg_slices[gid]
        if nb > na:
            n_e = all_ref_neg_emb[na:nb]
            if len(n_e) < len(p_e):
                n_e = np.concatenate([n_e, g_neg_emb[: len(p_e) - len(n_e)]], axis=0)
            proto_pn_same_by_game[gid] = PrototypeFilter.fit(p_e, n_e, mode="pos_minus_neg")
        else:
            proto_pn_same_by_game[gid] = proto_pn_global_by_game[gid]

    scores_pos_only = np.copy(global_pos_scores)
    scores_pn_same = np.copy(global_pn_scores)
    scores_pn_global = np.copy(global_pn_scores)

    det_gids = [group_by.get(d.image_id, d.image_id) for d in flat_dets]
    idx_by_game: dict[str, list[int]] = defaultdict(list)
    for idx, gid in enumerate(det_gids):
        if gid in proto_pos_by_game:
            idx_by_game[gid].append(idx)

    for gid, idxs in idx_by_game.items():
        emb_g = val_emb[idxs]
        scores_pos_only[idxs] = proto_pos_by_game[gid].score(emb_g)
        scores_pn_same[idxs] = proto_pn_same_by_game[gid].score(emb_g)
        scores_pn_global[idxs] = proto_pn_global_by_game[gid].score(emb_g)

    return {
        "positive_only": {"scores": scores_pos_only, "global_scores": global_pos_scores},
        "pos_minus_neg_same_game": {"scores": scores_pn_same, "global_scores": global_pn_scores},
        "pos_minus_neg_global": {"scores": scores_pn_global, "global_scores": global_pn_scores},
    }


def run_round3_experiment(
    config: Round3Config,
    *,
    val_detector: BallDetectorProtocol | None = None,
    mining_detector: BallDetectorProtocol | None = None,
) -> dict:
    splits = load_yolo_dataset(
        config.data_yaml,
        path_remap=config.path_remap,
        target_class=config.target_class,
        require_train=True,
    )
    train_split = splits["train"]
    val_split = splits["val"]
    assert_disjoint_splits(train_split, val_split)

    group_map = _load_group_map(config.group_map_path, (train_split, val_split))
    if group_map is None:
        group_map = {r.image_id: r.image_id for r in (*train_split.records, *val_split.records)}
    group_audit = audit_group_overlap(train_split, val_split, group_map)

    folds = make_group_kfold(val_split, n_folds=config.n_folds, seed=config.seed, group_by=group_map)
    fold_by_image = {r.image_id: f_idx for f_idx, fl in enumerate(folds) for r in fl.records}
    gt_by_image = val_split.gt_by_image()

    val_det = val_detector or YoloBallDetector(
        config.detector_weights,
        imgsz=config.imgsz,
        conf=config.val_mining_conf,
        iou=config.iou,
        max_det=config.max_det,
        target_class=config.target_class,
        device=config.device,
        batch_size=1,
    )
    val_cache = config.cache_dir / "dets_val_all_conf005.json"
    val_dets_low = val_det.predict_records(val_split.records, cache_path=val_cache)

    auto_refs = select_game_auto_references(
        val_split,
        val_dets_low,
        group_map,
        k_refs=config.n_refs_per_game,
        ref_min_conf=config.ref_min_conf,
        neg_conf_max=config.same_game_neg_max_conf,
        eval_iou=config.eval_iou,
    )
    ref_image_ids: set[str] = auto_refs["ref_image_ids"]
    elig_query_ids: list[str] = auto_refs["eligible_query_ids"]
    elig_no_v14_ids: list[str] = auto_refs["eligible_no_v14_query_ids"]
    elig_query_set = set(elig_query_ids)

    mine_det = mining_detector or YoloBallDetector(
        config.detector_weights,
        imgsz=config.train_mining_imgsz,
        conf=config.train_mining_conf,
        iou=config.iou,
        max_det=config.max_det,
        target_class=config.target_class,
        device=config.device,
        batch_size=1,
    )
    capped_train_records = _select_capped_train_records(
        train_split, group_map, cap_per_game=config.train_cap_per_game, seed=config.train_seed
    )
    mined_jsonl = config.cache_dir / f"train_mined_cap{config.train_cap_per_game}_imgsz{config.train_mining_imgsz}.jsonl"
    mining_stats = mine_training_boxes_incremental(
        capped_train_records,
        mine_det,
        mined_jsonl,
        eval_conf=0.35,
        mining_conf=config.train_mining_conf,
        eval_iou=config.eval_iou,
    )

    rng_g = random.Random(config.seed)
    tr_pos_cands = [
        (r.image_id, r.image_path, box)
        for r in sorted(train_split.records, key=lambda x: x.image_id)
        for box in r.gt_boxes
    ]
    rng_g.shuffle(tr_pos_cands)
    tr_pos_chosen = tr_pos_cands[: config.n_refs_per_game]

    tr_neg_cands: list[tuple[str, str, tuple[float, float, float, float], float]] = []
    with mined_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gt_boxes = [tuple(float(v) for v in b) for b in row["gt_boxes"]]
            gt_arr = np.asarray(gt_boxes, dtype=np.float64).reshape(len(gt_boxes), 4) if gt_boxes else np.empty((0, 4))
            for d in row["dets"]:
                if not d["is_tp"]:
                    box = tuple(float(v) for v in d["box"])
                    if len(gt_arr) > 0 and float(box_iou_matrix(np.asarray([box]), gt_arr).max()) > 0.0:
                        continue
                    tr_neg_cands.append((str(row["image_id"]), str(row["image_path"]), box, float(d["conf"])))
    rng_g.shuffle(tr_neg_cands)
    tr_neg_chosen = tr_neg_cands[: config.n_refs_per_game]

    global_pos_crops_by_ctx: dict[float, list[Image.Image]] = {ctx: [] for ctx in config.context_scales}
    global_neg_crops_by_ctx: dict[float, list[Image.Image]] = {ctx: [] for ctx in config.context_scales}
    for _, ipath, box in tr_pos_chosen:
        with Image.open(ipath) as img:
            rgb = img.convert("RGB")
            for ctx in config.context_scales:
                global_pos_crops_by_ctx[ctx].append(crop_box(rgb, box, context_scale=ctx, square=True))
    for _, ipath, box, _ in tr_neg_chosen:
        with Image.open(ipath) as img:
            rgb = img.convert("RGB")
            for ctx in config.context_scales:
                global_neg_crops_by_ctx[ctx].append(crop_box(rgb, box, context_scale=ctx, square=True))

    training_runs: list[dict] = []
    ckpt_dirs_by_ctx: dict[float, Path] = {}
    for ctx in config.context_scales:
        game_pools = build_per_game_train_pools(
            mined_jsonl,
            group_map,
            context_scale=ctx,
            canvas_wh=config.canvas_wh,
            eval_conf=0.35,
            mining_conf=config.train_mining_conf,
            seed=config.train_seed,
        )
        ckpt_dir = config.cache_dir / f"ckpts_per_game_ctx{int(ctx)}"
        ckpt_dirs_by_ctx[ctx] = ckpt_dir
        t_info = train_fewshot_ball_per_game(
            game_pools,
            config=BallTrainConfig(
                epochs=config.train_epochs,
                steps_per_epoch=config.train_steps_per_epoch,
                references_per_class=config.n_refs_per_game,
                queries_per_class=8,
                seed=config.train_seed,
                canvas_wh=config.canvas_wh,
                device=config.device,
                trainable_layers="layer3_layer4",
                imagenet_weights_path=config.imagenet_weights,
                epoch_checkpoints_dir=ckpt_dir,
            ),
        )
        t_info["context_scale"] = float(ctx)
        training_runs.append(t_info)

    min_op_conf = min(config.operating_confs)
    dets_min_op = filter_dets_by_conf(val_dets_low, min_op_conf)
    flat_all_dets: list[Detection] = []
    crops_all_by_ctx: dict[float, list[Image.Image]] = {ctx: [] for ctx in config.context_scales}
    for rec in val_split.records:
        ds = dets_min_op.get(rec.image_id, [])
        if not ds:
            continue
        flat_all_dets.extend(ds)
        with Image.open(rec.image_path) as img:
            rgb = img.convert("RGB")
            for det in ds:
                for ctx in config.context_scales:
                    crops_all_by_ctx[ctx].append(crop_box(rgb, det.box, context_scale=ctx, square=True))

    scores_store: dict[str, dict[int, np.ndarray]] = {}
    frozen_enc = FewShotR18Encoder(
        config.fewshot_weights, device=config.device, canvas_wh=config.canvas_wh
    )
    for ctx in config.context_scales:
        frz_map = _score_per_game_and_global(
            encoder=frozen_enc,
            val_split=val_split,
            flat_dets=flat_all_dets,
            val_crops=crops_all_by_ctx[ctx],
            auto_refs=auto_refs,
            group_by=group_map,
            global_pos_crops=global_pos_crops_by_ctx[ctx],
            global_neg_crops=global_neg_crops_by_ctx[ctx],
            context_scale=ctx,
        )
        for mode_name, pack in frz_map.items():
            scores_store[f"frozen_ctx{int(ctx)}_{mode_name}"] = {0: pack["scores"]}

        ckpt_dir = ckpt_dirs_by_ctx[ctx]
        for ep in range(1, config.train_epochs + 1):
            ep_enc = FewShotR18Encoder(
                ckpt_dir / f"epoch_{ep:02d}.pth",
                device=config.device,
                canvas_wh=config.canvas_wh,
            )
            tr_map = _score_per_game_and_global(
                encoder=ep_enc,
                val_split=val_split,
                flat_dets=flat_all_dets,
                val_crops=crops_all_by_ctx[ctx],
                auto_refs=auto_refs,
                group_by=group_map,
                global_pos_crops=global_pos_crops_by_ctx[ctx],
                global_neg_crops=global_neg_crops_by_ctx[ctx],
                context_scale=ctx,
            )
            for mode_name, pack in tr_map.items():
                scores_store.setdefault(f"trained_ctx{int(ctx)}_{mode_name}", {})[ep] = pack["scores"]

    det_index_map = {
        (d.image_id, d.box, d.conf): idx for idx, d in enumerate(flat_all_dets)
    }

    operating_point_results: dict[str, dict] = {}
    total_configs_tested = 0

    for op_conf in config.operating_confs:
        op_key = f"conf_{op_conf:.2f}"
        base_by_img = filter_dets_by_conf(val_dets_low, op_conf)
        base_all_m = evaluate_detections(
            _preds_to_eval_format(base_by_img), gt_by_image, val_split.image_ids, iou_threshold=config.eval_iou
        )
        base_elig_m = evaluate_detections(
            _preds_to_eval_format(base_by_img), gt_by_image, elig_query_ids, iou_threshold=config.eval_iou
        )
        base_no_v14_m = (
            evaluate_detections(
                _preds_to_eval_format(base_by_img), gt_by_image, elig_no_v14_ids, iou_threshold=config.eval_iou
            )
            if elig_no_v14_ids
            else None
        )

        cheap_baselines: dict[str, dict] = {}
        for cname, c_dets in (
            ("nms_050", apply_post_nms(base_by_img, 0.50)),
            ("nms_040", apply_post_nms(base_by_img, 0.40)),
            ("top1_per_frame", apply_top1_per_frame(base_by_img)),
        ):
            total_configs_tested += 1
            cheap_baselines[cname] = {
                "all_val": _compute_subset_summary(
                    subset_ids=val_split.image_ids,
                    gt_by_image=gt_by_image,
                    base_by_img=base_by_img,
                    filt_by_img=c_dets,
                    group_by=group_map,
                    eval_iou=config.eval_iou,
                    bootstrap_resamples=config.bootstrap_resamples,
                    seed=config.seed + 100,
                ),
                "eligible_queries": _compute_subset_summary(
                    subset_ids=elig_query_ids,
                    gt_by_image=gt_by_image,
                    base_by_img=base_by_img,
                    filt_by_img=c_dets,
                    group_by=group_map,
                    eval_iou=config.eval_iou,
                    bootstrap_resamples=config.bootstrap_resamples,
                    seed=config.seed + 200,
                ),
            }

        op_flat_dets: list[Detection] = []
        op_flat_tp: list[bool] = []
        op_global_idx: list[int] = []
        for rec in val_split.records:
            ds = base_by_img.get(rec.image_id, [])
            if not ds:
                continue
            matched = match_image_detections(
                [d.box for d in ds], [d.conf for d in ds], rec.gt_boxes, iou_threshold=config.eval_iou
            )
            for d, tp_flag in zip(ds, matched):
                op_flat_dets.append(d)
                op_flat_tp.append(bool(tp_flag))
                op_global_idx.append(det_index_map[(d.image_id, d.box, d.conf)])

        op_tp_arr = np.asarray(op_flat_tp, dtype=bool)
        op_conf_arr = np.asarray([d.conf for d in op_flat_dets], dtype=np.float64)
        op_fold_arr = np.asarray([fold_by_image[d.image_id] for d in op_flat_dets], dtype=np.int64)
        op_in_elig_query = np.asarray([d.image_id in elig_query_set for d in op_flat_dets], dtype=bool)
        op_is_ref_frame = np.asarray([d.image_id in ref_image_ids for d in op_flat_dets], dtype=bool)
        op_idx_arr = np.asarray(op_global_idx, dtype=np.int64)

        yolo_conf_kept = np.ones(len(op_flat_dets), dtype=bool)
        for f_idx in range(config.n_folds):
            t_mask = op_fold_arr == f_idx
            c_mask = (~t_mask) & (~op_is_ref_frame)
            c_gt = sum(len(gt_by_image[r.image_id]) for i, fl in enumerate(folds) if i != f_idx for r in fl.records if r.image_id not in ref_image_ids)
            cal = calibrate_threshold(op_conf_arr[c_mask], op_tp_arr[c_mask], c_gt)
            yolo_conf_kept[t_mask] = op_conf_arr[t_mask] >= float(cal["threshold"])
        yolo_conf_kept[op_is_ref_frame] = True
        total_configs_tested += 1
        yolo_conf_by_img = _apply_mask_to_split(val_split, op_flat_dets, yolo_conf_kept)
        cheap_baselines["yolo_conf_cv"] = {
            "all_val": _compute_subset_summary(
                subset_ids=val_split.image_ids,
                gt_by_image=gt_by_image,
                base_by_img=base_by_img,
                filt_by_img=yolo_conf_by_img,
                group_by=group_map,
                eval_iou=config.eval_iou,
                bootstrap_resamples=config.bootstrap_resamples,
                seed=config.seed + 300,
            ),
            "eligible_queries": _compute_subset_summary(
                subset_ids=elig_query_ids,
                gt_by_image=gt_by_image,
                base_by_img=base_by_img,
                filt_by_img=yolo_conf_by_img,
                group_by=group_map,
                eval_iou=config.eval_iou,
                bootstrap_resamples=config.bootstrap_resamples,
                seed=config.seed + 400,
            ),
        }

        def run_nested_strategy(
            prefix: str,
            strategy: str,
            fallback_keep_detector: bool,
            apply_top1_after: bool = False,
        ) -> dict:
            nonlocal total_configs_tested
            matching_keys = sorted(k for k in scores_store.keys() if k.startswith(prefix))
            total_configs_tested += len(matching_keys)
            oof_kept = np.ones(len(op_flat_dets), dtype=bool)
            oof_scores_elig = np.zeros(len(op_flat_dets), dtype=np.float64)
            fold_logs: list[dict] = []

            for f_idx in range(config.n_folds):
                t_mask = op_fold_arr == f_idx
                c_mask = (~t_mask) & op_in_elig_query if fallback_keep_detector else (~t_mask) & (~op_is_ref_frame)
                if not np.any(c_mask):
                    c_mask = (~t_mask) & (~op_is_ref_frame)
                c_img_ids = [
                    r.image_id
                    for i, fl in enumerate(folds)
                    if i != f_idx
                    for r in fl.records
                    if (r.image_id in elig_query_set if fallback_keep_detector else r.image_id not in ref_image_ids)
                ]
                c_gt = sum(len(gt_by_image[iid]) for iid in c_img_ids)

                best_choice = None
                best_key = None
                for cfg_key in matching_keys:
                    for ep, raw_scores in sorted(scores_store[cfg_key].items()):
                        s_op = raw_scores[op_idx_arr]
                        if strategy == "cut_only":
                            cal = calibrate_threshold(s_op[c_mask], op_tp_arr[c_mask], c_gt)
                            c_auc = compute_roc_auc(s_op[c_mask], op_tp_arr[c_mask])
                            key = (round(cal["calib_best_f1"], 12), round(cal["calib_best_recall"], 12), round(c_auc or -1.0, 12), -ep)
                            params = {"threshold": float(cal["threshold"])}
                        elif strategy == "doubt_zone":
                            dz = fit_doubt_zone_filter(op_conf_arr[c_mask], s_op[c_mask], op_tp_arr[c_mask], c_gt)
                            c_auc = compute_roc_auc(s_op[c_mask], op_tp_arr[c_mask])
                            key = (round(dz["calib_best_f1"], 12), round(dz["calib_best_recall"], 12), round(c_auc or -1.0, 12), -ep)
                            params = dz
                        else:
                            lf = fit_logistic_fusion(op_conf_arr[c_mask], s_op[c_mask], op_tp_arr[c_mask], c_gt)
                            cal = lf["calibration"]
                            c_fused = apply_logistic_fusion(lf, op_conf_arr[c_mask], s_op[c_mask])
                            c_auc = compute_roc_auc(c_fused, op_tp_arr[c_mask])
                            key = (round(cal["calib_best_f1"], 12), round(cal["calib_best_recall"], 12), round(c_auc or -1.0, 12), -ep)
                            params = lf
                        if best_key is None or key > best_key:
                            best_key = key
                            best_choice = (cfg_key, ep, params, c_auc)

                assert best_choice is not None
                cfg_key, ep, params, c_auc = best_choice
                s_test = scores_store[cfg_key][ep][op_idx_arr]
                if strategy == "cut_only":
                    kept_t = s_test[t_mask] >= params["threshold"]
                    t_score_eval = s_test[t_mask]
                elif strategy == "doubt_zone":
                    in_d = op_conf_arr[t_mask] < params["conf_high"]
                    kept_t = (~in_d) | (s_test[t_mask] >= params["threshold"])
                    t_score_eval = s_test[t_mask]
                else:
                    t_fused = apply_logistic_fusion(params, op_conf_arr[t_mask], s_test[t_mask])
                    kept_t = t_fused >= float(params["calibration"]["threshold"])
                    t_score_eval = t_fused

                oof_kept[t_mask] = kept_t
                oof_scores_elig[t_mask] = t_score_eval
                fold_logs.append(
                    {
                        "fold": f_idx,
                        "chosen_config": cfg_key,
                        "chosen_epoch": int(ep),
                        "calib_roc_auc": c_auc,
                        "fold_roc_auc_elig": compute_roc_auc(
                            oof_scores_elig[t_mask & op_in_elig_query],
                            op_tp_arr[t_mask & op_in_elig_query],
                        ),
                    }
                )

            oof_kept[op_is_ref_frame] = True
            if fallback_keep_detector:
                oof_kept[~op_in_elig_query] = True

            filt_by_img = _apply_mask_to_split(val_split, op_flat_dets, oof_kept)
            if apply_top1_after:
                filt_by_img = apply_top1_per_frame(filt_by_img)

            fold_aucs = [fl["fold_roc_auc_elig"] for fl in fold_logs if fl["fold_roc_auc_elig"] is not None]
            res = {
                "strategy": strategy,
                "fallback_mode": "keep_detector" if fallback_keep_detector else "global_train_prototype",
                "top1_after": apply_top1_after,
                "mean_fold_roc_auc_elig": float(np.mean(fold_aucs)) if fold_aucs else None,
                "eligible_queries": _compute_subset_summary(
                    subset_ids=elig_query_ids,
                    gt_by_image=gt_by_image,
                    base_by_img=base_by_img,
                    filt_by_img=filt_by_img,
                    group_by=group_map,
                    eval_iou=config.eval_iou,
                    bootstrap_resamples=config.bootstrap_resamples,
                    seed=config.seed + 600,
                ),
                "all_val": _compute_subset_summary(
                    subset_ids=val_split.image_ids,
                    gt_by_image=gt_by_image,
                    base_by_img=base_by_img,
                    filt_by_img=filt_by_img,
                    group_by=group_map,
                    eval_iou=config.eval_iou,
                    bootstrap_resamples=config.bootstrap_resamples,
                    seed=config.seed + 700,
                ),
                "folds": fold_logs,
            }
            if elig_no_v14_ids:
                res["eligible_no_v14_queries"] = _compute_subset_summary(
                    subset_ids=elig_no_v14_ids,
                    gt_by_image=gt_by_image,
                    base_by_img=base_by_img,
                    filt_by_img=filt_by_img,
                    group_by=group_map,
                    eval_iou=config.eval_iou,
                    bootstrap_resamples=config.bootstrap_resamples,
                    seed=config.seed + 800,
                )
            return res

        methods: dict[str, dict] = {
            "frozen_per_game_cut": run_nested_strategy("frozen_", "cut_only", fallback_keep_detector=True),
            "trained_per_game_cut_keep_fallback": run_nested_strategy("trained_", "cut_only", fallback_keep_detector=True),
            "trained_per_game_cut_global_fallback": run_nested_strategy("trained_", "cut_only", fallback_keep_detector=False),
            "trained_per_game_doubt_zone": run_nested_strategy("trained_", "doubt_zone", fallback_keep_detector=False),
            "trained_per_game_logistic_fusion": run_nested_strategy("trained_", "logistic_fusion", fallback_keep_detector=False),
            "trained_per_game_logistic_plus_top1": run_nested_strategy(
                "trained_", "logistic_fusion", fallback_keep_detector=False, apply_top1_after=True
            ),
        }

        mode_breakdown: dict[str, dict] = {}
        for mode_tag in ("positive_only", "pos_minus_neg_same_game", "pos_minus_neg_global"):
            mode_keys = [k for k in scores_store.keys() if k.startswith("trained_") and k.endswith(mode_tag)]
            if not mode_keys:
                continue
            oof_kept_m = np.ones(len(op_flat_dets), dtype=bool)
            fold_aucs_m: list[float] = []
            for f_idx in range(config.n_folds):
                t_mask = op_fold_arr == f_idx
                c_mask = (~t_mask) & op_in_elig_query
                if not np.any(c_mask):
                    c_mask = (~t_mask) & (~op_is_ref_frame)
                c_img_ids = [r.image_id for i, fl in enumerate(folds) if i != f_idx for r in fl.records if r.image_id in elig_query_set]
                c_gt = sum(len(gt_by_image[iid]) for iid in c_img_ids)
                best_c = None
                best_k = None
                for mk in mode_keys:
                    for ep, raw_s in sorted(scores_store[mk].items()):
                        s_op = raw_s[op_idx_arr]
                        cal = calibrate_threshold(s_op[c_mask], op_tp_arr[c_mask], c_gt)
                        c_auc = compute_roc_auc(s_op[c_mask], op_tp_arr[c_mask])
                        k_val = (round(cal["calib_best_f1"], 12), round(cal["calib_best_recall"], 12), round(c_auc or -1.0, 12), -ep)
                        if best_k is None or k_val > best_k:
                            best_k = k_val
                            best_c = (mk, ep, float(cal["threshold"]))
                assert best_c is not None
                mk, ep, th = best_c
                s_op = scores_store[mk][ep][op_idx_arr]
                oof_kept_m[t_mask] = s_op[t_mask] >= th
                a = compute_roc_auc(s_op[t_mask & op_in_elig_query], op_tp_arr[t_mask & op_in_elig_query])
                if a is not None:
                    fold_aucs_m.append(a)
            oof_kept_m[op_is_ref_frame] = True
            oof_kept_m[~op_in_elig_query] = True
            filt_m_img = _apply_mask_to_split(val_split, op_flat_dets, oof_kept_m)
            mode_breakdown[mode_tag] = {
                "mean_fold_roc_auc_elig": float(np.mean(fold_aucs_m)) if fold_aucs_m else None,
                "eligible_queries": _compute_subset_summary(
                    subset_ids=elig_query_ids,
                    gt_by_image=gt_by_image,
                    base_by_img=base_by_img,
                    filt_by_img=filt_m_img,
                    group_by=group_map,
                    eval_iou=config.eval_iou,
                    bootstrap_resamples=config.bootstrap_resamples,
                    seed=config.seed + 900,
                ),
            }

        operating_point_results[op_key] = {
            "conf": float(op_conf),
            "detector_conf_roc_auc_all": compute_roc_auc(op_conf_arr, op_tp_arr),
            "detector_conf_roc_auc_elig": compute_roc_auc(op_conf_arr[op_in_elig_query], op_tp_arr[op_in_elig_query]),
            "baseline_all_val": _strip_per_image(base_all_m),
            "baseline_eligible_queries": _strip_per_image(base_elig_m),
            "baseline_eligible_no_v14": _strip_per_image(base_no_v14_m) if base_no_v14_m else None,
            "cheap_baselines": cheap_baselines,
            "methods": methods,
            "mode_breakdown_eligible": mode_breakdown,
        }

    payload = {
        "round": 3,
        "protocol": {
            "data_yaml": str(config.data_yaml),
            "detector_weights": config.detector_weights.name,
            "fewshot_weights": config.fewshot_weights.name,
            "imgsz": config.imgsz,
            "operating_confs": list(config.operating_confs),
            "nms_iou": config.iou,
            "eval_iou": config.eval_iou,
            "context_scales": list(config.context_scales),
            "square_crop": True,
            "canvas_wh": list(config.canvas_wh),
            "n_refs_per_game": config.n_refs_per_game,
            "ref_min_conf": config.ref_min_conf,
            "same_game_neg_max_conf": config.same_game_neg_max_conf,
            "n_folds": config.n_folds,
            "seed": config.seed,
            "train_seed": config.train_seed,
            "train_epochs": config.train_epochs,
            "train_steps_per_epoch": config.train_steps_per_epoch,
            "train_total_episodes": config.train_epochs * config.train_steps_per_epoch,
            "train_cap_per_game": config.train_cap_per_game,
            "train_mining_imgsz": config.train_mining_imgsz,
            "bootstrap_resamples": config.bootstrap_resamples,
            "total_configs_tested": total_configs_tested,
        },
        "splits": {
            "train_images": len(train_split.records),
            "val_images": len(val_split.records),
            "val_gt_balls": count_gt_boxes(val_split.records),
            "train_val_group_audit": group_audit,
        },
        "auto_reference_audit": auto_refs["coverage"],
        "mining_stats": mining_stats,
        "training_runs": training_runs,
        "operating_points": operating_point_results,
    }
    sanitized = _sanitize_for_json(payload)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = config.output_dir / "rodada3_experiment_results.json"
    tmp_json = out_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_json.replace(out_json)
    return sanitized
