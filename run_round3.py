from __future__ import annotations

import argparse
from pathlib import Path

from bola_fewshot.round3_per_game import Round3Config, run_round3_experiment


def _parse_remap(spec: str) -> tuple[str, str]:
    old_p, new_p = spec.split("=", 1)
    return old_p.strip(), new_p.strip()


def _parse_canvas(spec: str) -> tuple[int, int]:
    w, h = spec.lower().split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Round 3 per-game auto-reference few-shot, low-conf fusion, and NMS baselines."
    )
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--fewshot-weights", type=Path, required=True)
    parser.add_argument("--imagenet-weights", type=Path, default=None)
    parser.add_argument("--group-map-json", type=Path, default=None)
    parser.add_argument("--remap-prefix", type=_parse_remap, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/r3_cache"))
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--operating-confs", type=float, nargs="+", default=[0.35, 0.15, 0.10])
    parser.add_argument("--context-scales", type=float, nargs="+", default=[2.0, 3.0])
    parser.add_argument("--canvas", type=_parse_canvas, default=(96, 96))
    parser.add_argument("--n-refs", type=int, default=6)
    parser.add_argument("--ref-min-conf", type=float, default=0.60)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=25)
    parser.add_argument("--train-epochs", type=int, default=16)
    parser.add_argument("--train-steps-per-epoch", type=int, default=360)
    parser.add_argument("--train-mining-imgsz", type=int, default=1280)
    parser.add_argument("--train-cap-per-game", type=int, default=80)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Round3Config(
        data_yaml=args.data_yaml,
        detector_weights=args.detector_weights,
        fewshot_weights=args.fewshot_weights,
        imagenet_weights=args.imagenet_weights,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        group_map_path=args.group_map_json,
        path_remap=args.remap_prefix,
        imgsz=args.imgsz,
        operating_confs=tuple(args.operating_confs),
        context_scales=tuple(args.context_scales),
        canvas_wh=args.canvas,
        n_refs_per_game=args.n_refs,
        ref_min_conf=args.ref_min_conf,
        n_folds=args.n_folds,
        seed=args.seed,
        train_seed=args.train_seed,
        train_epochs=args.train_epochs,
        train_steps_per_epoch=args.train_steps_per_epoch,
        train_mining_imgsz=args.train_mining_imgsz,
        train_cap_per_game=args.train_cap_per_game,
        bootstrap_resamples=args.bootstrap_resamples,
        device=args.device,
    )
    payload = run_round3_experiment(cfg)
    cov = payload["auto_reference_audit"]
    print(
        f"round3 done: eligible_games={cov['eligible_games']}/{cov['total_games']} "
        f"auto_ref_prec50={cov['auto_ref_precision_iou50']:.4f}",
        flush=True,
    )
    for op_key, op_val in payload["operating_points"].items():
        b_f1 = op_val["baseline_all_val"]["f1"]
        t1_f1 = op_val["cheap_baselines"]["top1_per_frame"]["all_val"]["filtered"]["f1"]
        lf_f1 = op_val["methods"]["trained_per_game_logistic_fusion"]["all_val"]["filtered"]["f1"]
        lt_f1 = op_val["methods"]["trained_per_game_logistic_plus_top1"]["all_val"]["filtered"]["f1"]
        print(
            f"  {op_key}: base={b_f1:.4f} top1={t1_f1:.4f} logistic={lf_f1:.4f} logistic+top1={lt_f1:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
