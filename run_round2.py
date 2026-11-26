from __future__ import annotations

import argparse
from pathlib import Path

from bola_fewshot.cv_round2 import Round2Config, run_round2_cv


def _parse_remap(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("remap-prefix must have format OLD_PREFIX=NEW_PREFIX")
    old_p, new_p = spec.split("=", 1)
    return old_p.strip(), new_p.strip()


def _parse_canvas(spec: str) -> tuple[int, int]:
    w, h = spec.lower().split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Round 2 nested 5-fold group CV with multi-context crops and full-train negative mining."
    )
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--fewshot-weights", type=Path, required=True)
    parser.add_argument("--imagenet-weights", type=Path, default=None)
    parser.add_argument("--group-map-json", type=Path, default=None)
    parser.add_argument("--remap-prefix", type=_parse_remap, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/r2_cache"))
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--eval-iou", type=float, default=0.50)
    parser.add_argument("--context-scales", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--canvas", type=_parse_canvas, default=(96, 96))
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=25)
    parser.add_argument("--train-epochs", type=int, default=16)
    parser.add_argument("--train-steps-per-epoch", type=int, default=360)
    parser.add_argument("--train-mining-conf", type=float, default=0.05)
    parser.add_argument("--train-mining-imgsz", type=int, default=1280)
    parser.add_argument("--train-max-pos-crops", type=int, default=12000)
    parser.add_argument("--train-max-neg-crops", type=int, default=12000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-all-layers", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Round2Config(
        data_yaml=args.data_yaml,
        detector_weights=args.detector_weights,
        fewshot_weights=args.fewshot_weights,
        imagenet_weights=args.imagenet_weights,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        group_map_path=args.group_map_json,
        path_remap=args.remap_prefix,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        eval_iou=args.eval_iou,
        context_scales=tuple(args.context_scales),
        canvas_wh=args.canvas,
        k_values=tuple(args.k_values),
        n_folds=args.n_folds,
        seed=args.seed,
        train_seed=args.train_seed,
        train_epochs=args.train_epochs,
        train_steps_per_epoch=args.train_steps_per_epoch,
        train_mining_conf=args.train_mining_conf,
        train_mining_imgsz=args.train_mining_imgsz,
        train_max_pos_crops=args.train_max_pos_crops,
        train_max_neg_crops=args.train_max_neg_crops,
        bootstrap_resamples=args.bootstrap_resamples,
        device=args.device,
        include_all_layers_variation=not args.skip_all_layers,
    )
    payload = run_round2_cv(config)
    for arm_name, res in payload["nested_arms"].items():
        b_img = res["bootstrap_image"]["delta_f1"]
        print(
            f"round2[{arm_name}]: oof_f1={res['pooled_oof_metrics']['f1']:.4f} "
            f"delta_f1={b_img['point_estimate']:+.4f} "
            f"ci_img=[{b_img['ci_95_low']:+.4f}, {b_img['ci_95_high']:+.4f}] "
            f"auc={res['pooled_oof_roc_auc']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
