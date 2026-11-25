from __future__ import annotations

import argparse
from pathlib import Path

from bola_fewshot.experiment import ExperimentConfig, run_experiment


def _parse_remap(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("remap-prefix must have format OLD_PREFIX=NEW_PREFIX")
    old_p, new_p = spec.split("=", 1)
    if not old_p.strip() or not new_p.strip():
        raise argparse.ArgumentTypeError("remap-prefix requires non-empty old and new prefixes")
    return old_p.strip(), new_p.strip()


def _parse_extra_dataset(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("extra dataset must have format NAME=PATH_TO_DATA_YAML")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise argparse.ArgumentTypeError("extra dataset name must be non-empty")
    return name, path


def _parse_canvas(spec: str) -> tuple[int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("canvas must have format WxH, e.g. 48x96 or 64x64")
    w, h = int(parts[0]), int(parts[1])
    if w < 8 or h < 8:
        raise argparse.ArgumentTypeError("canvas dimensions must be >= 8")
    return w, h


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO ball detector alone vs YOLO + FewShot-R18 FP filter."
    )
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--fewshot-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-path", type=Path, default=Path("RELATORIO.md"))
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--labels-root", type=Path, default=None)
    parser.add_argument("--remap-prefix", type=_parse_remap, default=None)
    parser.add_argument("--val-list", type=Path, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--eval-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calib-source",
        choices=("val_split", "train_holdout"),
        default="val_split",
    )
    parser.add_argument("--calib-fraction", type=float, default=0.50)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--group-map-json", type=Path, default=None)
    parser.add_argument("--val-status-note", type=str, default=None)
    parser.add_argument("--canvas", type=_parse_canvas, default=(48, 96))
    parser.add_argument("--context-scale", type=float, default=1.0)
    parser.add_argument("--square-crop", action="store_true")
    parser.add_argument(
        "--variations",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument("--auc-variation-trigger", type=float, default=0.70)
    parser.add_argument("--max-train-neg-images", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--extra-dataset",
        type=_parse_extra_dataset,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--extra-val-list",
        type=_parse_extra_dataset,
        action="append",
        default=[],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        data_yaml=args.data_yaml,
        detector_weights=args.detector_weights,
        fewshot_weights=args.fewshot_weights,
        output_dir=args.output_dir,
        report_path=args.report_path,
        images_root=args.images_root,
        labels_root=args.labels_root,
        path_remap=args.remap_prefix,
        val_list_override=args.val_list,
        max_val_images=args.max_val_images,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        eval_iou=args.eval_iou,
        max_det=args.max_det,
        target_class=args.target_class,
        k_values=tuple(args.k_values),
        seed=args.seed,
        calib_source=args.calib_source,
        calib_fraction=args.calib_fraction,
        bootstrap_resamples=args.bootstrap_resamples,
        group_map_path=args.group_map_json,
        val_status_note=args.val_status_note,
        canvas_wh=args.canvas,
        context_scale=args.context_scale,
        square_crop=args.square_crop,
        variations=args.variations,
        auc_variation_trigger=args.auc_variation_trigger,
        max_train_neg_images=args.max_train_neg_images,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        extra_datasets=tuple(args.extra_dataset),
        extra_val_lists=tuple(args.extra_val_list),
    )
    results = run_experiment(config)
    selected_id = results["selected_config_id"]
    selected = next(c for c in results["configurations"] if c["config_id"] == selected_id)
    bf1 = selected["bootstrap"]["delta_f1"]
    print(
        f"done: selected={selected_id} delta_f1={bf1['point_estimate']:+.4f} "
        f"ci95=[{bf1['ci_95_low']:+.4f}, {bf1['ci_95_high']:+.4f}] "
        f"report={config.report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
