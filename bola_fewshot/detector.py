from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Protocol, Sequence

import torch

from bola_fewshot.dataset import ImageRecord


@dataclass(frozen=True)
class Detection:
    image_id: str
    box: tuple[float, float, float, float]
    conf: float


class BallDetectorProtocol(Protocol):
    def predict_records(
        self,
        records: Sequence[ImageRecord],
        cache_path: Path | None = None,
    ) -> dict[str, list[Detection]]: ...


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloBallDetector:
    def __init__(
        self,
        weights_path: Path,
        *,
        imgsz: int = 1920,
        conf: float = 0.35,
        iou: float = 0.70,
        max_det: int = 300,
        target_class: int = 0,
        device: str = "auto",
        batch_size: int = 1,
    ) -> None:
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"detector weights not found: {self.weights_path}")
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.target_class = int(target_class)
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights_path))
        return self._model

    def _cache_signature(self, records: Sequence[ImageRecord]) -> dict:
        return {
            "weights_name": self.weights_path.name,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "target_class": self.target_class,
            "n_images": len(records),
            "image_ids": [r.image_id for r in records],
        }

    def _try_load_cache(
        self,
        cache_path: Path,
        records: Sequence[ImageRecord],
    ) -> dict[str, list[Detection]] | None:
        if not cache_path.is_file():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("signature") != self._cache_signature(records):
            return None
        raw = payload.get("detections", {})
        return {
            image_id: [
                Detection(
                    image_id=image_id,
                    box=tuple(float(v) for v in item["box"]),
                    conf=float(item["conf"]),
                )
                for item in raw.get(image_id, [])
            ]
            for image_id in (r.image_id for r in records)
        }

    def _save_cache(
        self,
        cache_path: Path,
        records: Sequence[ImageRecord],
        detections: dict[str, list[Detection]],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": self._cache_signature(records),
            "detections": {
                image_id: [
                    {"box": list(det.box), "conf": det.conf}
                    for det in detections.get(image_id, [])
                ]
                for image_id in (r.image_id for r in records)
            },
        }
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(cache_path)

    def predict_records(
        self,
        records: Sequence[ImageRecord],
        cache_path: Path | None = None,
    ) -> dict[str, list[Detection]]:
        if cache_path is not None:
            cached = self._try_load_cache(cache_path, records)
            if cached is not None:
                return cached

        model = self._load_model()
        output: dict[str, list[Detection]] = {}
        total = len(records)
        t0 = time.monotonic()
        last_log = t0
        for start in range(0, total, self.batch_size):
            chunk = records[start : start + self.batch_size]
            sources = [str(record.image_path) for record in chunk]
            results = model.predict(
                source=sources,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                classes=[self.target_class],
                device=self.device,
                verbose=False,
                workers=0,
                cache=False,
                stream=False,
                save=False,
            )
            for record, result in zip(chunk, results):
                items: list[Detection] = []
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy_arr = boxes.xyxy.detach().cpu().numpy()
                    conf_arr = boxes.conf.detach().cpu().numpy()
                    cls_arr = boxes.cls.detach().cpu().numpy()
                    for box_vals, conf_val, cls_val in zip(xyxy_arr, conf_arr, cls_arr):
                        if int(round(float(cls_val))) != self.target_class:
                            continue
                        x1, y1, x2, y2 = (float(v) for v in box_vals)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        items.append(
                            Detection(
                                image_id=record.image_id,
                                box=(x1, y1, x2, y2),
                                conf=float(conf_val),
                            )
                        )
                items.sort(key=lambda d: -d.conf)
                output[record.image_id] = items
            done = min(total, start + len(chunk))
            now = time.monotonic()
            if total >= 50 and (done == total or now - last_log >= 30.0):
                elapsed = max(1e-6, now - t0)
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0.0
                print(
                    f"detector[{self.weights_path.name}@{self.imgsz}]: "
                    f"{done}/{total} imgs ({rate:.2f} img/s, elapsed={elapsed:.1f}s, eta={eta:.1f}s)",
                    flush=True,
                )
                last_log = now

        if cache_path is not None:
            self._save_cache(cache_path, records, output)
        return output
