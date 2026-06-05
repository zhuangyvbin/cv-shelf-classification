"""模型懒加载、图像推理与 PredictResult。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import transforms

from utils.config_loader import get_project_root, load_config
from utils.model import MultiLabelClassifier

_ROOT = get_project_root()


def get_predict_cfg(cfg: dict | None = None) -> dict:
    """predict 段优先，缺失项回退到模块内默认值或顶层 predict_threshold。"""
    if cfg is None:
        cfg = load_config()
    p = cfg.get("predict") or {}
    backoff = p.get("retry_backoff_seconds", [0.5, 1.0])
    return {
        "threshold": p.get("threshold", cfg.get("predict_threshold", 0.6)),
        "mysql_batch_size": p.get("mysql_batch_size", 500),
        "mysql_fetch_page_size": p.get("mysql_fetch_page_size", 1000),
        "max_attempts": p.get("max_attempts", 3),
        "retry_backoff_seconds": tuple(backoff),
        "failure_log": p.get("failure_log", "predict_mysql_failures.log"),
        "beijing_tz_offset_hours": p.get("beijing_tz_offset_hours", 8),
        "backfill_create_time_range": p.get("backfill_create_time_range"),
    }


def _project_path(*parts: str) -> str:
    return os.path.join(_ROOT, *parts)


@dataclass(frozen=True)
class PredictResult:
    ok: bool
    labels: dict[str, tuple[int, float]] | None = None
    error: str | None = None
    retryable: bool = False

    @classmethod
    def from_legacy(cls, d: dict) -> PredictResult:
        if d.get("status") == 1:
            return cls(ok=True, labels=d.get("reason"))
        reason = d.get("reason", "unknown")
        return cls(ok=False, error=str(reason), retryable=False)


class Predictor:
    """单例懒加载推理器；测试可设 Predictor._instance = mock。"""

    _instance: Predictor | None = None

    def __init__(
        self,
        model: MultiLabelClassifier,
        transform: transforms.Compose,
        device: str,
        cfg: dict,
    ) -> None:
        self.model = model
        self.transform = transform
        self.device = device
        self.cfg = cfg

    @classmethod
    def get(cls) -> Predictor:
        if cls._instance is None:
            cls._instance = cls._load_from_config(load_config())
        return cls._instance

    @classmethod
    def _load_from_config(cls, cfg: dict) -> Predictor:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MultiLabelClassifier(
            backbone=cfg["model"]["backbone"],
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
        ).to(device)
        model.load_state_dict(
            torch.load(
                _project_path(cfg["model"]["save_path"]),
                map_location=device,
                weights_only=True,
            )
        )
        model.eval()
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (cfg["train"]["image_size"], cfg["train"]["image_size"])
                ),
                transforms.ToTensor(),
            ]
        )
        return cls(model=model, transform=transform, device=device, cfg=cfg)

    def predict(self, image_path: str, *, quiet: bool = False) -> PredictResult:
        try:
            if image_path.startswith("http"):
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

            image_tensor = (
                self.transform(image).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                logits = self.model(image_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            threshold = get_predict_cfg(self.cfg)["threshold"]
            preds = (probs > threshold).astype(int)
            labels = {
                cls: (int(pred), float(prob))
                for cls, pred, prob in zip(self.cfg["classes"], preds, probs)
            }
            return PredictResult(ok=True, labels=labels)

        except requests.exceptions.RequestException as e:
            if not quiet:
                print(f"预测失败: {e}")
            return PredictResult(ok=False, error=str(e), retryable=True)
        except FileNotFoundError as e:
            if not quiet:
                print(f"预测失败: {e}")
            return PredictResult(ok=False, error=str(e), retryable=False)
        except Exception as e:
            if not quiet:
                print(f"预测失败: {e}")
            return PredictResult(ok=False, error=str(e), retryable=False)


def get_predictor() -> Predictor:
    return Predictor.get()


def predict_image(image_path: str, *, quiet: bool = False) -> PredictResult:
    return Predictor.get().predict(image_path, quiet=quiet)


def predict_image_with_retry(
    image_url: str,
    max_attempts: int | None = None,
    *,
    quiet: bool = True,
    cfg: dict | None = None,
) -> PredictResult:
    predict_cfg = get_predict_cfg(cfg)
    if max_attempts is None:
        max_attempts = predict_cfg["max_attempts"]
    backoff_seconds = predict_cfg["retry_backoff_seconds"]
    last_result = PredictResult(ok=False, error="unknown")
    for attempt in range(max_attempts):
        last_result = predict_image(image_url, quiet=quiet)
        if last_result.ok:
            return last_result
        if not last_result.retryable:
            break
        if attempt < max_attempts - 1:
            backoff = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
            time.sleep(backoff)
    return last_result
