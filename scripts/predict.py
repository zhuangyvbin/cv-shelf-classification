"""预测入口：本地 folder/csv 演示；MySQL 与推理逻辑见 utils.predict_*。"""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.config_loader import build_resource_url, get_project_root, load_config
from utils.predict_inference import (
    PredictResult,
    Predictor,
    get_predictor,
    predict_image,
)
from utils.predict_mapping import (
    AUDIT_IMG_PLACES,
    PLACE_RULES,
    row_to_prediction_tuple,
)
from utils.predict_mysql import (
    predict_from_mysql,
    predict_yesterday_from_mysql,
    run_mysql_batch,
)

cfg = load_config()


def _project_path(*parts: str) -> str:
    return os.path.join(get_project_root(), *parts)


def predict_folder(image_folder: str) -> None:
    records = []
    failed = []
    for fname in os.listdir(image_folder):
        img_path = os.path.join(image_folder, fname)
        result = predict_image(img_path)
        if not result.ok:
            failed.append((fname, result.error))
            continue
        labels = result.labels or {}
        row = {"image_name": fname}
        row.update({cls: labels[cls][0] for cls in cfg["classes"]})
        row.update({f"{cls}_prob": labels[cls][1] for cls in cfg["classes"]})
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(_project_path(cfg["data"]["predictions_csv"]), index=False)
    print("预测结果已保存到" + cfg["data"]["predictions_csv"])
    if failed:
        print(
            f"跳过 {len(failed)} 张失败图像: {failed[:5]}"
            f"{'...' if len(failed) > 5 else ''}"
        )


def predict_from_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    records = []
    failed = []
    for _, row in df.iterrows():
        image_name = row["image_name"]
        img_path = _project_path(cfg["data"]["image_dir"], image_name)
        result = predict_image(img_path, quiet=True)
        if not result.ok:
            failed.append((image_name, result.error))
            continue
        labels = result.labels or {}
        records.append(
            {
                "image_name": image_name,
                **{cls: labels[cls][0] for cls in cfg["classes"]},
                **{f"{cls}_prob": labels[cls][1] for cls in cfg["classes"]},
            }
        )
    out_df = pd.DataFrame(records)
    out_df.to_csv(_project_path(cfg["data"]["test_predictions_csv"]), index=False)
    print("预测结果已保存到" + cfg["data"]["test_predictions_csv"])
    if failed:
        print(
            f"跳过 {len(failed)} 张失败图像: {failed[:5]}"
            f"{'...' if len(failed) > 5 else ''}"
        )


__all__ = [
    "PredictResult",
    "Predictor",
    "get_predictor",
    "predict_image",
    "predict_folder",
    "predict_from_csv",
    "predict_from_mysql",
    "predict_yesterday_from_mysql",
    "run_mysql_batch",
    "row_to_prediction_tuple",
    "PLACE_RULES",
    "AUDIT_IMG_PLACES",
    "cfg",
]


if __name__ == "__main__":
    # 1. 单张图像预测示例
    print(
        predict_image(
            build_resource_url(
                "resource/image/visit/fromrpphoto/2025/07/02/"
                "visitPhoto_10631389.jpg"
            )
        )
    )

    # 2. 文件夹批量预测
    # predict_folder("../data/images_test")

    # 3. CSV 文件批量预测
    # predict_from_csv("../" + cfg["data"]["split_csv_dir"] + "/test.csv")

    # 4. MySQL 自动预测（历史 create_time 补数窗口）
    # predict_from_mysql()

    # 5. 昨日 upload_time 批处理（适合 cron）
    # predict_yesterday_from_mysql()
    # predict_yesterday_from_mysql(target_date=date(2026, 5, 26))  # 补跑指定日
