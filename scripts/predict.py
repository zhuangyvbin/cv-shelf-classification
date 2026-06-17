"""预测入口：本地 folder/csv 演示；MySQL 与推理逻辑见 utils.predict_*。"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta

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
    beijing_tz,
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
    # print(
    #     predict_image(
    #         build_resource_url(
    #             "resource/image/visit/fromrpphoto/2025/07/02/"
    #             "visitPhoto_10631389.jpg"
    #         )
    #     )
    # )

    # 2. 文件夹批量预测
    # predict_folder("../data/images_test")

    # 3. CSV 文件批量预测
    # predict_from_csv("../" + cfg["data"]["split_csv_dir"] + "/test.csv")

    # 4. MySQL 自动预测（历史 create_time 补数窗口）
    # predict_from_mysql()

    # 5. 昨日 upload_time 批处理（适合 cron）
    parser = argparse.ArgumentParser(
        description="昨日 upload_time 批处理（rack_predict），写入 user_visit_img_predictions",
    )
    parser.add_argument(
        "-d",
        "--date",
        metavar="YYYY-MM-DD",
        help="目标日期，默认为昨天（北京时区）",
    )
    args = parser.parse_args()

    target_date: date | None = None
    if args.date is not None:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(
                f"日期格式错误: {args.date!r}，请使用 YYYY-MM-DD 格式，例如 2026-04-01",
                file=sys.stderr,
            )
            sys.exit(2)

    if target_date is None:
        tz = beijing_tz(cfg)
        display_date = datetime.now(tz).date() - timedelta(days=1)
    else:
        display_date = target_date
    date_label = f"目标日期: {display_date.isoformat()}"

    try:
        start_time = datetime.now()
        print(
            f"rack_predict 批处理脚本开始... "
            f"({date_label}, 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')})"
        )
        insert_count = predict_yesterday_from_mysql(target_date=target_date)
        end_time = datetime.now()
        duration = end_time - start_time
        print(
            f"rack_predict 批处理脚本结束... "
            f"({date_label}, 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"耗时: {duration.total_seconds():.2f}秒, 插入记录数: {insert_count})"
        )
    except Exception as exc:
        print(f"rack_predict 批处理失败 ({date_label}): {exc}", file=sys.stderr)
        sys.exit(1)
