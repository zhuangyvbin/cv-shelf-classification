"""MySQL 批量预测：查询、重试、插入与失败日志。"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import pymysql

from utils.config_loader import (
    build_resource_url,
    get_env_config,
    get_mysql_connect_kwargs,
    get_project_root,
    load_config,
)
from utils.predict_inference import get_predict_cfg, predict_image_with_retry
from utils.predict_mapping import AUDIT_IMG_PLACES, row_to_prediction_tuple

_ROOT = get_project_root()

_INSERT_PREDICTIONS_SQL = """
    INSERT INTO user_visit_img_predictions (
        img_id, visit_id, predict_prod, predict_img,
        predict_status, failure_reason, model_version, predict_time
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def _project_path(*parts: str) -> str:
    return os.path.join(_ROOT, *parts)


def setup_predict_logging() -> logging.Logger:
    env_cfg = get_env_config()
    log_level_name = (env_cfg.get("logging") or {}).get("level", "INFO")
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    return logging.getLogger(__name__)


def beijing_tz(cfg: dict | None = None) -> timezone:
    hours = get_predict_cfg(cfg)["beijing_tz_offset_hours"]
    return timezone(timedelta(hours=hours))


def yesterday_upload_time_range(
    target_date: date | None = None,
    cfg: dict | None = None,
) -> tuple[int, int]:
    """返回 [start_ts, end_ts)，左闭右开，对应 target_date 当日 00:00~24:00（默认=昨天）。"""
    if cfg is None:
        cfg = load_config()
    tz = beijing_tz(cfg)
    now = datetime.now(tz)
    if target_date is None:
        target_date = now.date() - timedelta(days=1)
    start_dt = datetime(
        target_date.year, target_date.month, target_date.day, tzinfo=tz
    )
    end_dt = start_dt + timedelta(days=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def predict_failure_log_path(cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = load_config()
    failure_log = get_predict_cfg(cfg)["failure_log"]
    if os.path.dirname(failure_log):
        return (
            failure_log
            if os.path.isabs(failure_log)
            else _project_path(failure_log)
        )
    output_dir = cfg.get("data", {}).get("output_dir", "outputs")
    return _project_path(output_dir, failure_log)


def _get_predict_failure_file_logger(cfg: dict | None = None) -> logging.Logger:
    log_path = predict_failure_log_path(cfg)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    failure_logger = logging.getLogger("utils.predict_mysql.predict_failures")
    failure_logger.setLevel(logging.INFO)
    failure_logger.propagate = False
    if not failure_logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        failure_logger.addHandler(handler)
    return failure_logger


def _iter_paginated_rows(
    cursor,
    base_sql: str,
    base_params: tuple | None,
    page_size: int,
    id_col: str,
):
    """按 id 游标分页拉取行，每页 yield (last_id, rows)。"""
    last_id = 0
    params = base_params or ()
    while True:
        page_sql = (
            f"{base_sql.strip()} AND {id_col} > %s ORDER BY {id_col} LIMIT %s"
        )
        cursor.execute(page_sql, params + (last_id, page_size))
        rows = cursor.fetchall()
        if not rows:
            break
        yield last_id, rows
        last_id = rows[-1][0]


def _flush_prediction_records(conn, records, logger, insert_count):
    conn.ping(reconnect=True)
    with conn.cursor() as insert_cursor:
        insert_cursor.executemany(_INSERT_PREDICTIONS_SQL, records)
        conn.commit()
        insert_count += len(records)
        logger.info(f"已插入 {insert_count} 条记录")
    return insert_count


def process_mysql_rows(
    conn,
    rows,
    logger: logging.Logger,
    batch_size: int | None = None,
    cfg: dict | None = None,
) -> int:
    if cfg is None:
        cfg = load_config()
    if batch_size is None:
        batch_size = get_predict_cfg(cfg)["mysql_batch_size"]
    predict_time = int(time.time())
    model_version = f"{cfg['model']['name']}_{cfg['model']['version']}"
    records = []
    insert_count = 0
    failed_count = 0
    failure_logger = _get_predict_failure_file_logger(cfg)
    max_attempts = get_predict_cfg(cfg)["max_attempts"]

    for row in rows:
        id_, _visit_id, picture_path, _img_place = row
        image_url = build_resource_url(picture_path)
        pre_result = predict_image_with_retry(image_url, cfg=cfg)

        if not pre_result.ok:
            failed_count += 1
            msg = (
                "图像预测失败，已重试 %d 次仍失败 - ID: %s, URL: %s, 错误: %s"
                % (max_attempts, id_, image_url, pre_result.error)
            )
            failure_logger.warning(msg)
            logger.warning(msg)
            continue

        result = pre_result.labels or {}
        records.append(
            row_to_prediction_tuple(row, result, model_version, predict_time)
        )

        if len(records) >= batch_size:
            insert_count = _flush_prediction_records(
                conn, records, logger, insert_count
            )
            records = []

    if records:
        insert_count = _flush_prediction_records(
            conn, records, logger, insert_count
        )

    logger.info(
        "批次处理完成: total=%d, inserted=%d, failed=%d",
        len(rows),
        insert_count,
        failed_count,
    )
    return insert_count


def run_mysql_batch(
    *,
    select_sql: str,
    params: tuple | None = None,
    log_context: str = "",
    batch_size: int | None = None,
    id_col: str = "id",
    cfg: dict | None = None,
) -> int:
    if cfg is None:
        cfg = load_config()
    if batch_size is None:
        batch_size = get_predict_cfg(cfg)["mysql_batch_size"]
    page_size = get_predict_cfg(cfg)["mysql_fetch_page_size"]
    logger = setup_predict_logging()
    if log_context:
        logger.info(log_context)

    conn = pymysql.connect(**get_mysql_connect_kwargs())
    try:
        total_insert_count = 0
        if page_size > 0:
            with conn.cursor() as cursor:
                for last_id, rows in _iter_paginated_rows(
                    cursor, select_sql, params, page_size, id_col
                ):
                    logger.info(
                        "分页查询: last_id=%d, page_size=%d, fetched=%d",
                        last_id,
                        page_size,
                        len(rows),
                    )
                    total_insert_count += process_mysql_rows(
                        conn, rows, logger, batch_size=batch_size, cfg=cfg
                    )
        else:
            fetch_sql = f"{select_sql.strip()} ORDER BY {id_col}"
            with conn.cursor() as cursor:
                cursor.execute(fetch_sql, params)
                rows = cursor.fetchall()
            total_insert_count = process_mysql_rows(
                conn, rows, logger, batch_size=batch_size, cfg=cfg
            )
        logger.info("数据库预测结果已全部更新完成")
        return total_insert_count
    except Exception:
        logger.exception("MySQL 批量预测失败")
        raise
    finally:
        conn.close()


def predict_from_mysql(cfg: dict | None = None) -> int:
    if cfg is None:
        cfg = load_config()
    backfill = get_predict_cfg(cfg)["backfill_create_time_range"]
    if not backfill or len(backfill) != 2:
        raise ValueError(
            "predict.backfill_create_time_range 须为 [start_ts, end_ts]，"
            "用于 predict_from_mysql 补数窗口"
        )
    start_ts, end_ts = int(backfill[0]), int(backfill[1])
    places = ", ".join(f"'{p}'" for p in AUDIT_IMG_PLACES)
    return run_mysql_batch(
        select_sql=f"""
            SELECT id, visit_id, picture_path, img_place
            FROM user_visit_imgs
            WHERE img_place IN ({places})
            AND picture_path IS NOT NULL
            AND create_time >= %s
            AND create_time < %s
        """,
        params=(start_ts, end_ts),
        log_context=f"create_time 补数窗口: [{start_ts}, {end_ts})",
        cfg=cfg,
    )


def predict_yesterday_from_mysql(
    target_date: date | None = None,
    cfg: dict | None = None,
) -> int:
    if cfg is None:
        cfg = load_config()
    start_ts, end_ts = yesterday_upload_time_range(target_date, cfg=cfg)
    tz = beijing_tz(cfg)
    if target_date is None:
        actual_date = datetime.now(tz).date() - timedelta(days=1)
    else:
        actual_date = target_date

    places = ", ".join(f"'{p}'" for p in AUDIT_IMG_PLACES)
    insert_count = run_mysql_batch(
        select_sql=f"""
            SELECT u.id, u.visit_id, u.picture_path, u.img_place
            FROM user_visit_imgs u
            WHERE u.img_place IN ({places})
              AND u.picture_path IS NOT NULL
              AND u.upload_time >= %s
              AND u.upload_time < %s
              AND NOT EXISTS (
                  SELECT 1 FROM user_visit_img_predictions p
                  WHERE p.img_id = u.id
              )
        """,
        params=(start_ts, end_ts),
        log_context=(
            f"upload_time 区间: date={actual_date}, start_ts={start_ts}, end_ts={end_ts}"
        ),
        id_col="u.id",
        cfg=cfg,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"昨日批量预测完成，共插入 {insert_count} 条记录")
    return insert_count
