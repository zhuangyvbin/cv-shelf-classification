"""验证 MySQL 连接超时重试与幂等性（mock 测试 + 可选 live 集成）。"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from datetime import date
from io import StringIO
from unittest.mock import MagicMock, patch

import pymysql

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import config_loader
from utils.config_loader import get_project_root, load_config
from utils.predict_inference import PredictResult
from utils.predict_mysql import (
    _is_mysql_retryable,
    _iter_paginated_rows,
    _with_mysql_retry,
    predict_yesterday_from_mysql,
)

_TEST_CFG = {
    "predict": {
        "max_attempts": 3,
        "retry_backoff_seconds": [0.01, 0.02],
        "mysql_fetch_page_size": 50,
        "mysql_batch_size": 500,
        "beijing_tz_offset_hours": 8,
    },
    "model": {"name": "rack_model", "version": "v4"},
}


def _capture_logger() -> tuple[logging.Logger, StringIO]:
    stream = StringIO()
    logger = logging.getLogger("verify_mysql_retry")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream


def verify_retryable_errno() -> None:
    print("== 可重试错误码判定 ==")
    for errno in (2003, 2006, 2013, 2055):
        exc = pymysql.err.OperationalError(errno, "test")
        assert _is_mysql_retryable(exc), f"errno {errno} 应可重试"
        print(f"  OK OperationalError({errno}) 可重试")
    assert _is_mysql_retryable(pymysql.err.InterfaceError(0, "test"))
    print("  OK InterfaceError 可重试")
    assert not _is_mysql_retryable(pymysql.err.OperationalError(1064, "syntax"))
    print("  OK OperationalError(1064) 不可重试")
    assert not _is_mysql_retryable(ValueError("bad value"))
    print("  OK ValueError 不可重试")


def verify_with_mysql_retry_success_after_2013() -> None:
    print("== _with_mysql_retry：2013 后重试成功 ==")
    logger, stream = _capture_logger()
    holder = MagicMock()
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise pymysql.err.OperationalError(
                2013, "Lost connection to MySQL server during query (timed out)"
            )
        return "ok"

    with patch("utils.predict_mysql.time.sleep"):
        result = _with_mysql_retry(
            operation,
            holder=holder,
            cfg=_TEST_CFG,
            logger=logger,
            op_name="分页查询 last_id=0",
        )
    assert result == "ok"
    assert len(calls) == 2
    holder.reconnect.assert_called_once()
    log_text = stream.getvalue()
    assert "MySQL 分页查询 last_id=0 重试" in log_text, log_text
    assert "2013" in log_text
    print("  OK 第 1 次 2013 → 重连 → 第 2 次成功，日志含重试信息")


def verify_non_retryable_immediate_fail() -> None:
    print("== 非可重试错误立即失败 ==")
    logger, stream = _capture_logger()
    holder = MagicMock()
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("SQL 语法错误")

    try:
        _with_mysql_retry(
            operation,
            holder=holder,
            cfg=_TEST_CFG,
            logger=logger,
            op_name="分页查询 last_id=0",
        )
        raise AssertionError("应抛出 ValueError")
    except ValueError:
        pass
    assert attempts == 1
    holder.reconnect.assert_not_called()
    assert "重试" not in stream.getvalue()
    print("  OK ValueError 不重试、不重连")


def verify_exhausted_retries() -> None:
    print("== 耗尽重试次数后抛出 ==")
    logger, _ = _capture_logger()
    holder = MagicMock()
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise pymysql.err.OperationalError(2013, "timed out")

    try:
        with patch("utils.predict_mysql.time.sleep"):
            _with_mysql_retry(
                operation,
                holder=holder,
                cfg=_TEST_CFG,
                logger=logger,
                op_name="批量插入",
            )
        raise AssertionError("应抛出 OperationalError")
    except pymysql.err.OperationalError as exc:
        assert exc.args[0] == 2013
    assert attempts == 3
    assert holder.reconnect.call_count == 2
    print("  OK 3 次均 2013，重连 2 次后向上抛出")


def verify_paginated_select_retry() -> None:
    print("== 分页 SELECT 重试（mock cursor） ==")
    logger, stream = _capture_logger()
    holder = MagicMock()
    execute_calls = 0
    page_rows = [(1, 10, "path/a.jpg", "rack")]

    class FakeCursor:
        def execute(self, sql, params):
            nonlocal execute_calls
            execute_calls += 1
            if execute_calls == 1:
                raise pymysql.err.OperationalError(2013, "timed out")
            # last_id > 0 时模拟无更多数据
            if params and params[-2] > 0:
                return

        def fetchall(self):
            if execute_calls == 1:
                return []
            if execute_calls == 2:
                return page_rows
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    holder.conn.cursor.return_value = FakeCursor()

    pages = list(
        _iter_paginated_rows(
            holder,
            "SELECT id FROM t WHERE 1=1",
            None,
            page_size=50,
            id_col="id",
            cfg=_TEST_CFG,
            logger=logger,
        )
    )
    assert len(pages) == 1
    assert pages[0][1] == page_rows
    assert execute_calls == 2
    holder.reconnect.assert_called_once()
    assert "MySQL 分页查询 last_id=0 重试" in stream.getvalue()
    print("  OK 分页 execute 第 1 次 2013 → 重试后拉取成功")


def verify_insert_retry() -> None:
    print("== 批量 INSERT 重试（mock conn） ==")
    from utils.predict_mysql import _flush_prediction_records

    logger, stream = _capture_logger()
    holder = MagicMock()
    ping_calls = 0
    exec_calls = 0
    records = [(1, 2, "{}", "RP_RACK", 1, "", "m_v1", 1234567890)]

    class FakeInsertCursor:
        def executemany(self, sql, rows):
            nonlocal exec_calls
            exec_calls += 1
            if exec_calls == 1:
                raise pymysql.err.OperationalError(2013, "timed out")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_ping(reconnect=True):
        nonlocal ping_calls
        ping_calls += 1

    holder.conn.ping = fake_ping
    holder.conn.cursor.return_value = FakeInsertCursor()
    holder.conn.commit = MagicMock()

    with patch("utils.predict_mysql.time.sleep"):
        count = _flush_prediction_records(
            holder, records, logger, insert_count=0, cfg=_TEST_CFG
        )
    assert count == 1
    assert exec_calls == 2
    holder.reconnect.assert_called_once()
    assert "MySQL 批量插入 重试" in stream.getvalue()
    print("  OK INSERT 第 1 次 2013 → 重连 → 第 2 次成功")


def verify_idempotency_sql() -> None:
    """确认 predict_yesterday_from_mysql 使用 NOT EXISTS，二次执行不会重复插入。"""
    print("== 幂等性 SQL（NOT EXISTS） ==")
    import inspect

    source = inspect.getsource(predict_yesterday_from_mysql)
    assert "NOT EXISTS" in source
    assert "user_visit_img_predictions" in source
    print("  OK predict_yesterday_from_mysql 含 NOT EXISTS 子查询")


def _can_run_live() -> bool:
    config_path = os.path.join(get_project_root(), "config.yaml")
    if not os.path.isfile(config_path):
        return False
    if not os.environ.get("RACK_MYSQL_PASSWORD"):
        secrets_path = os.path.join(get_project_root(), "config.secrets.yaml")
        if not os.path.isfile(secrets_path):
            return False
    return True


def verify_live(target_date: date, short_timeout: int = 1) -> None:
    """用小 read_timeout + page_size 跑真实 MySQL，验证重试日志与幂等性。"""
    print("== Live 集成：read_timeout=%d, page_size=50 ==" % short_timeout)
    config_loader.load_config.cache_clear()
    cfg = copy.deepcopy(load_config())
    cfg.setdefault("predict", {})
    cfg["predict"]["mysql_fetch_page_size"] = 50
    cfg["predict"]["max_attempts"] = 3
    cfg["predict"]["retry_backoff_seconds"] = [0.5, 1.0]

    original_get_kwargs = config_loader.get_mysql_connect_kwargs

    def patched_get_kwargs():
        kwargs = original_get_kwargs()
        kwargs["read_timeout"] = short_timeout
        return kwargs

    log_stream = StringIO()
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    mock_predict = MagicMock(
        return_value=PredictResult(
            ok=True,
            labels={"RC": (1, 0.9), "RP": (0, 0.1), "other": (0, 0.1)},
        )
    )

    def run_once() -> int:
        with (
            patch(
                "utils.predict_mysql.get_mysql_connect_kwargs",
                patched_get_kwargs,
            ),
            patch(
                "utils.predict_mysql.predict_image_with_retry",
                mock_predict,
            ),
        ):
            return predict_yesterday_from_mysql(
                target_date=target_date, cfg=cfg
            )

    try:
        count1 = run_once()
        log_text = log_stream.getvalue()
        if "重试" in log_text:
            print("  OK 观察到 MySQL 重试日志")
            for line in log_text.splitlines():
                if "重试" in line:
                    print(f"    {line.strip()}")
        else:
            print(
                "  注意: 未触发重试（数据量小或查询够快），"
                f"首次 insert_count={count1}"
            )

        count2 = run_once()
        assert count2 == 0, f"二次执行应 insert_count==0，实际 {count2}"
        print(f"  OK 幂等性: 首次={count1}, 二次={count2}")
    finally:
        root_logger.removeHandler(handler)
        config_loader.load_config.cache_clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 MySQL 超时重试机制")
    parser.add_argument(
        "--live",
        action="store_true",
        help="连接真实 MySQL（需 config.yaml + 密码）",
    )
    parser.add_argument(
        "-d",
        "--date",
        metavar="YYYY-MM-DD",
        help="Live 模式目标日期，默认昨天",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=1,
        help="Live 模式临时 read_timeout（秒），默认 1",
    )
    args = parser.parse_args()

    verify_retryable_errno()
    verify_with_mysql_retry_success_after_2013()
    verify_non_retryable_immediate_fail()
    verify_exhausted_retries()
    verify_paginated_select_retry()
    verify_insert_retry()
    verify_idempotency_sql()

    if args.live:
        if not _can_run_live():
            print(
                "\n跳过 Live 集成: 需要 config.yaml 及 "
                "RACK_MYSQL_PASSWORD 或 config.secrets.yaml",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.date:
            target = date.fromisoformat(args.date)
        else:
            from datetime import datetime, timedelta

            from utils.predict_mysql import beijing_tz

            target = datetime.now(beijing_tz()).date() - timedelta(days=1)
        verify_live(target, short_timeout=args.read_timeout)

    print("\n全部验证通过。")


if __name__ == "__main__":
    main()
