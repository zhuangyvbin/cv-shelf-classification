"""本地验证环境切换与密码缺失报错（不连接 MySQL）。"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import config_loader
from utils.config_loader import (
    build_resource_url,
    get_active_env,
    get_env_config,
    get_mysql_connect_kwargs,
)

_EXPECTED = {
    "development": {
        "resource_domain": "https://resource.example.cn",
        "mysql_host": "localhost",
    },
    "testing": {
        "resource_domain": "https://resource-test.example.cn",
        "mysql_host": "test-db-server.example.com",
    },
    "production": {
        "resource_domain": "https://resource.example.cn",
        "mysql_host": "production-db-server.example.com",
    },
}

_PASSWORD_ERROR = "未配置 {} 的 MySQL 密码，请设置 RACK_MYSQL_PASSWORD 或 config.secrets.yaml"


def _reset_env(*, rack_env: str | None = None) -> None:
    config_loader.load_config.cache_clear()
    os.environ.pop("RACK_MYSQL_PASSWORD", None)
    if rack_env is None:
        os.environ.pop("RACK_ENV", None)
    else:
        os.environ["RACK_ENV"] = rack_env


def verify_env_switch() -> None:
    print("== 环境切换 ==")
    cases = [
        ("default (config active_env)", None),
        ("RACK_ENV=testing", "testing"),
        ("RACK_ENV=production", "production"),
    ]
    for label, rack_env in cases:
        _reset_env(rack_env=rack_env)
        active = get_active_env()
        env_cfg = get_env_config()
        expected = _EXPECTED[active]
        url = build_resource_url("/resource/image/test.jpg")
        assert env_cfg["resource_domain"] == expected["resource_domain"], (
            f"{label}: resource_domain 不匹配"
        )
        assert env_cfg["mysql"]["host"] == expected["mysql_host"], (
            f"{label}: mysql.host 不匹配"
        )
        assert url == f"{expected['resource_domain']}/resource/image/test.jpg", (
            f"{label}: build_resource_url 不正确"
        )
        print(
            f"  OK [{label}] active={active}, "
            f"host={env_cfg['mysql']['host']}, url={url}"
        )


def verify_password_missing() -> None:
    print("== 密码缺失报错 ==")
    for env in ("development", "testing", "production"):
        _reset_env(rack_env=env)
        try:
            get_mysql_connect_kwargs()
            raise AssertionError(f"{env}: 应抛出 ValueError，但未抛出")
        except ValueError as exc:
            expected = _PASSWORD_ERROR.format(env)
            if str(exc) != expected:
                raise AssertionError(
                    f"{env}: 报错信息不符\n  期望: {expected}\n  实际: {exc}"
                ) from exc
            print(f"  OK [{env}] ValueError: {exc}")


def main() -> None:
    verify_env_switch()
    verify_password_missing()
    print("\n全部验证通过。")


if __name__ == "__main__":
    main()
