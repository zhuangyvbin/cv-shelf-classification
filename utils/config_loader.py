"""统一加载 config.yaml / config.secrets.yaml，解析环境与 MySQL 连接参数。"""

from __future__ import annotations

import copy
import logging
import os
from functools import lru_cache
from typing import Any

import pymysql
import yaml

logger = logging.getLogger(__name__)

_VALID_ENVS = frozenset({"development", "testing", "production"})
_PLACEHOLDER_PASSWORDS = frozenset({"", "${RACK_MYSQL_PASSWORD}"})
_CLIENT_FLAG_MAP = {
    "MULTI_STATEMENTS": pymysql.constants.CLIENT.MULTI_STATEMENTS,
}


def get_project_root() -> str:
    """基于本模块位置返回项目根目录（含 config.yaml 的目录）。"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """加载并缓存 config.yaml，若存在则深合并 config.secrets.yaml 的 environments。"""
    root = get_project_root()
    config_path = os.path.join(root, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    cfg = _load_yaml(config_path)

    secrets_path = os.path.join(root, "config.secrets.yaml")
    if os.path.isfile(secrets_path):
        secrets = _load_yaml(secrets_path)
        secret_envs = secrets.get("environments")
        if isinstance(secret_envs, dict):
            base_envs = cfg.setdefault("environments", {})
            if isinstance(base_envs, dict):
                cfg["environments"] = _deep_merge(base_envs, secret_envs)

    return cfg


def get_active_env() -> str:
    """解析当前环境名。优先级: RACK_ENV > config.active_env > development。"""
    cfg = load_config()
    env_from_var = os.environ.get("RACK_ENV", "").strip()
    env_from_config = (cfg.get("active_env") or "development").strip()

    if env_from_var:
        active = env_from_var
    else:
        active = env_from_config or "development"

    if active not in _VALID_ENVS:
        raise ValueError(
            f"无效环境 '{active}'，允许值: {', '.join(sorted(_VALID_ENVS))}"
        )

    environments = cfg.get("environments") or {}
    if active not in environments:
        raise ValueError(f"config.yaml 中未定义环境 '{active}' 的配置")

    if env_from_var and env_from_var != env_from_config:
        logger.info(
            "RACK_ENV=%s 已覆盖 config.yaml 中的 active_env=%s",
            env_from_var,
            env_from_config,
        )

    return active


def get_env_config() -> dict[str, Any]:
    """返回当前激活环境的配置（resource_domain、mysql、logging 等）。"""
    cfg = load_config()
    active = get_active_env()
    env_cfg = copy.deepcopy((cfg.get("environments") or {}).get(active) or {})
    if not env_cfg:
        raise ValueError(f"环境 '{active}' 的配置为空")
    return env_cfg


def _resolve_mysql_password(env_cfg: dict[str, Any], active_env: str) -> str:
    env_password = os.environ.get("RACK_MYSQL_PASSWORD")
    if env_password:
        return env_password

    cfg_password = (env_cfg.get("mysql") or {}).get("password") or ""
    if cfg_password in _PLACEHOLDER_PASSWORDS:
        cfg_password = ""

    if not cfg_password:
        raise ValueError(
            f"未配置 {active_env} 的 MySQL 密码，请设置 RACK_MYSQL_PASSWORD "
            "或 config.secrets.yaml"
        )
    return cfg_password


def _resolve_client_flag(mysql_cfg: dict[str, Any]) -> int | None:
    raw = mysql_cfg.get("client_flag")
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        key = raw.strip().upper()
        if key in _CLIENT_FLAG_MAP:
            return _CLIENT_FLAG_MAP[key]
        raise ValueError(f"不支持的 mysql.client_flag: {raw}")
    raise ValueError(f"mysql.client_flag 类型无效: {type(raw).__name__}")


def get_mysql_connect_kwargs() -> dict[str, Any]:
    """组装 pymysql.connect(**kwargs) 参数字典。"""
    active = get_active_env()
    env_cfg = get_env_config()
    mysql_cfg = env_cfg.get("mysql") or {}

    kwargs: dict[str, Any] = {
        "host": mysql_cfg["host"],
        "port": int(mysql_cfg.get("port", 3306)),
        "user": mysql_cfg["user"],
        "password": _resolve_mysql_password(env_cfg, active),
        "database": mysql_cfg["database"],
        "autocommit": True,
    }

    for timeout_key in ("connect_timeout", "read_timeout", "write_timeout"):
        if timeout_key in mysql_cfg:
            kwargs[timeout_key] = int(mysql_cfg[timeout_key])

    client_flag = _resolve_client_flag(mysql_cfg)
    if client_flag is not None:
        kwargs["client_flag"] = client_flag

    return kwargs


def build_resource_url(picture_path: str) -> str:
    """将相对 picture_path 拼成完整资源 URL。"""
    env_cfg = get_env_config()
    domain = (env_cfg.get("resource_domain") or "").rstrip("/")
    if not domain:
        raise ValueError("当前环境未配置 resource_domain")
    path = (picture_path or "").lstrip("/")
    return f"{domain}/{path}"
