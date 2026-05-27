# rack_multilabel

货架多标签图像分类项目。训练、评估与预测脚本位于 `scripts/`，统一配置由 `utils/config_loader.py` 加载。

## 环境配置

项目支持 **development**、**testing**、**production** 三套环境。非敏感配置在 [`config.yaml`](config.yaml)，MySQL 密码等敏感信息通过 [`config.secrets.yaml`](config.secrets.example.yaml) 或环境变量提供。

### 首次配置

1. 复制密码模板并填写真实密码：

   ```bash
   cp config.secrets.example.yaml config.secrets.yaml
   ```

2. 编辑 `config.secrets.yaml`，为各环境填写 `mysql.password`（该文件已被 `.gitignore` 忽略，不会提交）。

3. 或在部署/CI 中仅设置环境变量 `RACK_MYSQL_PASSWORD`（见下文），无需创建 `config.secrets.yaml`。

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `RACK_ENV` | 否 | 激活的环境名：`development` \| `testing` \| `production`。未设置时使用 `config.yaml` 中的 `active_env`（默认 `development`）。 |
| `RACK_MYSQL_PASSWORD` | 是* | MySQL 密码。**全环境统一覆盖**，优先级高于 `config.secrets.yaml` 与 `config.yaml` 占位符。适合容器、CI 与生产部署。 |

\* 若未设置 `RACK_MYSQL_PASSWORD`，则必须在 `config.secrets.yaml` 中为当前环境配置 `environments.<env>.mysql.password`。

### 优先级

**环境名**（高 → 低）：

1. `RACK_ENV`
2. `config.yaml` 的 `active_env`
3. 默认 `development`

**MySQL 密码**（高 → 低）：

1. `RACK_MYSQL_PASSWORD`
2. `config.secrets.yaml` 中对应环境的 `mysql.password`
3. `config.yaml` 中的占位符（`"${RACK_MYSQL_PASSWORD}"` 或空字符串）—— 视为未配置，会报错

当 `RACK_ENV` 与 `config.yaml` 的 `active_env` 不一致时，会在日志中提示当前实际使用的环境。

### 各环境差异（摘要）

| 环境 | resource_domain | 典型用途 |
|------|-----------------|----------|
| development | `https://resource.linno.cn` | 本地开发 |
| testing | `https://resource-test.linno.cn` | 测试联调 |
| production | `https://resource.linno.cn` | 生产 |

完整 MySQL 主机、库名等见 [`config.yaml`](config.yaml) 的 `environments` 节。

### 使用示例

**Windows（PowerShell / CMD）**

```powershell
# 开发环境（默认，依赖 config.yaml 的 active_env: development）
cd scripts
python predict.py

# 切换到测试环境
$env:RACK_ENV = "testing"
python predict.py

# 生产环境：密码仅通过环境变量（推荐）
$env:RACK_ENV = "production"
$env:RACK_MYSQL_PASSWORD = "your_prod_password"
python predict.py
```

**Linux / macOS**

```bash
# 开发
cd scripts && python predict.py

# 测试
RACK_ENV=testing python predict.py

# 生产
RACK_ENV=production RACK_MYSQL_PASSWORD='***' python predict.py
```

### 密码未配置时的报错

若当前环境既无 `RACK_MYSQL_PASSWORD`，也无 `config.secrets.yaml` 中的有效密码，调用 `get_mysql_connect_kwargs()`（例如 `predict_from_mysql()`）会抛出：

```text
ValueError: 未配置 <环境名> 的 MySQL 密码，请设置 RACK_MYSQL_PASSWORD 或 config.secrets.yaml
```

请按上述方式之一补全密码后重试。

### 本地验证环境与密码逻辑

在项目根目录执行：

```bash
python scripts/verify_config_env.py
```

脚本会检查三环境切换、资源 URL 拼接，以及无密码时的报错信息（不连接真实数据库）。

## 项目结构（简要）

```
config.yaml              # 业务配置 + 各环境非敏感项
config.secrets.yaml      # 本地密码（不提交，见 config.secrets.example.yaml）
utils/config_loader.py   # 统一配置加载
scripts/predict.py       # 预测脚本（MySQL 批处理 / 单图示例）
```
