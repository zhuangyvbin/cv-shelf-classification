import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import pymysql
import logging
from utils.model import MultiLabelClassifier
from utils.config_loader import (
    load_config,
    get_project_root,
    get_env_config,
    get_mysql_connect_kwargs,
    build_resource_url,
)
from io import BytesIO
import requests
import time
import json

_ROOT = get_project_root()
cfg = load_config()


def _project_path(*parts):
    return os.path.join(_ROOT, *parts)


device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model = MultiLabelClassifier(backbone="resnet50", num_classes=cfg["model"]["num_classes"], pretrained=False).to(device)

model.load_state_dict(torch.load(_project_path(cfg["model"]["save_path"]), map_location=device, weights_only=True))
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((cfg["train"]["image_size"], cfg["train"]["image_size"])),
    transforms.ToTensor()
])

def predict_image(image_path):
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # 本地路径
            image = Image.open(image_path).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        preds = (probs > cfg["predict_threshold"]).astype(int)
        result = {cls: (int(pred), float(prob)) for cls, pred, prob in zip(cfg["classes"], preds, probs)}
        return {
            "status": 1,
            "reason": result
        }

    except Exception as e:
        print(f"预测失败: {e}")
        return {
            "status": 2,
            "reason": str(e)
        }


def predict_folder(image_folder):
    records = []
    for fname in os.listdir(image_folder):
        img_path = os.path.join(image_folder, fname)
        result = predict_image(img_path)
        row = {"image_name": fname}
        row.update({cls: result[cls][0] for cls in cfg["classes"]})
        row.update({f"{cls}_prob": result[cls][1] for cls in cfg["classes"]})
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(_project_path(cfg["data"]["predictions_csv"]), index=False)
    print("预测结果已保存到" + cfg["data"]["predictions_csv"])


def predict_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    records = []
    for idx, row in df.iterrows():
        image_name = row["image_name"]
        img_path = _project_path(cfg["data"]["image_dir"], image_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        result  = {
            "image_name": image_name,
            **{cls: row[cls] for cls in cfg["classes"]},
            **{f"{cls}_prob": prob for cls, prob in zip(cfg["classes"], probs)}
        }
        records.append(result )
    df = pd.DataFrame(records)
    df.to_csv(_project_path(cfg["data"]["test_predictions_csv"]), index=False)
    print("预测结果已保存到" + cfg["data"]["test_predictions_csv"])

def predict_from_mysql():
    env_cfg = get_env_config()
    log_level_name = (env_cfg.get("logging") or {}).get("level", "INFO")
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    batch_size = 500
    predict_time = int(time.time())
    model_version = f"{cfg['model']['name']}_{cfg['model']['version']}"

    conn = pymysql.connect(**get_mysql_connect_kwargs())

    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT id, visit_id, picture_path, img_place
            FROM user_visit_imgs
            WHERE img_place IN ('fromrpphoto', 'fromrcphoto')
            AND picture_path IS NOT NULL
	        AND create_time >= 1772294400
	        AND create_time < 1774972800
	        ORDER BY id
        """)
        rows = cursor.fetchall()
        records = []
        insert_count = 0

        for row in rows:
            id_, visit_id, picture_path, img_place = row
            image_url = build_resource_url(picture_path)
            pre_result = predict_image(image_url)

            # 检查预测状态，status为2表示失败，跳过该条记录
            if pre_result.get("status") == 2:
                # logger.warning(f"图像预测失败，跳过 - ID: {id_}, URL: {image_url}, 错误: {pre_result.get('reason')}")
                print(f"图像预测失败，跳过 - ID: {id_}, URL: {image_url}, 错误: {pre_result.get('reason')}")
                continue
            # status为1时，从reason中获取预测结果
            result = pre_result.get("reason", {})

            rp_pred = result.get("RP", (0,))[0]
            rc_pred = result.get("RC", (0,))[0]

            predict_img = None
            predict_status = 0
            if img_place == "fromrpphoto":
                predict_img = "RP_RACK"
                if rp_pred == 1:
                    predict_status = 1
                    failure_reason = ""
                elif rc_pred == 1:
                    predict_status = 2
                    failure_reason = "RC展架"
                else:
                    predict_status = 3
                    failure_reason = "非RP/RC展架"
            elif img_place == "fromrcphoto":
                predict_img = "RC_RACK"
                if rc_pred == 1:
                    predict_status = 1
                    failure_reason = ""
                elif rp_pred == 1:
                    predict_status = 2
                    failure_reason = "RP展架"
                else:
                    predict_status = 3
                    failure_reason = "非RP/RC展架"
            else:
                failure_reason = f"非审核图片: {img_place}"

            records.append((
                id_,
                visit_id,
                json.dumps(result, ensure_ascii=False),
                predict_img,
                predict_status,
                failure_reason,
                model_version,
                predict_time
            ))

            # 每 batch_size 次执行一次批量插入
            if len(records) >= batch_size:
                with conn.cursor() as insert_cursor:
                    insert_cursor.executemany("""
                        INSERT INTO user_visit_img_predictions (
                            img_id, visit_id, predict_prod, predict_img, 
                            predict_status, failure_reason, model_version, predict_time
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, records)
                    conn.commit()
                    insert_count += len(records)
                    logger.info(f"已插入 {insert_count} 条记录")
                records = []

        # 插入剩余未满 batch 的记录
        if records:
            with conn.cursor() as insert_cursor:
                insert_cursor.executemany("""
                    INSERT INTO user_visit_img_predictions (
                        img_id, visit_id, predict_prod, predict_img, 
                        predict_status, failure_reason, model_version, predict_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, records)
                conn.commit()
                insert_count += len(records)
                logger.info(f"已插入 {insert_count} 条记录")

        logger.info("数据库预测结果已全部更新完成")

    except Exception as e:
        logger.exception(f"异常: {e}")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    # 1. 单张图像预测示例
    print(predict_image(build_resource_url(
        "resource/image/visit/fromrpphoto/2025/07/02/visitPhoto_10631389.jpg"
    )))

    # 2. 文件夹批量预测
    # predict_folder("../data/images_test")

    # 3. CSV 文件批量预测
    # predict_from_csv("../" + cfg["data"]["split_csv_dir"] + "/test.csv")

    # 4. MySQL 自动预测
    # predict_from_mysql()
