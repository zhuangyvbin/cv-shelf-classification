import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.dataset import MultiLabelDataset
from utils.model import MultiLabelClassifier
from utils.imageset import compute_pos_weight_from_csv
import numpy as np
import os
import yaml

# ========================
# 配置 & 数据加载部分
# ========================

with open('../config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
pos_weight = compute_pos_weight_from_csv("../" + cfg["data"]["label_csv"])

def get_data_loaders():
    train_dataset = MultiLabelDataset(
        csv_path="../" + cfg["data"]["split_csv_dir"] + "/train.csv",
        image_root="../" + cfg["data"]["image_dir"],
        classes=cfg["classes"],
        mode="train",
        image_size=cfg["train"]["image_size"]
    )
    val_dataset = MultiLabelDataset(
        csv_path="../" + cfg['data']["split_csv_dir"] + "/val.csv",
        image_root="../" + cfg["data"]["image_dir"],
        classes=cfg["classes"],
        mode="val",
        image_size=cfg["train"]["image_size"]
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4)
    return train_loader, val_loader

# ========================
# 模型 & 损失 & 优化器
# ========================

model = MultiLabelClassifier(backbone="resnet50", num_classes=cfg["model"]["num_classes"], pretrained=True, freeze_layers=False)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ========================
# 训练循环封装成 main()
# ========================

def main():
    train_loader, val_loader = get_data_loaders()

    # 训练循环
    best_f1 = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        binarized_preds = (all_preds > 0.5).astype(int)

        f1 = f1_score(all_labels, binarized_preds, average="macro")
        prec = precision_score(all_labels, binarized_preds, average="macro")
        rec = recall_score(all_labels, binarized_preds, average="macro")

        print(
            f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "../"+cfg["model"]["save_path"])
            print(f"✅ 新最佳模型保存，F1: {best_f1:.4f}")

    print(f"训练完成，最佳 F1: {best_f1:.4f}")

# ========================
# ✅ 程序入口保护
# ========================

if __name__ == '__main__':
    main()
