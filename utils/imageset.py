import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import shutil


def dir_to_label(classes, raw_dir, csv_file):
    """
    分类好的数据集生成多标签文件 label.csv
    支持多标签图像（出现在多个类中）
    """
    image_labels = {}
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        for img_name in os.listdir(cls_dir):
            if img_name not in image_labels:
                image_labels[img_name] = {cat: 0 for cat in classes}
            image_labels[img_name][cls] = 1

    records = []
    for img_name, labels in image_labels.items():
        record = {"image_name": img_name}
        record.update(labels)
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(csv_file, index=False)
    print(f"标签文件已保存：{csv_file}")
    return df


def oversample_minority(df, classes, target_ratio=1.0):
    """
    对少数类（如 other）进行过采样以平衡类别
    target_ratio = 1.0 表示所有类都将复制至与最大类相同数量
    """
    class_counts = {cls: df[cls].sum() for cls in classes}
    max_count = max(class_counts.values())

    oversampled_df = df.copy()
    for cls in classes:
        count = class_counts[cls]
        if count < max_count * target_ratio:
            needed = int(max_count * target_ratio - count)
            minority_samples = df[df[cls] == 1]
            replicated = minority_samples.sample(n=needed, replace=True, random_state=42)
            oversampled_df = pd.concat([oversampled_df, replicated], ignore_index=True)

    oversampled_df = oversampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"数据过采样完成，样本总数：{len(oversampled_df)}")
    return oversampled_df


def split_dataset(df, split_csv_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    拆分为 train/val/test CSV 文件
    """
    os.makedirs(split_csv_dir, exist_ok=True)

    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=53, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=53)

    train_df.to_csv(os.path.join(split_csv_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_csv_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_csv_dir, "test.csv"), index=False)

    print(f"数据拆分完成：训练 {len(train_df)}，验证 {len(val_df)}，测试 {len(test_df)}")

def copy_images_from_csv(csv_path, src_dir, dst_dir, overwrite=True):
    """
    根据 CSV 文件中的 image_name 列，将图像从 src_dir 复制到 dst_dir
    :param csv_path: CSV 文件路径，包含 'image_name' 列
    :param src_dir: 图像源目录
    :param dst_dir: 目标目录
    :param overwrite: 是否覆盖已有文件，默认为 True
    """
    import os
    import pandas as pd
    import shutil

    # 读取 CSV
    df = pd.read_csv(csv_path)
    image_names = df['image_name'].unique()

    # 创建目标目录
    os.makedirs(dst_dir, exist_ok=True)

    # 复制文件
    for img_name in image_names:
        src_file = os.path.join(src_dir, img_name)
        dst_file = os.path.join(dst_dir, img_name)

        if os.path.exists(src_file):
            if overwrite or not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"已复制: {img_name}")
            else:
                print(f"已存在（跳过）: {img_name}")
        else:
            print(f"源文件不存在: {img_name}")

    print(f"图片复制完成，共处理 {len(image_names)} 张图片。")


def compute_pos_weight_from_csv(csv_path):
    """
    从多标签CSV文件读取标签数据并计算 pos_weight。
    参数:
        csv_path (str): CSV 文件路径
    返回:
        pos_weight (Tensor): 每个类别的 pos_weight
    """
    df = pd.read_csv(csv_path)

    # 排除 image_name 列
    label_columns = df.columns.tolist()
    if 'image_name' in label_columns:
        label_columns.remove('image_name')
    # 只对标签列进行求和
    label_counts = df[label_columns].sum(axis=0).values.astype(float)

    total_samples = len(df)
    label_counts_tensor = torch.tensor(label_counts, dtype=torch.float32)
    pos_weight = (total_samples - label_counts_tensor) / (label_counts_tensor + 1e-6)
    return pos_weight
