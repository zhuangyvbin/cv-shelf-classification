import yaml
from utils.imageset import dir_to_label, split_dataset, oversample_minority, copy_images_from_csv

with open('../config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

label_csv = '../' + cfg['data']['label_csv']
raw_dir = '../' + cfg['data']['raw_dir']
split_csv_dir = '../' + cfg['data']['split_csv_dir']
image_dir = '../' + cfg['data']['image_dir']
image_test_dir = '../' + cfg['data']['image_test_dir']



# 步骤1：生成标签 CSV
df = dir_to_label(cfg['classes'], raw_dir, label_csv)

# 步骤2：类别不均衡过采样（other 类）
df_balanced = oversample_minority(df, cfg['classes'], target_ratio=1.0)

# 步骤3：拆分数据集
split_dataset(df_balanced, split_csv_dir)

copy_images_from_csv(split_csv_dir + "/test.csv", image_dir, image_test_dir)

