import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import yaml

with open('../config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 加载数据
df = pd.read_csv("../" + cfg["data"]["test_predictions_csv"])

y_true = df[cfg["classes"]].values
y_scores = df[[f"{cls}_prob" for cls in cfg["classes"]]].values
y_pred = (y_scores > cfg["predict_threshold"]).astype(int)

# 生成 classification report
report = classification_report(y_true, y_pred, target_names=cfg["classes"], output_dict=True)
report_str = classification_report(y_true, y_pred, target_names=cfg["classes"])

print("==== Classification Report ====")
print(report_str)

if cfg["save_report"]:
    with open("../" + cfg["report_file"], "w") as f:
        f.write(report_str)
    print(f"报告已保存至 {cfg['report_file']}")

# 绘制 ROC 曲线
plt.figure(figsize=(8,6))
for i, cls in enumerate(cfg["classes"]):
    if np.all(y_true[:, i] == 1):
        print(f"类别 {cls} 没有负样本，跳过 ROC 曲线绘制")
        continue
    elif np.all(y_true[:, i] == 0):
        print(f"类别 {cls} 没有正样本，跳过 ROC 曲线绘制")
        continue
    fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-label ROC Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/roc_curve.png")
plt.show()
print("ROC 曲线已保存至 roc_curve.png")
