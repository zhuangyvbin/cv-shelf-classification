import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import yaml

with open('../config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

save_visuals = True
df = pd.read_csv("../" + cfg["data"]["predictions_csv"])
def show_image_with_probs(img_path, probs, labels, image_name):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # 显示图片
    img = Image.open(img_path).convert("RGB")
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title(f"Image: {image_name}")

    # 显示概率条形图
    axs[1].bar(cfg["classes"], probs, color=['#ff9999','#66b3ff','#99ff99'])
    axs[1].set_ylim(0, 1)
    axs[1].set_title(f"Predicted: {[cls for cls, lab in zip(cfg['classes'], labels) if lab==1]}")

    plt.tight_layout()
    if save_visuals:
        save_path = "../" + cfg["data"]["output_dir"] + "/" + image_name + "_viz.png"
        plt.savefig(save_path)
    plt.show()
    plt.close()

for idx, row in df.iterrows():
    image_name = row["image_name"]
    img_path = "../" + cfg["data"]["image_dir"] + "/" + image_name

    probs = [row[f"{cls}_prob"] for cls in cfg["classes"]]
    labels = [row[cls] for cls in cfg["classes"]]

    show_image_with_probs(img_path, probs, labels, image_name)
