import torch
import torch.nn as nn
from torchvision import models


class MultiLabelClassifier(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=3, pretrained=True, freeze_layers=False):
        super(MultiLabelClassifier, self).__init__()

        # 加载预训练模型
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        else:
            raise ValueError(f"未支持的backbone: {backbone}")

        # 冻结前面的卷积层
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 修改最后的分类头
        if "efficientnet" in backbone:
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        # 多标签 Sigmoid 输出
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.backbone(x)
        probs = self.sigmoid(logits)
        return probs
