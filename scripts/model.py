"""
Transfer-learning model: pretrained ResNet18 (timm) with early layers frozen,
fine-tuning only the last residual block (layer4) and the classifier head.
"""
import timm
import torch.nn as nn

NUM_CLASSES = 2
UNFREEZE_PREFIXES = ("layer4", "fc")


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    model = timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith(UNFREEZE_PREFIXES):
            param.requires_grad = True

    return model


def trainable_summary(model: nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)"


if __name__ == "__main__":
    m = build_model()
    print(trainable_summary(m))
