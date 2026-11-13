"""Model definitions: a compact VGG-style CNN (used for Dogs-vs-Cats and
CIFAR-10) and a fine-tuned pretrained ResNet-18."""
import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, pool=True):
        layers = [
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class SimpleCNN(nn.Module):
    """Five-conv Dogs-vs-Cats or eight-conv CIFAR network with BatchNorm.

    Dogs-vs-Cats config (SimpleCNN-DVC), input 3x128x128:
        Block1 32  (no pool)  -> 32x128x128   (conv only)
        Block1b 32 (pool)     -> 32x64x64
        Block2 64  (pool)     -> 64x32x32
        Block3 128 (pool)     -> 128x16x16
        Block4 128 (pool)     -> 128x8x8
        GAP -> dropout -> FC 128->num_classes
    CIFAR-10 config (SimpleCNN-C10), input 3x32x32, base=64:
        [64,64,M] [128,128,M] [256,256,M] [512,512,M] -> 512x2x2, GAP -> FC 10
    """

    def __init__(self, num_classes=2, base=32, dropout=0.5, cifar=False):
        super().__init__()
        self.num_classes = num_classes
        if cifar:
            self.features = nn.Sequential(
                ConvBlock(3, base, pool=False), ConvBlock(base, base),
                ConvBlock(base, base * 2, pool=False), ConvBlock(base * 2, base * 2),
                ConvBlock(base * 2, base * 4, pool=False), ConvBlock(base * 4, base * 4),
                ConvBlock(base * 4, base * 8, pool=False), ConvBlock(base * 8, base * 8),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(dropout), nn.Linear(base * 8, num_classes))
        else:
            self.features = nn.Sequential(
                ConvBlock(3, base, pool=False),
                ConvBlock(base, base),
                ConvBlock(base, base * 2),
                ConvBlock(base * 2, base * 4),
                ConvBlock(base * 4, base * 4),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(dropout), nn.Linear(base * 4, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(num_classes=2, pretrained=True):
    """ImageNet-pretrained ResNet-18 with a fresh classifier head."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.num_classes = num_classes
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m1 = SimpleCNN(num_classes=2)
    print("SimpleCNN-DVC params:", count_params(m1))
    print("out:", m1(torch.randn(2, 3, 128, 128)).shape)
    m2 = SimpleCNN(num_classes=10, base=64, cifar=True)
    print("SimpleCNN-C10 params:", count_params(m2))
    print("out:", m2(torch.randn(2, 3, 32, 32)).shape)
    m3 = build_resnet18(2, pretrained=False)
    print("ResNet-18 params:", count_params(m3))
    print("out:", m3(torch.randn(2, 3, 160, 160)).shape)
