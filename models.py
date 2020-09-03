import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self, encoder, num_classes=4, pretrained=True):
        super(BaseModel, self).__init__()

        self.bn0 = nn.BatchNorm2d(3)

        if encoder.startswith("efficientnet"):  # efficientnet-b{0,1,2,3,4,5,6,7}
            self.encoder = efficientnet(encoder, num_classes, pretrained)
        else:
            self.encoder = base_models(encoder, num_classes, pretrained)

    def forward(self, x):
        x = self.bn0(x)
        x = self.encoder(x)
        return x


def efficientnet(encoder, num_classes, pretrained):
    base = EfficientNet.from_pretrained(encoder)#, dropout_rate=0.0)
    base._fc = nn.Linear(base._fc.in_features, num_classes)
    return base


def base_models(encoder, num_classes, pretrained):
    base = getattr(models, encoder)(pretrained=pretrained)

    if encoder in set(
        [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ]
    ):
        # base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)

    elif encoder in set(["densenet121", "densenet169", "densenet201", "densenet161"]):
        base.classifier = nn.Linear(base.classifier.in_features, num_classes)

    elif encoder in set(["mobilenet_v2"]):
        in_features = 1280
        base.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError("{} - unknown model".format(encoder))

    return base


def build_model(**kwargs):
    return BaseModel(**kwargs)
