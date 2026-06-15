import torch
import torch.nn as nn
from torchvision import models

from models.attn import SelfAttention, BahdanauAttention


def _resnet18(pretrained: bool):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=pretrained)


class ResNet18AttnBackbone(nn.Module):
    def __init__(self, num_classes: int, pretrained=True, attn_type="bahdanau"):
        super().__init__()
        resnet = _resnet18(pretrained)
        self.hidden_size = resnet.fc.in_features

        self.embedding_module = nn.Sequential(resnet.conv1,
                                              resnet.bn1,
                                              resnet.relu,
                                              resnet.maxpool,
                                              resnet.layer1,
                                              resnet.layer2,
                                              resnet.layer3,
                                              resnet.layer4)
        self.classify_module = nn.Sequential(nn.Linear(resnet.fc.in_features, 512),
                                             nn.LeakyReLU(),
                                             nn.Linear(512, num_classes))
        if attn_type == "self":
            self.attention_module = SelfAttention(in_channels=3)
        elif attn_type == "bahdanau":
            self.attention_module = BahdanauAttention(in_channels=3)
        else:
            raise NotImplementedError(f"Unknown attention type {attn_type}")
        self.avgpool = resnet.avgpool

    def forward(self, x, mask=None):
        y, hidden = self.forward_with_hidden(x, mask=mask)
        return y
    
    def forward_with_hidden(self, x, mask=None):
        if mask is None:
            z, hidden = self.attention_module.forward_with_hidden(x)
        else:
            z = x * mask
            hidden = {"attn": mask}
        z = self.embedding_module(z)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        y = self.classify_module(z)
        hidden.update({"latent": z})
        return y, hidden


def test_resnet18_attn_backbone():
    batch_size, channels, height, width = 4, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)

    num_classes = 10
    model = ResNet18AttnBackbone(num_classes=num_classes, pretrained=False)

    y = model(x)

    print("Output shape:", y.shape)
    assert y.shape == (batch_size, num_classes), "Output shape is not as expected!"

    y, hidden = model.forward_with_hidden(x)
    print("Output shape with hidden:", y.shape, "\nHidden states:", hidden)
    assert y.shape == (batch_size, num_classes), "Output shape with hidden is not as expected!"


if __name__ == "__main__":
    test_resnet18_attn_backbone()
