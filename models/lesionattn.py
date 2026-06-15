import torch
import torch.nn as nn

from models.backbones import ResNet18AttnBackbone


class LesionAttn(nn.Module):
    def __init__(self, backbone: str, num_classes, pretrained=True, attn_type="bahdanau"):
        super().__init__()
        self.num_classes = num_classes

        if backbone == "resnet18-attn":
            self.net = ResNet18AttnBackbone(num_classes=self.num_classes, pretrained=pretrained, attn_type=attn_type)
        else:
            raise NotImplementedError(f"backbone {backbone} not implemented.")

        self.latent_size = self.net.hidden_size

    def get_latent_size(self):
        return self.latent_size

    def forward_with_hidden(self, x):
        y_logits, hidden = self.net.forward_with_hidden(x)
        return y_logits, hidden

    def forward(self, x):
        return self.net(x)
