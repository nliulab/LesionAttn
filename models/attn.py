import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        attention_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        return out + x

    def forward_with_hidden(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        attention = torch.bmm(value, attention.permute(0, 2, 1))
        attention = attention.view(batch_size, C, width, height)
        attention = torch.mean(attention, dim=1, keepdim=True)

        hidden = {"attn": attention}

        return attention + x, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.attn = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.v = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)

    def forward_with_hidden(self, feature_map):
        batch_size, channels, height, width = feature_map.size()

        energy = torch.tanh(self.attn(feature_map)) 
        attention = self.v(energy)
        attention = F.softmax(attention.view(batch_size, -1), dim=1)
        attention = attention.view(batch_size, 1, height, width)

        attended_features = attention * feature_map
        hidden = {"attn": attention}

        return attended_features, hidden
    
    def forward(self, x):
        attended_features, _ = self.forward_with_hidden(x)
        return attended_features
    

def test_attention(attn_module):
    # Assuming the dimensions of the input data are [batch_size, channels, width, height]
    batch_size, channels, width, height = 4, 64, 32, 32  # These parameters can be adjusted as needed

    # Create a random input tensor
    x = torch.randn(batch_size, channels, width, height)

    # Pass the input tensor through the self-attention module
    output, hidden = attn_module.forward_with_hidden(x)
    attn = hidden["attn"]

    # Print the shape of the output tensor
    print("Output shape:", output.shape)
    print("Attention shape:", attn.shape)

    # Ensure the shape of the output tensor is the same as the input tensor
    assert output.shape == x.shape, "Output shape is not the same as input shape!"


if __name__ == "__main__":
    channels = 64
    attn_module = SelfAttention(in_channels=channels)
    # attn_module = BahdanauAttention(channels=channels)
    test_attention(attn_module)
