import torch
import torch.nn as nn


class Conv2D(nn.Module):
    """"""

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, bias=False, ac=False, dropout=0.0):
        """Constructor for Conv2D"""
        super(Conv2D, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias))
        layers.append(nn.BatchNorm2d(out_channel))
        if ac:
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    """"""

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, bias=False, dropout=0.0):
        """Constructor for ResBlock"""
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.ac = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(self.ac(out2)))
        return self.ac(out1 + out2)


class Attention(nn.Module):
    """"""

    def __init__(self, feature_dim, hidden, patches):
        """Constructor for Attention"""
        super(Attention, self).__init__()
        self.v = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Tanh()
        )
        self.w = nn.Sequential(
            nn.Linear(hidden, 1),
        )
        self.patches = patches

    def forward(self, x):
        out = self.v(x)
        out = self.w(out)
        # out = torch.tanh(out)
        out = out.view(-1, self.patches)
        # print(out.size())
        out = torch.softmax(out, 1)
        # print(out.size())
        return out

# --------------------------------------------------
# Channel Attention Module (Squeeze-and-Excitation)
# --------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Global pooling: average and max
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]
        out = avg_out + max_out
        return self.sigmoid(out)  # [B, C, 1, 1]


# --------------------------------------------------
# Spatial Attention Module
# --------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        pool = torch.cat([avg_out, max_out], dim=1)     # [B,2,H,W]
        out = self.conv(pool)                          # [B,1,H,W]
        return self.sigmoid(out)                       # [B,1,H,W]


# --------------------------------------------------
# Convolutional Block Attention Module (CBAM)
# --------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # Apply channel attention
        ca = self.channel_att(x)    # [B,C,1,1]
        x = x * ca                  # channel-weighted
        # Apply spatial attention
        sa = self.spatial_att(x)    # [B,1,H,W]
        x = x * sa                  # spatial-weighted
        # x = torch.sum(x, dim=1)
        return x


if __name__ == '__main__':
    x = torch.randn(20, 128)
    net = Attention(128, 256, 10)
    att = net(x)
    print(att)
    print(torch.sum(att, 1))
