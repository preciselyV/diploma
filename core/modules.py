import torch
import torch.nn as nn


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.maxPool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxPool(x)
        return x


class DownEmb(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(DownEmb, self).__init__()
        self.down = Down(in_channels=in_channels, out_channels=out_channels)
        self.emb = nn.Sequential(
            nn.Linear(in_features=time_dim, out_features=out_channels),
            nn.SiLU()
        )

    def forward(self, x, t):
        x = self.down(x)
        # transform [batch_size,channels] to -> [batch_size,chanels,feature_map_h,feature_map_w]
        emb = self.emb(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2,
                               kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, residual):
        x = self.tconv1(x)
        # so yeah, main idea is that we want to have in_channels / out_channels == 2
        # thats why we upsample x4 then concat with residiual info and have x2.
        x = torch.cat([x, residual], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpEmb(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(UpEmb, self).__init__()

        self.up = Up(in_channels=in_channels, out_channels=out_channels)

        self.emb = nn.Sequential(
            nn.Linear(in_features=time_dim, out_features=out_channels),
            nn.SELU()
        )

    def forward(self, x, residual, t):
        x = self.up(x, residual)
        emb = self.emb(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SinPositionalEncoding(nn.Module):
    def __init__(self, time_dim: int):
        super(SinPositionalEncoding, self).__init__()
        self.time_dim = time_dim
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.time_dim, 2).float() / self.time_dim)
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, t):
        pos_enc_a = torch.sin(t.repeat(1, self.time_dim // 2) * self.inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.time_dim // 2) * self.inv_freq)
        pos_enc = torch.flatten(torch.stack([pos_enc_a, pos_enc_b], dim=-1), -2, -1)
        return pos_enc
