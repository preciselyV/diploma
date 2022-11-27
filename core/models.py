import logging 
import torch
import torch.nn as nn
from modules import UpEmb, DownEmb, SinPositionalEncoding


class UnetV1(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super(UnetV1, self).__init__()

        self.channels = channels
        self.time_dim = time_dim
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.down1 = DownEmb(in_channels=64, out_channels=128, time_dim=self.time_dim)
        self.down2 = DownEmb(in_channels=128, out_channels=256, time_dim=self.time_dim)
        self.down3 = DownEmb(in_channels=256, out_channels=512, time_dim=self.time_dim)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpEmb(in_channels=512, out_channels=256, time_dim=self.time_dim)
        self.up2 = UpEmb(in_channels=256, out_channels=128, time_dim=self.time_dim)
        self.up3 = UpEmb(in_channels=128, out_channels=64, time_dim=self.time_dim)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.encode = SinPositionalEncoding(time_dim=self.time_dim)
        self.encode.requires_grad_(False)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.encode(t)
        x1 = self.input(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x4 = self.bottleneck(x4)
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        x = self.output(x)
        return x


def main():
    model = UnetV1(channels=3, time_dim=256).to('cuda')
    minibatch = torch.randn(3, 3, 64, 64).to('cuda')
    ts = torch.tensor([228] * minibatch.shape[0], dtype=torch.long).to('cuda')
    model.eval()
    # yeah... OS is killing the process as it is drainig all of the
    # memory while dragging this derivative DUG. Gotta do it without
    # gradients :P
    with torch.no_grad():
        res = model(minibatch, ts)
        res = res.to('cpu')
        print(res.shape)


if __name__ == '__main__':
    main()
