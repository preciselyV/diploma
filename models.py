import torch
import torch.nn as nn
from modules import Up, Down


class UnetV1(nn.Module):
    def __init__(self, channels):
        super(UnetV1, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024)

        self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.output(x)
        return x


def main():
    model = UnetV1(channels=3)
    minibatch = torch.randn(3, 3, 1024, 1024)
    model.eval()
    # yeah... OS is killing the process as it is drainig all of the
    # memory while dragging this derivative DUG. Gotta do it without
    # gradients :P
    with torch.no_grad():
        res = model(minibatch)
        print(res.shape)


if __name__ == '__main__':
    main()
