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
