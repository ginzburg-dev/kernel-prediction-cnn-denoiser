import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, base=64) -> None:
        super().__init__()
        self.e1 = self._conv_block(in_channels, base, 2)
        self.pool1 = nn.MaxPool2d(2)

        self.e2 = self._conv_block(base, base * 2, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.e2 = self._conv_block(base * 2, base * 4, 2)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base*4, base*2, 2)

        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base*2, base, 2)

        self.out_conv = nn.Conv2d(base, out_channels=in_channels, kernel_size=3, padding=1)
    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self):
