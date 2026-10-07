"""A small U-Net: 3 input channels (recoat, melt, expected slice), 5 classes.
About 0.47 M parameters, fully convolutional, so it trains on 256 px crops and
runs on the full 512 px frame. Only ops OpenCV's DNN importer supports
(Conv, BatchNorm, ReLU, MaxPool, ConvTranspose, Concat)."""
from __future__ import annotations

import torch
import torch.nn as nn


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin: int = 3, ncls: int = 5, w: int = 16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(cin, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.mid = block(4 * w, 8 * w)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2)
        self.d3 = block(8 * w, 4 * w)
        self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.d2 = block(4 * w, 2 * w)
        self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.d1 = block(2 * w, w)
        self.head = nn.Conv2d(w, ncls, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)
