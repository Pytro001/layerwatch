"""Stage (e): learned segmenter, run through OpenCV's own DNN module.

A small U-Net (layerwatch/train/unet.py) trained in PyTorch on the synthetic
training split, exported to ONNX, loaded with cv.dnn.readNetFromONNX. Input is
three channels in the rectified frame: the flattened recoat frame, the
flattened melt frame and the expected slice mask, so the network knows where
the part should be. Output: 5 class logits per pixel (background + 4 defects).
"""
from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

S = 512
NAMES = {1: "streak", 2: "spatter", 3: "super_elevation", 4: "short_feed"}
# minimum component area (px) per class before the DNN calls a defect present
MIN_AREA = {1: 60, 2: 12, 3: 30, 4: 400}
MIN_STREAK_LEN = 70  # px, as stages.T["streak_min_len"]


def make_input(recoat_flat: np.ndarray, melt_flat: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """float32 1 x 3 x H x W, the exact same normalisation is used in training"""
    a = np.clip(recoat_flat, 0, 2).astype(np.float32) - 1.0
    b = np.clip(melt_flat, 0, 2).astype(np.float32) - 1.0
    c = (expected > 0).astype(np.float32)
    return np.stack([a, b, c])[None]


class Segmenter:
    def __init__(self, onnx_path: str | Path):
        self.path = str(onnx_path)
        self.net = cv.dnn.readNetFromONNX(self.path)
        # OpenCV 5 runs ONNX models on its new graph engine; the default backend is used as is

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        self.net.setInput(blob)
        out = self.net.forward()  # 1 x 5 x H x W
        return out[0]

    @staticmethod
    def components(logits: np.ndarray, plate: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        cls = np.argmax(logits, axis=0).astype(np.uint8)
        cls[plate == 0] = 0
        found = []
        for k, name in NAMES.items():
            m = (cls == k).astype(np.uint8)
            if not m.any():
                continue
            if k == 2:
                # spatter is many small droplets: one detection per cluster (35 px linking)
                n, lab, stats, _ = cv.connectedComponentsWithStats(cv.dilate(m, np.ones((35, 35), np.uint8)))
                for i in range(1, n):
                    a = int(((lab == i) & (m > 0)).sum())
                    if a >= MIN_AREA[k]:
                        ys, xs = np.nonzero((lab == i) & (m > 0))
                        found.append({"type": name, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "area_px": a})
                continue
            n, lab, stats, _ = cv.connectedComponentsWithStats(m)
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if a < MIN_AREA[k]:
                    continue
                if k == 1 and w < MIN_STREAK_LEN:
                    continue  # same length rule as the Hough path
                found.append({"type": name, "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)], "area_px": int(a),
                              **({"length_px": int(w)} if k == 1 else {})})
        return cls, found
