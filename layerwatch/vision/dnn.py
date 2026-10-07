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
MIN_AREA = {1: 60, 2: 6, 3: 30, 4: 400}


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
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

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
            n, lab, stats, _ = cv.connectedComponentsWithStats(m)
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if a >= MIN_AREA[k]:
                    found.append({"type": name, "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)], "area_px": int(a)})
        # spatter is many tiny blobs: also count the class total
        sp = int((cls == 2).sum())
        if sp >= 12 and not any(f["type"] == "spatter" for f in found):
            ys, xs = np.nonzero(cls == 2)
            found.append({"type": "spatter", "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "area_px": sp})
        return cls, found
