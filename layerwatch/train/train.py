"""Train the U-Net on the synthetic training split, export it to ONNX and check
that OpenCV's DNN module gives the same answer as PyTorch.

    python -m layerwatch.train.train --minutes 25

CPU only, 4 threads, 256 px crops (60 % centred on a defect), class weighted
cross entropy, AdamW + cosine schedule, early stop on validation loss (the last
5 training builds are held out as validation; the test split is never touched).
Inputs are prepared with the same functions the pipeline uses, registered with
the true vibration homography.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from layerwatch import part
from layerwatch.train.unet import UNet
from layerwatch.vision import register as reg
from layerwatch.vision import stages as st
from layerwatch.vision.dnn import make_input
from layerwatch.vision.pipeline import expected_slice

ROOT = Path(__file__).resolve().parents[2]
CLASS_W = torch.tensor([1.0, 6.0, 10.0, 8.0, 2.5])


class BuildCache:
    def __init__(self, build: dict, data: Path):
        self.maps = reg.rectify_maps(np.array(build["camera"]["K"]), np.array(build["camera"]["dist"]),
                                     np.array(build["camera"]["H"]))
        ref = reg.rectify(cv.imread(str(data / build["reference"]), 0), self.maps)
        self.light = st.lighting(ref)


def prepare(sample: dict, bc: BuildCache, data: Path) -> tuple[np.ndarray, np.ndarray]:
    J = np.array(sample["jitter_H"], np.float32)
    r = reg.apply_warp(reg.rectify(cv.imread(str(data / sample["recoat"]), 0), bc.maps), J)
    m = reg.apply_warp(reg.rectify(cv.imread(str(data / sample["melt"]), 0), bc.maps), J)
    exp = expected_slice(sample["h_m"])
    x = make_input(st.flat(r, bc.light), st.flat(m, bc.light), exp)[0]
    y = cv.imread(str(data / sample["mask"]), 0)
    y[st.PLATE_INNER == 0] = 0
    return x, y


class EpochSampler(Sampler):
    """indices ep*n .. ep*n+n-1, so every epoch draws new crops while the
    persistent workers keep their caches"""

    def __init__(self, n: int):
        self.n, self.epoch = n, 0

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        return iter(range(self.epoch * self.n, (self.epoch + 1) * self.n))


class Crops(Dataset):
    def __init__(self, builds: list[dict], data: Path, n: int, seed: int, crop: int = 256, train: bool = True):
        self.items = [(s, b) for b in builds for s in b["samples"]]
        self.data, self.n, self.seed, self.crop, self.train = data, n, seed, crop, train
        self.bc: dict[str, BuildCache] = {}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng(self.seed * 100003 + i)
        s, b = self.items[int(rng.integers(len(self.items)))]
        if b["build"] not in self.bc:
            self.bc[b["build"]] = BuildCache(b, self.data)
        x, y = prepare(s, self.bc[b["build"]], self.data)
        c = self.crop
        ys, xs = np.nonzero(y)
        if len(ys) and rng.random() < 0.6:
            j = int(rng.integers(len(ys)))
            cy, cx = ys[j] + rng.integers(-c // 3, c // 3), xs[j] + rng.integers(-c // 3, c // 3)
        else:
            cy, cx = rng.integers(c // 2, 512 - c // 2, 2)
        y0 = int(np.clip(cy - c // 2, 0, 512 - c))
        x0 = int(np.clip(cx - c // 2, 0, 512 - c))
        x = x[:, y0:y0 + c, x0:x0 + c]
        y = y[y0:y0 + c, x0:x0 + c]
        if self.train:
            # small shifts emulate registration error; vertical flips keep the recoat axis
            if rng.random() < 0.5:
                x, y = x[:, ::-1], y[::-1]
            dx, dy = rng.integers(-1, 2, 2)
            x = np.roll(x, (int(dy), int(dx)), (1, 2))
            x = x * rng.uniform(0.9, 1.1) + rng.normal(0, 0.02, x.shape).astype(np.float32) * (rng.random() < 0.5)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--iters-per-epoch", type=int, default=120)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "models"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    data = Path(a.data)
    meta = json.load(open(data / "train" / "labels.json"))
    builds = meta["builds"]
    train_b, val_b = builds[:-5], builds[-5:]
    sampler = EpochSampler(a.iters_per_epoch * a.batch)
    tr = DataLoader(Crops(train_b, data, 10 ** 9, 1), batch_size=a.batch, sampler=sampler, num_workers=2,
                    persistent_workers=True)

    va = DataLoader(Crops(val_b, data, 48, 2, train=False), batch_size=a.batch, num_workers=1, persistent_workers=True)
    net = UNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    max_epochs = 60
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()
    best, bad, log = math.inf, 0, []
    for ep in range(max_epochs):
        sampler.epoch = ep  # new crops every epoch
        net.train()
        tl = 0.0
        for x, y in tr:
            loss = F.cross_entropy(net(x), y, weight=CLASS_W)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()
        net.eval()
        vl, inter, union = 0.0, np.zeros(5), np.zeros(5)
        with torch.no_grad():
            for x, y in va:
                lo = net(x)
                vl += F.cross_entropy(lo, y, weight=CLASS_W).item()
                p = lo.argmax(1)
                for k in range(1, 5):
                    inter[k] += ((p == k) & (y == k)).sum().item()
                    union[k] += ((p == k) | (y == k)).sum().item()
        vl /= len(va)
        iou = (inter / np.maximum(union, 1))[1:]
        el = (time.time() - t0) / 60
        rec = {"epoch": ep, "train_loss": round(tl / len(tr), 4), "val_loss": round(vl, 4),
               "val_iou": {n: round(float(v), 3) for n, v in zip(["streak", "spatter", "super_elevation", "short_feed"], iou)},
               "minutes": round(el, 2)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(net.state_dict(), out / "layerwatch_unet.pt")
        else:
            bad += 1
        if bad >= a.patience:
            print("early stop", flush=True)
            break
        if el * (ep + 2) / (ep + 1) > a.minutes:
            print("time budget reached", flush=True)
            break
    json.dump(log, open(out / "train_log.json", "w"), indent=1)
    export(out)


def export(out: Path) -> None:
    net = UNet()
    net.load_state_dict(torch.load(out / "layerwatch_unet.pt"))
    net.eval()
    x = torch.randn(1, 3, 512, 512)
    onnx_path = out / "layerwatch_unet.onnx"
    torch.onnx.export(net, x, str(onnx_path), opset_version=17, input_names=["input"], output_names=["logits"],
                      dynamo=False)
    cvnet = cv.dnn.readNetFromONNX(str(onnx_path))
    cvnet.setInput(x.numpy())
    a = cvnet.forward()
    with torch.no_grad():
        b = net(x).numpy()
    diff = float(np.abs(a - b).max())
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB), cv.dnn vs torch max abs diff {diff:.2e}")
    json.dump({"onnx": onnx_path.name, "cv_dnn_vs_torch_max_abs_diff": diff,
               "params": sum(p.numel() for p in net.parameters())}, open(out / "export_check.json", "w"))


if __name__ == "__main__":
    main()
