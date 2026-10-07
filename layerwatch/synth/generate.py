"""Generate the synthetic powder bed dataset.

    python -m layerwatch.synth.generate --out data/synth --train-builds 50 --test-builds 10 --layers 40

Each build is one simulated machine set up (camera, lamp, powder, rim) with its
own layer 0 reference image and 40 layers at random heights through the part,
in print order. Deterministic: build b of split s uses seed hash(s, b).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2 as cv
import numpy as np

from layerwatch import part
from layerwatch.synth.render import Build, LayerSpec, random_defects, random_nuisance, render_layer, render_reference, slice_at

# "test" was looked at while tuning thresholds; "holdout" is generated once tuning is frozen
SPLIT_SEED = {"train": 1000, "test": 9000, "holdout": 7000}


def make_build(args: tuple[str, int, int, str]) -> dict:
    split, b, n_layers, out = args
    cv.setNumThreads(1)
    seed = SPLIT_SEED[split] + b
    build = Build(seed)
    rng = np.random.default_rng(seed * 31 + 7)
    d = Path(out) / split / f"b{b:03d}"
    d.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(d / "ref.jpg"), render_reference(build), [cv.IMWRITE_JPEG_QUALITY, 92])
    # layer indices on the 480 layer demo grid, sorted = print order
    idx = np.sort(rng.choice(part.DEMO_LAYERS, n_layers, replace=False))
    samples = []
    for k, li in enumerate(idx):
        h = part.layer_height(int(li))
        sl = slice_at(h)
        spec = LayerSpec(int(li), h, defects=random_defects(rng, sl))
        random_nuisance(rng, spec)
        r = render_layer(build, spec, rng)
        stem = f"l{k:02d}"
        cv.imwrite(str(d / f"{stem}_recoat.jpg"), r["recoat"], [cv.IMWRITE_JPEG_QUALITY, 95])
        cv.imwrite(str(d / f"{stem}_melt.jpg"), r["melt"], [cv.IMWRITE_JPEG_QUALITY, 95])
        cv.imwrite(str(d / f"{stem}_mask.png"), r["mask"])
        samples.append({
            "id": f"{split}/b{b:03d}/{stem}", "build": f"{split}/b{b:03d}", "order": k, "layer_index": int(li),
            "h_m": round(h, 6),
            "recoat": f"{split}/b{b:03d}/{stem}_recoat.jpg", "melt": f"{split}/b{b:03d}/{stem}_melt.jpg",
            "mask": f"{split}/b{b:03d}/{stem}_mask.png",
            "defects": r["defects"], "nuisances": r["nuisances"], "jitter_H": np.round(r["J"], 8).tolist(),
        })
    return {"build": f"{split}/b{b:03d}", "seed": seed, "camera": build.cam.to_dict(),
            "reference": f"{split}/b{b:03d}/ref.jpg", "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--train-builds", type=int, default=50)
    ap.add_argument("--test-builds", type=int, default=10)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--holdout-builds", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [("train", b, a.layers, a.out) for b in range(a.train_builds)] + \
           [("test", b, a.layers, a.out) for b in range(a.test_builds)] + \
           [("holdout", b, a.layers, a.out) for b in range(a.holdout_builds)]
    with Pool(min(4, a.workers)) as pool:
        builds = pool.map(make_build, jobs, chunksize=1)
    for split in ("train", "test", "holdout"):
        bs = [b for b in builds if b["build"].startswith(split)]
        if not bs:
            continue
        n = sum(len(b["samples"]) for b in bs)
        counts: dict[str, int] = {}
        for b in bs:
            for s in b["samples"]:
                for t in {d["type"] for d in s["defects"]}:
                    counts[t] = counts.get(t, 0) + 1
                for nu in s["nuisances"]:
                    key = nu["type"] + ("_bad" if nu["bad"] else "_mild")
                    counts[key] = counts.get(key, 0) + 1
        meta = {"split": split, "n_layers": n, "image_size": 512, "frame_half_m": part.FRAME_HALF,
                "classes": {"1": "streak", "2": "spatter", "3": "super_elevation", "4": "short_feed"},
                "layers_with": counts, "builds": bs}
        Path(a.out, split).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(a.out, split, "labels.json"), "w") as f:
            json.dump(meta, f)
        print(split, n, "layers", counts)
    print(f"generated in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
