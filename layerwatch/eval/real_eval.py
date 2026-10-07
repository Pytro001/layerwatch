"""Run the rule based recoat checks on REAL powder bed images.

Dataset: "Annotated Image Dataset for defects detection in Laser Powder Bed
Fusion" (Aalto University, EOS M290, 316L, EOSTATE PowderBed camera,
1280 x 1024, CC-BY 4.0, doi 10.5281/zenodo.14996805). Only the folder
"Original Images and Manual Labels" is used: 250 labelled powder bed images
sampled every 6th from the 1,530 that have a LabelImg box file. The labels have
one class ("defects"), drawn as boxes around visibly disturbed powder, mostly
recoater streaks and the debris around them.

What runs: calibration (4 bed corners picked once by hand, the same for all
three builds in the set) -> cv.warpPerspective into the same 512 px plate frame
as the synthetic pipeline -> lighting from the per build median frame ->
streak detector, short feed (dark patch) detector and bright blob detector.
No expected slice is available (no job files), no registration (no layer 0
frame), no DNN (it was trained on synthetic three channel input only).

Scoring, per GT box: hit if any detection box intersects it. Per detection:
correct if it intersects any GT box. Reported separately from synthetic.

    python -m layerwatch.eval.real_eval
"""
from __future__ import annotations

import glob
import json
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from layerwatch.vision import stages as st

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real"
S = 512
P0, P1 = 27, 485  # PLATE_INNER extent in the rectified frame
# bed corners in the 1280 x 1024 camera image (TL, TR, BR, BL), picked once by hand
BED = np.float32([[235, 150], [1270, 125], [1270, 890], [170, 895]])
DST = np.float32([[P0, P0], [P1, P0], [P1, P1], [P0, P1]])
H = cv.getPerspectiveTransform(BED, DST)
Hinv = np.linalg.inv(H)


def boxes(xml: Path) -> list[list[int]]:
    r = ET.parse(xml).getroot()
    return [[int(o.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")] for o in r.iter("bndbox")]


def to_cam(b: list[int]) -> list[int]:
    pts = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2)
    q = cv.perspectiveTransform(pts, Hinv).reshape(-1, 2)
    return [int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())]


def inter(a: list[int], b: list[int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def detect(rect: np.ndarray, light: np.ndarray) -> list[dict]:
    f = st.flat(rect, light)
    empty = np.zeros((S, S), np.uint8)
    out = []
    sk = st.streaks(f, empty)
    for s in sk["streaks"]:
        out.append({"type": "streak", "bbox": [s["x1"], int(min(s["y1"], s["y2"])) - 3, s["x2"], int(max(s["y1"], s["y2"])) + 3]})
    sf = st.short_feed(f)
    for r in sf["regions"]:
        out.append({"type": "dark_patch", "bbox": r["bbox"]})
    th = st.bright_detail(f)
    sp = st.spatter(f, empty, None, th)
    if sp["detected"]:
        # one box per cluster of blobs
        pts = np.array([[b["x"], b["y"]] for b in sp["blobs"]], np.float32)
        n, lab = cv.connectedComponents(cv.dilate(_points(pts), np.ones((35, 35), np.uint8)))
        for k in range(1, n):
            sel = [p for p in pts if lab[int(p[1]), int(p[0])] == k]
            if len(sel) >= 2:
                q = np.array(sel)
                out.append({"type": "bright_blobs", "bbox": [int(q[:, 0].min()) - 3, int(q[:, 1].min()) - 3,
                                                             int(q[:, 0].max()) + 3, int(q[:, 1].max()) + 3]})
    return out


def _points(pts: np.ndarray) -> np.ndarray:
    m = np.zeros((S, S), np.uint8)
    for x, y in pts:
        m[int(y), int(x)] = 255
    return m


def main() -> dict:
    files = sorted(glob.glob(str(REAL / "images" / "*.jpg")))
    by_build = defaultdict(list)
    for f in files:
        by_build[Path(f).name[:20]].append(f)
    t0 = time.time()
    lights = {}
    for b, fs in by_build.items():
        stack = np.stack([cv.warpPerspective(cv.imread(f, 0), H, (S, S)) for f in fs[:: max(1, len(fs) // 15)][:15]])
        lights[b] = st.lighting(np.median(stack, 0).astype(np.uint8))
    gt_n = gt_hit = det_n = det_ok = 0
    img_with_hit = 0
    per_type = defaultdict(lambda: [0, 0])
    rows = []
    lat = []
    for f in files:
        name = Path(f).stem
        gts = boxes(REAL / "labels" / f"{name}.xml")
        img = cv.imread(f, 0)
        t1 = time.perf_counter()
        rect = cv.warpPerspective(img, H, (S, S))
        dets = detect(rect, lights[name[:20]])
        lat.append((time.perf_counter() - t1) * 1000)
        dcam = [dict(d, bbox_cam=to_cam(d["bbox"])) for d in dets]
        hits = [any(inter(g, d["bbox_cam"]) for d in dcam) for g in gts]
        oks = [any(inter(g, d["bbox_cam"]) for g in gts) for d in dcam]
        gt_n += len(gts)
        gt_hit += sum(hits)
        det_n += len(dcam)
        det_ok += sum(oks)
        img_with_hit += int(any(hits))
        for d, ok in zip(dcam, oks):
            per_type[d["type"]][0] += 1
            per_type[d["type"]][1] += int(ok)
        rows.append({"image": name, "gt_boxes": gts, "detections": [{"type": d["type"], "bbox_cam": d["bbox_cam"]} for d in dcam],
                     "gt_hit": hits, "det_correct": oks})
    res = {
        "dataset": "Aalto LPBF annotated powder bed images (EOS M290), CC-BY 4.0, doi:10.5281/zenodo.14996805",
        "images": len(files), "gt_boxes": gt_n,
        "box_recall": round(gt_hit / max(gt_n, 1), 3), "detection_precision": round(det_ok / max(det_n, 1), 3),
        "images_with_at_least_one_hit": round(img_with_hit / max(len(files), 1), 3),
        "detections": det_n, "detections_per_image": round(det_n / max(len(files), 1), 2),
        "by_detector": {k: {"detections": v[0], "on_a_gt_box": v[1], "precision": round(v[1] / max(v[0], 1), 3)} for k, v in per_type.items()},
        "latency_ms_median": round(float(np.median(lat)), 1),
        "builds": {k: len(v) for k, v in by_build.items()},
        "wall_s": round(time.time() - t0, 1),
    }
    out = ROOT / "docs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(out / "real.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def figure(out_path: Path, names: list[str]) -> None:
    """side by side: GT boxes (green) vs detections (red) on real frames"""
    d = json.load(open(ROOT / "docs" / "eval" / "real.json"))
    rows = {r["image"]: r for r in d["rows"]}
    tiles = []
    for n in names:
        img = cv.cvtColor(cv.imread(str(REAL / "images" / f"{n}.jpg"), 0), cv.COLOR_GRAY2BGR)
        r = rows[n]
        for g in r["gt_boxes"]:
            cv.rectangle(img, tuple(g[:2]), tuple(g[2:]), (90, 220, 90), 3)
        for x in r["detections"]:
            b = x["bbox_cam"]
            cv.rectangle(img, tuple(b[:2]), tuple(b[2:]), (60, 60, 255), 2)
        tiles.append(cv.resize(img[100:940, 140:1280], (570, 420)))
    cv.imwrite(str(out_path), np.hstack(tiles), [cv.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
