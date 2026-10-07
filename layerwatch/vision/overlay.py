"""One overlay image per pipeline stage, for the dashboard and the report.
All 512 x 512 BGR in the rectified plate frame."""
from __future__ import annotations

import cv2 as cv
import numpy as np

from layerwatch.vision import register as reg
from layerwatch.vision import stages as st

# BGR, the same palette as the dashboard
COL = {
    "streak": (79, 77, 255),
    "spatter": (102, 209, 255),
    "super_elevation": (66, 140, 255),
    "short_feed": (255, 163, 77),
    "expected": (140, 230, 120),
    "observed": (230, 120, 220),
    "ref": (230, 210, 60),
    "glare": (200, 200, 200),
}
DNN_COL = {1: COL["streak"], 2: COL["spatter"], 3: COL["super_elevation"], 4: COL["short_feed"]}


def _bgr(g: np.ndarray, dim: float = 1.0) -> np.ndarray:
    return cv.cvtColor((g.astype(np.float32) * dim).clip(0, 255).astype(np.uint8), cv.COLOR_GRAY2BGR)


def _tint(img: np.ndarray, mask: np.ndarray, color: tuple, a: float) -> np.ndarray:
    m = mask > 0
    out = img.copy()
    out[m] = (out[m] * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
    return out


def stage_overlays(r_al, m_al, ref, expected, sc, sk, sf, sp, se, cls, evidence, W, rr) -> dict:
    o: dict[str, np.ndarray] = {}

    # (a) registration: reference structure (cyan) on the registered frame, ECC mask rim lit
    img = _bgr(m_al, 0.85)
    edges = cv.Canny(cv.GaussianBlur(ref, (3, 3), 0), 40, 110)
    edges[reg._MASK == 0] = 0
    img = _tint(img, reg._MASK, (60, 50, 20), 0.35)
    img[edges > 0] = COL["ref"]
    o["registration"] = img

    # (b) slice compare: expected contour green, observed melt magenta
    img = _bgr(m_al, 0.8)
    img = _tint(img, sc["observed"], COL["observed"], 0.35)
    cnts, _ = cv.findContours(expected, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, cnts, -1, COL["expected"], 1, cv.LINE_AA)
    o["slice"] = img

    # (c) streaks: directional response as heat, merged Hough lines in red
    img = _bgr(r_al, 0.8)
    resp = sk["response"]
    heat = np.clip(resp / 0.08, 0, 1)
    img = (img * (1 - heat[..., None] * 0.8) + np.array([30, 120, 255], np.float32) * heat[..., None] * 0.8).astype(np.uint8)
    for s in sk["streaks"]:
        cv.line(img, (s["x1"], int(s["y1"])), (s["x2"], int(s["y2"])), COL["streak"], 2, cv.LINE_AA)
    sfm = sf["mask"]
    img = _tint(img, sfm, COL["short_feed"], 0.45)
    cnts, _ = cv.findContours(sfm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, cnts, -1, COL["short_feed"], 1, cv.LINE_AA)
    o["streaks"] = img

    # (d) spatter blobs and super elevation crescents
    img = _bgr(m_al, 0.8)
    cnts, _ = cv.findContours(expected, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, cnts, -1, (70, 110, 70), 1, cv.LINE_AA)
    img = _tint(img, se["mask"], COL["super_elevation"], 0.75)
    for b in sp["blobs"]:
        cv.circle(img, (b["x"], b["y"]), int(b["r"]) + 4, COL["spatter"], 1, cv.LINE_AA)
    o["spatter"] = img

    # (e) DNN classes
    img = _bgr(m_al, 0.75)
    if cls is not None:
        for k, c in DNN_COL.items():
            img = _tint(img, (cls == k).astype(np.uint8), c, 0.7)
    o["dnn"] = img

    # (f) evidence: texture energy of the powder (blur) and saturated pixels (glare)
    lap = np.abs(cv.Laplacian(cv.pyrDown(m_al), cv.CV_32F, ksize=3))
    lap = cv.resize(lap, (512, 512))
    lap8 = np.clip(lap / 12 * 255, 0, 255).astype(np.uint8)
    img = cv.applyColorMap(lap8, cv.COLORMAP_BONE)
    img = (img * 0.9).astype(np.uint8)
    sat = (m_al >= 250).astype(np.uint8)
    img = _tint(img, cv.dilate(sat, np.ones((3, 3), np.uint8)), (60, 60, 255), 0.85)
    o["evidence"] = img
    return o
