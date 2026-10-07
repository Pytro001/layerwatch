"""Stage (a): put every camera frame into the rectified plate frame.

1. Calibration (once per machine): lens distortion + keystone homography map
   the camera image to the plate frame. One cv.remap does both.
2. Per layer: the camera vibrates a little and the recoater shakes the machine,
   so each frame is registered to the layer 0 reference with
   cv.findTransformECC (MOTION_HOMOGRAPHY), coarse to fine, using only the
   static chamber floor rim and plate edge (the powder itself changes every
   layer), then resampled with cv.warpPerspective.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from layerwatch import part

S = 512
_PX0 = int(round((part.FRAME_HALF - part.PLATE_HALF) / (2 * part.FRAME_HALF) * S))


def rectify_maps(K: np.ndarray, dist: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """remap tables from rectified plate pixels to distorted camera pixels"""
    xx, yy = np.meshgrid(np.arange(S, dtype=np.float64), np.arange(S, dtype=np.float64))
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(S * S)], 1) @ H.T
    u = p[:, :2] / p[:, 2:3]
    # pinhole + radial distortion (OpenCV model, k1 k2)
    xn = (u[:, 0] - K[0, 2]) / K[0, 0]
    yn = (u[:, 1] - K[1, 2]) / K[1, 1]
    r2 = xn * xn + yn * yn
    k = 1 + dist[0] * r2 + dist[1] * r2 * r2
    mx = (xn * k * K[0, 0] + K[0, 2]).astype(np.float32).reshape(S, S)
    my = (yn * k * K[1, 1] + K[1, 2]).astype(np.float32).reshape(S, S)
    return mx, my


def rectify(img: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv.remap(img, maps[0], maps[1], cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)


def ecc_mask() -> np.ndarray:
    """static structure: chamber floor rim + the plate edge, not the powder"""
    m = np.zeros((S, S), np.uint8)
    m[4:S - 4, 4:S - 4] = 255
    m[_PX0 + 6:S - _PX0 - 6, _PX0 + 6:S - _PX0 - 6] = 0
    return m


_MASK = ecc_mask()
_CRIT = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 60, 1e-5)
_CRIT_FINE = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 1e-4)
_CRIT_FULL = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 8, 1e-5)


def _prep(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    return cv.GaussianBlur(f, (0, 0), 1.0)


MAX_SHIFT = 15.0


def displacement(W: np.ndarray) -> float:
    return corner_error(W, np.eye(3))


def _ecc_level(t, i, W, scale, motion, crit):
    n = S // scale
    ts = cv.resize(t, (n, n), interpolation=cv.INTER_AREA)
    im = cv.resize(i, (n, n), interpolation=cv.INTER_AREA)
    ms = cv.resize(_MASK, (n, n), interpolation=cv.INTER_NEAREST)
    Sc = np.diag([scale, scale, 1.0]).astype(np.float32)
    Ws = (np.linalg.inv(Sc) @ W @ Sc).astype(np.float32)
    if motion != cv.MOTION_HOMOGRAPHY:
        Ws = Ws[:2].copy()
    cc, Ws = cv.findTransformECC(ts, im, Ws, motion, crit, ms, 3)
    if Ws.shape[0] == 2:
        Ws = np.vstack([Ws, [0, 0, 1]]).astype(np.float32)
    return float(cc), (Sc @ Ws @ np.linalg.inv(Sc)).astype(np.float32)


def register(img_rect: np.ndarray, ref_rect: np.ndarray) -> dict:
    """ECC homography from the reference to this frame over a pyramid
    (1/4 scale translation to get close, 1/2 scale homography, a few full scale steps to finish).
    If the homography does not converge at the coarse level, a rigid (Euclidean)
    ECC gives it a starting point; a level that fails keeps the previous level's
    warp. Returns the aligned image, the warp and the final correlation."""
    W = np.eye(3, dtype=np.float32)
    cc, levels = 0.0, 0
    t, i = _prep(ref_rect), _prep(img_rect)
    # 1/4 scale: translation only (the rim is ~6 px wide there, too thin for 8 parameters)
    try:
        _, W = _ecc_level(t, i, W, 4, cv.MOTION_TRANSLATION, _CRIT)
    except cv.error:
        W = np.eye(3, dtype=np.float32)
    for scale, crit in ((2, _CRIT_FINE), (1, _CRIT_FULL)):
        try:
            cc_k, W_k = _ecc_level(t, i, W, scale, cv.MOTION_HOMOGRAPHY, crit)
            # vibration moves the bed by a few pixels; a warp that moves the plate corners
            # by more than MAX_SHIFT px has diverged, keep the previous level instead
            if displacement(W_k) > MAX_SHIFT:
                continue
            cc, W = cc_k, W_k
            levels += 1
        except cv.error:
            if levels == 0:
                try:
                    cc, W = _ecc_level(t, i, W, scale, cv.MOTION_EUCLIDEAN, _CRIT)
                    cc, W = _ecc_level(t, i, W, scale, cv.MOTION_HOMOGRAPHY, crit)
                    levels += 1
                except cv.error:
                    pass
    ok = levels > 0
    if not ok:
        W, cc = np.eye(3, dtype=np.float32), 0.0
    aligned = cv.warpPerspective(img_rect, W, (S, S), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                                 borderMode=cv.BORDER_REPLICATE)
    return {"aligned": aligned, "W": W, "cc": float(cc), "ok": ok, "levels": levels}


def apply_warp(img_rect: np.ndarray, W: np.ndarray) -> np.ndarray:
    return cv.warpPerspective(img_rect, W, (S, S), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                              borderMode=cv.BORDER_REPLICATE)


def corner_error(W: np.ndarray, J: np.ndarray) -> float:
    """mean corner displacement (px) between the estimated warp and the true
    vibration homography, measured over the plate corners"""
    c = np.float64([[_PX0, _PX0, 1], [S - _PX0, _PX0, 1], [S - _PX0, S - _PX0, 1], [_PX0, S - _PX0, 1]])
    a = c @ np.asarray(W, np.float64).T
    b = c @ np.asarray(J, np.float64).T
    a = a[:, :2] / a[:, 2:3]
    b = b[:, :2] / b[:, 2:3]
    return float(np.linalg.norm(a - b, axis=1).mean())
