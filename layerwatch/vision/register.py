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
_CRIT_FINE = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 25, 1e-5)
_CRIT_FULL = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 8, 1e-5)


def _prep(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    return cv.GaussianBlur(f, (0, 0), 1.0)


def register(img_rect: np.ndarray, ref_rect: np.ndarray) -> dict:
    """ECC homography from the reference to this frame, coarse (half size) then fine.
    Returns the aligned image, the warp and the final correlation coefficient."""
    W = np.eye(3, dtype=np.float32)
    cc = 0.0
    ok = True
    try:
        # pyramid: 1/4 scale to get close, 1/2 scale to refine, a few full scale steps to finish
        t, i = _prep(ref_rect), _prep(img_rect)
        for k, (scale, crit) in enumerate(((4, _CRIT), (2, _CRIT_FINE), (1, _CRIT_FULL))):
            n = S // scale
            ts = cv.resize(t, (n, n), interpolation=cv.INTER_AREA)
            im = cv.resize(i, (n, n), interpolation=cv.INTER_AREA)
            ms = cv.resize(_MASK, (n, n), interpolation=cv.INTER_NEAREST)
            Sc = np.diag([scale, scale, 1.0]).astype(np.float32)
            Ws = (np.linalg.inv(Sc) @ W @ Sc).astype(np.float32)
            cc, Ws = cv.findTransformECC(ts, im, Ws, cv.MOTION_HOMOGRAPHY, crit, ms, 3)
            W = (Sc @ Ws @ np.linalg.inv(Sc)).astype(np.float32)
    except cv.error:
        ok = False
        W = np.eye(3, dtype=np.float32)
    aligned = cv.warpPerspective(img_rect, W, (S, S), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                                 borderMode=cv.BORDER_REPLICATE)
    return {"aligned": aligned, "W": W, "cc": float(cc), "ok": ok}


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
