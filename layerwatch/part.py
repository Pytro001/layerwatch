"""The test part: a small regeneratively cooled rocket thrust chamber, printed in one
piece with the nozzle exit down. Bell nozzle, throat, combustion chamber, injector
head with its manifold, and 36 cooling channels inside the wall that run from a
ring manifold at the exit to the top. Four tensile test bars stand in the corners.

`solid(x, z, h)` says what is metal at any point (x, z across the plate, h up from
it, all in metres, plate 0.4 m square centred on the origin). Everything else in
LayerWatch (the expected slice, the synthetic camera images, the 3D defect map in
the dashboard) is derived from this one function, so they can never disagree.

Numpy port of a TypeScript part function written for an earlier printer
simulation by the same author. The dashboard keeps a TypeScript copy
(dashboard/src/part.ts) that must stay in step with this file.
"""
from __future__ import annotations

import numpy as np

RE = 0.13  # exit radius
RT = 0.032  # throat radius
RC = 0.07  # chamber radius
HT = 0.16  # throat height
HC = 0.22  # end of the converging section
HJ = 0.3  # chamber ends, injector head starts
RH = 0.085  # injector head radius
HTOP = 0.355
HPIPE = 0.385  # total part height
WALL = 0.011
N_CHANNELS = 36
BAR = 0.17  # test bar position (both axes)
BAR_R = 0.006
BAR_H = 0.25
TOR_RC = RE + WALL + 0.016
TOR_H = 0.022
TOR_O = 0.016
TOR_I = 0.0115

PLATE_HALF = 0.2  # metres, the powder bed is 0.4 m square
FRAME_HALF = 0.22  # the rectified camera frame also shows a 20 mm rim of chamber floor


def _sm(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def r_in(h: float) -> float:
    """inner (hot gas) radius of the chamber and nozzle"""
    if h < HT:
        u = (HT - h) / HT
        return RT + (RE - RT) * (1 - (1 - u) ** 1.7)
    if h < HC:
        return RT + (RC - RT) * _sm((h - HT) / (HC - HT))
    return RC


def r_out(h: float) -> float:
    """outer skin radius below the injector head"""
    r = r_in(h) + WALL
    if h > HJ - 0.02:
        r = max(r, RC + WALL + (RH - RC - WALL) * _sm((h - (HJ - 0.02)) / 0.03))
    for b in (0.055, 0.115):  # stiffening bands on the nozzle
        if abs(h - b) < 0.006:
            r += 0.005 * _sm((0.006 - abs(h - b)) / 0.002)
    return r


def solid(x: np.ndarray, z: np.ndarray, h: float) -> np.ndarray:
    """Boolean array, True where there is metal at height h (vectorised over x, z)."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros(np.broadcast(x, z).shape, dtype=bool)
    if h < 0 or h > HPIPE:
        return out
    r = np.hypot(x, z)
    decided = np.zeros_like(out)

    def put(mask: np.ndarray, value: np.ndarray | bool) -> None:
        m = mask & ~decided
        if isinstance(value, np.ndarray):
            out[m] = value[m]
        else:
            out[m] = value
        decided[m] = True

    # test bars
    if h < BAR_H:
        ax, az = np.abs(x) - BAR, np.abs(z) - BAR
        near = (np.abs(ax) < 0.02) & (np.abs(az) < 0.02) & (np.hypot(ax, az) < BAR_R)
        put(near, True)
    # coolant manifold torus at the exit, hollow
    dt = np.hypot(r - TOR_RC, h - TOR_H)
    put(dt < TOR_O, dt > TOR_I)
    # the bridge from the torus into the wall
    ri_h = r_in(h)
    if TOR_H - 0.006 < h < TOR_H + 0.006:
        bridge = (r > ri_h) & (r < TOR_RC)
        put(bridge, ~((abs(h - TOR_H) < 0.0025) & (r > ri_h + 0.002)))
    if h < 0.312:
        ri, ro = ri_h, r_out(h)
        wall = (r >= ri) & (r <= ro)
        a = np.arctan2(z, x)
        f = np.abs(np.mod((a * N_CHANNELS) / (2 * np.pi) + 0.5, 1.0) - 0.5)
        c0, c1 = ri + 0.0022, min(ri + 0.0075, ro - 0.0025)
        channel = (f < 0.24) & (r > c0) & (r < c1) & (TOR_H - 0.002 < h < 0.305)
        put(np.ones_like(out), wall & ~channel)
        return out
    if h < HTOP:  # injector head
        inside = r <= RH
        if h < 0.33:
            val = inside.copy()
            a = np.arctan2(z, x)
            for rr, n in ((0.0, 1), (0.022, 8), (0.042, 14), (0.06, 20)):
                ring = (np.abs(r - rr) <= 0.004) & (r < RC)
                if rr == 0:
                    val &= ~ring
                    continue
                k = np.round(a / (2 * np.pi) * n)
                ex, ez = np.cos(k / n * 2 * np.pi) * rr, np.sin(k / n * 2 * np.pi) * rr
                val &= ~(ring & (np.hypot(x - ex, z - ez) < 0.0032))
            put(np.ones_like(out), val)
        elif h < 0.344:
            put(np.ones_like(out), inside & (r > 0.066))  # manifold cavity
        else:
            put(np.ones_like(out), inside & ((r > 0.0125) | (h > 0.351)))  # top plate + inlet
        return out
    put(np.ones_like(out), (r < 0.019) & (r > 0.0125))  # inlet pipe
    return out


def frame_grid(size: int = 512, half: float = FRAME_HALF) -> tuple[np.ndarray, np.ndarray]:
    """x, z coordinates of pixel centres of the rectified frame (row = z, col = x)."""
    c = (np.arange(size) + 0.5) / size * 2 * half - half
    return np.meshgrid(c, c)


_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def slice_mask(h: float, size: int = 512, half: float = FRAME_HALF) -> np.ndarray:
    """uint8 mask (0/255) of the slice at height h in the rectified frame."""
    key = size
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = frame_grid(size, half)
    x, z = _GRID_CACHE[key]
    # supersample 2x2 so thin walls and bars come out anti aliased then thresholded
    d = half / size
    acc = np.zeros(x.shape, np.float32)
    for ox in (-0.5, 0.5):
        for oz in (-0.5, 0.5):
            acc += solid(x + ox * d, z + oz * d, h)
    return ((acc >= 2) * 255).astype(np.uint8)


def plate_mask(size: int = 512, half: float = FRAME_HALF) -> np.ndarray:
    """uint8 mask of the powder bed inside the rectified frame."""
    x, z = frame_grid(size, half)
    return (((np.abs(x) < PLATE_HALF) & (np.abs(z) < PLATE_HALF)) * 255).astype(np.uint8)


def px_to_plate(col: float, row: float, size: int = 512, half: float = FRAME_HALF) -> tuple[float, float]:
    """pixel (col, row) of the rectified frame to plate metres (x, z)."""
    return ((col + 0.5) / size * 2 * half - half, (row + 0.5) / size * 2 * half - half)


# The demo build images 480 evenly spaced layer heights through the part.
DEMO_LAYERS = 480


def layer_height(i: int, n: int = DEMO_LAYERS) -> float:
    return (i + 0.5) / n * HPIPE
