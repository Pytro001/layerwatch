"""Synthetic powder bed camera.

For one layer of one build it renders two 8-bit grey camera frames, the way a
real LPBF machine images every layer:

* recoat: right after the recoater spreads fresh powder (streaks and short feed
  show up here, the previous layer bleeds through faintly);
* melt: right after the laser scanned the slice (hatch texture rotated 67 degrees
  per layer, contour scan, spatter, super elevation).

Everything is first drawn in the rectified plate frame (512 x 512 px over
+-0.22 m, so 0.86 mm per pixel, with a 20 mm rim of chamber floor). Then a
camera model maps it into the image the camera would see: a per build keystone
homography (the camera looks down at an angle from the front), radial lens
distortion, and a small per layer vibration homography. That is why the
pipeline must register before it can compare.

Defects come with exact pixel masks in the rectified frame, because we know
where we injected them. Class ids: 1 streak, 2 spatter, 3 super elevation,
4 short feed. Glare and motion blur are nuisances, not defects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from layerwatch import part

S = 512  # rectified frame and camera image size
CLASSES = {1: "streak", 2: "spatter", 3: "super_elevation", 4: "short_feed"}
CLASS_ID = {v: k for k, v in CLASSES.items()}
_PLATE = part.plate_mask(S) > 0
_XX, _ZZ = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
_PLATE_PX0 = int(round((part.FRAME_HALF - part.PLATE_HALF) / (2 * part.FRAME_HALF) * S))  # 23
_SLICE_CACHE: dict[float, np.ndarray] = {}


def slice_at(h: float) -> np.ndarray:
    key = round(h, 6)
    m = _SLICE_CACHE.get(key)
    if m is None:
        m = part.slice_mask(h, S)
        if len(_SLICE_CACHE) > 600:
            _SLICE_CACHE.clear()
        _SLICE_CACHE[key] = m
    return m


def value_noise(rng: np.random.Generator, cell: int, size: int = S) -> np.ndarray:
    n = max(2, size // cell + 2)
    g = rng.standard_normal((n, n)).astype(np.float32)
    return cv.resize(g, (size, size), interpolation=cv.INTER_CUBIC)


def fbm(rng: np.random.Generator, cells=(96, 32, 10, 3), amps=(0.3, 0.45, 0.6, 0.6)) -> np.ndarray:
    out = np.zeros((S, S), np.float32)
    for c, a in zip(cells, amps):
        out += a * value_noise(rng, c)
    return out / np.sqrt(sum(a * a for a in amps))


@dataclass
class Camera:
    """Fixed per build: keystone homography (rectified -> undistorted camera px),
    intrinsics and radial distortion. A real machine gets these from a one time
    calibration with a target on the plate; the pipeline is given them."""

    H: np.ndarray
    K: np.ndarray
    dist: np.ndarray

    @staticmethod
    def random(rng: np.random.Generator) -> "Camera":
        src = np.float32([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]])
        top_in = rng.uniform(30, 60)
        bot_in = rng.uniform(4, 16)
        top_y = rng.uniform(26, 46)
        bot_y = S - 1 - rng.uniform(6, 18)
        dst = np.float32([
            [top_in, top_y], [S - 1 - top_in, top_y],
            [S - 1 - bot_in, bot_y], [bot_in, bot_y],
        ]) + rng.uniform(-5, 5, (4, 2)).astype(np.float32)
        H = cv.getPerspectiveTransform(src, dst)
        f = rng.uniform(480, 560)
        K = np.array([[f, 0, S / 2 + rng.uniform(-6, 6)], [0, f, S / 2 + rng.uniform(-6, 6)], [0, 0, 1]], np.float64)
        dist = np.array([rng.uniform(-0.16, -0.06), rng.uniform(0.0, 0.05), 0, 0, 0], np.float64)
        return Camera(H.astype(np.float64), K, dist)

    def to_dict(self) -> dict:
        return {"H": self.H.tolist(), "K": self.K.tolist(), "dist": self.dist.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        return Camera(np.array(d["H"], np.float64), np.array(d["K"], np.float64), np.array(d["dist"], np.float64))


_MAP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def camera_maps(cam: Camera, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """remap tables: for every distorted camera pixel, where to sample the rectified scene."""
    key = (cam.H.tobytes(), cam.K.tobytes(), cam.dist.tobytes(), J.tobytes())
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    pts = np.stack([_XX.ravel(), _ZZ.ravel()], 1).reshape(-1, 1, 2).astype(np.float64)
    und = cv.undistortPoints(pts, cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    M = np.linalg.inv(cam.H @ J)
    hom = np.c_[und, np.ones(len(und))] @ M.T
    src = (hom[:, :2] / hom[:, 2:3]).astype(np.float32)
    mx, my = src[:, 0].reshape(S, S), src[:, 1].reshape(S, S)
    if len(_MAP_CACHE) > 64:
        _MAP_CACHE.clear()
    _MAP_CACHE[key] = (mx, my)
    return mx, my


def random_jitter(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """small vibration homography about the frame centre"""
    a = np.deg2rad(rng.normal(0, 0.35)) * scale
    tx, ty = rng.normal(0, 2.2, 2) * scale
    s = 1 + rng.normal(0, 0.004) * scale
    c = S / 2
    T1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
    R = np.array([[s * np.cos(a), -s * np.sin(a), 0], [s * np.sin(a), s * np.cos(a), 0], [0, 0, 1]])
    P = np.eye(3)
    P[2, :2] = rng.normal(0, 1.5e-5, 2) * scale
    T2 = np.array([[1, 0, c + tx], [0, 1, c + ty], [0, 0, 1]])
    return T2 @ P @ R @ T1


@dataclass
class Look:
    """Per build appearance (domain randomisation)."""

    powder: float
    powder_tex: float
    grain: float
    melt: float
    hatch_amp: float
    hatch_period: float
    contour: float
    bleed: float
    lamp_dir: tuple[float, float]
    lamp_grad: float
    vignette: float
    gain: float
    noise: float
    gamma: float
    rim: float
    jpeg: int

    @staticmethod
    def random(rng: np.random.Generator) -> "Look":
        a = rng.uniform(0, 2 * np.pi)
        return Look(
            powder=rng.uniform(0.44, 0.58), powder_tex=rng.uniform(0.018, 0.04), grain=rng.uniform(0.02, 0.045),
            melt=rng.uniform(0.2, 0.32), hatch_amp=rng.uniform(0.035, 0.07), hatch_period=rng.uniform(2.6, 4.2),
            contour=rng.uniform(0.06, 0.14), bleed=rng.uniform(0.025, 0.07),
            lamp_dir=(float(np.cos(a)), float(np.sin(a))), lamp_grad=rng.uniform(0.08, 0.25),
            vignette=rng.uniform(0.05, 0.22), gain=rng.uniform(0.85, 1.12), noise=rng.uniform(0.008, 0.022),
            gamma=rng.uniform(0.85, 1.15), rim=rng.uniform(0.12, 0.24), jpeg=int(rng.integers(80, 96)),
        )


@dataclass
class Build:
    """One simulated build: a camera, a look and a static chamber floor."""

    seed: int
    cam: Camera = field(init=False)
    look: Look = field(init=False)
    floor: np.ndarray = field(init=False)
    light: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.cam = Camera.random(rng)
        self.look = Look.random(rng)
        self.floor = self._chamber_floor(rng)
        lx, lz = self.look.lamp_dir
        u = ((_XX - S / 2) * lx + (_ZZ - S / 2) * lz) / (S / 2)
        r2 = ((_XX - S / 2) ** 2 + (_ZZ - S / 2) ** 2) / (S / 2) ** 2
        self.light = (self.look.gain * (1 + self.look.lamp_grad * u) * (1 - self.look.vignette * r2)).astype(np.float32)

    def _chamber_floor(self, rng: np.random.Generator) -> np.ndarray:
        """static rim around the plate: brushed steel, bolt holes, engraved fiducial crosses"""
        L = self.look
        brushed = cv.GaussianBlur(rng.standard_normal((S, S)).astype(np.float32), (0, 0), sigmaX=9, sigmaY=0.8)
        f = L.rim + 0.05 * brushed / (brushed.std() + 1e-6) + 0.03 * value_noise(rng, 40)
        p0, p1 = _PLATE_PX0, S - _PLATE_PX0
        for k in np.linspace(p0 + 30, p1 - 30, 6):
            for (cx, cy) in ((k, p0 / 2), (k, S - p0 / 2), (p0 / 2, k), (S - p0 / 2, k)):
                cv.circle(f, (int(cx), int(cy)), 4, 0.03, -1, cv.LINE_AA)
        for (cx, cy) in ((p0 / 2, p0 / 2), (S - p0 / 2, p0 / 2), (S - p0 / 2, S - p0 / 2), (p0 / 2, S - p0 / 2)):
            cv.line(f, (int(cx - 8), int(cy)), (int(cx + 8), int(cy)), 0.9, 2, cv.LINE_AA)
            cv.line(f, (int(cx), int(cy - 8)), (int(cx), int(cy + 8)), 0.9, 2, cv.LINE_AA)
        # the plate edge gap
        cv.rectangle(f, (p0 - 2, p0 - 2), (p1 + 1, p1 + 1), 0.04, 2)
        return f.astype(np.float32)

    # ------------------------------------------------------------------ scene

    def powder(self, rng: np.random.Generator) -> np.ndarray:
        L = self.look
        tex = fbm(rng)
        grain = rng.standard_normal((S, S)).astype(np.float32)
        return L.powder + L.powder_tex * tex + L.grain * grain * 0.5

    def hatch(self, layer_index: int, rng: np.random.Generator) -> np.ndarray:
        L = self.look
        th = np.deg2rad((67.0 * layer_index) % 180.0)
        u = _XX * np.cos(th) + _ZZ * np.sin(th)
        ripple = np.sin(2 * np.pi * u / L.hatch_period + rng.uniform(0, 6.3)).astype(np.float32)
        tex = 0.5 * value_noise(rng, 6) + 0.5 * rng.standard_normal((S, S)).astype(np.float32)
        return L.melt + L.hatch_amp * ripple + 0.02 * tex

    def compose(self, base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """plate content inside, chamber floor outside, lamp on top"""
        img = np.where(_PLATE, base, self.floor)
        return img * self.light

    def to_camera(self, rect: np.ndarray, J: np.ndarray, rng: np.random.Generator, blur: dict | None = None,
                  glare: dict | None = None) -> np.ndarray:
        L = self.look
        mx, my = camera_maps(self.cam, J)
        img = cv.remap(rect.astype(np.float32), mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0.05)
        # outside the scene: dark chamber walls, a little texture
        outside = (mx < 0) | (my < 0) | (mx > S - 1) | (my > S - 1)
        img[outside] = 0.05 + 0.02 * value_noise(rng, 30)[outside]
        if glare:
            img = img + glare_layer(glare)
        if blur:
            img = motion_blur(img, blur["length"], blur["angle"])
        img = np.clip(img, 0, None) ** L.gamma
        img = img + L.noise * rng.standard_normal(img.shape).astype(np.float32)
        u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        ok, enc = cv.imencode(".jpg", u8, [cv.IMWRITE_JPEG_QUALITY, L.jpeg])
        return cv.imdecode(enc, cv.IMREAD_GRAYSCALE)


def motion_blur(img: np.ndarray, length: float, angle: float) -> np.ndarray:
    n = max(3, int(round(length)) | 1)
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1
    R = cv.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle, 1)
    k = cv.warpAffine(k, R, (n, n))
    k /= k.sum()
    return cv.filter2D(img, -1, k, borderType=cv.BORDER_REFLECT)


def glare_layer(g: dict) -> np.ndarray:
    """specular reflection of the lamp: a bright soft ellipse, saturating in the core.
    Drawn in camera space (it is a property of the optics, not the bed)."""
    cx, cy, rx, ry, peak, ang = g["cx"], g["cy"], g["rx"], g["ry"], g["peak"], g["angle"]
    a = np.deg2rad(ang)
    dx, dy = _XX - cx, _ZZ - cy
    u = (dx * np.cos(a) + dy * np.sin(a)) / rx
    v = (-dx * np.sin(a) + dy * np.cos(a)) / ry
    return (peak * np.exp(-(u * u + v * v) * 1.6)).astype(np.float32)


# ---------------------------------------------------------------------- defects


def draw_streak(rng, d, recoat, melt, mask, sl):
    """A damaged or contaminated recoater blade drags a groove through the fresh
    powder along the recoat direction (+x in the rectified frame)."""
    y0 = d["y"]
    x0, x1 = d["x0"], d["x1"]
    w = d["width"]
    depth = d["depth"]
    ang = np.deg2rad(d.get("angle", 0.0))
    xs = np.arange(int(x0), int(x1))
    wob = np.cumsum(rng.normal(0, 0.08, len(xs))) if len(xs) else xs
    ys = y0 + (xs - x0) * np.tan(ang) + wob
    groove = np.zeros((S, S), np.float32)
    pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
    cv.polylines(groove, [pts], False, 1.0, max(1, int(round(w))), cv.LINE_AA)
    groove = cv.GaussianBlur(groove, (0, 0), 0.6 + w * 0.25)
    groove /= max(groove.max(), 1e-6)
    ridge = cv.GaussianBlur(groove, (0, 0), 1.6 + w * 0.5) - groove * 0.7
    # the streak fades in where the particle catches the blade
    ramp = np.clip((_XX - x0) / 25.0, 0, 1)
    eff = (-depth * groove + 0.35 * depth * np.clip(ridge, 0, None)) * ramp
    recoat += eff
    melt += eff * np.where(sl > 0, 0.45, 0.9)
    m = (groove * ramp > 0.25).astype(np.uint8)
    m = cv.dilate(m, np.ones((3, 3), np.uint8))
    mask[(m > 0) & _PLATE] = 1


def draw_spatter(rng, d, recoat, melt, mask, sl, lamp):
    """ejected melt droplets that land on the powder next to the part"""
    cx, cy, n, spread = d["cx"], d["cy"], d["n"], d["spread"]
    lx, lz = lamp
    for _ in range(n):
        px = cx + rng.normal(0, spread)
        py = cy + rng.normal(0, spread)
        r = rng.uniform(1.1, 3.2)
        b = rng.uniform(0.3, 0.55) * d.get("strength", 1.0)
        blob = np.zeros((S, S), np.float32)
        cv.circle(blob, (int(px * 4), int(py * 4)), int(r * 4), 1.0, -1, cv.LINE_AA, shift=2)
        sh = np.zeros((S, S), np.float32)
        cv.circle(sh, (int((px - lx * r) * 4), int((py - lz * r) * 4)), int(r * 4), 1.0, -1, cv.LINE_AA, shift=2)
        melt += b * blob - 0.12 * np.clip(sh - blob, 0, 1)
        mm = blob > 0.3
        mask[mm & _PLATE & (mask == 0)] = 2


def draw_super_elevation(rng, d, recoat, melt, mask, sl, lamp):
    """the edge of the slice curls up above the powder: a bright crescent with a
    shadow, along part of the contour. Pokes through the next recoat too."""
    cnts, _ = cv.findContours((sl > 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        return False
    cnt = max(cnts, key=len)[:, 0, :]
    if len(cnt) < 40:
        return False
    k0 = int(rng.integers(0, len(cnt)))
    n = int(min(len(cnt) - 1, d["arc"]))
    idx = (k0 + np.arange(n)) % len(cnt)
    arc = cnt[idx]
    band = np.zeros((S, S), np.float32)
    cv.polylines(band, [arc.reshape(-1, 1, 2).astype(np.int32)], False, 1.0, int(d["width"]), cv.LINE_AA)
    band = cv.GaussianBlur(band, (0, 0), 1.0)
    lx, lz = lamp
    shadow = np.roll(np.roll(band, int(round(-lz * 3)), 0), int(round(-lx * 3)), 1)
    melt += d["bright"] * band - 0.15 * np.clip(shadow - band, 0, 1)
    recoat += 0.6 * d["bright"] * band - 0.08 * np.clip(shadow - band, 0, 1)
    m = band > 0.3
    mask[m & _PLATE & (mask == 0)] = 3
    return True


def draw_short_feed(rng, d, recoat, melt, mask, sl_prev_tex):
    """not enough powder reached the far end of the stroke: a ragged bare patch
    where the previous, already melted layer shows through"""
    x0, y0, y1 = d["x0"], d["y0"], d["y1"]
    edge = x0 + 18 * value_noise(rng, 24)
    ragged = 10 * value_noise(rng, 12)
    region = (_XX > edge) & (_ZZ > y0 + ragged) & (_ZZ < y1 - ragged) & _PLATE
    region = cv.morphologyEx(region.astype(np.uint8), cv.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    soft = cv.GaussianBlur(region.astype(np.float32), (0, 0), 2.0)
    # bare: previous layer metal (darker, hatch) where it was melted, compacted darker powder elsewhere
    bare = sl_prev_tex
    recoat[:] = recoat * (1 - soft) + (bare - d["drop"]) * soft
    melt[:] = melt * (1 - 0.7 * soft) + (melt - d["drop"]) * 0.7 * soft
    mask[region & (mask == 0)] = 4


@dataclass
class LayerSpec:
    layer_index: int
    h: float
    defects: list[dict] = field(default_factory=list)
    glare: dict | None = None
    blur: dict | None = None
    glare_on: str = "melt"
    blur_on: str = "melt"
    jitter_scale: float = 1.0


def random_defects(rng: np.random.Generator, sl: np.ndarray, p_clean: float = 0.42) -> list[dict]:
    """sample defect specs for one layer"""
    out: list[dict] = []
    if rng.random() < p_clean:
        return out
    n = 1 if rng.random() < 0.75 else 2
    types = rng.choice(["streak", "spatter", "super_elevation", "short_feed"], n, replace=False)
    ys, xs = np.nonzero(sl)
    for t in types:
        if t == "streak":
            k = int(rng.integers(1, 3))
            y = rng.uniform(40, S - 40)
            for j in range(k):
                x0 = rng.uniform(_PLATE_PX0 + 2, S * 0.6)
                x1 = min(S - _PLATE_PX0 - 2, x0 + rng.uniform(90, 420))
                out.append({"type": "streak", "y": y + j * rng.uniform(5, 25), "x0": x0, "x1": x1,
                            "width": rng.uniform(1.0, 3.2), "depth": rng.uniform(0.07, 0.2),
                            "angle": rng.normal(0, 0.6)})
        elif t == "spatter":
            if len(xs) and rng.random() < 0.8:
                j = int(rng.integers(0, len(xs)))
                cx, cy = xs[j] + rng.normal(0, 25), ys[j] + rng.normal(0, 25)
            else:
                cx, cy = rng.uniform(60, S - 60, 2)
            out.append({"type": "spatter", "cx": float(cx), "cy": float(cy), "n": int(rng.integers(3, 28)),
                        "spread": float(rng.uniform(4, 22)), "strength": float(rng.uniform(0.7, 1.1))})
        elif t == "super_elevation":
            if len(xs) < 50:
                continue
            out.append({"type": "super_elevation", "arc": float(rng.uniform(25, 140)), "width": float(rng.uniform(2, 5)),
                        "bright": float(rng.uniform(0.2, 0.45))})
        elif t == "short_feed":
            y0 = rng.uniform(30, S - 140)
            out.append({"type": "short_feed", "x0": float(rng.uniform(S * 0.55, S * 0.85)), "y0": float(y0),
                        "y1": float(y0 + rng.uniform(50, 200)), "drop": float(rng.uniform(0.08, 0.2))})
    return out


def random_nuisance(rng: np.random.Generator, spec: LayerSpec) -> None:
    r = rng.random()
    if r < 0.12:
        strong = rng.random() < 0.55
        spec.glare = {"cx": float(rng.uniform(80, S - 80)), "cy": float(rng.uniform(80, S - 80)),
                      "rx": float(rng.uniform(25, 70)), "ry": float(rng.uniform(15, 45)),
                      "peak": float(rng.uniform(0.75, 1.3) if strong else rng.uniform(0.12, 0.3)),
                      "angle": float(rng.uniform(0, 180))}
        spec.glare_on = "melt" if rng.random() < 0.6 else "recoat"
    elif r < 0.22:
        strong = rng.random() < 0.55
        spec.blur = {"length": float(rng.uniform(7, 15) if strong else rng.uniform(2, 3.5)),
                     "angle": float(rng.uniform(0, 180))}
        spec.blur_on = "melt" if rng.random() < 0.6 else "recoat"


def render_layer(build: Build, spec: LayerSpec, rng: np.random.Generator, J: np.ndarray | None = None) -> dict:
    """Render both camera frames of one layer. Returns camera images, the
    rectified ground truth mask, the vibration homography and per defect labels."""
    L = build.look
    h = spec.h
    dh = part.HPIPE / part.DEMO_LAYERS
    sl = slice_at(h)
    sl_prev = slice_at(max(0.0, h - dh)) if h > dh else np.zeros((S, S), np.uint8)
    powder = build.powder(rng)
    # previous layer bleeding through the thin new powder
    prev_soft = cv.GaussianBlur(sl_prev.astype(np.float32) / 255, (0, 0), 1.3)
    recoat = powder * (1 - L.bleed * prev_soft)
    # post melt: hatch inside the slice, a contour scan on its edge, a faint heat tint around it
    slf = sl.astype(np.float32) / 255
    slf_aa = cv.GaussianBlur(slf, (0, 0), 0.6)
    hatch = build.hatch(spec.layer_index, rng)
    edge = cv.morphologyEx(sl, cv.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32) / 255
    halo = cv.GaussianBlur(slf, (0, 0), 3.0)
    melt = powder * (1 - 0.06 * np.clip(halo - slf, 0, 1)) * (1 - slf_aa) + hatch * slf_aa + L.contour * edge * slf_aa
    # the already melted previous layer, exposed if the powder is missing
    prev_tex = powder * 0.92 * (1 - cv.GaussianBlur(sl_prev.astype(np.float32) / 255, (0, 0), 0.6)) + \
        (L.melt + 0.5 * L.hatch_amp * np.sin(_XX / 3.1)) * cv.GaussianBlur(sl_prev.astype(np.float32) / 255, (0, 0), 0.6)

    mask = np.zeros((S, S), np.uint8)
    labels = []
    for d in spec.defects:
        t = d["type"]
        before = mask.copy()
        if t == "streak":
            draw_streak(rng, d, recoat, melt, mask, sl)
        elif t == "spatter":
            draw_spatter(rng, d, recoat, melt, mask, sl, L.lamp_dir)
        elif t == "super_elevation":
            if not draw_super_elevation(rng, d, recoat, melt, mask, sl, L.lamp_dir):
                continue
        elif t == "short_feed":
            draw_short_feed(rng, d, recoat, melt, mask, prev_tex)
        new = (mask != before) & (mask == CLASS_ID[t])
        if new.sum() < 4:
            continue
        ys, xs = np.nonzero(new)
        labels.append({
            "type": t, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "area_px": int(new.sum()), "inside_part_px": int((new & (sl > 0)).sum()),
        })

    J = random_jitter(rng, spec.jitter_scale) if J is None else J
    rect_recoat = build.compose(recoat, rng)
    rect_melt = build.compose(melt, rng)
    cam_recoat = build.to_camera(rect_recoat, J, rng,
                                 blur=spec.blur if spec.blur_on == "recoat" else None,
                                 glare=spec.glare if spec.glare_on == "recoat" else None)
    cam_melt = build.to_camera(rect_melt, J, rng,
                               blur=spec.blur if spec.blur_on == "melt" else None,
                               glare=spec.glare if spec.glare_on == "melt" else None)
    nuis = []
    if spec.glare:
        cam = cam_melt if spec.glare_on == "melt" else cam_recoat
        sat = float((cam >= 250).mean())
        nuis.append({"type": "glare", "on": spec.glare_on, "peak": spec.glare["peak"], "saturated_frac": sat,
                     "bad": bool(spec.glare["peak"] >= 0.6)})
    if spec.blur:
        nuis.append({"type": "motion_blur", "on": spec.blur_on, "length_px": spec.blur["length"],
                     "bad": bool(spec.blur["length"] >= 6.0)})
    return {"recoat": cam_recoat, "melt": cam_melt, "mask": mask, "slice": sl, "J": J, "defects": labels,
            "nuisances": nuis}


def render_reference(build: Build) -> np.ndarray:
    """layer 0 reference: fresh powder on the empty plate, no vibration"""
    rng = np.random.default_rng(build.seed * 7 + 1)
    rect = build.compose(build.powder(rng), rng)
    return build.to_camera(rect, np.eye(3), rng)
