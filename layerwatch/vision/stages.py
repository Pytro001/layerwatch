"""Stages (b) to (d) and (f) of the LayerWatch pipeline, all classic OpenCV 5.

All inputs are registered 8-bit images in the rectified plate frame. Brightness
is normalised by a lighting estimate taken from the layer 0 reference (the lamp
and lens vignetting do not change between layers), so thresholds are relative
to fresh powder = 1.0.
"""
from __future__ import annotations

import math

import cv2 as cv
import numpy as np

from layerwatch import part

S = 512
PLATE = part.plate_mask(S)
PLATE_INNER = cv.erode(PLATE, np.ones((9, 9), np.uint8))

# thresholds; see docs/LayerWatch_Report.md for how they were chosen (train split only)
T = {
    "streak_open_len": 41,  # px, opening with a 1 x L line keeps only long horizontal structure
    "streak_resp": 0.03,  # opened dark response, fraction of the fresh powder level
    "streak_min_len": 70,  # px, merged Hough length to call it a streak
    "streak_max_angle": 4.0,  # degrees from the recoat axis
    "short_feed_ratio": 0.85,  # local brightness relative to fresh powder
    "short_feed_min_area": 500,  # px
    "spatter_tophat": 0.16,  # local contrast (fraction of powder level)
    "spatter_min_blobs": 2,
    "spatter_link": 35,  # px, blobs closer than this belong to one ejection event
    "spatter_strong": 0.45,  # a single blob this bright counts on its own
    "se_tophat": 0.14,
    "se_min_area": 40,
    "se_abs": 1.15,  # a curled edge is brighter than fresh powder
    "glare_zone": 1.12,  # smoothed brightness above fresh powder: lamp reflection, not a defect
    "se_min_len": 14,
    "blur_ratio": 0.8,  # half scale Laplacian variance of the powder relative to the reference
    "blur_ratio_soft": 0.95,  # below this, a strongly directional texture also counts as blur
    "aniso_min": 0.6,  # structure tensor lambda_min / lambda_max
    "sat_frac": 0.0012,  # fraction of saturated pixels in the frame
    "ecc_min_cc": 0.6,
}


def lighting(ref_rect: np.ndarray) -> np.ndarray:
    """smooth lamp + vignette field from the reference powder, plate only"""
    f = ref_rect.astype(np.float32)
    m = (PLATE_INNER > 0).astype(np.float32)
    num = cv.GaussianBlur(f * m, (0, 0), 14)
    den = cv.GaussianBlur(m, (0, 0), 14)
    light = num / np.maximum(den, 1e-3)
    return np.maximum(light, 8.0)


def flat(img: np.ndarray, light: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / light


def to8(f: np.ndarray, lo: float = 0.0, hi: float = 2.0) -> np.ndarray:
    return np.clip((f - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- (b) slice


def slice_compare(melt_flat: np.ndarray, expected: np.ndarray) -> dict:
    """observed melt region vs the expected slice from the part function"""
    m8 = to8(melt_flat)
    # melted metal is darker than fresh powder: global Otsu on the flattened image
    vals = m8[PLATE_INNER > 0]
    otsu, _ = cv.threshold(vals.reshape(-1, 1), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu = min(otsu, 0.82 * 127.5)  # never call fresh powder metal on an empty-ish layer
    glob = (m8 < otsu).astype(np.uint8) * 255
    # plus local adaptive threshold for thin walls where the global level is not enough
    ad = cv.adaptiveThreshold(cv.GaussianBlur(m8, (3, 3), 0), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 18)
    obs = cv.bitwise_or(glob, ad)
    obs = cv.bitwise_and(obs, PLATE_INNER)
    obs = cv.morphologyEx(obs, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    obs = cv.morphologyEx(obs, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv.findContours(obs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(obs)
    for c in cnts:
        if cv.contourArea(c) >= 12:
            cv.drawContours(clean, [c], -1, 255, -1)
    obs = cv.bitwise_and(clean, obs)
    # compare only near the expected part: far away dark patches belong to short feed
    near = cv.dilate(expected, cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21)))
    obs_near = cv.bitwise_and(obs, near)
    e = expected > 0
    o = obs_near > 0
    inter = int((e & o).sum())
    union = int((e | o).sum())
    iou = inter / union if union else 1.0
    ec, _ = cv.findContours(expected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    oc, _ = cv.findContours(obs_near, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    shape = None
    if ec and oc:
        a = max(ec, key=cv.contourArea)
        b = max(oc, key=cv.contourArea)
        if cv.contourArea(a) > 30 and cv.contourArea(b) > 30:
            shape = float(cv.matchShapes(a, b, cv.CONTOURS_MATCH_I1, 0))
    return {"iou": round(iou, 4), "match_shapes": None if shape is None else round(shape, 4),
            "expected_px": int(e.sum()), "observed_px": int(o.sum()), "observed": obs_near}


# -------------------------------------------------------------- (c) streaks


def streaks(recoat_flat: np.ndarray, expected: np.ndarray) -> dict:
    """recoater streaks: long thin dark grooves along the recoat axis (+x)"""
    f = cv.GaussianBlur(recoat_flat, (0, 0), 0.8)
    dark = cv.GaussianBlur(f, (0, 0), 6) - f  # positive where darker than surroundings
    dark[PLATE_INNER == 0] = 0
    # directional filter: an opening with a long horizontal line keeps only structure
    # that is at least L px long along the recoat direction; isotropic noise vanishes
    L = T["streak_open_len"]
    line = cv.getStructuringElement(cv.MORPH_RECT, (L, 1))
    resp = cv.morphologyEx(dark, cv.MORPH_OPEN, line)
    # also a slightly tilted copy so streaks a few degrees off axis survive
    resp = np.maximum(resp, _open_tilted(dark, L, 2.0))
    resp = np.maximum(resp, _open_tilted(dark, L, -2.0))
    # noise level of the raw dark detail (for reporting contrast) and an absolute
    # floor on the opened response: after the opening, isotropic grain is gone, so
    # what survives above a few percent of the powder level is real line structure
    v = dark[PLATE_INNER > 0]
    sigma = 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-6
    binm = ((resp > T["streak_resp"]) * 255).astype(np.uint8)
    lines = cv.HoughLinesP(binm, 1, np.pi / 360, threshold=25, minLineLength=30, maxLineGap=14)
    segs = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 5 returns N x 4, 4.x returned N x 1 x 4
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            ang = (ang + 90) % 180 - 90
            if abs(ang) <= T["streak_max_angle"]:
                if x2 < x1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                segs.append([int(x1), int(y1), int(x2), int(y2), float(ang)])
    merged = _merge_segments(segs)
    exp_d = cv.dilate(expected, np.ones((3, 3), np.uint8))
    found = []
    for m in merged:
        length = m["x2"] - m["x1"]
        if length < T["streak_min_len"]:
            continue
        lm = np.zeros((S, S), np.uint8)
        cv.line(lm, (m["x1"], int(m["y1"])), (m["x2"], int(m["y2"])), 255, 3)
        inside = int(((lm > 0) & (exp_d > 0)).sum() / 3)
        contrast = float(resp[lm > 0].mean() / sigma)
        found.append({"x1": m["x1"], "y1": round(m["y1"], 1), "x2": m["x2"], "y2": round(m["y2"], 1),
                      "length_px": int(length), "angle_deg": round(m["angle"], 2), "inside_part_px": inside,
                      "contrast_sigma": round(contrast, 2)})
    found.sort(key=lambda d: -d["length_px"])
    return {"streaks": found, "count": len(found),
            "max_length_px": max([d["length_px"] for d in found], default=0),
            "inside_part_px": sum(d["inside_part_px"] for d in found), "response": resp, "binary": binm,
            "sigma": sigma}


def _open_tilted(img: np.ndarray, L: int, deg: float) -> np.ndarray:
    k = np.zeros((9, L), np.uint8)
    cv.line(k, (0, 4 - int(round(math.tan(math.radians(deg)) * L / 2))),
            (L - 1, 4 + int(round(math.tan(math.radians(deg)) * L / 2))), 1, 1)
    return cv.morphologyEx(img, cv.MORPH_OPEN, k)


def _merge_segments(segs: list) -> list:
    """join Hough segments that lie on the same streak (same row band, overlapping or close in x)"""
    segs = sorted(segs, key=lambda s: (s[1] + s[3]) / 2)
    groups: list[dict] = []
    for x1, y1, x2, y2, a in segs:
        yc = (y1 + y2) / 2
        for g in groups:
            if abs(g["yc"] - yc) <= 4 and x1 <= g["x2"] + 20 and x2 >= g["x1"] - 20:
                g["x1"], g["x2"] = min(g["x1"], x1), max(g["x2"], x2)
                if g["x1"] == x1:
                    g["y1"] = y1
                if g["x2"] == x2:
                    g["y2"] = y2
                g["yc"] = (g["yc"] * g["n"] + yc) / (g["n"] + 1)
                g["n"] += 1
                g["angle"] = (g["angle"] * (g["n"] - 1) + a) / g["n"]
                break
        else:
            groups.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "yc": yc, "n": 1, "angle": a})
    return groups


# ----------------------------------------------------------- short feed


def short_feed(recoat_flat: np.ndarray, expected_prev: np.ndarray | None = None) -> dict:
    """bare patches: the recoat left the far end of the bed without enough powder"""
    f = cv.GaussianBlur(recoat_flat, (0, 0), 3)
    med = float(np.median(f[PLATE_INNER > 0]))
    low = ((f < T["short_feed_ratio"] * med) & (PLATE_INNER > 0)).astype(np.uint8) * 255
    low = cv.morphologyEx(low, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, cent = cv.connectedComponentsWithStats(low)
    regs = []
    keep = np.zeros_like(low)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < T["short_feed_min_area"]:
            continue
        depth = float(1 - f[lab == i].mean() / med)
        regs.append({"bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)], "area_px": int(a), "depth": round(depth, 3)})
        keep[lab == i] = 255
    return {"regions": regs, "area_px": int(sum(r["area_px"] for r in regs)), "mask": keep}


# ------------------------------------------------ (d) spatter + super elevation


def _blob_detector() -> cv.SimpleBlobDetector:
    p = cv.SimpleBlobDetector_Params()
    p.filterByColor = True
    p.blobColor = 255
    p.minThreshold = 30
    p.maxThreshold = 200
    p.thresholdStep = 15
    p.filterByArea = True
    p.minArea = 3
    p.maxArea = 160
    p.filterByCircularity = True
    p.minCircularity = 0.35
    p.filterByConvexity = False
    p.filterByInertia = True
    p.minInertiaRatio = 0.2
    p.minDistBetweenBlobs = 2
    return cv.SimpleBlobDetector_create(p)


_BLOB = _blob_detector()


def bright_detail(melt_flat: np.ndarray) -> np.ndarray:
    """white top hat: small things brighter than their surroundings"""
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    return cv.morphologyEx(cv.GaussianBlur(melt_flat, (0, 0), 0.7), cv.MORPH_TOPHAT, k)


def spatter(melt_flat: np.ndarray, expected: np.ndarray, fg: np.ndarray | None, th: np.ndarray,
            glare: np.ndarray | None = None) -> dict:
    """ejected droplets on the powder: bright compact blobs outside the slice,
    confirmed against the running MOG2 background model of previous layers"""
    keep_out = cv.dilate(expected, cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))
    th8 = np.clip(th / 0.6 * 255, 0, 255).astype(np.uint8)
    kps = _BLOB.detect(th8)
    blobs = []
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if not (0 <= x < S and 0 <= y < S) or PLATE_INNER[y, x] == 0 or keep_out[y, x] > 0:
            continue
        if glare is not None and glare[y, x] > 0:
            continue
        r = max(1, int(kp.size / 2))
        c = float(th[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].max())
        new = True if fg is None else bool(fg[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].max() > 0)
        if c >= T["spatter_tophat"] and (new or c >= 2 * T["spatter_tophat"]):
            blobs.append({"x": x, "y": y, "r": round(kp.size / 2, 1), "contrast": round(c, 3), "mog2_new": new})
    # real spatter lands in clusters around the melt pool; lone bright grains do not count
    if blobs:
        P = np.array([[b["x"], b["y"]] for b in blobs], np.float32)
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        nb = (D < T["spatter_link"]).sum(1) - 1
        blobs = [b for b, k in zip(blobs, nb) if k >= 1 or b["contrast"] >= T["spatter_strong"]]
    exp_pts = np.column_stack(np.nonzero(expected))
    near = 0
    if len(blobs) and len(exp_pts):
        sub = exp_pts[:: max(1, len(exp_pts) // 2000)]
        for b in blobs:
            d = np.min(np.hypot(sub[:, 1] - b["x"], sub[:, 0] - b["y"]))
            near += int(d < 40)
    detected = len(blobs) >= T["spatter_min_blobs"]
    return {"blobs": blobs, "count": len(blobs), "near_part": near, "detected": detected}


def super_elevation(melt_flat: np.ndarray, expected: np.ndarray, th: np.ndarray, glare: np.ndarray | None = None) -> dict:
    """edge of the slice curling up: a bright elongated crescent on the contour"""
    band = cv.subtract(cv.dilate(expected, cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))),
                       cv.erode(expected, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))))
    sm = cv.GaussianBlur(melt_flat, (0, 0), 0.8)
    hot = ((th > T["se_tophat"]) & (sm > T["se_abs"]) & (band > 0)).astype(np.uint8) * 255
    if glare is not None:
        hot[glare > 0] = 0
    hot = cv.morphologyEx(hot, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv.findContours(hot, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    regs = []
    mask = np.zeros((S, S), np.uint8)
    for c in cnts:
        a = cv.contourArea(c)
        if a < T["se_min_area"]:
            continue
        (cx, cy), (w, h), ang = cv.minAreaRect(c)
        length = max(w, h)
        if length < T["se_min_len"]:
            continue
        x, y, bw, bh = cv.boundingRect(c)
        regs.append({"bbox": [int(x), int(y), int(x + bw - 1), int(y + bh - 1)], "area_px": int(a),
                     "arc_len_px": round(float(length), 1)})
        cv.drawContours(mask, [c], -1, 255, -1)
    return {"regions": regs, "max_arc_px": max([r["arc_len_px"] for r in regs], default=0.0), "mask": mask}


# ------------------------------------------------------- (f) evidence check


def lap_var(img: np.ndarray, region: np.ndarray) -> float:
    """Laplacian variance at half resolution. Full resolution mostly measures per
    pixel sensor grain, which resampling or a harmless 2 px shake already removes;
    one pyramid level down it measures the powder texture a real motion blur kills."""
    small = cv.pyrDown(img)
    reg = cv.erode(cv.resize(region, (small.shape[1], small.shape[0]), interpolation=cv.INTER_NEAREST),
                   np.ones((3, 3), np.uint8))
    lap = cv.Laplacian(small, cv.CV_32F, ksize=3)
    return float(lap[reg > 0].var())


def anisotropy(img: np.ndarray, region: np.ndarray) -> float:
    """lambda_min / lambda_max of the mean gradient structure tensor over the powder.
    Fresh powder is isotropic (close to 1); motion blur removes gradients along
    the motion direction only, so the ratio drops."""
    small = cv.pyrDown(img).astype(np.float32)
    reg = cv.resize(region, (small.shape[1], small.shape[0]), interpolation=cv.INTER_NEAREST) > 0
    gx = cv.Sobel(small, cv.CV_32F, 1, 0)
    gy = cv.Sobel(small, cv.CV_32F, 0, 1)
    a, b, c = float((gx * gx)[reg].mean()), float((gy * gy)[reg].mean()), float((gx * gy)[reg].mean())
    tr, d = a + b, math.sqrt((a - b) ** 2 + 4 * c * c)
    return (tr - d) / max(tr + d, 1e-9)


def glare_zone(melt_flat: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """broad regions brighter than fresh powder: a reflection of the lamp.
    Bright defect searches skip them (and a saturated one fails the evidence check)."""
    f = melt_flat.copy()
    f[cv.dilate(expected, np.ones((9, 9), np.uint8)) > 0] = 1.0
    g = cv.GaussianBlur(f, (0, 0), 10)
    z = ((g > T["glare_zone"]) & (PLATE > 0)).astype(np.uint8) * 255
    return cv.dilate(z, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))


def evidence(cam_img: np.ndarray, rect_img: np.ndarray, expected: np.ndarray, ref_lap: float, ecc_cc: float) -> dict:
    """is this frame good enough to decide on? blur, glare and registration"""
    powder = cv.subtract(PLATE_INNER, cv.dilate(expected, np.ones((9, 9), np.uint8)))
    lv = lap_var(rect_img, powder)
    ratio = lv / max(ref_lap, 1e-6)
    an = anisotropy(rect_img, powder)
    sat = float((cam_img >= 250).mean())
    reasons = []
    if ratio < T["blur_ratio"]:
        reasons.append(f"blurred: Laplacian variance {lv:.0f} is {ratio:.2f} of the reference (min {T['blur_ratio']})")
    elif ratio < T["blur_ratio_soft"] and an < T["aniso_min"]:
        reasons.append(f"motion blur: texture {ratio:.2f} of the reference and directional (gradient anisotropy {an:.2f} < {T['aniso_min']})")
    if sat > T["sat_frac"]:
        reasons.append(f"glare: {sat * 100:.2f}% of pixels saturated (max {T['sat_frac'] * 100:.2f}%)")
    if ecc_cc < T["ecc_min_cc"]:
        reasons.append(f"registration weak: ECC correlation {ecc_cc:.2f} (min {T['ecc_min_cc']})")
    return {"lap_var": round(lv, 1), "lap_ratio": round(ratio, 3), "anisotropy": round(an, 3),
            "saturated_frac": round(sat, 5), "ecc_cc": round(ecc_cc, 4), "ok": not reasons, "reasons": reasons}
