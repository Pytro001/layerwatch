"""The LayerWatch pipeline for one build: two camera frames per layer in,
measurements (and optionally an overlay per stage) out.

    lw = LayerWatch(camera, reference_image, dnn="models/layerwatch_unet.onnx")
    res = lw.process(recoat_img, melt_img, h=0.12, layer_index=150)

`res["detections"]` is the fused defect list the agent decides on; every stage
also reports its raw numbers so the decision can always be traced back.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2 as cv
import numpy as np

from layerwatch import part
from layerwatch.vision import register as reg
from layerwatch.vision import stages as st
from layerwatch.vision.dnn import Segmenter, make_input

S = 512
DEFAULT_ONNX = Path(__file__).resolve().parents[2] / "models" / "layerwatch_unet.onnx"


_SLICES: dict[float, np.ndarray] = {}


def expected_slice(h: float) -> np.ndarray:
    """the expected slice from the part function (a real machine has it from the job file);
    cached because the same heights repeat across builds"""
    k = round(h, 6)
    if k not in _SLICES:
        if len(_SLICES) > 1000:
            _SLICES.clear()
        _SLICES[k] = part.slice_mask(h, S)
    return _SLICES[k]


class LayerWatch:
    def __init__(self, camera: dict, reference: np.ndarray, dnn: str | Path | None = DEFAULT_ONNX,
                 mode: str = "fused"):
        """camera: {"K", "dist", "H"} from calibration. reference: layer 0 camera frame.
        mode: "rules", "dnn" or "fused" (rules plus DNN)."""
        cv.setNumThreads(2)
        self.K = np.array(camera["K"], np.float64)
        self.dist = np.array(camera["dist"], np.float64)
        self.H = np.array(camera["H"], np.float64)
        self.maps = reg.rectify_maps(self.K, self.dist, self.H)
        self.ref = reg.rectify(reference, self.maps)
        self.light = st.lighting(self.ref)
        # the reference goes through the same sub pixel resampling as every registered frame
        shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], np.float32)
        self.ref_lap = st.lap_var(reg.apply_warp(self.ref, shift), st.PLATE_INNER)
        self.mog = cv.createBackgroundSubtractorMOG2(history=12, varThreshold=20, detectShadows=False)
        self.mode = mode
        self.seg = None
        if mode != "rules" and dnn is not None and Path(dnn).exists():
            self.seg = Segmenter(dnn)
        self.n_seen = 0

    def process(self, recoat: np.ndarray, melt: np.ndarray, h: float, layer_index: int = 0,
                overlays: bool = False, update_background: bool = True) -> dict:
        t = {}
        t0 = time.perf_counter()
        expected = expected_slice(h)
        t["slice_render"] = time.perf_counter() - t0

        # (a) registration
        t1 = time.perf_counter()
        r_rect = reg.rectify(recoat, self.maps)
        m_rect = reg.rectify(melt, self.maps)
        rr = reg.register(m_rect, self.ref)
        W = rr["W"]
        m_al = rr["aligned"]
        r_al = reg.apply_warp(r_rect, W)
        t["registration"] = time.perf_counter() - t1

        # (f) evidence first: if the frame is bad nothing below is trusted
        t1 = time.perf_counter()
        ev_m = st.evidence(melt, m_al, expected, self.ref_lap, rr["cc"])
        ev_r = st.evidence(recoat, r_al, expected, self.ref_lap, rr["cc"])
        evidence = {"melt": ev_m, "recoat": ev_r, "ok": ev_m["ok"] and ev_r["ok"],
                    "reasons": [f"melt frame {x}" for x in ev_m["reasons"]] + [f"recoat frame {x}" for x in ev_r["reasons"]]}
        t["evidence"] = time.perf_counter() - t1

        rf = st.flat(r_al, self.light)
        mf = st.flat(m_al, self.light)

        # (b) slice compare
        t1 = time.perf_counter()
        sc = st.slice_compare(mf, expected)
        t["slice_compare"] = time.perf_counter() - t1

        # (c) streaks and short feed on the recoat frame
        t1 = time.perf_counter()
        sk = st.streaks(rf, expected)
        sf = st.short_feed(rf)
        st.drop_streaks_on_patch_edges(sk, sf["mask"])
        t["streaks"] = time.perf_counter() - t1

        # (d) spatter and super elevation on the melt frame, MOG2 over previous layers
        t1 = time.perf_counter()
        th = st.bright_detail(mf)
        m8 = st.to8(mf)
        fg = self.mog.apply(m8, learningRate=0) if self.n_seen >= 3 else None
        gz = st.glare_zone(mf, expected)
        sp = st.spatter(mf, expected, fg, th, gz)
        se = st.super_elevation(mf, expected, th, gz)
        sp["glare_zone_px"] = int((gz > 0).sum())
        if update_background and evidence["ok"]:
            # learn only the powder: paint the slice with the median powder level first
            bg = m8.copy()
            bg[cv.dilate(expected, np.ones((7, 7), np.uint8)) > 0] = int(np.median(m8[st.PLATE_INNER > 0]))
            self.mog.apply(bg, learningRate=0.25)
            self.n_seen += 1
        t["spatter_super_elevation"] = time.perf_counter() - t1

        # (e) DNN through cv.dnn
        dnn_found: list[dict] = []
        cls = None
        if self.seg is not None:
            t1 = time.perf_counter()
            logits = self.seg(make_input(rf, mf, expected))
            cls, dnn_found = Segmenter.components(logits, st.PLATE_INNER)
            # a lamp reflection is not a bright defect, for the network either
            dnn_found = [d for d in dnn_found if d["type"] not in ("spatter", "super_elevation")
                         or gz[(d["bbox"][1] + d["bbox"][3]) // 2, (d["bbox"][0] + d["bbox"][2]) // 2] == 0]
            t["dnn"] = time.perf_counter() - t1

        rules_found = self._rules_detections(sk, sf, sp, se)
        det = self._fuse(rules_found, dnn_found, expected)
        t["total"] = time.perf_counter() - t0

        out = {
            "layer_index": layer_index, "h_m": round(h, 6),
            "registration": {"W": np.round(W, 6).tolist(), "ecc_cc": round(rr["cc"], 4), "ok": rr["ok"]},
            "evidence": evidence,
            "slice": {k: v for k, v in sc.items() if k != "observed"},
            "streaks": {k: v for k, v in sk.items() if k not in ("response", "binary")},
            "short_feed": {k: v for k, v in sf.items() if k != "mask"},
            "spatter": sp,
            "super_elevation": {k: v for k, v in se.items() if k != "mask"},
            "dnn": dnn_found,
            "rules": rules_found,
            "detections": det,
            "timing_ms": {k: round(v * 1000, 2) for k, v in t.items()},
        }
        if overlays:
            from layerwatch.vision.overlay import stage_overlays
            out["_overlays"] = stage_overlays(r_al, m_al, self.ref, expected, sc, sk, sf, sp, se, cls, evidence, W, rr)
        self._last = {"recoat": r_al, "melt": m_al, "expected": expected, "cls": cls}
        return out

    def warm(self, melt: np.ndarray, h: float) -> None:
        """feed a previous layer into the MOG2 background (Lambda cold start)"""
        m_al = reg.register(reg.rectify(melt, self.maps), self.ref)["aligned"]
        m8 = st.to8(st.flat(m_al, self.light))
        expected = expected_slice(h)
        m8[cv.dilate(expected, np.ones((7, 7), np.uint8)) > 0] = int(np.median(m8[st.PLATE_INNER > 0]))
        self.mog.apply(m8, learningRate=0.25)
        self.n_seen += 1

    # ------------------------------------------------------------------

    @staticmethod
    def _rules_detections(sk: dict, sf: dict, sp: dict, se: dict) -> list[dict]:
        out = []
        for s in sk["streaks"]:
            out.append({"type": "streak", "bbox": [s["x1"], int(min(s["y1"], s["y2"])) - 2, s["x2"], int(max(s["y1"], s["y2"])) + 2],
                        "length_px": s["length_px"], "inside_part_px": s["inside_part_px"], "source": "rules"})
        for r in sf["regions"]:
            out.append({"type": "short_feed", "bbox": r["bbox"], "area_px": r["area_px"], "source": "rules"})
        if sp["detected"]:
            xs = [b["x"] for b in sp["blobs"]]
            ys = [b["y"] for b in sp["blobs"]]
            out.append({"type": "spatter", "bbox": [min(xs), min(ys), max(xs), max(ys)], "count": sp["count"],
                        "near_part": sp["near_part"], "source": "rules"})
        for r in se["regions"]:
            out.append({"type": "super_elevation", "bbox": r["bbox"], "area_px": r["area_px"], "arc_len_px": r["arc_len_px"],
                        "source": "rules"})
        return out

    def _fuse(self, rules: list[dict], dnn: list[dict], expected: np.ndarray) -> list[dict]:
        if self.mode == "rules" or self.seg is None:
            base = rules
        elif self.mode == "dnn":
            base = [dict(d, source="dnn") for d in dnn]
        else:
            # rules plus DNN: keep every rule detection, mark the ones the DNN agrees
            # with, and add DNN only finds for classes where it is the stronger detector
            base = []
            for r in rules:
                agree = any(d["type"] == r["type"] and _overlap(d["bbox"], r["bbox"], 6) for d in dnn)
                base.append(dict(r, source="rules+dnn" if agree else "rules"))
            for d in dnn:
                if any(r["type"] == d["type"] and _overlap(d["bbox"], r["bbox"], 6) for r in rules):
                    continue
                base.append(dict(d, source="dnn"))
        out = []
        for d in base:
            x0, y0, x1, y1 = d["bbox"]
            box = np.zeros((S, S), np.uint8)
            box[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 1
            d = dict(d)
            d.setdefault("inside_part_px", int((box & (expected > 0)).sum()))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            px, pz = part.px_to_plate(cx, cy, S)
            d["plate_xz_m"] = [round(px, 4), round(pz, 4)]
            out.append(d)
        return out


def _overlap(a: list, b: list, pad: int = 0) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])
