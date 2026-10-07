"""The LayerWatch agent: a transparent policy over the vision measurements.

It never looks at pixels. It reads the numbers the OpenCV pipeline measured
for this layer, the attempt count for this layer and the history of the last
N layers, and returns one action with a reason string and the exact numbers
that triggered it:

    CONTINUE   print the next layer
    RECOAT     spread powder again over this layer, then re-check
    RE_IMAGE   take the picture again (evidence too poor to decide)
    STOP       pause the build and call a person

Order of checks: evidence first (never guess on a bad frame), then defects by
severity, then trends. Every threshold lives in POLICY below.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

POLICY = {
    "max_reimage": 2,  # re-image attempts before giving up on the camera
    "max_recoat": 2,  # recoat attempts per layer
    "streak_critical_len": 150,  # px (0.86 mm/px): long enough to starve the next layers
    "streak_part_px": 10,  # streak crosses the part by at least this many px
    "streak_trend": 3,  # streak layers in the window that mean the blade is worn
    "short_feed_area": 500,  # px of bare bed
    "se_stop_arc": 60,  # px of curled edge: risk of a recoater crash
    "se_trend": 2,  # consecutive super elevation layers: stop
    "spatter_heavy": 25,  # blobs on one layer worth a note
    "slice_iou_min": 0.55,  # melted region must match the expected slice
    "slice_min_px": 400,  # only judge IoU on slices big enough to measure
    "window": 10,
}

CONTINUE, RECOAT, RE_IMAGE, STOP = "CONTINUE", "RECOAT", "RE_IMAGE", "STOP"
SEVERITY = {CONTINUE: 0, RE_IMAGE: 1, RECOAT: 2, STOP: 3}


@dataclass
class Decision:
    action: str
    reason: str
    evidence: dict
    severity: str = "info"  # info, warn, critical
    watch: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason, "evidence": self.evidence, "severity": self.severity,
                "watch": self.watch}


class Agent:
    def __init__(self, policy: dict | None = None):
        self.p = dict(POLICY, **(policy or {}))
        self.history: deque[dict] = deque(maxlen=self.p["window"])

    def decide(self, m: dict, attempt: int = 0, recoats: int = 0) -> Decision:
        """m: pipeline result for this layer. attempt: how many times this layer
        was already re-imaged, recoats: how many times it was recoated."""
        p = self.p
        ev = m["evidence"]
        if not ev["ok"]:
            if attempt >= p["max_reimage"]:
                return Decision(STOP, f"camera evidence still unusable after {attempt} re-images: " + "; ".join(ev["reasons"]),
                                {"reasons": ev["reasons"], "attempt": attempt}, "critical")
            return Decision(RE_IMAGE, "frame not good enough to judge, taking it again: " + "; ".join(ev["reasons"]),
                            {"reasons": ev["reasons"], "lap_ratio": min(ev["melt"]["lap_ratio"], ev["recoat"]["lap_ratio"]),
                             "saturated_frac": max(ev["melt"]["saturated_frac"], ev["recoat"]["saturated_frac"]),
                             "attempt": attempt}, "warn")

        det = m["detections"]
        streaks = [d for d in det if d["type"] == "streak"]
        feeds = [d for d in det if d["type"] == "short_feed"]
        ses = [d for d in det if d["type"] == "super_elevation"]
        spats = [d for d in det if d["type"] == "spatter"]
        watch: list[str] = []

        def bbox_len(d: dict) -> int:
            return int(d.get("length_px", d["bbox"][2] - d["bbox"][0]))

        # slice mismatch: the laser did not melt what the part file says
        sl = m["slice"]
        if sl["expected_px"] >= p["slice_min_px"] and sl["iou"] < p["slice_iou_min"]:
            return Decision(STOP, f"melted region does not match the slice: IoU {sl['iou']:.2f} < {p['slice_iou_min']}",
                            {"iou": sl["iou"], "match_shapes": sl["match_shapes"], "expected_px": sl["expected_px"],
                             "observed_px": sl["observed_px"]}, "critical")

        # super elevation: an edge curling up can hit the recoater blade
        if ses:
            arc = max(d.get("arc_len_px", bbox_len(d)) for d in ses)
            run = 1 + self._consecutive("super_elevation")
            if arc >= p["se_stop_arc"] or run >= p["se_trend"]:
                return Decision(STOP, f"super elevation on the part edge, arc {arc:.0f} px over {run} layer(s): recoater crash risk",
                                {"arc_len_px": arc, "consecutive_layers": run, "regions": len(ses)}, "critical")
            watch.append(f"super elevation arc {arc:.0f} px, watching the next layer")

        # recoater streaks
        if streaks:
            longest = max(bbox_len(d) for d in streaks)
            inside = sum(int(d.get("inside_part_px", 0)) for d in streaks)
            trend = 1 + self._count("streak")
            critical = longest >= p["streak_critical_len"] or inside >= p["streak_part_px"]
            nums = {"longest_px": longest, "inside_part_px": inside, "streaks": len(streaks),
                    "streak_layers_in_window": trend, "recoats": recoats}
            if critical and recoats >= 1:
                return Decision(STOP, f"recoater streak {longest} px survives {recoats} recoat(s) and crosses the part by {inside} px: blade damage likely",
                                nums, "critical")
            if trend >= p["streak_trend"] and critical:
                return Decision(STOP, f"streaks on {trend} of the last {p['window']} layers, latest {longest} px: blade wear",
                                nums, "critical")
            if critical:
                return Decision(RECOAT, f"recoater streak {longest} px, {inside} px across the part: recoating this layer",
                                nums, "warn")
            watch.append(f"short streak {longest} px outside the part")

        # short feed
        if feeds:
            area = sum(int(d.get("area_px", 0)) for d in feeds)
            if area >= p["short_feed_area"]:
                if recoats >= p["max_recoat"]:
                    return Decision(STOP, f"short feed {area} px still bare after {recoats} recoats: powder supply problem",
                                    {"area_px": area, "recoats": recoats}, "critical")
                return Decision(RECOAT, f"short feed: {area} px of the bed without fresh powder, recoating with more dose",
                                {"area_px": area, "regions": len(feeds), "recoats": recoats}, "warn")

        if spats:
            n = max(int(d.get("count", 0)) for d in spats)
            watch.append(f"spatter, {n} blobs" + (" (heavy, flag for CT)" if n >= p["spatter_heavy"] else ""))

        if watch:
            return Decision(CONTINUE, "continue; noted: " + "; ".join(watch), {"notes": watch}, "info", watch)
        return Decision(CONTINUE, "layer clean", {}, "info")

    def commit(self, m: dict, d: Decision) -> None:
        """record the final measurement of a layer (after any recoat or re-image)"""
        types = sorted({x["type"] for x in m["detections"]}) if m["evidence"]["ok"] else []
        self.history.append({"layer": m.get("layer_index"), "types": types, "action": d.action})

    def _count(self, t: str) -> int:
        return sum(1 for h in self.history if t in h["types"])

    def _consecutive(self, t: str) -> int:
        n = 0
        for h in reversed(self.history):
            if t in h["types"]:
                n += 1
            else:
                break
        return n
