"""The demo build: 480 layers of the thrust chamber on one simulated machine,
with a scripted set of injected events, run through the real handler.

    python -m layerwatch.demo.run_build

The "printer" here is the synthetic renderer. It reacts to the agent the way a
machine would: RECOAT spreads powder again (transient problems clear,
blade damage does not), RE_IMAGE takes the frame again (the glare or shake is
gone), STOP pauses the build until a scripted operator replaces the blade.
Writes dashboard/public/run/run.json plus frames and per stage overlays.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from layerwatch import part
from layerwatch.agent.incident import bedrock_note
from layerwatch.synth.render import S, Build, LayerSpec, render_layer, render_reference
from server import handler
from server.store import LocalStore

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dashboard" / "public" / "run"
BUILD_ID = "demo-thrust-chamber"
SEED = 4242
N = part.DEMO_LAYERS
Y_STREAK = S // 2 + 22

# layer: list of (defects or nuisances) as they appear on attempt 0; "persist" survives a recoat
SCRIPT: dict[int, dict] = {
    18: {"defects": [{"type": "spatter", "cx": 300.0, "cy": 186.0, "n": 11, "spread": 9.0, "strength": 1.0}]},
    46: {"glare": {"cx": 330.0, "cy": 300.0, "rx": 60.0, "ry": 35.0, "peak": 0.2, "angle": 30.0}, "glare_on": "melt"},
    97: {"defects": [{"type": "short_feed", "x0": 360.0, "y0": 120.0, "y1": 300.0, "drop": 0.16}], "clears_after": 1},
    150: {"glare": {"cx": 240.0, "cy": 230.0, "rx": 62.0, "ry": 38.0, "peak": 1.15, "angle": 25.0}, "glare_on": "melt",
          "clears_after_reimage": 1},
    171: {"defects": [{"type": "super_elevation", "arc": 34.0, "width": 3.0, "bright": 0.32}]},
    190: {"blur": {"length": 11.0, "angle": 12.0}, "blur_on": "recoat", "clears_after_reimage": 1},
    205: {"defects": [{"type": "streak", "y": 70.0, "x0": 40.0, "x1": 120.0, "width": 1.6, "depth": 0.12, "angle": 0.2}]},
    212: {"defects": [{"type": "streak", "y": float(Y_STREAK), "x0": 48.0, "x1": 470.0, "width": 2.4, "depth": 0.17, "angle": 0.25},
                      {"type": "streak", "y": float(Y_STREAK + 9), "x0": 150.0, "x1": 430.0, "width": 1.4, "depth": 0.1, "angle": 0.25}],
          "persist": True},
    260: {"defects": [{"type": "spatter", "cx": 300.0, "cy": 236.0, "n": 26, "spread": 16.0, "strength": 1.05}]},
    300: {"defects": [{"type": "super_elevation", "arc": 40.0, "width": 3.0, "bright": 0.3}]},
    355: {"defects": [{"type": "short_feed", "x0": 330.0, "y0": 300.0, "y1": 470.0, "drop": 0.18}], "clears_after": 2},
    420: {"defects": [{"type": "streak", "y": 440.0, "x0": 360.0, "x1": 450.0, "width": 1.4, "depth": 0.12, "angle": -0.3}]},
    455: {"defects": [{"type": "spatter", "cx": 230.0, "cy": 300.0, "n": 9, "spread": 8.0, "strength": 1.0}]},
}


def spec_for(i: int, attempt: int, recoats: int, reimages: int, blade_ok: bool, rng: np.random.Generator) -> tuple[LayerSpec, list[str]]:
    h = part.layer_height(i)
    s = SCRIPT.get(i, {})
    spec = LayerSpec(i, h)
    injected: list[str] = []
    defects = [dict(d) for d in s.get("defects", [])]
    if defects:
        if s.get("persist"):
            keep = not blade_ok  # blade damage: a recoat does not help, a new blade does
        else:
            keep = recoats < s.get("clears_after", 99)  # transient: clears after n recoats
        if keep:
            spec.defects = defects
            injected += [d["type"] for d in defects]
    if "glare" in s and reimages < s.get("clears_after_reimage", 99):
        spec.glare, spec.glare_on = dict(s["glare"]), s.get("glare_on", "melt")
        injected.append("glare" if s["glare"]["peak"] >= 0.6 else "glare (mild)")
    if "blur" in s and reimages < s.get("clears_after_reimage", 99):
        spec.blur, spec.blur_on = dict(s["blur"]), s.get("blur_on", "melt")
        injected.append("motion blur")
    return spec, injected


def main() -> None:
    t0 = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "img").mkdir(parents=True)
    (OUT / "ov").mkdir(parents=True)
    store_root = ROOT / "data" / "demo_store"
    if store_root.exists():
        shutil.rmtree(store_root)
    store = LocalStore(store_root)
    build = Build(SEED)
    ref = render_reference(build)
    ok, ref_jpg = cv.imencode(".jpg", ref, [cv.IMWRITE_JPEG_QUALITY, 95])
    handler._WORKERS.clear()
    handler.register_build(store, BUILD_ID, build.cam.to_dict(), ref_jpg.tobytes(), N)
    cv.imwrite(str(OUT / "img" / "ref.jpg"), ref, [cv.IMWRITE_JPEG_QUALITY, 85])

    layers = []
    stops = []
    blade_ok = False
    for i in range(N):
        rng = np.random.default_rng(SEED * 1000 + i)
        attempt, recoats, reimages = 0, 0, 0
        attempts = []
        while True:
            spec, injected = spec_for(i, attempt, recoats, reimages, blade_ok, rng)
            r = render_layer(build, spec, rng)
            rj = cv.imencode(".jpg", r["recoat"], [cv.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            mj = cv.imencode(".jpg", r["melt"], [cv.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            key = SCRIPT.get(i) is not None or i % 24 == 0 or i == N - 1
            rec = handler.process_layer(store, BUILD_ID, i, attempt, recoats, rj, mj, overlays=key)
            ov = rec.pop("_overlays", None)
            a = rec["decision"]["action"]
            stem = f"{i:03d}_{attempt}"
            cv.imwrite(str(OUT / "img" / f"{stem}_recoat.jpg"), r["recoat"], [cv.IMWRITE_JPEG_QUALITY, 80])
            cv.imwrite(str(OUT / "img" / f"{stem}_melt.jpg"), r["melt"], [cv.IMWRITE_JPEG_QUALITY, 80])
            if ov:
                for k, v in ov.items():
                    cv.imwrite(str(OUT / "ov" / f"{stem}_{k}.jpg"), v, [cv.IMWRITE_JPEG_QUALITY, 82])
            entry = {
                "attempt": attempt, "recoats": recoats, "action": a, "reason": rec["decision"]["reason"],
                "severity": rec["decision"]["severity"], "numbers": rec["decision"]["evidence"],
                "injected": injected, "detections": rec["detections"], "evidence": rec["evidence"],
                "evidence_numbers": rec["evidence_numbers"], "slice": rec["slice"], "ecc_cc": rec["registration"]["ecc_cc"],
                "streaks": rec["streaks"], "timing_ms": rec["timing_ms"], "overlays": bool(ov),
            }
            attempts.append(entry)
            if a == "RECOAT":
                recoats += 1
                attempt += 1
                continue
            if a == "RE_IMAGE":
                reimages += 1
                attempt += 1
                continue
            if a == "STOP":
                note, src = bedrock_note({"layer_index": i, "h_m": rec["h_m"], "decision": rec["decision"],
                                          "detections": rec["detections"], "attempts": attempt + 1})
                stops.append({"layer": i, "reason": rec["decision"]["reason"], "numbers": rec["decision"]["evidence"],
                              "note": note, "note_source": src})
                entry["note"] = note
                entry["note_source"] = src
                # operator: replaces the recoater blade, resumes the build on this layer
                blade_ok = True
                attempts.append({"attempt": attempt, "action": "OPERATOR",
                                 "reason": "operator replaced the recoater blade and resumed the build"})
                handler._WORKERS[BUILD_ID][1].history.clear()
                attempt += 1
                recoats = 0
                continue
            break
        final = attempts[-1]
        layers.append({"i": i, "h_m": round(part.layer_height(i), 6), "final": final["action"],
                       "worst": max((x["action"] for x in attempts), key=lambda a: {"CONTINUE": 0, "OPERATOR": 0, "RE_IMAGE": 1, "RECOAT": 2, "STOP": 3}[a]),
                       "attempts": attempts})
        if i % 40 == 0:
            print(f"layer {i} {final['action']} {time.time() - t0:.0f}s", flush=True)

    # points for the 3D defect map: every final detection at its height and bed position
    points = []
    for L in layers:
        for att in L["attempts"]:
            for d in att.get("detections", []):
                points.append({"layer": L["i"], "h_m": L["h_m"], "type": d["type"], "xz_m": d["plate_xz_m"],
                               "bbox": d["bbox"], "action": att["action"], "source": d.get("source", "rules"),
                               "inside_part_px": d.get("inside_part_px", 0)})
    lat = [a["timing_ms"]["total"] for L in layers for a in L["attempts"] if "timing_ms" in a]
    acts: dict[str, int] = {}
    for L in layers:
        for a in L["attempts"]:
            acts[a["action"]] = acts.get(a["action"], 0) + 1
    run = {
        "build_id": BUILD_ID, "n_layers": N, "part_height_m": part.HPIPE, "frame_half_m": part.FRAME_HALF,
        "plate_half_m": part.PLATE_HALF, "image_size": S, "layers": layers, "points": points, "stops": stops,
        "actions": acts, "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                                        "n": len(lat)},
        "generated_s": round(time.time() - t0, 1),
    }
    with open(OUT / "run.json", "w") as f:
        json.dump(run, f, separators=(",", ":"))
    print(json.dumps({k: run[k] for k in ("actions", "latency_ms", "generated_s")}), flush=True)
    for s in stops:
        print("STOP", s["layer"], s["reason"], "|", s["note_source"], "|", s["note"])


if __name__ == "__main__":
    main()
