"""Evaluation on the synthetic test split (400 layers, 10 builds never used in
training or threshold tuning).

    python -m layerwatch.eval.run_eval            # writes docs/eval/synthetic.json

Per layer it records the ground truth (injected defect types, whether the frame
is unusable, whether the layer needs an intervention), what the rules, the DNN
and rules plus DNN detect, the agent decision, registration error and timing.
Layer level metrics: a defect type counts as detected on a layer if the
detector reports at least one region of that type; layers whose frame is
unusable (strong glare or motion blur) are scored by the evidence gate only.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from layerwatch.agent.policy import CONTINUE, RE_IMAGE, STOP, Agent
from layerwatch.vision.pipeline import DEFAULT_ONNX, LayerWatch
from layerwatch.vision.register import corner_error

ROOT = Path(__file__).resolve().parents[2]
TYPES = ["streak", "spatter", "super_elevation", "short_feed"]


def gt_bad(s: dict) -> bool:
    return any(n["bad"] for n in s["nuisances"])


def gt_critical(s: dict) -> list[str]:
    """the label side of 'this layer needs an intervention', from injected
    geometry only (never from detector output)"""
    out = []
    for d in s["defects"]:
        w = d["bbox"][2] - d["bbox"][0]
        if d["type"] == "streak" and (w >= 150 or d["inside_part_px"] >= 10):
            out.append("streak")
        elif d["type"] == "short_feed" and d["area_px"] >= 500:
            out.append("short_feed")
        elif d["type"] == "super_elevation" and d["area_px"] >= 250:
            out.append("super_elevation")
    return sorted(set(out))


def dnn_pixel_stats(cls: np.ndarray | None, mask: np.ndarray, acc: dict) -> None:
    if cls is None:
        return
    for k in range(1, 5):
        p, g = cls == k, mask == k
        acc.setdefault(k, [0, 0])
        acc[k][0] += int((p & g).sum())
        acc[k][1] += int((p | g).sum())


def main(split: str = "test") -> dict:
    data = ROOT / "data" / "synth"
    meta = json.load(open(data / split / "labels.json"))
    rows = []
    pix: dict = {}
    t0 = time.time()
    for b in meta["builds"]:
        lw = LayerWatch(b["camera"], cv.imread(str(data / b["reference"]), 0), dnn=DEFAULT_ONNX, mode="fused")
        agent = Agent()
        for s in b["samples"]:
            r = lw.process(cv.imread(str(data / s["recoat"]), 0), cv.imread(str(data / s["melt"]), 0), s["h_m"],
                           s["layer_index"])
            d = agent.decide(r)
            agent.commit(r, d)
            mask = cv.imread(str(data / s["mask"]), 0)
            if r["evidence"]["ok"] and not gt_bad(s):
                dnn_pixel_stats(lw._last["cls"], mask, pix)
            rows.append({
                "id": s["id"], "gt": sorted({x["type"] for x in s["defects"]}), "gt_bad": gt_bad(s),
                "gt_critical": gt_critical(s), "nuisances": [n["type"] + (":bad" if n["bad"] else ":mild") for n in s["nuisances"]],
                "rules": sorted({x["type"] for x in r["rules"]}), "dnn": sorted({x["type"] for x in r["dnn"]}),
                "fused": sorted({x["type"] for x in r["detections"]}), "evidence_ok": r["evidence"]["ok"],
                "action": d.action, "reason": d.reason, "reg_err_px": corner_error(r["registration"]["W"], s["jitter_H"]),
                "ecc_cc": r["registration"]["ecc_cc"], "iou": r["slice"]["iou"], "expected_px": r["slice"]["expected_px"],
                "timing_ms": r["timing_ms"],
            })
        print(f"{b['build']} done, {time.time() - t0:.0f}s", flush=True)
    res = score(rows, pix)
    res["meta"] = {"split": split, "n_layers": len(rows), "builds": len(meta["builds"]), "machine": platform.machine(),
                   "platform": platform.platform(), "opencv": cv.__version__, "onnx": DEFAULT_ONNX.name,
                   "wall_s": round(time.time() - t0, 1)}
    out = ROOT / "docs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(out / f"synthetic_{split}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def score(rows: list[dict], pix: dict) -> dict:
    good = [r for r in rows if not r["gt_bad"]]
    det = {}
    for mode in ("rules", "dnn", "fused"):
        det[mode] = {}
        for t in TYPES:
            tp = sum(1 for r in good if t in r["gt"] and t in r[mode])
            fp = sum(1 for r in good if t not in r["gt"] and t in r[mode])
            fn = sum(1 for r in good if t in r["gt"] and t not in r[mode])
            det[mode][t] = prf(tp, fp, fn)
        mt = [det[mode][t]["f1"] for t in TYPES]
        det[mode]["macro_f1"] = round(float(np.mean(mt)), 3)
    # evidence gate
    ev_tp = sum(1 for r in rows if r["gt_bad"] and not r["evidence_ok"])
    ev_fp = sum(1 for r in rows if not r["gt_bad"] and not r["evidence_ok"])
    ev_fn = sum(1 for r in rows if r["gt_bad"] and r["evidence_ok"])
    mild = [r for r in rows if any(n.endswith(":mild") for n in r["nuisances"]) and not r["gt_bad"]]
    # agent
    n = len(rows)
    needs = [r for r in rows if not r["gt_bad"] and r["gt_critical"]]
    fine = [r for r in rows if not r["gt_bad"] and not r["gt_critical"]]
    missed = [r for r in needs if r["action"] == CONTINUE]
    false_stop = [r for r in fine if r["action"] == STOP]
    false_int = [r for r in fine if r["action"] != CONTINUE]
    guessed_on_bad = [r for r in rows if r["gt_bad"] and r["action"] != RE_IMAGE]
    t_all = {k: [r["timing_ms"][k] for r in rows if k in r["timing_ms"]] for k in rows[0]["timing_ms"]}
    lat = {k: {"median": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)} for k, v in t_all.items()}
    reg = np.array([r["reg_err_px"] for r in rows if not r["gt_bad"]])
    big = [r for r in rows if r["expected_px"] >= 400 and not r["gt_bad"]]
    return {
        "detection": det,
        "dnn_pixel_iou": {["streak", "spatter", "super_elevation", "short_feed"][k - 1]: round(v[0] / max(v[1], 1), 3) for k, v in sorted(pix.items())},
        "evidence_gate": dict(prf(ev_tp, ev_fp, ev_fn), mild_nuisance_layers=len(mild),
                              mild_nuisance_flagged=sum(1 for r in mild if not r["evidence_ok"])),
        "agent": {
            "layers": n, "layers_needing_intervention": len(needs), "missed_critical": len(missed),
            "missed_critical_ids": [r["id"] for r in missed],
            "false_stops": len(false_stop), "false_stop_per_1000": round(1000 * len(false_stop) / max(len(fine), 1), 1),
            "false_interventions": len(false_int), "false_intervention_per_1000": round(1000 * len(false_int) / max(len(fine), 1), 1),
            "bad_frames": sum(r["gt_bad"] for r in rows), "decided_on_bad_frame": len(guessed_on_bad),
            "actions": {a: sum(1 for r in rows if r["action"] == a) for a in ("CONTINUE", "RECOAT", "RE_IMAGE", "STOP")},
        },
        "registration_px": {"median": round(float(np.median(reg)), 3), "p95": round(float(np.percentile(reg, 95)), 3),
                            "max": round(float(reg.max()), 3)},
        "slice_iou": {"median": round(float(np.median([r["iou"] for r in big])), 3),
                      "p5": round(float(np.percentile([r["iou"] for r in big], 5)), 3)},
        "latency_ms": lat,
    }


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "test")
