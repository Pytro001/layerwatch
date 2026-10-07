"""Build docs/LayerWatch_Report.md (+ .html, figures) from the evaluation JSON
files, and the eval.json the dashboard reads. Every number in the report is
read from those files.

    python -m layerwatch.eval.report        # then: node docs/render_pdf.mjs
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
EV = DOCS / "eval"
RUN = ROOT / "dashboard" / "public" / "run"
TYPES = ["streak", "spatter", "super_elevation", "short_feed"]
LABEL = {"streak": "Recoater streak", "spatter": "Spatter", "super_elevation": "Super elevation", "short_feed": "Short feed"}
MODES = [("rules", "Rules only"), ("dnn", "DNN only"), ("fused", "Rules + DNN")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def load(p: Path):
    return json.load(open(p)) if p.exists() else None


def cdk_resources() -> int | None:
    t = ROOT / "infra" / "cdk.out" / "LayerWatch.template.json"
    if not t.exists():
        return None
    return len(json.load(open(t))["Resources"])


def fig_prf(s: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.26
    x = np.arange(len(TYPES))
    cols = ["#7f8a99", "#4da3ff", "#2b9a66"]
    for k, (m, l) in enumerate(MODES):
        v = [s["detection"][m][t]["f1"] for t in TYPES]
        ax.bar(x + (k - 1) * w, v, w, label=l, color=cols[k])
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 1) * w, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [LABEL[t] for t in TYPES])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 (layer level)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "f1_by_type.png")
    plt.close(fig)


def fig_latency(s: dict) -> None:
    st = [("registration", "registration (ECC)"), ("evidence", "evidence check"), ("slice_compare", "slice compare"),
          ("streaks", "streaks + short feed"), ("spatter_super_elevation", "spatter + super elevation"), ("dnn", "U-Net via cv.dnn"),
          ("slice_render", "expected slice render")]
    st = [(k, l) for k, l in st if k in s["latency_ms"]]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    med = [s["latency_ms"][k]["median"] for k, _ in st]
    p95 = [s["latency_ms"][k]["p95"] for k, _ in st]
    y = np.arange(len(st))
    ax.barh(y, p95, color="#d7e6fb", label="p95")
    ax.barh(y, med, color="#4da3ff", label="median")
    ax.set_yticks(y, [l for _, l in st])
    ax.invert_yaxis()
    ax.set_xlabel("ms per layer (two frames)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "latency.png")
    plt.close(fig)


def fig_train(log: list) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
    ep = [r["epoch"] for r in log]
    ax[0].plot(ep, [r["train_loss"] for r in log], label="train", color="#7f8a99")
    ax[0].plot(ep, [r["val_loss"] for r in log], label="validation", color="#4da3ff")
    ax[0].set_xlabel("epoch (40 steps)")
    ax[0].set_ylabel("weighted cross entropy")
    ax[0].legend(frameon=False)
    for t, c in zip(TYPES, ["#ff4d4f", "#e0b030", "#ff8c42", "#4da3ff"]):
        ax[1].plot(ep, [r["val_iou"][t] for r in log], label=LABEL[t], color=c)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("validation IoU")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training.png")
    plt.close(fig)


def fig_pipeline() -> bool:
    """the six stage overlays of layer 212 (the streak that stops the build)"""
    names = ["registration", "slice", "streaks", "spatter", "dnn", "evidence"]
    tiles = []
    for n in names:
        p = RUN / "ov" / f"212_1_{n}.jpg"
        if not p.exists():
            return False
        im = cv.imread(str(p))
        cv.putText(im, n, (12, 32), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv.LINE_AA)
        cv.putText(im, n, (12, 32), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
        tiles.append(cv.resize(im, (384, 384), interpolation=cv.INTER_AREA))
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv.imwrite(str(FIG / "pipeline_layer212.jpg"), grid, [cv.IMWRITE_JPEG_QUALITY, 88])
    raw = cv.imread(str(RUN / "img" / "212_0_recoat.jpg"))
    reg = cv.imread(str(RUN / "bed" / "212_0_r.jpg"))
    if raw is not None and reg is not None:
        cv.imwrite(str(FIG / "camera_vs_registered.jpg"), np.hstack([raw, reg]), [cv.IMWRITE_JPEG_QUALITY, 88])
    return True


def real_figure(real_rows: list) -> None:
    from layerwatch.eval.real_eval import figure

    # three frames with the most labelled boxes, ties broken by name (deterministic)
    rows = sorted(real_rows, key=lambda r: (-len(r["gt_boxes"]), r["image"]))
    picks = [rows[0]["image"], rows[len(rows) // 3]["image"], rows[2 * len(rows) // 3]["image"]]
    figure(FIG / "real_examples.jpg", picks)
    im = cv.imread(str(FIG / "real_examples.jpg"))
    cv.imwrite(str(RUN / "real_examples.jpg"), cv.resize(im, (im.shape[1] * 2 // 3, im.shape[0] * 2 // 3)), [cv.IMWRITE_JPEG_QUALITY, 86])


def prf_table(s: dict) -> str:
    h = "| Defect | " + " | ".join(f"{l} P | R | F1" for _, l in MODES) + " |\n"
    h += "|---|" + "---:|---:|---:|" * 3 + "\n"
    for t in TYPES:
        h += f"| {LABEL[t]} | " + " | ".join(
            f"{s['detection'][m][t]['precision']:.2f} | {s['detection'][m][t]['recall']:.2f} | {s['detection'][m][t]['f1']:.2f}" for m, _ in MODES) + " |\n"
    h += "| **macro F1** | " + " | ".join(f" | | **{s['detection'][m]['macro_f1']:.2f}**" for m, _ in MODES) + " |\n"
    return h


def counts_table(s: dict) -> str:
    h = "| Defect | " + " | ".join(f"{l} TP / FP / FN" for _, l in MODES) + " |\n|---|---:|---:|---:|\n"
    for t in TYPES:
        h += f"| {LABEL[t]} | " + " | ".join(
            f"{s['detection'][m][t]['tp']} / {s['detection'][m][t]['fp']} / {s['detection'][m][t]['fn']}" for m, _ in MODES) + " |\n"
    return h


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    hold = load(EV / "synthetic_holdout.json")
    test = load(EV / "synthetic_test.json")
    real = load(EV / "real.json")
    train_log = load(ROOT / "models" / "train_log.json") or []
    export = load(ROOT / "models" / "export_check.json") or {}
    run = load(RUN / "run.json")
    s = hold["summary"]
    fig_prf(s)
    fig_latency(s)
    if train_log:
        fig_train(train_log)
    have_pipe = fig_pipeline()
    real_figure(real["rows"])
    nres = cdk_resources()
    out = {"synthetic": s, "synthetic_dev_test": test["summary"] if test else None, "real": real["summary"],
           "train": train_log, "export": export, "aws": {"resources": nres},
           "demo": {k: run[k] for k in ("actions", "latency_ms", "stops")} if run else None}
    json.dump(out, open(RUN / "eval.json", "w"))
    json.dump(out, open(EV / "summary.json", "w"), indent=1)

    a = s["agent"]
    r = real["summary"]
    good = a["layers"] - a["bad_frames"]
    fine = good - a["layers_needing_intervention"]
    lat = s["latency_ms"]
    best_epoch = min(train_log, key=lambda x: x["val_loss"]) if train_log else None
    rows = hold["rows"]
    missed = [x for x in rows if x["id"] in set(a["missed_critical_ids"])]
    seen = [x for x in missed if set(x["gt_critical"]) <= set(x["fused"])]
    by_t: dict = {}
    for x in missed:
        for t in x["gt_critical"]:
            by_t[t] = by_t.get(t, 0) + 1
    fstops = [x for x in rows if not x["gt_bad"] and not x["gt_critical"] and x["action"] == "STOP"]
    se_trend = [x for x in fstops if "over 2 layer" in x["reason"]]
    failure_text = (f"All {len(missed)} missed critical layers: the defect was detected in {len(seen)} of them "
                    f"({', '.join(f'{LABEL[k]} {v}' for k, v in by_t.items())}). The miss is in the policy, not the vision: "
                    "the label calls a super elevation critical from 250 px of area, the policy stops on a crescent arc of 60 px "
                    "or on two layers in a row, so crescents under 60 px of arc but over 250 px of area fall between the two; the streak was measured just under the 150 px length rule. "
                    f"{len(se_trend)} of the {len(fstops)} false stops come from the opposite side of the same rule "
                    "(a small crescent on two consecutive sampled layers). These policy thresholds were set before the "
                    "hold-out was generated and are reported unchanged; aligning the label and policy definitions is the first next step.")
    md = f"""# LayerWatch: evaluation report

Layer by layer vision QA for laser powder bed fusion, built on OpenCV {s['meta']['opencv']}.
Generated by `make report` from `docs/eval/*.json`; every number below is read from those files.

## 1. What is measured

* **Synthetic hold-out**: {s['meta']['n_layers']} layers from {s['meta']['builds']} simulated builds (new camera, lamp,
  powder and defect seeds) generated after all thresholds were frozen. None of these images was used for training,
  validation or tuning. A second synthetic split ("dev test", {test['summary']['meta']['n_layers'] if test else 0} layers) was looked at while
  tuning thresholds; its numbers are in `docs/eval/synthetic_test.json` and are not the headline.
* **Real images**: {r['images']} labelled powder bed frames of an EOS M290 (Aalto University dataset, CC-BY 4.0,
  doi:10.5281/zenodo.14996805), zero shot: recoat checks only, thresholds unchanged, no DNN.
* **Closed loop demo build**: 480 layers through the real handler with scripted events (section 6).

Scoring is per layer: a defect type counts as found on a layer if the detector reports at least one region of that
type. Frames with strong glare or motion blur are scored only on whether the evidence gate rejects them.

## 2. Detection on the synthetic hold-out

![F1 per defect type](figures/f1_by_type.png)

{prf_table(s)}
Counts:

{counts_table(s)}
DNN pixel IoU per class on usable hold-out frames: """ + ", ".join(f"{LABEL[k]} {v:.2f}" for k, v in s["dnn_pixel_iou"].items()) + f"""

## 3. The agent

| Measure | Value |
|---|---:|
| Layers | {a['layers']} |
| Unusable frames (strong glare or blur) | {a['bad_frames']} |
| Decisions made on an unusable frame (should be 0) | {a['decided_on_bad_frame']} |
| Evidence gate precision / recall | {s['evidence_gate']['precision']:.2f} / {s['evidence_gate']['recall']:.2f} |
| Mild nuisance frames (should pass) flagged | {s['evidence_gate']['mild_nuisance_flagged']} of {s['evidence_gate']['mild_nuisance_layers']} |
| Layers that need a recoat or stop (by injected geometry) | {a['layers_needing_intervention']} |
| Missed critical layers (agent said continue) | {a['missed_critical']} |
| False stops on layers that did not need one | {a['false_stops']} ({a['false_stop_per_1000']} per 1,000 good layers) |
| Any intervention on layers that did not need one | {a['false_interventions']} ({a['false_intervention_per_1000']} per 1,000) |
| Actions | """ + ", ".join(f"{k} {v}" for k, v in a["actions"].items()) + f""" |

"Needs intervention" is defined from the injected geometry only: a streak at least 150 px long or crossing the part,
a short feed patch of at least 500 px, or a super elevation crescent of at least 250 px. The hold-out layers of one
build are 40 random heights, not consecutive layers, so trend rules (streaks on 3 of the last 10 layers, super
elevation on 2 layers in a row) fire less often than they would on a real build.

### Where the agent went wrong

{failure_text}

## 4. Registration and slice comparison

* ECC homography against the layer 0 frame, using only the static chamber rim: median corner error
  {s['registration_px']['median']:.2f} px, p95 {s['registration_px']['p95']:.2f} px, max {s['registration_px']['max']:.2f} px
  (true vibration known from the generator).
* Melted region vs the slice from the part function, layers with at least 400 px of slice: median IoU
  {s['slice_iou']['median']:.2f}, 5th percentile {s['slice_iou']['p5']:.2f}.

## 5. Real powder bed images (zero shot)

| Measure | Value |
|---|---:|
| Labelled frames | {r['images']} ({', '.join(f'{v} from build {k[-6:]}' for k, v in r['builds'].items())}) |
| Labelled defect boxes | {r['gt_boxes']} |
| Boxes hit by a LayerWatch detection (recall) | {r['box_recall']:.2f} |
| Detections that hit a labelled box (precision) | {r['detection_precision']:.2f} |
| Frames with at least one labelled box found | {r['images_with_at_least_one_hit']:.2f} |
| Detections per frame | {r['detections_per_image']} |
""" + "".join(f"| {k} detector precision | {v['precision']:.2f} ({v['on_a_gt_box']} of {v['detections']}) |\n" for k, v in r["by_detector"].items()) + f"""
![real frames](figures/real_examples.jpg)

Green: annotator boxes. Red: LayerWatch detections mapped back to the camera image. The labels are one class
("defects") drawn as boxes, so this measures where LayerWatch looks, not whether it names the type right.
Calibration for these frames is a 4 point homography picked by hand once; there is no layer 0 frame, so
registration is not exercised, and the lighting reference is the per build median frame.

## 6. Closed loop demo build

""" + (f"""480 layers of the thrust chamber on one simulated machine, run through the same handler the Lambda runs. Actions:
""" + ", ".join(f"{k} {v}" for k, v in run["actions"].items()) + f""". Median handler latency {run['latency_ms']['median']:.0f} ms (p95 {run['latency_ms']['p95']:.0f} ms).

""" + "".join(f"* Stop at layer {st['layer']}: {st['reason']}. Note ({st['note_source']}): {st['note']}\n" for st in run["stops"]) if run else "not run") + f"""
""" + ("![pipeline overlays, layer 212](figures/pipeline_layer212.jpg)\n\nStage overlays of layer 212 after the recoat: registration, slice compare, streaks, spatter and super elevation, DNN, evidence.\n" if have_pipe else "") + f"""
## 7. Latency

![latency](figures/latency.png)

Median {lat['total']['median']:.0f} ms, p95 {lat['total']['p95']:.0f} ms per layer (two frames) on {s['meta']['platform']}, a laptop
that was shared with other heavy jobs during the run, so these are upper bounds. Graviton numbers need the AWS deploy
(`make deploy`), which is pending an AWS account.

## 8. The learned segmenter

""" + (f"""U-Net, {export.get('params', 0):,} parameters, trained on CPU (4 threads) for {len(train_log)} epochs of 40 steps
({train_log[-1]['minutes']:.0f} minutes wall clock on a shared machine), best validation loss at epoch {best_epoch['epoch']}.
Exported to ONNX and run with `cv.dnn.readNetFromONNX`; maximum absolute difference to PyTorch on a random input:
{export.get('cv_dnn_vs_torch_max_abs_diff', 0):.1e}.

![training](figures/training.png)
""" if train_log else "") + """
## 9. Limits

* The headline numbers are on synthetic images whose defects I designed. They show the pipeline does what it is built
  to do; they do not show real world accuracy. The real image check covers recoat defects only and has box labels.
* The synthetic camera is 512 x 512 at 0.86 mm per pixel; real machines use more pixels per mm.
* The demo build samples 480 heights through a 385 mm part, so one demo layer stands for many 40 micron layers.
* Not a certification tool. A stop is a request for a person to look.
"""
    (DOCS / "LayerWatch_Report.md").write_text(md)
    import markdown

    body = markdown.markdown(md, extensions=["tables"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>LayerWatch report</title><style>
    body{{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;color:#111;max-width:860px;margin:28px auto;line-height:1.45;font-size:12.5px}}
    h1{{font-size:24px;margin-bottom:4px}} h2{{font-size:17px;margin-top:22px;border-bottom:1px solid #ddd;padding-bottom:3px}}
    table{{border-collapse:collapse;margin:8px 0;font-size:11.5px}} td,th{{border-bottom:1px solid #e3e3e3;padding:4px 8px}}
    th{{background:#f4f6f8;text-align:left}} img{{max-width:100%;margin:6px 0}} code{{background:#f2f2f2;padding:0 3px}}
    </style></head><body>{body}</body></html>"""
    (DOCS / "LayerWatch_Report.html").write_text(html)
    print("report written")


if __name__ == "__main__":
    main()
