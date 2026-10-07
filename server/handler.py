"""The LayerWatch worker. The same code runs as the AWS Lambda function (arm64,
Graviton, container image) and behind FastAPI on a laptop (`make serve`).

Two entry points:
  register_build(store, build_id, camera, reference_jpg, n_layers)
  process_layer(store, build_id, layer_index, attempt, recoats, recoat_jpg=None, melt_jpg=None)

Lambda events understood by `lambda_handler`:
  * S3 ObjectCreated for builds/<id>/layers/<layer>_<attempt>_melt.jpg (the
    recoat frame with the same prefix must already be there): the printer
    uploads, the worker decides, the decision lands in DynamoDB;
  * a Function URL / API call with a JSON body {"op": "layer" | "register", ...}.

Metrics go to CloudWatch with the Embedded Metric Format (one JSON line on
stdout per layer), so the function needs no cloudwatch:PutMetricData permission.
"""
from __future__ import annotations

import base64
import json
import os
import platform
import re
import time

import cv2 as cv
import numpy as np

from layerwatch import part
from layerwatch.agent.policy import RECOAT, RE_IMAGE, STOP, Agent
from layerwatch.vision.pipeline import DEFAULT_ONNX, LayerWatch

NAMESPACE = "LayerWatch"
_WORKERS: dict[str, tuple[LayerWatch, Agent, dict]] = {}


def _key(build_id: str, layer: int, attempt: int, kind: str) -> str:
    return f"builds/{build_id}/layers/{layer:05d}_{attempt}_{kind}.jpg"


def _decode(b: bytes) -> np.ndarray:
    img = cv.imdecode(np.frombuffer(b, np.uint8), cv.IMREAD_GRAYSCALE)
    if img is None or img.shape != (512, 512):
        raise ValueError("expected a 512 x 512 greyscale JPEG or PNG")
    return img


def register_build(store, build_id: str, camera: dict, reference_jpg: bytes, n_layers: int = part.DEMO_LAYERS) -> dict:
    _decode(reference_jpg)
    store.put_bytes(f"builds/{build_id}/ref.jpg", reference_jpg)
    job = {"build_id": build_id, "camera": camera, "n_layers": n_layers, "part_height_m": part.HPIPE}
    store.put_bytes(f"builds/{build_id}/job.json", json.dumps(job).encode())
    _WORKERS.pop(build_id, None)
    return job


def _worker(store, build_id: str, dnn: str | None) -> tuple[LayerWatch, Agent, dict]:
    if build_id in _WORKERS:
        return _WORKERS[build_id]
    job = json.loads(store.get_bytes(f"builds/{build_id}/job.json"))
    ref = _decode(store.get_bytes(f"builds/{build_id}/ref.jpg"))
    lw = LayerWatch(job["camera"], ref, dnn=dnn or os.environ.get("LAYERWATCH_ONNX") or DEFAULT_ONNX,
                    mode=os.environ.get("LAYERWATCH_MODE", "fused"))
    agent = Agent()
    # cold start: rebuild the agent history and the MOG2 background from the log
    hist = [r for r in store.records(build_id, last=40) if r.get("final")]
    for r in hist[-agent.p["window"]:]:
        agent.history.append({"layer": r["layer_index"], "types": r.get("types", []), "action": r["decision"]["action"]})
    for r in hist[-4:]:
        k = _key(build_id, r["layer_index"], r.get("attempt", 0), "melt")
        if store.exists(k):
            lw.warm(_decode(store.get_bytes(k)), r["h_m"])
    _WORKERS[build_id] = (lw, agent, job)
    return _WORKERS[build_id]


def process_layer(store, build_id: str, layer_index: int, attempt: int = 0, recoats: int = 0,
                  recoat_jpg: bytes | None = None, melt_jpg: bytes | None = None, h_m: float | None = None,
                  dnn: str | None = None, overlays: bool = False) -> dict:
    t0 = time.perf_counter()
    if recoat_jpg is not None:
        store.put_bytes(_key(build_id, layer_index, attempt, "recoat"), recoat_jpg)
    if melt_jpg is not None:
        store.put_bytes(_key(build_id, layer_index, attempt, "melt"), melt_jpg)
    lw, agent, job = _worker(store, build_id, dnn)
    recoat = _decode(recoat_jpg if recoat_jpg is not None else store.get_bytes(_key(build_id, layer_index, attempt, "recoat")))
    melt = _decode(melt_jpg if melt_jpg is not None else store.get_bytes(_key(build_id, layer_index, attempt, "melt")))
    h = h_m if h_m is not None else part.layer_height(layer_index, job["n_layers"])
    res = lw.process(recoat, melt, h, layer_index, overlays=overlays,
                     update_background=True)
    d = agent.decide(res, attempt=attempt, recoats=recoats)
    final = d.action not in (RECOAT, RE_IMAGE)
    if final:
        agent.commit(res, d)
    rec = {
        "build_id": build_id, "layer_index": layer_index, "attempt": attempt, "recoats": recoats, "h_m": round(h, 6),
        "decision": d.to_dict(), "final": final,
        "types": sorted({x["type"] for x in res["detections"]}) if res["evidence"]["ok"] else [],
        "detections": res["detections"], "evidence": {k: res["evidence"][k] for k in ("ok", "reasons")},
        "evidence_numbers": {f: {k: res["evidence"][f][k] for k in ("lap_ratio", "anisotropy", "saturated_frac")}
                             for f in ("melt", "recoat")},
        "slice": res["slice"], "registration": {"ecc_cc": res["registration"]["ecc_cc"]},
        "streaks": res["streaks"]["streaks"][:4], "spatter_count": res["spatter"]["count"],
        "timing_ms": res["timing_ms"], "arch": platform.machine(), "opencv": cv.__version__,
    }
    rec["timing_ms"]["handler_total"] = round((time.perf_counter() - t0) * 1000, 2)
    store.put_record(build_id, rec)
    _emit_metrics(build_id, rec)
    if overlays:
        rec["_overlays"] = res["_overlays"]
    return rec


def _emit_metrics(build_id: str, rec: dict) -> None:
    a = rec["decision"]["action"]
    m = {"LayerLatencyMs": rec["timing_ms"]["total"], "StopEvents": int(a == STOP), "RecoatEvents": int(a == RECOAT),
         "ReImageEvents": int(a == RE_IMAGE), "Defects": len(rec["detections"])}
    emf = {"_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
        "Namespace": NAMESPACE, "Dimensions": [["BuildId"], []],
        "Metrics": [{"Name": "LayerLatencyMs", "Unit": "Milliseconds"}] +
                   [{"Name": k, "Unit": "Count"} for k in m if k != "LayerLatencyMs"]}]},
           "BuildId": build_id, "Layer": rec["layer_index"], "Action": a, **m}
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("LAYERWATCH_EMF"):
        print(json.dumps(emf))  # Lambda ships stdout to CloudWatch Logs, EMF turns it into metrics
    if os.environ.get("LAYERWATCH_PUT_METRICS"):  # EC2 / COOL worker: no EMF agent, publish directly
        import boto3

        data = [{"MetricName": k, "Value": float(v), "Unit": "Milliseconds" if k == "LayerLatencyMs" else "Count"}
                for k, v in m.items()]
        boto3.client("cloudwatch").put_metric_data(Namespace=NAMESPACE, MetricData=data)


# ------------------------------------------------------------------ Lambda

_LAYER_KEY = re.compile(r"builds/(?P<b>[^/]+)/layers/(?P<l>\d+)_(?P<a>\d+)_melt\.jpg$")


def lambda_handler(event, context=None):
    from server.store import AwsStore

    store = AwsStore()
    if "Records" in event:  # S3 upload of the melt frame
        out = []
        for r in event["Records"]:
            key = r["s3"]["object"]["key"]
            m = _LAYER_KEY.search(key)
            if not m:
                continue
            rec = process_layer(store, m["b"], int(m["l"]), attempt=int(m["a"]))
            out.append({"layer": rec["layer_index"], "action": rec["decision"]["action"]})
        return {"processed": out}
    body = event.get("body", event)
    if isinstance(body, str):
        body = json.loads(base64.b64decode(body) if event.get("isBase64Encoded") else body)
    op = body.get("op", "layer")
    if op == "health":
        return _http(200, health())
    if op == "register":
        job = register_build(store, body["build_id"], body["camera"], base64.b64decode(body["reference_b64"]),
                             int(body.get("n_layers", part.DEMO_LAYERS)))
        return _http(200, job)
    rec = process_layer(store, body["build_id"], int(body["layer_index"]), int(body.get("attempt", 0)),
                        int(body.get("recoats", 0)),
                        base64.b64decode(body["recoat_b64"]) if "recoat_b64" in body else None,
                        base64.b64decode(body["melt_b64"]) if "melt_b64" in body else None)
    return _http(200, rec)


def _http(code: int, obj: dict) -> dict:
    return {"statusCode": code, "headers": {"content-type": "application/json"}, "body": json.dumps(obj)}


def health() -> dict:
    return {"ok": True, "opencv": cv.__version__, "arch": platform.machine(), "python": platform.python_version(),
            "onnx": os.environ.get("LAYERWATCH_ONNX", "models/layerwatch_unet.onnx")}
