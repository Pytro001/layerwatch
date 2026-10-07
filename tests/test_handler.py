import base64

import cv2 as cv
import numpy as np

from layerwatch.synth.render import Build, LayerSpec, render_layer, render_reference
from server import handler
from server.store import LocalStore


def test_handler_roundtrip_local(tmp_path):
    store = LocalStore(tmp_path)
    b = Build(5)
    ref = cv.imencode(".jpg", render_reference(b))[1].tobytes()
    handler._WORKERS.clear()
    handler.register_build(store, "t1", b.cam.to_dict(), ref, 480)
    r = render_layer(b, LayerSpec(100, 0.08), np.random.default_rng(1))
    rec = handler.process_layer(store, "t1", 100, 0, 0, cv.imencode(".jpg", r["recoat"])[1].tobytes(),
                                cv.imencode(".jpg", r["melt"])[1].tobytes())
    assert rec["decision"]["action"] in ("CONTINUE", "RECOAT", "RE_IMAGE", "STOP")
    assert store.records("t1")[-1]["layer_index"] == 100
    # cold start: a new worker rebuilds its state from the log and stored frames
    handler._WORKERS.clear()
    rec2 = handler.process_layer(store, "t1", 101, 0, 0, cv.imencode(".jpg", r["recoat"])[1].tobytes(),
                                 cv.imencode(".jpg", r["melt"])[1].tobytes())
    assert rec2["layer_index"] == 101


def test_lambda_api_event(tmp_path, monkeypatch):
    import server.store as st
    monkeypatch.setattr(st, "AwsStore", lambda: LocalStore(tmp_path))
    b = Build(6)
    ref = base64.b64encode(cv.imencode(".jpg", render_reference(b))[1].tobytes()).decode()
    handler._WORKERS.clear()
    out = handler.lambda_handler({"body": {"op": "register", "build_id": "t2", "camera": b.cam.to_dict(), "reference_b64": ref}})
    assert out["statusCode"] == 200
    out = handler.lambda_handler({"body": {"op": "health"}})
    assert "opencv" in out["body"]
