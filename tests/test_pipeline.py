import numpy as np

from layerwatch.synth.render import Build, LayerSpec, render_layer, render_reference
from layerwatch.vision.pipeline import LayerWatch
from layerwatch.vision.register import corner_error


def _run(defects, glare=None, seed=11):
    b = Build(seed)
    lw = LayerWatch(b.cam.to_dict(), render_reference(b), dnn=None, mode="rules")
    rng = np.random.default_rng(3)
    r = render_layer(b, LayerSpec(150, 0.12, defects=defects, glare=glare), rng)
    return lw.process(r["recoat"], r["melt"], 0.12, 150, overlays=True), r


def test_registration_recovers_vibration():
    res, r = _run([])
    assert corner_error(res["registration"]["W"], r["J"]) < 1.0
    assert res["evidence"]["ok"]


def test_streak_is_found_and_crosses_part():
    res, _ = _run([{"type": "streak", "y": 280.0, "x0": 40.0, "x1": 470.0, "width": 2.4, "depth": 0.17, "angle": 0.2}])
    s = [d for d in res["detections"] if d["type"] == "streak"]
    assert s and max(d["length_px"] for d in s) > 300


def test_glare_fails_evidence():
    res, _ = _run([], glare={"cx": 250.0, "cy": 250.0, "rx": 60.0, "ry": 40.0, "peak": 1.2, "angle": 10.0})
    assert not res["evidence"]["ok"]
    assert any("glare" in x for x in res["evidence"]["reasons"])


def test_every_stage_writes_an_overlay():
    res, _ = _run([])
    assert set(res["_overlays"]) == {"registration", "slice", "streaks", "spatter", "dnn", "evidence"}
    assert all(v.shape == (512, 512, 3) for v in res["_overlays"].values())
