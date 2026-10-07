"""Local runner: the same Lambda handler behind FastAPI, plus the dashboard.

    make serve        # http://localhost:8304

POST /api/builds/{build_id}                 register a build (camera calibration + layer 0 reference)
POST /api/builds/{build_id}/layers          one layer: two frames in, measurements + decision out
GET  /api/builds/{build_id}/log             the layer log
GET  /api/health                            OpenCV version, CPU architecture
GET  /                                      the dashboard (dashboard/dist), /run/* the recorded demo build
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2 as cv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server import handler
from server.store import LocalStore

ROOT = Path(__file__).resolve().parents[1]
STORE = LocalStore(os.environ.get("LAYERWATCH_STORE", ROOT / "data" / "local_store"))
app = FastAPI(title="LayerWatch", version="1.0")


class Register(BaseModel):
    camera: dict
    reference_b64: str
    n_layers: int = 480


class Layer(BaseModel):
    layer_index: int = Field(ge=0)
    attempt: int = Field(0, ge=0, le=5)
    recoats: int = Field(0, ge=0, le=5)
    recoat_b64: str
    melt_b64: str
    h_m: float | None = None
    overlays: bool = False


@app.get("/api/health")
def health():
    return handler.health()


@app.post("/api/builds/{build_id}")
def register(build_id: str, body: Register):
    _check_id(build_id)
    try:
        return handler.register_build(STORE, build_id, body.camera, base64.b64decode(body.reference_b64), body.n_layers)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/builds/{build_id}/layers")
def layer(build_id: str, body: Layer):
    _check_id(build_id)
    if not STORE.exists(f"builds/{build_id}/job.json"):
        raise HTTPException(404, "unknown build, register it first")
    try:
        rec = handler.process_layer(STORE, build_id, body.layer_index, body.attempt, body.recoats,
                                    base64.b64decode(body.recoat_b64), base64.b64decode(body.melt_b64), body.h_m,
                                    overlays=body.overlays)
    except ValueError as e:
        raise HTTPException(400, str(e))
    ov = rec.pop("_overlays", None)
    if ov:
        rec["overlays"] = {k: "data:image/jpeg;base64," + base64.b64encode(cv.imencode(".jpg", v, [cv.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
                           for k, v in ov.items()}
    return JSONResponse(rec)


@app.get("/api/builds/{build_id}/log")
def log(build_id: str, last: int | None = None):
    _check_id(build_id)
    return STORE.records(build_id, last)


def _check_id(b: str) -> None:
    if not b.replace("-", "").replace("_", "").isalnum() or len(b) > 64:
        raise HTTPException(400, "build id: letters, digits, - and _ only")


run_dir = ROOT / "dashboard" / "public" / "run"
if run_dir.exists():
    app.mount("/run", StaticFiles(directory=run_dir), name="run")
dist = ROOT / "dashboard" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
