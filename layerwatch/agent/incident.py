"""Optional: a human readable incident note for a STOP, written by Claude on
Amazon Bedrock from the structured agent log.

The model only rewrites numbers that the OpenCV pipeline already measured into
a short note for the operator. It is told not to add measurements, it never
sees the image, and its output is never fed back into a decision. Without a
Bedrock token a deterministic template note is written instead.

Env: AWS_BEARER_TOKEN_BEDROCK, AWS_REGION, BEDROCK_CLAUDE_MODEL (see .env.example).
"""
from __future__ import annotations

import json
import os
import urllib.request

SYSTEM = (
    "You write incident notes for a metal 3D printer operator. You get a JSON log of one stopped layer from "
    "an automated vision system. Write at most 70 words in plain sentences: what was found, where, the numbers "
    "that triggered the stop, and the next check for the operator. Use only numbers present in the JSON. "
    "Do not speculate about root cause beyond the reason field. No headings, no lists, no exclamation marks, "
    "no dashes as punctuation."
)


def template_note(rec: dict) -> str:
    d = rec["decision"]
    ev = d.get("evidence", {})
    nums = ", ".join(f"{k} {v}" for k, v in ev.items() if isinstance(v, (int, float)))
    return (f"Layer {rec['layer_index']} stopped. {d['reason']}. Measured: {nums}. "
            f"Check the recoater blade and the bed before resuming.")


def _load_env() -> None:
    p = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def bedrock_note(rec: dict, timeout: float = 30.0) -> tuple[str, str]:
    """returns (note, source) where source is the model id or 'template'"""
    _load_env()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    model = os.environ.get("BEDROCK_CLAUDE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    region = os.environ.get("AWS_REGION", "us-east-1")
    if not token:
        return template_note(rec), "template"
    slim = {k: rec[k] for k in ("layer_index", "h_m", "decision", "detections", "attempts") if k in rec}
    body = {
        "system": [{"text": SYSTEM}],
        "messages": [{"role": "user", "content": [{"text": json.dumps(slim)}]}],
        "inferenceConfig": {"maxTokens": 220, "temperature": 0.2},
    }
    req = urllib.request.Request(
        f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        text = "".join(c.get("text", "") for c in out["output"]["message"]["content"]).strip()
        text = text.replace("—", ",").replace("–", ",")
        return text, model
    except Exception as e:  # network, auth, throttling: never block the machine on the note
        return template_note(rec) + f" (note service unavailable: {type(e).__name__})", "template"
