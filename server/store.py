"""Where layer images, calibration and the layer log live.

LocalStore: a folder on disk (used by `make serve` and the tests).
AwsStore:   S3 for images and calibration, DynamoDB for the layer log.
Both expose the same five methods, so the handler does not know which it runs on.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def put_record(self, build_id: str, rec: dict) -> None:
        p = self.root / "log" / f"{build_id}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def records(self, build_id: str, last: int | None = None) -> list[dict]:
        p = self.root / "log" / f"{build_id}.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        return rows[-last:] if last else rows


class AwsStore:
    """S3 bucket + DynamoDB table (partition key build_id, sort key seq = layer * 10 + attempt)."""

    def __init__(self, bucket: str | None = None, table: str | None = None):
        import boto3

        self.bucket = bucket or os.environ["LAYER_BUCKET"]
        self.s3 = boto3.client("s3")
        self.table = boto3.resource("dynamodb").Table(table or os.environ["LAYER_TABLE"])

    def put_bytes(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)  # bucket default encryption applies

    def get_bytes(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def put_record(self, build_id: str, rec: dict) -> None:
        item = json.loads(json.dumps(rec), parse_float=Decimal)
        item["build_id"] = build_id
        item["seq"] = int(rec["layer_index"]) * 10 + int(rec.get("attempt", 0))
        self.table.put_item(Item=item)

    def records(self, build_id: str, last: int | None = None) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        kw = {"KeyConditionExpression": Key("build_id").eq(build_id), "ScanIndexForward": False}
        if last:
            kw["Limit"] = last
        items = self.table.query(**kw)["Items"]
        rows = json.loads(json.dumps(items, default=lambda d: float(d) if isinstance(d, Decimal) else str(d)))
        return list(reversed(rows))
