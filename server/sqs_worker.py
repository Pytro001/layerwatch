"""Long running worker for the COOL path: an EC2 Graviton instance started from
the Cloud-Optimized OpenCV Library AMI polls an SQS queue that receives the S3
upload notifications, and runs exactly the same handler as the Lambda function.

    LAYER_BUCKET=... LAYER_TABLE=... LAYER_QUEUE_URL=... python -m server.sqs_worker
"""
from __future__ import annotations

import json
import os

import boto3

from server import handler


def main() -> None:
    sqs = boto3.client("sqs")
    url = os.environ["LAYER_QUEUE_URL"]
    while True:
        r = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=1, WaitTimeSeconds=20, VisibilityTimeout=120)
        for msg in r.get("Messages", []):
            event = json.loads(msg["Body"])
            if "Records" in event:
                print(json.dumps(handler.lambda_handler(event)), flush=True)
            sqs.delete_message(QueueUrl=url, ReceiptHandle=msg["ReceiptHandle"])


if __name__ == "__main__":
    main()
