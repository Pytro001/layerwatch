#!/usr/bin/env python3
"""LayerWatch on AWS.

    cdk synth                                  # default: Lambda arm64 container (Graviton)
    cdk synth -c compute=cool -c coolAmiId=ami-...   # EC2 Graviton from the COOL AMI
    cdk deploy -c alarmEmail=you@example.com
"""
import os

import aws_cdk as cdk

from layerwatch_stack import LayerWatchStack

app = cdk.App()
LayerWatchStack(
    app, "LayerWatch",
    compute=app.node.try_get_context("compute") or "lambda",
    cool_ami_id=app.node.try_get_context("coolAmiId"),
    alarm_email=app.node.try_get_context("alarmEmail"),
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                        region=app.node.try_get_context("region") or os.environ.get("CDK_DEFAULT_REGION") or "us-east-1"),
)
app.synth()
