"""One stack: S3 (layer frames in), the OpenCV worker on Graviton, DynamoDB
(layer log), CloudWatch metrics, dashboard and a STOP alarm.

Least privilege: the worker may read and write objects under builds/ in its
own bucket, put and query items in its own table, and (EC2 path only) use its
own queue and publish metrics in the LayerWatch namespace. No image ever
leaves the bucket: the worker has no other network target, the bucket blocks
public access and enforces TLS, the Function URL requires IAM auth.
"""
from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cwa
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from constructs import Construct

REPO = Path(__file__).resolve().parents[1]
NS = "LayerWatch"


class LayerWatchStack(cdk.Stack):
    def __init__(self, scope: Construct, cid: str, compute: str = "lambda", cool_ami_id: str | None = None,
                 alarm_email: str | None = None, **kw) -> None:
        super().__init__(scope, cid, **kw)

        bucket = s3.Bucket(
            self, "LayerFrames",
            encryption=s3.BucketEncryption.S3_MANAGED, enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL, object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            lifecycle_rules=[s3.LifecycleRule(prefix="builds/", expiration=cdk.Duration.days(180))],
            removal_policy=cdk.RemovalPolicy.RETAIN,
        )
        table = ddb.Table(
            self, "LayerLog",
            partition_key=ddb.Attribute(name="build_id", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="seq", type=ddb.AttributeType.NUMBER),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=ddb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=True),
            encryption=ddb.TableEncryption.AWS_MANAGED, removal_policy=cdk.RemovalPolicy.RETAIN,
        )
        env = {"LAYER_BUCKET": bucket.bucket_name, "LAYER_TABLE": table.table_name, "LAYERWATCH_MODE": "fused"}
        layer_objects = bucket.arn_for_objects("builds/*")
        data_policy = [
            iam.PolicyStatement(actions=["s3:GetObject", "s3:PutObject"], resources=[layer_objects]),
            iam.PolicyStatement(actions=["dynamodb:PutItem", "dynamodb:Query"], resources=[table.table_arn]),
        ]
        notify_filter = s3.NotificationKeyFilter(prefix="builds/", suffix="_melt.jpg")

        if compute == "cool":
            worker_name = self._cool_worker(bucket, env, data_policy, notify_filter, cool_ami_id)
        else:
            fn = lambda_.DockerImageFunction(
                self, "Worker",
                code=lambda_.DockerImageCode.from_image_asset(
                    str(REPO), file="Dockerfile", platform=ecr_assets.Platform.LINUX_ARM64),
                architecture=lambda_.Architecture.ARM_64,  # Graviton
                memory_size=2048, timeout=cdk.Duration.seconds(60), environment=env,
                reserved_concurrent_executions=20,
                log_group=logs.LogGroup(self, "WorkerLogs", retention=logs.RetentionDays.ONE_MONTH,
                                        removal_policy=cdk.RemovalPolicy.DESTROY),
                description="LayerWatch OpenCV 5 worker: registration, slice compare, streaks, spatter, cv.dnn, agent",
            )
            for st in data_policy:
                fn.add_to_role_policy(st)
            bucket.add_event_notification(s3.EventType.OBJECT_CREATED, s3n.LambdaDestination(fn), notify_filter)
            url = fn.add_function_url(auth_type=lambda_.FunctionUrlAuthType.AWS_IAM)
            cdk.CfnOutput(self, "WorkerUrl", value=url.url, description="IAM authenticated endpoint (POST JSON, op=layer|register|health)")
            worker_name = fn.function_name
            self._lambda_widgets = [
                cw.GraphWidget(title="Lambda duration (ms)", left=[fn.metric_duration(statistic="p50"), fn.metric_duration(statistic="p95")]),
                cw.GraphWidget(title="Lambda errors / throttles", left=[fn.metric_errors(), fn.metric_throttles()]),
            ]

        # metrics come from the worker (EMF on Lambda, PutMetricData on EC2)
        def m(name: str, stat: str = "Sum", unit: cw.Unit | None = None) -> cw.Metric:
            return cw.Metric(namespace=NS, metric_name=name, statistic=stat, period=cdk.Duration.minutes(1))

        stops = m("StopEvents")
        topic = sns.Topic(self, "StopTopic", display_name="LayerWatch build stopped")
        if alarm_email:
            topic.add_subscription(subs.EmailSubscription(alarm_email))
        alarm = cw.Alarm(
            self, "BuildStopped", metric=stops, threshold=1, evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="LayerWatch agent stopped a build. Read the layer log in DynamoDB for the reason and numbers.",
        )
        alarm.add_alarm_action(cwa.SnsAction(topic))
        latency_alarm = cw.Alarm(
            self, "SlowLayers", metric=m("LayerLatencyMs", "p95"), threshold=5000, evaluation_periods=5,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="p95 layer latency above 5 s: the check is slower than a recoat.",
        )
        dash = cw.Dashboard(self, "Dashboard", dashboard_name="LayerWatch")
        dash.add_widgets(
            cw.GraphWidget(title="Layer latency (ms)", left=[m("LayerLatencyMs", "p50"), m("LayerLatencyMs", "p95")]),
            cw.GraphWidget(title="Agent actions per minute", left=[stops, m("RecoatEvents"), m("ReImageEvents")]),
            cw.GraphWidget(title="Defects detected", left=[m("Defects")]),
            cw.AlarmStatusWidget(title="Alarms", alarms=[alarm, latency_alarm]),
        )
        if getattr(self, "_lambda_widgets", None):
            dash.add_widgets(*self._lambda_widgets)
        cdk.CfnOutput(self, "BucketName", value=bucket.bucket_name)
        cdk.CfnOutput(self, "TableName", value=table.table_name)
        cdk.CfnOutput(self, "WorkerName", value=worker_name)

    def _cool_worker(self, bucket, env, data_policy, notify_filter, ami_id):
        """EC2 Graviton instance from the Cloud-Optimized OpenCV Library (COOL)
        Marketplace AMI, polling an SQS queue fed by the bucket notifications."""
        if not ami_id:
            raise ValueError("compute=cool needs -c coolAmiId=ami-... (subscribe to COOL in AWS Marketplace first)")
        dlq = sqs.Queue(self, "LayerDLQ", retention_period=cdk.Duration.days(14), enforce_ssl=True)
        q = sqs.Queue(self, "LayerQueue", visibility_timeout=cdk.Duration.seconds(180), enforce_ssl=True,
                      dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=3, queue=dlq))
        bucket.add_event_notification(s3.EventType.OBJECT_CREATED, s3n.SqsDestination(q), notify_filter)
        vpc = ec2.Vpc(self, "Vpc", max_azs=1, nat_gateways=0,
                      subnet_configuration=[ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC)])
        role = iam.Role(self, "WorkerRole", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
                        managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")])
        for st in data_policy:
            role.add_to_policy(st)
        q.grant_consume_messages(role)
        role.add_to_policy(iam.PolicyStatement(actions=["cloudwatch:PutMetricData"], resources=["*"],
                                               conditions={"StringEquals": {"cloudwatch:namespace": NS}}))
        sg = ec2.SecurityGroup(self, "WorkerSg", vpc=vpc, allow_all_outbound=True, description="no inbound")
        ud = ec2.UserData.for_linux()
        ud.add_commands(
            "set -e", "apt-get update -y && apt-get install -y git python3-venv",
            "git clone https://github.com/Pytro001/layerwatch /opt/layerwatch",
            "cd /opt/layerwatch && python3 -m venv --system-site-packages .venv",
            # COOL provides cv2; only the rest is installed from PyPI
            ".venv/bin/pip install numpy boto3",
            f"cat > /etc/layerwatch.env <<EOT\nLAYER_BUCKET={env['LAYER_BUCKET']}\nLAYER_TABLE={env['LAYER_TABLE']}\n"
            f"LAYER_QUEUE_URL={q.queue_url}\nLAYERWATCH_PUT_METRICS=1\nAWS_DEFAULT_REGION={self.region}\nEOT",
            "cat > /etc/systemd/system/layerwatch.service <<EOT\n[Service]\nEnvironmentFile=/etc/layerwatch.env\n"
            "WorkingDirectory=/opt/layerwatch\nExecStart=/opt/layerwatch/.venv/bin/python -m server.sqs_worker\nRestart=always\n"
            "[Install]\nWantedBy=multi-user.target\nEOT",
            "systemctl daemon-reload && systemctl enable --now layerwatch",
        )
        inst = ec2.Instance(
            self, "CoolWorker", vpc=vpc, security_group=sg, role=role, user_data=ud,
            instance_type=ec2.InstanceType("c7g.large"),  # Graviton3
            machine_image=ec2.MachineImage.generic_linux({self.region: ami_id}),
            require_imdsv2=True,
            block_devices=[ec2.BlockDevice(device_name="/dev/sda1", volume=ec2.BlockDeviceVolume.ebs(20, encrypted=True))],
        )
        cdk.CfnOutput(self, "QueueUrl", value=q.queue_url)
        return inst.instance_id
