# LayerWatch: Devpost submission (OpenCV AI Competition 2026, powered by AWS)

Manage page: https://devpost.com/submit-to/30984-opencv-ai-competition-2026-powered-by-aws/manage/submissions
Deadline: Oct 26, 2026, 11:59 pm Pacific.

## Project name

LayerWatch

## Elevator pitch (under 200 characters)

Layer by layer vision QA for metal 3D printing. OpenCV 5 finds recoater streaks, spatter and warping, and an agent recoats, re-images or stops the build, with its reasons.

## Links

* Repository: [REPO] (https://github.com/Pytro001/layerwatch, public, Apache 2.0)
* Video: [VIDEO]
* Demo / web endpoint: [DEMO] (local: `make serve` then http://localhost:8304; the AWS endpoint is not deployed, see "Web endpoint")
* Report: docs/LayerWatch_Report.pdf in the repository

## Built with

opencv, opencv-5, python, numpy, pytorch, onnx, fastapi, aws-cdk, aws-lambda, aws-graviton, amazon-s3, amazon-dynamodb, amazon-cloudwatch, amazon-bedrock, three.js, typescript, vite, playwright

## Sponsor and special prizes

* Agentic Vision Award: **yes**. The vision output drives a physical action on the machine (continue, recoat, re-image, stop) through a transparent policy, with the reason and the triggering numbers logged for every layer.
* Best Use of COOL Award: **yes, as code, unverified**. The CDK app has a COOL variant (`-c compute=cool -c coolAmiId=...`): an EC2 c7g Graviton instance from the Cloud-Optimized OpenCV Library AMI, fed by SQS, running the same handler. It synthesizes; it has not been launched, because no AWS account was used. The default path is a Lambda container on arm64 (Graviton) with OpenCV 5 headless; its arm64 dependencies resolve, the Docker image was not built.

## Special award consideration (text)

**Agentic Vision.** LayerWatch is an agent whose eyes are measured, classic computer vision. Each layer it reads two camera frames, measures, and acts: recoat this layer, take the picture again, stop the build, or continue. Evidence comes first: if the frame is blurred (Laplacian variance at half scale, gradient anisotropy for motion blur) or has glare (saturated pixels) or registration is weak (ECC correlation), the action is always "re-image", never a guess. On the 400 layer hold-out, the evidence gate caught 50 of 51 unusable frames and rejected none of the 42 mildly blurred or lit frames. The policy has unit tests, every threshold is in one table, and every decision is logged with the numbers that triggered it. A language model (Claude on Amazon Bedrock) only writes the operator's incident note from those logged numbers; it never sees the image and never decides.

**COOL.** The worker is OpenCV heavy (ECC homography, morphology, Hough, blob detection, MOG2, a DNN) and runs per layer on a schedule set by the printer, which is exactly the kind of steady CPU load Graviton is for. The COOL variant of the stack puts the same handler on a c7g instance from the COOL AMI. I have not measured it on Graviton; the numbers in the report are from an arm64 laptop.

## Testing instructions (for judges)

```bash
git clone [REPO] && cd layerwatch
make setup                 # Python 3.12 (uv) + Node deps
make test                  # 23 tests, includes: OpenCV major version is 5
make demo                  # 480 layer demo build through the handler (about 5 minutes)
make serve                 # http://localhost:8304: dashboard + local endpoint
```

In the dashboard: press N to jump between events, 1 to 8 for pipeline stages, W for the evaluation page, R for the 3D defect map, A to send the current layer through the local endpoint. To reproduce the numbers: `make data && make eval && make report` (training is optional, the ONNX model is committed; `make train` retrains in about 40 minutes on CPU). To see the AWS stack: `make synth`.

## Web endpoint

Local, working: `make serve` exposes the same handler the Lambda runs: `GET /api/health`, `POST /api/builds/{id}`, `POST /api/builds/{id}/layers`, `GET /api/builds/{id}/log` on http://localhost:8304.

AWS: **not deployed.** No AWS account was used for this entry. The CDK app (`infra/`) synthesizes an IAM authenticated Lambda Function URL; `make deploy` creates it once credentials and Docker are available.

## About the project

### Inspiration

Laser powder bed fusion prints aerospace and medical parts one layer of tens of microns at a time, often for days. A single bad layer, a streak dragged by a nicked recoater blade, a patch the powder did not reach, an edge that curls up and hits the blade, can scrap the whole build, and often nobody knows until the part is cut open or scanned. Most machines already have a camera over the bed. I wanted to see how far careful, measured OpenCV gets toward turning those pictures into decisions, not just pictures.

### What it does

For every layer LayerWatch takes the two frames a machine records (after the recoater spreads powder, and after the laser melts the slice), and:

1. registers them to layer 0 (lens and keystone from calibration, vibration by ECC homography on the static rim);
2. checks the evidence: blur, glare, registration quality; bad frames mean "re-image";
3. compares the melted region with the slice computed from the part file;
4. finds recoater streaks, short feed, spatter and super elevation with classic OpenCV, plus a small U-Net run by `cv.dnn`;
5. lets an agent decide: CONTINUE, RECOAT, RE_IMAGE or STOP, with the reason and numbers;
6. logs everything, emits metrics, and places every finding on a 3D map of the part.

The dashboard shows the camera frame, an overlay for each pipeline stage, the agent log, a timeline of actions and the 3D map. The demo build is a copper rocket thrust chamber with 36 cooling channels: at layer 212 a streak crosses the part, the agent asks for a recoat, the streak survives, and it stops the build with an incident note.

### OpenCV 5 pipeline

* **Registration**: calibrated undistort + keystone in one `cv.remap`, then `cv.findTransformECC` (translation at 1/4 scale, homography at 1/2 and full scale) with a mask on the static chamber rim, a plausibility bound and a Euclidean fallback, then `cv.warpPerspective`. Median corner error 0.13 px, max 0.38 px on the hold-out.
* **Slice comparison**: lighting flattened with a field from the layer 0 frame; melted region by Otsu plus `cv.adaptiveThreshold`, `cv.morphologyEx`, `cv.findContours`; IoU against the expected slice and `cv.matchShapes` (Hu moments). Median IoU 0.87.
* **Recoater streaks**: dark detail, morphological opening with a 41 px line kernel along the recoat axis (plus two tilted copies), `cv.HoughLinesP`, segment merging, scored by length, angle and how far they cross the part. Streak like edges of a short feed patch are rejected.
* **Short feed**: smoothed dark patches with a threshold that adapts to the measured powder texture of each frame.
* **Spatter and super elevation**: white top hat, `cv.SimpleBlobDetector`, confirmation against `cv.createBackgroundSubtractorMOG2` learned over previous layers (with the slice masked out), single linkage clustering, texture adaptive contrast floor; super elevation as bright elongated regions on the slice edge band, brighter than fresh powder, outside any lamp reflection zone.
* **DNN**: a 482,805 parameter U-Net (input: registered recoat, registered melt, expected slice; 5 classes) trained in PyTorch on CPU, exported to ONNX, run with `cv.dnn.readNetFromONNX` (max difference to PyTorch 9.5e-6). Fused with the rules.
* **Evidence**: `cv.Laplacian` variance at half scale relative to the reference, structure tensor anisotropy from `cv.Sobel`, saturated pixel fraction, ECC correlation.

### AWS architecture

S3 (layer frames, SSE, TLS only, no public access, 180 day expiry) triggers the worker: an AWS Lambda container image on arm64 (Graviton) with OpenCV 5 headless, 2 GB, 60 s. It writes every decision to a DynamoDB layer log and emits CloudWatch metrics through the Embedded Metric Format (latency, stops, recoats, re-images, defects); a CloudWatch alarm on stop events notifies the operator through SNS. A CloudWatch dashboard shows latency percentiles and actions. Least privilege IAM: get and put on `builds/*` in its own bucket, put and query on its own table. An IAM authenticated Function URL serves direct calls. Optional COOL path: EC2 c7g from the COOL AMI with an SQS queue. All of it is AWS CDK in Python, `make deploy`. Status: synthesizes (18 and 29 resources), not deployed.

### Evaluation

Synthetic hold-out: 400 layers from 10 builds generated after all thresholds were frozen. Layer level, per defect type.

| Defect | Rules P / R / F1 | DNN P / R / F1 | Rules + DNN P / R / F1 |
|---|---|---|---|
| Recoater streak | 0.96 / 0.98 / 0.97 | 0.98 / 1.00 / 0.99 | 0.94 / 1.00 / 0.97 |
| Spatter | 1.00 / 0.75 / 0.85 | 0.98 / 0.78 / 0.87 | 0.98 / 0.87 / 0.92 |
| Super elevation | 0.87 / 0.82 / 0.84 | 0.97 / 0.97 / 0.97 | 0.88 / 0.99 / 0.93 |
| Short feed | 1.00 / 0.97 / 0.99 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Macro F1 | 0.91 | 0.96 | 0.95 |

| Agent | Value |
|---|---|
| Unusable frames caught (evidence gate) | 50 of 51 (precision 1.00) |
| Mild glare or blur wrongly rejected | 0 of 42 |
| Layers needing recoat or stop, let through | 13 of 170 |
| False stops | 5, that is 27.9 per 1,000 good layers |
| Latency per layer (two frames), arm64 laptop shared with other jobs | median 472 ms, p95 794 ms (registration 180 ms, U-Net 164 ms) |

Every one of the 13 let-through layers was detected; the miss is a policy calibration gap (the label calls a super elevation critical by area, the policy stops on arc length or on two layers in a row). The 5 false stops come from the same rule. I report this unchanged rather than retune on the hold-out.

Real images, zero shot (Aalto University LPBF dataset, EOS M290, CC-BY 4.0, 250 labelled frames, 803 boxes; recoat checks only, thresholds untouched, no DNN):

| Measure | Value |
|---|---|
| Labelled boxes found (recall) | 0.45 |
| Detections on a labelled box (precision) | 0.92 |
| Streak detector precision | 0.96 (200 of 208) |
| Bright blob detector precision | 0.89 (190 of 214) |
| Frames with at least one box found | 0.60 |

Closed loop demo build (480 layers through the real handler): every scripted event found, 4 recoats, 2 re-images, 1 stop, no unscripted action.

### How I built it

Python 3.12, OpenCV 5.0.0 for all vision, PyTorch CPU for training only, FastAPI for the local runner, AWS CDK in Python, a Vite + TypeScript + Three.js dashboard, Playwright to record the film frame by frame and render the PDF report. The synthetic data generator renders 2,400 layers with exact pixel masks in about two and a half minutes on four cores: powder texture, an angled lamp, keystone and lens distortion, per layer vibration, hatch rotated 67 degrees per layer, previous layer bleed, injected defects and nuisances, all randomized per build. The part geometry is one function, ported from a printer simulation I wrote earlier, so the expected slice, the training data and the 3D map always agree.

### Challenges

* ECC on a powder bed: the powder changes every layer, so it cannot align on texture. Masking ECC to the static rim fixed that, but at quarter scale the rim is six pixels wide and the 8 parameter homography sometimes diverged (in one case by 176 px). Translation first, homography at half and full scale, a plausibility bound and a Euclidean fallback took the worst case on the hold-out to 0.38 px.
* Laplacian variance at full resolution could not tell a harmless 2 px shake from a real 10 px motion blur: both remove the per pixel grain. Measuring at half scale plus a structure tensor anisotropy test separated them (0 of 42 mild frames rejected).
* A blotchy powder bed turned grains into "spatter" and dark patches into "short feed". Thresholds that adapt to each frame's measured texture removed every false detection on the demo machine's clean layers.
* OpenCV 5 changed small things: `HoughLinesP` returns N x 4 instead of N x 1 x 4, and the new DNN graph engine ignores `setPreferableTarget`.
* Training on a CPU shared with other jobs: 16 epochs in the 40 minute budget, still improving when it stopped.

### Accomplishments that I'm proud of

* The agent never decides on a bad picture: 50 of 51 unusable frames sent back for a re-image, and no good frame rejected.
* Sub pixel registration on a surface that changes every layer.
* The DNN runs inside OpenCV with the same answer as PyTorch, and rules plus DNN matches or beats rules alone on every class (macro F1 0.91 to 0.95).
* Honest numbers: a frozen hold-out, a separate dev split, real images reported zero shot, and the failures written down.

### What I learned

Most of the work in vision QA is knowing when not to trust the picture. The evidence gate, the plausibility bounds and the texture adaptive thresholds mattered more than any single detector. And a policy is only as good as the agreement between what you label "critical" and what you tell the agent to stop on.

### Responsible use and limits

* Headline numbers are on synthetic images whose defects I designed. They are not real world accuracy. On real images, zero shot, LayerWatch finds 45% of labelled defect boxes; it would need tuning and labelled data from the target machine.
* Camera placement matters: the pipeline assumes a fixed camera with a one time calibration and a visible static rim.
* The demo build samples 480 heights through the part; its events are scripted.
* Not a certification tool. A stop is a request for a person to look. Images stay in the customer's bucket; the worker has no other network target, and the language model only words a note from numbers.
* AWS: written as code and synthesized, not deployed.

### What's next

Align the policy with labelled severity on real builds; train on real machine data (the ORNL Peregrine releases need a Globus account, which I did not set up); deploy on Graviton and measure latency with and without COOL; layer to layer tracking so a streak or crescent is followed in 3D, not only flagged.

### How to reproduce

`make setup && make data && make train && make eval && make demo && make report`, then `make synth` for the AWS stack (`make deploy` with an AWS account).
