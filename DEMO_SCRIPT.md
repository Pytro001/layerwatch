# LayerWatch demo (2:50)

Screen film: `output/layerwatch-screen.mp4` (1920x1080, 30 fps, 170 s, no audio). Recorded frame by frame from the
dashboard in `?film=1` mode by `output/record.mjs`. Captions on screen name each step; Konstantin speaks over it.

## Intro to camera (before the screen starts)

> Metal 3D printers build a part one thin layer at a time, for days.
> One bad layer can ruin the whole build, and today a person often finds out at the end.
> I built LayerWatch: it checks every layer with OpenCV 5 and stops the build when it has to.

## Shot list

| Time | On screen | Konstantin says |
|---|---|---|
| 0:00 | 3D printer view: the copper thrust chamber growing under the powder, laser tracing the hatch, layers 150 to 204. Caption "Laser powder bed fusion: one layer at a time" | "This is a rocket thrust chamber printed in copper. Every layer, the machine spreads powder and a laser melts the slice. The camera takes two pictures each time." |
| 0:14 | Monitor view, raw camera frames of layers 205 to 211 after recoat, then layer 212 registered with red boxes. Caption "Layer 212: recoater streak across the part" | "Layer 212. A groove runs right across the bed and across the part. LayerWatch caught it: 413 pixels long, 58 of them on the part." |
| 0:29 | Agent log: RECOAT, then the re-check, then STOP with numbers and the incident note. Caption changes to "The streak survives the recoat: STOP, with the reason" | "The agent first asks for a recoat. The streak is still there, so the blade is likely damaged, and it stops the build. It says why, with the numbers. Claude writes the note for the operator from those numbers, it never measures anything." |
| 0:45 | Stage 1: registration overlay, cyan rim edges. Caption "1 Registration: cv.findTransformECC, cv.warpPerspective" | "Under the hood, all OpenCV 5. First, every frame is lined up with layer zero, using only the steel rim around the bed." |
| 0:52 | Stage 2: slice overlay, green expected contour, magenta melt. Caption "2 Slice vs part file" | "Then it checks the melted area against the slice from the part file. Overlap and shape match." |
| 0:59 | Stage 3: streak response heat and the Hough line. Caption "3 Streaks: line kernel opening, cv.HoughLinesP" | "Streaks are long dark lines along the recoat direction. A long line kernel keeps only those, then Hough finds them." |
| 1:06 | Stage 4: layer 260 spatter circles. Caption "4 Spatter: cv.SimpleBlobDetector, MOG2 over past layers" | "Spatter is bright droplets on the powder. A blob detector, checked against a background model of the previous layers." |
| 1:13 | Stage 5: layer 97 DNN class overlay, short feed in blue. Caption "5 U-Net through cv.dnn.readNetFromONNX" | "A small U-Net runs inside OpenCV's DNN module and catches what the rules miss." |
| 1:20 | Stage 6: layer 171 super elevation crescent. Caption "6 Super elevation: bright crescent on the slice edge" | "When an edge curls up, it shows as a bright crescent. If it grows, the recoater can hit it." |
| 1:27 | Layer 150: raw melt frame with a lamp reflection. Caption "Glare on layer 150" | "Here the lamp reflects right into the camera." |
| 1:33 | Evidence overlay, saturated pixels in red, numbers. Caption "Evidence check: saturated pixels, Laplacian variance" | "LayerWatch measures that first: 1.24 percent of the pixels are saturated." |
| 1:38 | Layer 150 retake, clean, CONTINUE. Caption "Bad evidence means re-image, never a guess" | "So it asks for a new picture instead of guessing. The retake is clean and the build goes on." |
| 1:45 | Evaluation view: P, R, F1 table for rules, DNN, both; agent cards; latency bars; real image panel. Caption "Measured on 400 unseen layers and 250 real frames" | "The numbers. On 400 synthetic layers it never saw, macro F1 is 0.91 with rules, 0.95 with rules plus the network. On 250 real frames from an EOS machine, with no retuning, 92 percent of its detections land on a defect a person marked. It also misses things, and the report says which." |
| 2:11 | AWS view: S3, Lambda arm64, DynamoDB, CloudWatch, alarm, COOL option, status column. Caption "S3, Lambda on Graviton, DynamoDB, CloudWatch" | "On AWS it is one CDK stack: frames into S3, the same worker on Graviton, a layer log, and an alarm when a build stops. It synthesizes; it is not deployed yet." |
| 2:31 | Defect map: ghost part, every finding at its height, slow orbit, hold. Caption "Every layer, checked" | "And at the end, every finding sits where it happened inside the part. Every layer, checked." |
| 2:50 | end | |

## Close to camera

> LayerWatch is open source, with the data generator, the model and the evaluation, so anyone can check the numbers.
> Next step is real machine data and a deploy on Graviton.

## Notes for recording

* The numbers spoken above are the ones on screen (from `docs/eval/summary.json` and the demo run). If you re-run
  `make eval` or `make demo`, re-check them.
* The demo events are scripted injections into a simulated build; say "simulated build" if asked.
