import type { Run } from './types'
import { TYPE_COLOR, TYPE_LABEL } from './types'

type PRF = { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any

const f2 = (x: number) => x.toFixed(2)
const pct = (x: number) => `${(x * 100).toFixed(0)}%`

export function renderMetrics(el: HTMLElement, ev: Any, run: Run) {
  if (!ev) { el.innerHTML = `<div class="card pad">Run <code>make eval</code> to fill this page.</div>`; return }
  const s = ev.synthetic
  const types = ['streak', 'spatter', 'super_elevation', 'short_feed']
  const modes: [string, string][] = [['rules', 'Rules only'], ['dnn', 'DNN only'], ['fused', 'Rules + DNN']]
  const head = `<tr><th></th>${modes.map(([, l]) => `<th colspan="3" class="grp">${l}</th>`).join('')}</tr>
    <tr><th>defect</th>${modes.map(() => '<th>P</th><th>R</th><th>F1</th>').join('')}</tr>`
  const body = types.map((t) => {
    const best = Math.max(...modes.map(([m]) => (s.detection[m][t] as PRF).f1))
    return `<tr><td><i class="dot" style="background:${TYPE_COLOR[t]}"></i>${TYPE_LABEL[t]}</td>${modes.map(([m]) => {
      const r = s.detection[m][t] as PRF
      return `<td>${f2(r.precision)}</td><td>${f2(r.recall)}</td><td class="${r.f1 === best ? 'best' : ''}">${f2(r.f1)}</td>`
    }).join('')}</tr>`
  }).join('') + `<tr class="macro"><td>macro F1</td>${modes.map(([m]) => `<td></td><td></td><td>${f2(s.detection[m].macro_f1)}</td>`).join('')}</tr>`
  const a = s.agent
  const k = (v: string, n: string, sub = '') => `<div class="big"><b>${v}</b><span>${n}</span>${sub ? `<small>${sub}</small>` : ''}</div>`
  const r = ev.real
  const lat = s.latency_ms
  const stages: [string, string][] = [['registration', 'registration (ECC)'], ['evidence', 'evidence'], ['slice_compare', 'slice compare'],
    ['streaks', 'streaks + short feed'], ['spatter_super_elevation', 'spatter + super elevation'], ['dnn', 'U-Net (cv.dnn)']]
  const maxL = Math.max(...stages.map(([id]) => lat[id]?.median ?? 0))
  el.innerHTML = `
  <section class="card pad m-left">
    <div class="m-title">Synthetic hold-out <span class="muted">${s.meta.n_layers} layers from ${s.meta.builds} simulated builds never used for training or tuning</span></div>
    <table class="prf">${head}${body}</table>
    <div class="bigs">
      ${k(`${a.false_stop_per_1000}`, 'false stops per 1,000 good layers', `${a.false_stops} of ${a.layers - a.bad_frames - a.layers_needing_intervention} layers`)}
      ${k(`${a.missed_critical}`, 'critical layers missed', `of ${a.layers_needing_intervention} that needed recoat or stop`)}
      ${k(`${a.decided_on_bad_frame}`, 'decisions made on a bad frame', `${a.bad_frames} glare or blur frames, gate recall ${f2(s.evidence_gate.recall)}`)}
      ${k(`${s.registration_px.median.toFixed(2)} px`, 'registration error, median', `p95 ${s.registration_px.p95.toFixed(2)} px`)}
    </div>
    <div class="m-sub">Latency per layer on this laptop (${s.meta.machine}), median</div>
    <div class="bars">${stages.map(([id, l]) => `<div class="bar"><span>${l}</span><i style="width:${(100 * (lat[id]?.median ?? 0)) / maxL}%"></i><b>${(lat[id]?.median ?? 0).toFixed(0)} ms</b></div>`).join('')}
      <div class="bar total"><span>whole layer</span><i style="width:0"></i><b>${lat.total.median.toFixed(0)} ms</b></div></div>
  </section>
  <section class="card pad m-right">
    <div class="m-title">Real powder bed images <span class="muted">zero shot</span></div>
    <p class="muted">${r.images} labelled frames from an EOS M290 (Aalto University, CC-BY 4.0). Recoat checks only, thresholds left exactly as tuned on synthetic data, no DNN.</p>
    <div class="bigs two">
      ${k(pct(r.box_recall), 'labelled defect boxes found', `${r.gt_boxes} boxes`)}
      ${k(pct(r.detection_precision), 'detections on a labelled box', `${r.detections} detections`)}
      ${k(pct(r.by_detector.streak?.precision ?? 0), 'streak detector precision', `${r.by_detector.streak?.detections ?? 0} streaks`)}
      ${k(pct(r.images_with_at_least_one_hit), 'frames with a defect found', '')}
    </div>
    <img class="real" src="run/real_examples.jpg" alt="real frames: labelled boxes green, LayerWatch detections red" />
    <p class="muted small">green: annotator boxes, red: LayerWatch. Labels are one class and boxes only, so this measures localisation, not defect type.</p>
  </section>`
}

export function renderAws(el: HTMLElement, ev: Any, run: Run) {
  const s = ev?.synthetic
  const aws = ev?.aws ?? {}
  const box = (x: number, y: number, w: number, h: number, t: string, sub: string, c = '#4da3ff') =>
    `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="#0f141b" stroke="${c}" stroke-width="2"/>
     <text x="${x + 18}" y="${y + 34}" fill="#e8edf3" font-size="22" font-weight="600">${t}</text>
     ${sub.split('|').map((l, i) => `<text x="${x + 18}" y="${y + 62 + i * 24}" fill="#8f9bab" font-size="16">${l}</text>`).join('')}</g>`
  const arrow = (x1: number, y1: number, x2: number, y2: number, label = '', dash = false) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#5d6b7d" stroke-width="2" ${dash ? 'stroke-dasharray="7 6"' : ''} marker-end="url(#ah)"/>` +
    (label ? `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 10}" fill="#8f9bab" font-size="15" text-anchor="middle">${label}</text>` : '')
  el.innerHTML = `
  <section class="card pad aws-main">
    <div class="m-title">On AWS <span class="muted">AWS CDK infrastructure as code: synthesized, not deployed</span></div>
    <svg viewBox="0 0 1180 600" class="arch" role="img" aria-label="architecture: printer camera to S3 to Lambda arm64 to DynamoDB and CloudWatch">
      <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#5d6b7d"/></marker></defs>
      ${box(20, 40, 230, 120, 'Printer camera', '2 frames per layer|recoat + melt, JPEG', '#7f8a99')}
      ${box(330, 40, 250, 120, 'Amazon S3', 'builds/&lt;id&gt;/layers/*.jpg|SSE, TLS only, no public', '#3ddc97')}
      ${box(660, 20, 500, 230, 'AWS Lambda, arm64 (Graviton)', 'container image, OpenCV 5.0 headless|ECC registration, slice compare|streaks, spatter, cv.dnn U-Net|evidence gate, agent policy|2 GB, 60 s, 20 concurrent', '#4da3ff')}
      ${box(660, 330, 240, 130, 'DynamoDB', 'layer log: build, layer, try|decision + numbers', '#ffb020')}
      ${box(930, 330, 230, 130, 'CloudWatch', 'EMF metrics: latency,|stops, recoats, re-images', '#ff8c42')}
      ${box(930, 490, 230, 90, 'Alarm + SNS', 'StopEvents >= 1', '#ff4d4f')}
      ${box(330, 330, 250, 130, 'COOL path (option)', 'EC2 c7g from the COOL AMI|SQS queue, same handler', '#8ab4ff')}
      ${box(20, 330, 230, 130, 'Operator', 'incident note|stop reason + numbers', '#c7cdd6')}
      ${arrow(250, 100, 328, 100, 'PUT')}
      ${arrow(580, 100, 658, 100, 'ObjectCreated')}
      ${arrow(780, 250, 780, 328)}
      ${arrow(1045, 250, 1045, 328)}
      ${arrow(1045, 460, 1045, 488)}
      <path d="M930,650 L140,650 L140,584" fill="none" stroke="#5d6b7d" stroke-width="2" marker-end="url(#ah)"/><text x="540" y="640" fill="#8f9bab" font-size="15" text-anchor="middle">email</text>
      ${arrow(455, 160, 455, 328, 'S3 to SQS', true)}
    </svg>
  </section>
  <section class="card pad aws-side">
    <div class="m-title">Status</div>
    <div class="row"><span>cdk synth</span><b class="ok">passes, ${aws.resources ?? 18} resources</b></div>
    <div class="row"><span>same handler, local endpoint</span><b class="ok">FastAPI :8304, tested</b></div>
    <div class="row"><span>arm64 dependencies (OpenCV 5, numpy, boto3)</span><b class="ok">resolve for Linux aarch64</b></div>
    <div class="row"><span>cdk deploy</span><b class="warn">not deployed, no AWS account used</b></div>
    <div class="row"><span>arm64 container image build</span><b class="warn">not built here (no Docker)</b></div>
    <div class="row"><span>Graviton latency</span><b class="warn">not measured</b></div>
    <div class="m-sub">Latency per layer today</div>
    <div class="row"><span>this laptop (${s?.meta?.machine ?? 'arm64'}), median</span><b>${s ? s.latency_ms.total.median.toFixed(0) : run.latency_ms.median.toFixed(0)} ms</b></div>
    <div class="row"><span>p95</span><b>${s ? s.latency_ms.total.p95.toFixed(0) : run.latency_ms.p95.toFixed(0)} ms</b></div>
    <div class="m-sub">Responsible ops</div>
    <ul class="ops">
      <li>Frames never leave the bucket; the worker can only get and put <code>builds/*</code> and write its own table.</li>
      <li>IAM authenticated endpoint, TLS enforced, 180 day retention on frames.</li>
      <li>The LLM never measures: Claude on Bedrock only words the stop note from logged numbers.</li>
      <li>Bad evidence means re-image, never a guess.</li>
    </ul>
  </section>`
}
