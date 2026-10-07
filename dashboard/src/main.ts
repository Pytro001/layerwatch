import './style.css'
import { Scene3D } from './scene3d'
import { installFilm } from './film'
import { renderAws, renderMetrics } from './panels'
import type { Attempt, Layer, Run } from './types'
import { ACTION_COLOR, TYPE_COLOR, TYPE_LABEL } from './types'

export type View = 'monitor' | 'metrics' | 'aws' | 'map'
export type Stage = 'camera' | 'registered' | 'registration' | 'slice' | 'streaks' | 'spatter' | 'dnn' | 'evidence'
export const STAGES: { id: Stage; label: string; cv: string }[] = [
  { id: 'camera', label: 'Camera', cv: 'raw frame' },
  { id: 'registered', label: 'Result', cv: 'detections' },
  { id: 'registration', label: 'Registration', cv: 'findTransformECC' },
  { id: 'slice', label: 'Slice', cv: 'matchShapes' },
  { id: 'streaks', label: 'Streaks', cv: 'HoughLinesP' },
  { id: 'spatter', label: 'Spatter', cv: 'SimpleBlobDetector' },
  { id: 'dnn', label: 'DNN', cv: 'cv.dnn' },
  { id: 'evidence', label: 'Evidence', cv: 'Laplacian' },
]

export interface State {
  layer: number
  attempt: number
  frame: 'melt' | 'recoat'
  stage: Stage
  view: View
  playing: boolean
  progress: number
  caption: string
  highlight: string | null
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const pad = (n: number) => String(n).padStart(3, '0')

async function boot() {
  const run: Run = await fetch('run/run.json').then((r) => r.json())
  const evalData = await fetch('run/eval.json').then((r) => (r.ok ? r.json() : null)).catch(() => null)
  const film = new URLSearchParams(location.search).has('film')
  document.body.classList.toggle('film', film)

  const state: State = { layer: 212, attempt: 0, frame: 'recoat', stage: 'registered', view: 'monitor', playing: false, progress: 0, caption: '', highlight: null }
  $('buildLabel').textContent = `${run.build_id} · copper thrust chamber · ${run.n_layers} layers`

  // 3D
  const canvas = document.createElement('canvas')
  $('threeHost').appendChild(canvas)
  const scene = new Scene3D(canvas, run)

  // stage tabs
  const tabs = $('stageTabs')
  STAGES.forEach((s, k) => {
    const b = document.createElement('button')
    b.setAttribute('role', 'tab')
    b.dataset.stage = s.id
    b.innerHTML = `<span>${s.label}</span><small>${s.cv}</small><kbd>${k + 1}</kbd>`
    b.onclick = () => { state.stage = s.id; render() }
    tabs.appendChild(b)
  })
  document.querySelectorAll<HTMLButtonElement>('.views button').forEach((b) => {
    b.onclick = () => { state.view = b.dataset.view as View; render() }
  })
  renderMetrics($('metrics'), evalData, run)
  renderAws($('aws'), evalData, run)

  // events: layers where the agent did more than continue, or found something
  const events = run.layers.filter((L) => L.worst !== 'CONTINUE' || L.attempts.some((a) => (a.detections?.length ?? 0) > 0)).map((L) => L.i)

  const att = (): Attempt => {
    const L = run.layers[state.layer]
    const real = L.attempts.filter((a) => a.action !== 'OPERATOR')
    return real[Math.min(state.attempt, real.length - 1)]
  }

  const imgCache = new Map<string, HTMLImageElement>()
  const load = (url: string): HTMLImageElement => {
    let im = imgCache.get(url)
    if (!im) {
      im = new Image()
      im.src = url
      imgCache.set(url, im)
      if (imgCache.size > 400) imgCache.delete(imgCache.keys().next().value as string)
    }
    return im
  }
  const pending: Promise<unknown>[] = []
  const urls = (L: number, a: Attempt) => {
    const stem = `${pad(L)}_${a.attempt}`
    return {
      raw: `run/img/${stem}_${state.frame}.jpg`,
      reg: `run/bed/${stem}_${state.frame === 'melt' ? 'm' : 'r'}.jpg`,
      bed: `run/bed/${stem}_m.jpg`,
      ov: (k: string) => `run/ov/${stem}_${k}.jpg`,
    }
  }

  // ---------------------------------------------------------------- render
  let lastBed = ''
  function render() {
    const L: Layer = run.layers[state.layer]
    const a = att()
    const u = urls(state.layer, a)
    document.querySelectorAll<HTMLElement>('.view').forEach((v) => (v.hidden = v.id !== state.view))
    document.querySelectorAll<HTMLButtonElement>('.views button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === state.view)))
    $('layerNo').innerHTML = `Layer <b>${state.layer}</b> / ${run.n_layers - 1} <span class="muted">h ${(L.h_m * 1000).toFixed(1)} mm</span>`
    const pill = $('statusPill')
    pill.textContent = a.action.replace('_', '-')
    pill.style.setProperty('--c', ACTION_COLOR[a.action])

    // frame
    const stage = state.stage
    const hasOv = !!a.overlays
    let src = u.raw
    let tag = `camera frame, ${state.frame === 'melt' ? 'after melting' : 'after recoat'} (lens + keystone + vibration)`
    if (stage === 'registered') { src = u.reg; tag = `registered to layer 0, ${state.frame === 'melt' ? 'after melting' : 'after recoat'}` }
    else if (stage !== 'camera') {
      if (hasOv) { src = u.ov(stage); tag = STAGES.find((s) => s.id === stage)!.label + ' overlay' }
      else { src = u.reg; tag = 'overlays are recorded on event layers and every 24th layer' }
    }
    const fi = $<HTMLImageElement>('frameImg')
    if (fi.getAttribute('src') !== src) { fi.setAttribute('src', src); pending.push(fi.decode().catch(() => null)) }
    $('frameTag').textContent = tag
    drawBoxes(stage === 'registered' ? a : null)
    $('measure').innerHTML = measureHtml(stage, a, L)

    // tabs
    tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', String((b as HTMLElement).dataset.stage === stage)))

    // agent log
    renderLog()
    renderKpis()
    drawTimeline()

    // 3D
    const host = state.view === 'map' ? $('mapHost') : $('threeHost')
    if (canvas.parentElement !== host) host.appendChild(canvas)
    if (state.view === 'monitor' || state.view === 'map') {
      const r = host.getBoundingClientRect()
      if (canvas.width !== Math.round(r.width) || canvas.height !== Math.round(r.height)) scene.setSize(Math.round(r.width), Math.round(r.height))
      scene.mode = state.view === 'map' ? 'map' : 'printer'
      if (u.bed !== lastBed) {
        const im = load(u.bed)
        lastBed = u.bed
        if (im.complete) scene.setBed(im)
        else pending.push(im.decode().then(() => scene.setBed(im)).catch(() => null))
      }
      scene.update(state.layer, state.progress)
      $('threeTag').innerHTML = scene.mode === 'printer'
        ? `printing layer ${state.layer}, hatch ${(67 * state.layer) % 180}° &middot; defects found so far shown in place`
        : ''
    }
    if (state.view === 'map') renderMapLegend()
    const cap = $('caption')
    cap.hidden = !state.caption
    cap.textContent = state.caption
  }

  function drawBoxes(a: Attempt | null) {
    const svg = $('boxes')
    if (!a || !a.detections) { svg.innerHTML = ''; return }
    const placed: [number, number, number][] = []
    svg.innerHTML = a.detections.map((d) => {
      const [x0, y0, x1, y1] = d.bbox
      const c = TYPE_COLOR[d.type] ?? '#fff'
      const p = 4
      const label = `${TYPE_LABEL[d.type] ?? d.type}${d.source && d.source !== 'rules' ? ' \u00b7 ' + d.source : ''}`
      const w = label.length * 6.2
      let ty = Math.max(10, y0 - p - 4)
      if (placed.some(([px, py, pw]) => Math.abs(py - ty) < 13 && x0 - p < px + pw && px < x0 - p + w)) ty = Math.min(508, y1 + p + 12)
      const tx = Math.min(x0 - p, 508 - w)
      placed.push([tx, ty, w])
      return `<rect x="${x0 - p}" y="${y0 - p}" width="${x1 - x0 + 2 * p}" height="${y1 - y0 + 2 * p}" fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke" rx="2"/>` +
        `<text x="${tx}" y="${ty}" fill="${c}" font-size="11">${label}</text>`
    }).join('')
  }

  function row(k: string, v: string, hl = false) { return `<div class="row${hl ? ' hl' : ''}"><span>${k}</span><b>${v}</b></div>` }
  function measureHtml(stage: Stage, a: Attempt, L: Layer): string {
    const t = a.timing_ms ?? {}
    const ev = a.evidence_numbers
    const head = `<div class="m-h">${STAGES.find((s) => s.id === stage)!.label}</div>`
    const inj = a.injected?.length ? `<div class="inj">injected in this frame: ${a.injected.join(', ')}</div>` : ''
    switch (stage) {
      case 'camera':
        return head + row('frame', '512 x 512 grey') + row('layer height', `${(L.h_m * 1000).toFixed(2)} mm`) + row('attempt', String(a.attempt)) + row('total time', `${(t.total ?? 0).toFixed(0)} ms`) + inj +
          `<p class="m-p">Two frames per layer: after the recoater spreads powder, and after the laser melts the slice. F switches.</p>`
      case 'registered': {
        const ds = a.detections ?? []
        return head + (ds.length ? ds.map((d) => row(TYPE_LABEL[d.type] ?? d.type, sizeOf(d), true)).join('') : row('detections', 'none')) +
          row('ECC correlation', (a.ecc_cc ?? 0).toFixed(3)) + inj
      }
      case 'registration':
        return head + row('cv.findTransformECC', 'MOTION_HOMOGRAPHY') + row('pyramid', '1/4, 1/2, full') + row('ECC correlation', (a.ecc_cc ?? 0).toFixed(3)) +
          row('ECC mask', 'chamber rim + plate edge') + row('time', `${(t.registration ?? 0).toFixed(0)} ms`) +
          `<p class="m-p">Lens and keystone come from calibration; per layer vibration is solved against the layer 0 frame, using only the static rim (cyan edges).</p>`
      case 'slice':
        return head + row('expected (part function)', `${a.slice?.expected_px ?? 0} px`) + row('observed melt', `${a.slice?.observed_px ?? 0} px`) +
          row('IoU', (a.slice?.iou ?? 0).toFixed(3), true) + row('cv.matchShapes (I1)', a.slice?.match_shapes == null ? 'n/a' : a.slice.match_shapes.toFixed(4)) +
          row('time', `${(t.slice_compare ?? 0).toFixed(0)} ms`) + `<p class="m-p">Green: expected slice. Magenta: melted region found with Otsu + adaptiveThreshold, morphology and contours.</p>`
      case 'streaks': {
        const ss = a.streaks ?? []
        return head + (ss.length ? ss.map((s, k) => row(`streak ${k + 1}`, `${s.length_px} px, ${s.angle_deg.toFixed(1)}°, ${s.inside_part_px} px on part`, true)).join('') : row('streaks', 'none')) +
          row('time', `${(t.streaks ?? 0).toFixed(0)} ms`) + `<p class="m-p">Dark detail, opened with a 41 px line kernel along the recoat axis, then cv.HoughLinesP; segments merged and scored by length and angle.</p>`
      }
      case 'spatter': {
        const sp = (a.detections ?? []).filter((d) => d.type === 'spatter' || d.type === 'super_elevation')
        return head + (sp.length ? sp.map((d) => row(TYPE_LABEL[d.type], sizeOf(d), true)).join('') : row('spatter, super elevation', 'none')) +
          row('time', `${(t.spatter_super_elevation ?? 0).toFixed(0)} ms`) + `<p class="m-p">White top hat + cv.SimpleBlobDetector, confirmed against a MOG2 background of previous layers; bright crescents on the slice edge.</p>`
      }
      case 'dnn': {
        const dd = (a.detections ?? []).filter((d) => (d.source ?? '').includes('dnn'))
        return head + row('model', 'U-Net, 3 in, 5 classes') + row('runtime', 'cv.dnn.readNetFromONNX') + row('time', `${(t.dnn ?? 0).toFixed(0)} ms`) +
          (dd.length ? dd.map((d) => row(TYPE_LABEL[d.type], `${d.source}`, true)).join('') : row('DNN agrees or adds', 'nothing')) +
          `<p class="m-p">Input: registered recoat, registered melt, expected slice. Trained on synthetic layers only.</p>`
      }
      case 'evidence':
        return head + (ev ? row('melt: Laplacian ratio', ev.melt.lap_ratio.toFixed(2)) + row('melt: anisotropy', ev.melt.anisotropy.toFixed(2)) +
          row('melt: saturated', `${(ev.melt.saturated_frac * 100).toFixed(2)} %`) + row('recoat: Laplacian ratio', ev.recoat.lap_ratio.toFixed(2)) +
          row('recoat: saturated', `${(ev.recoat.saturated_frac * 100).toFixed(2)} %`) : '') +
          row('usable', a.evidence?.ok ? 'yes' : 'no', !a.evidence?.ok) + (a.evidence?.reasons ?? []).map((r) => `<p class="m-p warn">${r}</p>`).join('')
    }
  }
  function sizeOf(d: { type: string; length_px?: number; area_px?: number; count?: number; arc_len_px?: number; inside_part_px?: number }) {
    if (d.type === 'streak') return `${d.length_px ?? '?'} px long, ${d.inside_part_px ?? 0} px on part`
    if (d.type === 'spatter') return `${d.count ?? '?'} blobs`
    if (d.type === 'super_elevation') return d.arc_len_px ? `arc ${Math.round(d.arc_len_px)} px` : `${d.area_px ?? '?'} px`
    return `${d.area_px ?? '?'} px bare`
  }

  function renderLog() {
    // chronological cursor: (layer, index in that layer's attempts, operator steps included)
    const curLayer = run.layers[state.layer]
    const curA = att()
    const kCur = curLayer.attempts.indexOf(curA)
    const items: string[] = []
    for (let i = state.layer; i >= 0 && items.length < 7; i--) {
      const at = run.layers[i].attempts
      for (let k = (i === state.layer ? kCur : at.length - 1); k >= 0 && items.length < 7; k--) {
        const a = at[k]
        const isCur = i === state.layer && k === kCur
        if (!isCur && a.action === 'CONTINUE' && a.reason === 'layer clean') continue
        const nums = a.numbers ? Object.entries(a.numbers).filter(([, v]) => typeof v === 'number').map(([k2, v]) => `<code>${k2} ${v}</code>`).join('') : ''
        const note = a.note ? `<div class="note"><span>incident note${a.note_source && a.note_source !== 'template' ? ' \u00b7 written by Claude on Bedrock from these numbers' : ''}</span>${a.note}</div>` : ''
        items.push(`<li class="${isCur ? 'cur' : ''}" style="--c:${ACTION_COLOR[a.action]}"><div class="l-h"><span>Layer ${i}${a.action === 'OPERATOR' ? '' : ` \u00b7 try ${a.attempt + 1}`}</span><b>${a.action.replace('_', '-')}</b></div><p>${a.reason}</p>${nums ? `<div class="nums">${nums}</div>` : ''}${note}</li>`)
      }
    }
    $('log').innerHTML = items.join('')
  }

  function renderKpis() {
    let stops = 0, recoats = 0, reim = 0
    const lat: number[] = []
    const curA = att()
    for (let i = 0; i <= state.layer; i++) for (const a of run.layers[i].attempts) {
      if (i === state.layer && a.attempt > curA.attempt) continue
      if (a.action === 'STOP') stops++
      if (a.action === 'RECOAT') recoats++
      if (a.action === 'RE_IMAGE') reim++
      if (a.timing_ms) lat.push(a.timing_ms.total)
    }
    lat.sort((x, y) => x - y)
    const med = lat.length ? lat[Math.floor(lat.length / 2)] : 0
    const k = (n: string, v: string, c?: string) => `<div class="kpi"${c ? ` style="--c:${c}"` : ''}><b>${v}</b><span>${n}</span></div>`
    $('kpis').innerHTML = k('layers checked', String(state.layer + 1)) + k('stops', String(stops), ACTION_COLOR.STOP) + k('recoats', String(recoats), ACTION_COLOR.RECOAT) +
      k('re-images', String(reim), ACTION_COLOR.RE_IMAGE) + k('median per layer', `${med.toFixed(0)} ms`)
  }

  const tl = $<HTMLCanvasElement>('tl')
  function drawTimeline() {
    const g = tl.getContext('2d')!
    const W = tl.width, H = tl.height, n = run.n_layers, bw = W / n
    g.clearRect(0, 0, W, H)
    for (let i = 0; i < n; i++) {
      const L = run.layers[i]
      const done = i <= state.layer
      let c = L.worst === 'CONTINUE' ? (L.attempts.some((a) => a.detections?.length) ? '#4f6b5f' : '#24342e') : ACTION_COLOR[L.worst]
      g.globalAlpha = done ? 1 : 0.28
      g.fillStyle = c
      const hh = L.worst === 'CONTINUE' ? (c === '#4f6b5f' ? 26 : 16) : 44
      g.fillRect(i * bw, H - hh - 6, Math.max(1, bw - 0.6), hh)
    }
    g.globalAlpha = 1
    g.fillStyle = '#e8edf3'
    g.fillRect(state.layer * bw + bw / 2 - 1, 0, 2, H)
  }
  tl.onclick = (e) => {
    const r = tl.getBoundingClientRect()
    state.layer = Math.max(0, Math.min(run.n_layers - 1, Math.floor(((e.clientX - r.left) / r.width) * run.n_layers)))
    state.attempt = 0
    render()
  }

  function renderMapLegend() {
    const counts: Record<string, number> = {}
    for (const p of run.points) counts[p.type] = (counts[p.type] ?? 0) + 1
    $('mapLegend').innerHTML = `<div class="m-h">Every layer, checked</div><p class="muted">${run.n_layers} layers, ${run.points.length} findings placed at their height and bed position inside the part</p>` +
      Object.entries(TYPE_LABEL).map(([k, v]) => `<div class="row"><span><i class="dot" style="background:${TYPE_COLOR[k]}"></i>${v}</span><b>${counts[k] ?? 0}</b></div>`).join('') +
      `<div class="row"><span><i class="ring" style="border-color:${ACTION_COLOR.STOP}"></i>stop</span><b>${run.stops.length}</b></div>`
  }

  // ---------------------------------------------------------------- input
  function step(dl: number) {
    state.layer = Math.max(0, Math.min(run.n_layers - 1, state.layer + dl))
    state.attempt = 0
    render()
  }
  function nextEvent(dir: 1 | -1) {
    const e = dir > 0 ? events.find((i) => i > state.layer) : [...events].reverse().find((i) => i < state.layer)
    if (e !== undefined) { state.layer = e; state.attempt = 0; render() }
  }
  window.addEventListener('keydown', (e) => {
    if (film) return
    const k = e.key
    if (k === 'ArrowRight') step(e.shiftKey ? 10 : 1)
    else if (k === 'ArrowLeft') step(e.shiftKey ? -10 : -1)
    else if (k === 'ArrowUp') { state.attempt = Math.max(0, state.attempt - 1); render() }
    else if (k === 'ArrowDown') { const n = run.layers[state.layer].attempts.filter((a) => a.action !== 'OPERATOR').length; state.attempt = Math.min(n - 1, state.attempt + 1); render() }
    else if (k === ' ') { state.playing = !state.playing; $('playBtn').textContent = state.playing ? 'Pause' : 'Play'; e.preventDefault() }
    else if (k >= '1' && k <= '8') { state.stage = STAGES[+k - 1].id; render() }
    else if (k === 'f' || k === 'F') { state.frame = state.frame === 'melt' ? 'recoat' : 'melt'; render() }
    else if (k === 'n' || k === 'N') nextEvent(1)
    else if (k === 'p' || k === 'P') nextEvent(-1)
    else if (k === 'q' || k === 'Q') { state.view = 'monitor'; render() }
    else if (k === 'w' || k === 'W') { state.view = 'metrics'; render() }
    else if (k === 'e' || k === 'E') { state.view = 'aws'; render() }
    else if (k === 'r' || k === 'R') { state.view = 'map'; render() }
    else if (k === 'a' || k === 'A') apiCheck()
    else return
  })
  $('playBtn').onclick = () => { state.playing = !state.playing; $('playBtn').textContent = state.playing ? 'Pause' : 'Play' }
  $('apiBtn').onclick = () => apiCheck()

  // live endpoint: send this layer's two frames through the same handler the Lambda runs
  async function apiCheck() {
    const toast = $('toast')
    const show = (s: string) => { toast.hidden = false; toast.innerHTML = s }
    try {
      const b64 = async (u: string) => {
        const buf = new Uint8Array(await (await fetch(u)).arrayBuffer())
        let s = ''
        for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000))
        return btoa(s)
      }
      show('sending to /api ...')
      const id = 'dashboard-check'
      const reg = await fetch(`/api/builds/${id}`, { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ camera: (run as unknown as { camera: object }).camera, reference_b64: await b64('run/img/ref.jpg'), n_layers: run.n_layers }) })
      if (!reg.ok) throw new Error(`register ${reg.status}`)
      const a = att()
      const stem = `${pad(state.layer)}_${a.attempt}`
      const res = await fetch(`/api/builds/${id}/layers`, { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ layer_index: state.layer, attempt: a.attempt, recoats: a.recoats ?? 0,
          recoat_b64: await b64(`run/img/${stem}_recoat.jpg`), melt_b64: await b64(`run/img/${stem}_melt.jpg`) }) })
      if (!res.ok) throw new Error(`layer ${res.status}`)
      const j = await res.json()
      show(`<b style="color:${ACTION_COLOR[j.decision.action]}">${j.decision.action}</b> from the local endpoint in ${j.timing_ms.handler_total.toFixed(0)} ms (${j.arch}, OpenCV ${j.opencv})<br><span class="muted">${j.decision.reason}</span>`)
    } catch (err) {
      show(`endpoint not reachable (${(err as Error).message}). Start it with <code>make serve</code>.`)
    }
    setTimeout(() => (toast.hidden = true), 7000)
  }

  // playback
  let lastT = performance.now()
  function tick(now: number) {
    const dt = Math.min(0.1, (now - lastT) / 1000)
    lastT = now
    if (!film) {
      state.progress += dt * 0.6
      if (state.playing) {
        state.layer += 1
        state.attempt = 0
        if (state.layer >= run.n_layers) { state.layer = run.n_layers - 1; state.playing = false }
        render()
      } else if (state.view === 'monitor' || state.view === 'map') {
        if (state.view === 'map') scene.az += dt * 0.12
        scene.update(state.layer, state.progress)
      }
    }
    requestAnimationFrame(tick)
  }
  window.addEventListener('resize', () => render())
  render()
  if (film) installFilm({ run, state, scene, render, pending, load, urls })
  else requestAnimationFrame(tick)
}

boot()
