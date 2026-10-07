/**
 * ?film=1: a deterministic, frame stepped tour for the screen recording.
 * State is a pure function of film time, so any frame can be rendered alone.
 *   window.__film = { ready, chapters, total, goto(i), step(dt, n) }
 */
import type { Scene3D } from './scene3d'
import type { State, Stage, View } from './main'
import type { Attempt, Run } from './types'

interface Ctx {
  run: Run
  state: State
  scene: Scene3D
  render: () => void
  pending: Promise<unknown>[]
  load: (u: string) => HTMLImageElement
  urls: (L: number, a: Attempt) => { raw: string; reg: string; bed: string; ov: (k: string) => string }
}
interface Shot { view: View; layer: number; attempt?: number; stage?: Stage; frame?: 'melt' | 'recoat'; caption: string; cam?: [number, number, number, number] }
interface Chapter { name: string; length: number; at: (t: number) => Shot }

const ease = (x: number) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x) }
const lerp = (a: number, b: number, u: number) => a + (b - a) * u

const CHAPTERS: Chapter[] = [
  {
    name: 'print', length: 14,
    at: (t) => ({ view: 'map', layer: Math.floor(lerp(150, 204.99, t / 14)), caption: 'Laser powder bed fusion: one layer at a time',
      cam: [lerp(0.15, 0.75, ease(t / 14)), lerp(0.42, 0.26, ease(t / 14)), lerp(10, 7.8, ease(t / 14)), 1] }),
  },
  {
    name: 'catch', length: 15,
    at: (t) => {
      const L = t < 5.6 ? 205 + Math.floor(t / 0.8) : 212
      return { view: 'monitor', layer: L, attempt: 0, stage: t < 7.5 ? 'camera' : 'registered', frame: 'recoat',
        caption: t < 6 ? 'Every layer, two camera frames' : 'Layer 212: recoater streak across the part' }
    },
  },
  {
    name: 'stop', length: 16,
    at: (t) => ({ view: 'monitor', layer: 212, attempt: t < 5.5 ? 0 : 1, stage: 'registered', frame: 'recoat',
      caption: t < 5.5 ? 'Agent: recoat and check again' : 'The streak survives the recoat: STOP, with the reason' }),
  },
  {
    name: 'pipeline', length: 42,
    at: (t) => {
      const k = Math.min(5, Math.floor(t / 7))
      const steps: Shot[] = [
        { view: 'monitor', layer: 212, attempt: 1, stage: 'registration', frame: 'melt', caption: '1  Registration: cv.findTransformECC, cv.warpPerspective' },
        { view: 'monitor', layer: 212, attempt: 1, stage: 'slice', frame: 'melt', caption: '2  Slice vs part file: thresholds, contours, IoU, cv.matchShapes' },
        { view: 'monitor', layer: 212, attempt: 1, stage: 'streaks', frame: 'recoat', caption: '3  Streaks: line kernel opening, cv.HoughLinesP' },
        { view: 'monitor', layer: 260, attempt: 0, stage: 'spatter', frame: 'melt', caption: '4  Spatter: cv.SimpleBlobDetector, MOG2 over past layers' },
        { view: 'monitor', layer: 97, attempt: 0, stage: 'dnn', frame: 'recoat', caption: '5  U-Net through cv.dnn.readNetFromONNX' },
        { view: 'monitor', layer: 171, attempt: 0, stage: 'spatter', frame: 'melt', caption: '6  Super elevation: bright crescent on the slice edge' },
      ]
      return steps[k]
    },
  },
  {
    name: 'glare', length: 18,
    at: (t) => {
      if (t < 6) return { view: 'monitor', layer: 150, attempt: 0, stage: 'camera', frame: 'melt', caption: 'Glare on layer 150' }
      if (t < 11) return { view: 'monitor', layer: 150, attempt: 0, stage: 'evidence', frame: 'melt', caption: 'Evidence check: saturated pixels, Laplacian variance' }
      return { view: 'monitor', layer: 150, attempt: 1, stage: 'registered', frame: 'melt', caption: 'Bad evidence means re-image, never a guess' }
    },
  },
  { name: 'metrics', length: 26, at: () => ({ view: 'metrics', layer: 479, caption: 'Measured on 400 unseen layers and 250 real frames' }) },
  { name: 'aws', length: 20, at: () => ({ view: 'aws', layer: 479, caption: 'S3, Lambda on Graviton, DynamoDB, CloudWatch' }) },
  {
    name: 'map', length: 19,
    at: (t) => ({ view: 'map', layer: 479, caption: t < 3 ? '' : 'Every layer, checked',
      cam: [lerp(0.9, 2.3, ease(t / 17)), lerp(0.36, 0.26, ease(t / 17)), lerp(11.5, 10.2, ease(t / 17)), 0] }),
  },
]

export function installFilm(c: Ctx) {
  const total = CHAPTERS.reduce((a, ch) => a + ch.length, 0)
  let T = 0
  const starts: number[] = []
  CHAPTERS.reduce((a, ch) => { starts.push(a); return a + ch.length }, 0)

  async function apply() {
    let k = CHAPTERS.length - 1
    for (let i = 0; i < CHAPTERS.length; i++) if (T < starts[i] + CHAPTERS[i].length) { k = i; break }
    const ch = CHAPTERS[k]
    const s = ch.at(Math.min(T - starts[k], ch.length))
    const st = c.state
    st.view = s.view
    st.layer = s.layer
    st.attempt = s.attempt ?? 0
    if (s.stage) st.stage = s.stage
    if (s.frame) st.frame = s.frame
    st.caption = s.caption
    st.progress = T * 0.9
    const printer = ch.name === 'print'
    document.body.classList.toggle('film-print', printer)
    if (s.cam) {
      c.scene.az = s.cam[0]; c.scene.el = s.cam[1]; c.scene.dist = s.cam[2]
      const h = c.run.layers[s.layer].h_m * 10
      c.scene.target.set(0, s.cam[3] ? h * 0.8 : 1.9, 0)
    } else {
      c.scene.az = 0.5 + T * 0.04; c.scene.el = 0.4; c.scene.dist = 8.4
      c.scene.target.set(0, c.run.layers[s.layer].h_m * 10 * 0.8, 0)
    }
    c.render()
    if (printer || s.view === 'map') {
      c.scene.mode = printer ? 'printer' : 'map'
      c.scene.update(st.layer, st.progress)
    }
    const p = c.pending.splice(0)
    if (p.length) {
      await Promise.all(p)
      c.render()
      if (printer || s.view === 'map') { c.scene.mode = printer ? 'printer' : 'map'; c.scene.update(st.layer, st.progress) }
    }
  }

  // preload every frame the tour shows
  const pre: Promise<unknown>[] = []
  for (let L = 150; L <= 212; L++) for (const a of c.run.layers[L].attempts) {
    if (a.action === 'OPERATOR') continue
    const stem = `${String(L).padStart(3, '0')}_${a.attempt}`
    for (const u of [`run/img/${stem}_recoat.jpg`, `run/img/${stem}_melt.jpg`, `run/bed/${stem}_m.jpg`, `run/bed/${stem}_r.jpg`]) pre.push(c.load(u).decode().catch(() => null))
  }
  ;(window as unknown as { __film: unknown }).__film = {
    ready: false,
    chapters: CHAPTERS.map((ch) => ({ name: ch.name, length: ch.length })),
    total,
    async goto(i: number) { T = starts[i] ?? 0; await apply() },
    async step(dt: number, n = 1) { T += dt * n; await apply() },
    time: () => T,
  }
  Promise.all(pre).then(async () => {
    await apply()
    ;(window as unknown as { __film: { ready: boolean } }).__film.ready = true
  })
}
