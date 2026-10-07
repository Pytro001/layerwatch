// node output/record.mjs [--stills]   needs the dashboard on http://localhost:5304 (make dashboard)
// Deterministic: steps window.__film frame by frame, JPEG frames piped to ffmpeg, 1920x1080, 30 fps.
import { chromium } from '/Users/pytro/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs'
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
const FPS = 30
const STILLS = process.argv.includes('--stills')
const dir = new URL('.', import.meta.url).pathname
const URL_ = process.env.FILM_URL || 'http://localhost:5304/?film=1'
const b = await chromium.launch({ headless: true, args: ['--use-angle=metal', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'] })
const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 })
p.on('pageerror', (e) => console.log('ERR', e.message))
await p.goto(URL_)
await p.waitForFunction(() => window.__film?.ready, null, { timeout: 120000 })
const chs = await p.evaluate(() => window.__film.chapters)
const total = chs.reduce((a, c) => a + c.length, 0)
console.log('chapters', chs.map((c) => `${c.name}:${c.length}`).join(' '), 'total', total, 's')

if (STILLS) {
  mkdirSync(dir + 'stills', { recursive: true })
  // one still per chapter, at a moment that shows its idea
  const at = { print: 11, catch: 11, stop: 12, pipeline: 16, glare: 8, metrics: 10, aws: 8, map: 12 }
  let t0 = 0
  for (const c of chs) {
    await p.evaluate(async ([i, t]) => { await window.__film.goto(i); await window.__film.step(t) }, [chs.indexOf(c), at[c.name] ?? c.length / 2])
    await p.waitForTimeout(150)
    await p.screenshot({ path: `${dir}stills/${String(chs.indexOf(c) + 1).padStart(2, '0')}-${c.name}.jpg`, type: 'jpeg', quality: 85 })
    t0 += c.length
  }
  await b.close()
  console.log('stills done')
  process.exit(0)
}

await p.evaluate(() => window.__film.goto(0))
const n = Math.round(total * FPS)
const ff = spawn('/opt/homebrew/bin/ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-vcodec', 'mjpeg', '-i', 'pipe:0',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dir + 'layerwatch-screen.mp4'])
for (let i = 0; i < n; i++) {
  if (i) await p.evaluate(() => window.__film.step(1 / 30))
  const jpg = await p.screenshot({ type: 'jpeg', quality: 92 })
  if (!ff.stdin.write(jpg)) await new Promise((r) => ff.stdin.once('drain', r))
  if (i % 300 === 0) console.log('frame', i, '/', n)
}
ff.stdin.end()
await new Promise((r) => ff.on('close', r))
await b.close()
console.log('done', dir + 'layerwatch-screen.mp4')
