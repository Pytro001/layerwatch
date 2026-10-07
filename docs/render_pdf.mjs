// node docs/render_pdf.mjs  renders docs/LayerWatch_Report.html to docs/LayerWatch_Report.pdf with Playwright
import { chromium } from '/Users/pytro/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs'
const dir = new URL('.', import.meta.url).pathname
const b = await chromium.launch({ headless: true })
const p = await b.newPage()
await p.goto('file://' + dir + 'LayerWatch_Report.html', { waitUntil: 'load' })
await p.pdf({ path: dir + 'LayerWatch_Report.pdf', format: 'A4', margin: { top: '14mm', bottom: '14mm', left: '12mm', right: '12mm' }, printBackground: true })
await b.close()
console.log('pdf written')
