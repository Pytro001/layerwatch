/**
 * The test part (regeneratively cooled thrust chamber, 36 cooling channels,
 * four tensile test bars). TypeScript copy of layerwatch/part.py: keep in step.
 * Units: metres, plate centred on the origin, h up from the plate.
 */
export const PART = {
  RE: 0.13, RT: 0.032, RC: 0.07, HT: 0.16, HC: 0.22, HJ: 0.3, RH: 0.085,
  HTOP: 0.355, HPIPE: 0.385, WALL: 0.011, N: 36, BAR: 0.17, BAR_R: 0.006, BAR_H: 0.25,
}
const P = PART
export const TOR_RC = P.RE + P.WALL + 0.016, TOR_H = 0.022, TOR_O = 0.016, TOR_I = 0.0115
const sm = (x: number) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x) }

export function rIn(h: number) {
  if (h < P.HT) { const u = (P.HT - h) / P.HT; return P.RT + (P.RE - P.RT) * (1 - Math.pow(1 - u, 1.7)) }
  if (h < P.HC) return P.RT + (P.RC - P.RT) * sm((h - P.HT) / (P.HC - P.HT))
  return P.RC
}
export function rOut(h: number) {
  let r = rIn(h) + P.WALL
  if (h > P.HJ - 0.02) r = Math.max(r, P.RC + P.WALL + (P.RH - P.RC - P.WALL) * sm((h - (P.HJ - 0.02)) / 0.03))
  for (const b of [0.055, 0.115]) if (Math.abs(h - b) < 0.006) r += 0.005 * sm((0.006 - Math.abs(h - b)) / 0.002)
  return r
}

/** true where there is metal */
export function solid(x: number, z: number, h: number): boolean {
  if (h < 0 || h > P.HPIPE) return false
  const r = Math.hypot(x, z)
  if (h < P.BAR_H && Math.abs(Math.abs(x) - P.BAR) < 0.02 && Math.abs(Math.abs(z) - P.BAR) < 0.02) {
    if (Math.hypot(Math.abs(x) - P.BAR, Math.abs(z) - P.BAR) < P.BAR_R) return true
  }
  const dt = Math.hypot(r - TOR_RC, h - TOR_H)
  if (dt < TOR_O) return dt > TOR_I
  if (h > TOR_H - 0.006 && h < TOR_H + 0.006 && r > rIn(h) && r < TOR_RC) return !(Math.abs(h - TOR_H) < 0.0025 && r > rIn(h) + 0.002)
  if (h < 0.312) {
    const ri = rIn(h), ro = rOut(h)
    if (r < ri || r > ro) return false
    const a = Math.atan2(z, x)
    const f = Math.abs(((((a * P.N) / (Math.PI * 2) + 0.5) % 1) + 1) % 1 - 0.5)
    const c0 = ri + 0.0022, c1 = Math.min(ri + 0.0075, ro - 0.0025)
    if (f < 0.24 && r > c0 && r < c1 && h > TOR_H - 0.002 && h < 0.305) return false
    return true
  }
  if (h < P.HTOP) {
    if (r > P.RH) return false
    if (h < 0.33) {
      if (r < P.RC) {
        for (const [rr, n] of [[0, 1], [0.022, 8], [0.042, 14], [0.06, 20]] as const) {
          if (Math.abs(r - rr) > 0.004) continue
          if (rr === 0) return false
          const a = Math.atan2(z, x), k = Math.round((a / (Math.PI * 2)) * n)
          if (Math.hypot(x - Math.cos((k / n) * Math.PI * 2) * rr, z - Math.sin((k / n) * Math.PI * 2) * rr) < 0.0032) return false
        }
      }
      return true
    }
    if (h < 0.344) return r > 0.066
    return r > 0.0125 || h > 0.351
  }
  return r < 0.019 && r > 0.0125
}

/** outer (r, h) profile for a lathe: the outer skin, the top, the inner bore */
export function outerProfile(n = 140): [number, number][] {
  const pts: [number, number][] = []
  for (let i = 0; i <= n; i++) { const h = (i / n) * P.HTOP; pts.push([rOut(h), h]) }
  pts.push([0.019, P.HTOP], [0.019, P.HPIPE], [0.0125, P.HPIPE], [0.0125, 0.344], [0.0001, 0.344])
  return pts
}
export function innerProfile(n = 100): [number, number][] {
  const pts: [number, number][] = []
  for (let i = 0; i <= n; i++) { const h = (i / n) * 0.312; pts.push([rIn(h), h]) }
  return pts
}

/** hatch segments (plate metres) for the laser animation: lines at angle a, spacing s */
export function hatch(h: number, a: number, s = 0.004, half = 0.2): [number, number, number, number][] {
  const ca = Math.cos(a), sa = Math.sin(a), out: [number, number, number, number][] = []
  const step = 0.0016
  for (let v = -0.29; v <= 0.29; v += s) {
    let inside = false, u0 = 0
    for (let u = -0.29; u <= 0.29; u += step) {
      const x = u * ca - v * sa, z = u * sa + v * ca
      const m = Math.abs(x) <= half && Math.abs(z) <= half && solid(x, z, h)
      if (m && !inside) { inside = true; u0 = u }
      if (!m && inside) { inside = false; out.push([u0 * ca - v * sa, u0 * sa + v * ca, (u - step) * ca - v * sa, (u - step) * sa + v * ca]) }
    }
  }
  return out
}
