import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { PART, TOR_H, TOR_O, TOR_RC, hatch, innerProfile, outerProfile } from './part'
import type { Point, Run } from './types'
import { TYPE_COLOR } from './types'

const K = 10 // scene units per metre
export type Mode3D = 'printer' | 'map'

export class Scene3D {
  renderer: THREE.WebGLRenderer
  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(32, 1, 0.05, 200)
  private printed: THREE.Group
  private ghostMat!: THREE.MeshStandardMaterial
  private ghost: THREE.Group
  private clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0)
  private powder: THREE.Mesh
  private bedTop: THREE.Mesh
  private bedMat: THREE.MeshStandardMaterial
  private laser: THREE.Mesh
  private spot: THREE.Sprite
  private markers = new THREE.Group()
  private layerPlane: THREE.Mesh
  private hatchCache = new Map<number, [number, number, number, number][]>()
  mode: Mode3D = 'printer'
  h = 0
  az = 0.6
  el = 0.42
  dist = 9
  target = new THREE.Vector3(0, 1.4, 0)

  constructor(private canvas: HTMLCanvasElement, private run: Run) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, alpha: false })
    this.renderer.setPixelRatio(1)
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.05
    this.renderer.localClippingEnabled = true
    this.scene.background = new THREE.Color(0x07090c)
    const pm = new THREE.PMREMGenerator(this.renderer)
    this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture
    this.scene.fog = new THREE.Fog(0x07090c, 18, 40)

    const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(4, 9, 6); this.scene.add(key)
    const rim = new THREE.DirectionalLight(0x8fb4ff, 0.7); rim.position.set(-6, 4, -5); this.scene.add(rim)
    this.scene.add(new THREE.HemisphereLight(0x9aa8c0, 0x101418, 0.35))

    // build plate and a floor disc
    const plate = new THREE.Mesh(new THREE.BoxGeometry(0.44 * K, 0.25, 0.44 * K),
      new THREE.MeshStandardMaterial({ color: 0x3c4148, metalness: 0.85, roughness: 0.45 }))
    plate.position.y = -0.125
    this.scene.add(plate)
    const floor = new THREE.Mesh(new THREE.CircleGeometry(14, 64),
      new THREE.MeshStandardMaterial({ color: 0x0c0f13, metalness: 0.2, roughness: 0.9 }))
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.26
    this.scene.add(floor)

    // the part, printed (copper, clipped at the current layer) and ghost (whole part, faint)
    const copper = new THREE.MeshStandardMaterial({ color: 0xc8794a, metalness: 1, roughness: 0.32, clippingPlanes: [this.clip], side: THREE.DoubleSide })
    this.printed = this.partGroup(copper)
    this.scene.add(this.printed)
    const ghostMat = this.ghostMat = new THREE.MeshStandardMaterial({ color: 0x9fc3ff, metalness: 0, roughness: 1, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide })
    this.ghost = this.partGroup(ghostMat)
    const edges = new THREE.LineBasicMaterial({ color: 0x6f8fbf, transparent: true, opacity: 0.22 })
    for (const m of [...this.ghost.children]) {
      if ((m as THREE.Mesh).geometry) this.ghost.add(new THREE.LineSegments(new THREE.EdgesGeometry((m as THREE.Mesh).geometry, 25), edges))
    }
    this.scene.add(this.ghost)

    // powder cake up to the current layer, with the registered bed frame on top
    this.powder = new THREE.Mesh(new THREE.BoxGeometry(0.4 * K, 1, 0.4 * K),
      new THREE.MeshStandardMaterial({ color: 0x6d6d6a, roughness: 0.95, metalness: 0.1, transparent: true, opacity: 0.55 }))
    this.scene.add(this.powder)
    this.bedMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, metalness: 0.05 })
    this.bedTop = new THREE.Mesh(new THREE.PlaneGeometry(0.44 * K, 0.44 * K), this.bedMat)
    this.bedTop.rotation.x = -Math.PI / 2
    this.scene.add(this.bedTop)

    // a thin glowing plane marking the current layer in map mode
    this.layerPlane = new THREE.Mesh(new THREE.PlaneGeometry(0.42 * K, 0.42 * K),
      new THREE.MeshBasicMaterial({ color: 0x4da3ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false }))
    this.layerPlane.rotation.x = -Math.PI / 2
    this.scene.add(this.layerPlane)

    // laser beam and melt pool spot
    this.laser = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 1, 8, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x9fd0ff, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }))
    this.scene.add(this.laser)
    const c = document.createElement('canvas'); c.width = c.height = 64
    const g = c.getContext('2d')!
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32)
    grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.25, 'rgba(255,214,150,0.9)'); grd.addColorStop(1, 'rgba(255,120,40,0)')
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64)
    this.spot = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }))
    this.spot.scale.setScalar(0.32)
    this.scene.add(this.spot)

    this.buildMarkers(run.points)
    this.scene.add(this.markers)
  }

  private partGroup(mat: THREE.Material): THREE.Group {
    const g = new THREE.Group()
    const lathe = (pts: [number, number][]) => new THREE.LatheGeometry(pts.map(([r, h]) => new THREE.Vector2(r * K, h * K)), 128)
    g.add(new THREE.Mesh(lathe(outerProfile()), mat))
    g.add(new THREE.Mesh(lathe(innerProfile()), mat))
    const tor = new THREE.Mesh(new THREE.TorusGeometry(TOR_RC * K, TOR_O * K, 24, 128), mat)
    tor.rotation.x = Math.PI / 2; tor.position.y = TOR_H * K
    g.add(tor)
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(PART.BAR_R * K, PART.BAR_R * K, PART.BAR_H * K, 20), mat)
      bar.position.set(sx * PART.BAR * K, (PART.BAR_H * K) / 2, sz * PART.BAR * K)
      g.add(bar)
    }
    return g
  }

  private buildMarkers(points: Point[]) {
    const sphere = new THREE.SphereGeometry(1, 16, 12)
    for (const p of points) {
      const col = new THREE.Color(TYPE_COLOR[p.type] ?? '#ffffff')
      const mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.95 })
      let m: THREE.Mesh
      const y = p.h_m * K
      if (p.type === 'streak') {
        // a streak is a line along the recoat axis: draw it from its box
        const [x0, , x1] = p.bbox
        const a = this.px(x0), b = this.px(x1)
        const len = Math.max(0.05, b - a)
        m = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, len, 8), mat)
        m.rotation.z = Math.PI / 2
        m.position.set((a + b) / 2, y, p.xz_m[1] * K)
      } else {
        const r = p.type === 'short_feed' ? 0.11 : p.type === 'super_elevation' ? 0.08 : 0.05
        m = new THREE.Mesh(sphere, mat)
        m.scale.setScalar(r)
        m.position.set(p.xz_m[0] * K, y, p.xz_m[1] * K)
      }
      m.userData = { h: p.h_m, action: p.action }
      if (p.action === 'STOP' || p.action === 'RECOAT') {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.012, 8, 48),
          new THREE.MeshBasicMaterial({ color: p.action === 'STOP' ? 0xff4d4f : 0xffb020, transparent: true, opacity: 0.9 }))
        ring.rotation.x = Math.PI / 2
        ring.position.copy(m.position)
        ring.userData = { h: p.h_m, ring: true }
        this.markers.add(ring)
      }
      this.markers.add(m)
    }
  }

  /** rectified frame pixel (512 over +-0.22 m) to scene x */
  private px(c: number) { return (((c + 0.5) / this.run.image_size) * 2 * this.run.frame_half_m - this.run.frame_half_m) * K }

  setBed(img: HTMLImageElement | null) {
    if (!img) return
    const t = new THREE.Texture(img)
    t.colorSpace = THREE.SRGBColorSpace
    t.needsUpdate = true
    this.bedMat.map?.dispose()
    this.bedMat.map = t
    this.bedMat.needsUpdate = true
  }

  setSize(w: number, h: number) {
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  /** layer: current layer, t: seconds (for the laser), progress: 0..1 through this layer's scan */
  update(layer: number, progress: number) {
    const h = this.run.layers[Math.min(layer, this.run.layers.length - 1)]?.h_m ?? 0
    this.h = h
    const y = h * K
    const printer = this.mode === 'printer'
    this.clip.constant = y
    this.printed.visible = printer
    this.ghostMat.opacity = printer ? 0.07 : 0.16
    this.ghost.visible = true
    this.powder.visible = printer
    this.bedTop.visible = printer
    this.laser.visible = printer
    this.spot.visible = printer
    this.layerPlane.visible = !printer
    this.powder.scale.y = Math.max(0.001, y)
    this.powder.position.y = y / 2
    this.bedTop.position.y = y + 0.004
    this.layerPlane.position.y = y
    for (const m of this.markers.children) m.visible = !printer ? true : (m.userData.h as number) <= h + 1e-6
    // laser along the hatch of this layer
    if (printer) {
      const segs = this.hatchFor(layer)
      if (segs.length) {
        let total = 0
        for (const s of segs) total += Math.hypot(s[2] - s[0], s[3] - s[1])
        let d = (progress % 1) * total
        let px = segs[0][0], pz = segs[0][1]
        for (const s of segs) {
          const L = Math.hypot(s[2] - s[0], s[3] - s[1])
          if (d <= L) { const u = d / L; px = s[0] + (s[2] - s[0]) * u; pz = s[1] + (s[3] - s[1]) * u; break }
          d -= L
        }
        const top = 6.5
        this.spot.position.set(px * K, y + 0.02, pz * K)
        const from = new THREE.Vector3(0, top, 0), to = this.spot.position.clone()
        const mid = from.clone().add(to).multiplyScalar(0.5)
        this.laser.position.copy(mid)
        this.laser.scale.y = from.distanceTo(to)
        this.laser.lookAt(to)
        this.laser.rotateX(Math.PI / 2)
      } else {
        this.laser.visible = this.spot.visible = false
      }
    }
    // camera
    const c = this.camera
    c.position.set(
      this.target.x + this.dist * Math.cos(this.el) * Math.sin(this.az),
      this.target.y + this.dist * Math.sin(this.el),
      this.target.z + this.dist * Math.cos(this.el) * Math.cos(this.az))
    c.lookAt(this.target)
    this.renderer.render(this.scene, c)
  }

  private hatchFor(layer: number) {
    let s = this.hatchCache.get(layer)
    if (!s) {
      const L = this.run.layers[layer]
      s = hatch(L.h_m, ((67 * layer) % 180) * (Math.PI / 180), 0.006)
      if (this.hatchCache.size > 60) this.hatchCache.clear()
      this.hatchCache.set(layer, s)
    }
    return s
  }
}
