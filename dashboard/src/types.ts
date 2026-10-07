export type Action = 'CONTINUE' | 'RECOAT' | 'RE_IMAGE' | 'STOP' | 'OPERATOR'

export interface Detection {
  type: string
  bbox: [number, number, number, number]
  source?: string
  length_px?: number
  area_px?: number
  count?: number
  arc_len_px?: number
  inside_part_px?: number
  plate_xz_m: [number, number]
}

export interface Attempt {
  attempt: number
  recoats?: number
  action: Action
  reason: string
  severity?: string
  numbers?: Record<string, unknown>
  injected?: string[]
  detections?: Detection[]
  evidence?: { ok: boolean; reasons: string[] }
  evidence_numbers?: Record<string, { lap_ratio: number; anisotropy: number; saturated_frac: number }>
  slice?: { iou: number; match_shapes: number | null; expected_px: number; observed_px: number }
  ecc_cc?: number
  streaks?: { x1: number; y1: number; x2: number; y2: number; length_px: number; angle_deg: number; inside_part_px: number }[]
  timing_ms?: Record<string, number>
  overlays?: boolean
  note?: string
  note_source?: string
}

export interface Layer { i: number; h_m: number; final: Action; worst: Action; attempts: Attempt[] }
export interface Point { layer: number; h_m: number; type: string; xz_m: [number, number]; bbox: [number, number, number, number]; action: Action; source: string }

export interface Run {
  build_id: string
  n_layers: number
  part_height_m: number
  frame_half_m: number
  plate_half_m: number
  image_size: number
  layers: Layer[]
  points: Point[]
  stops: { layer: number; reason: string; numbers: Record<string, unknown>; note: string; note_source: string }[]
  actions: Record<string, number>
  latency_ms: { median: number; p95: number; n: number }
}

export const TYPE_COLOR: Record<string, string> = {
  streak: '#ff4d4f',
  spatter: '#ffd166',
  super_elevation: '#ff8c42',
  short_feed: '#4da3ff',
}
export const TYPE_LABEL: Record<string, string> = {
  streak: 'Recoater streak',
  spatter: 'Spatter',
  super_elevation: 'Super elevation',
  short_feed: 'Short feed',
}
export const ACTION_COLOR: Record<string, string> = {
  CONTINUE: '#3ddc97',
  RECOAT: '#ffb020',
  RE_IMAGE: '#8ab4ff',
  STOP: '#ff4d4f',
  OPERATOR: '#c7cdd6',
}
