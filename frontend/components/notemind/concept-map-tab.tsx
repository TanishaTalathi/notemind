'use client'

import { useMemo, useRef, useState } from 'react'
import { RotateCcw, X } from 'lucide-react'
import type { ConceptMapResult, ConceptNode } from '@/lib/api'
import { Button } from '@/components/ui/button'

const W = 760
const H = 460
type Point = { x: number; y: number }

/** Small force-directed layout (repulsion + springs + centring), run once when the data changes. */
function layout(map: ConceptMapResult): Record<string, Point> {
  const ids = map.nodes.map((n) => n.id)
  const pos: Record<string, Point & { vx: number; vy: number }> = {}
  ids.forEach((id, i) => {
    const angle = (i / ids.length) * Math.PI * 2
    const r = 150 + (i % 2) * 40
    pos[id] = { x: W / 2 + Math.cos(angle) * r, y: H / 2 + Math.sin(angle) * r * 0.7, vx: 0, vy: 0 }
  })
  const edges = map.edges.filter((e) => pos[e.source] && pos[e.target])
  for (let step = 0; step < 400; step++) {
    const cooling = 1 - step / 400
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const a = pos[ids[i]], b = pos[ids[j]]
        let dx = a.x - b.x, dy = a.y - b.y
        const d2 = Math.max(dx * dx + dy * dy, 100)
        const f = 9000 / d2
        const d = Math.sqrt(d2)
        dx /= d; dy /= d
        a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f
      }
    }
    for (const e of edges) {
      const a = pos[e.source], b = pos[e.target]
      const dx = b.x - a.x, dy = b.y - a.y
      const d = Math.sqrt(dx * dx + dy * dy) || 1
      const f = (d - 170) * 0.02
      a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f
    }
    for (const id of ids) {
      const p = pos[id]
      p.vx += (W / 2 - p.x) * 0.004
      p.vy += (H / 2 - p.y) * 0.006
      p.x = Math.min(W - 70, Math.max(70, p.x + Math.max(-20, Math.min(20, p.vx)) * cooling))
      p.y = Math.min(H - 40, Math.max(40, p.y + Math.max(-20, Math.min(20, p.vy)) * cooling))
      p.vx *= 0.6; p.vy *= 0.6
    }
  }
  return Object.fromEntries(ids.map((id) => [id, { x: pos[id].x, y: pos[id].y }]))
}

export function ConceptMapTab({ map }: { map: ConceptMapResult }) {
  const initial = useMemo(() => layout(map), [map])
  const [positions, setPositions] = useState(initial)
  const [selected, setSelected] = useState<ConceptNode | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const dragging = useRef<{ id: string; moved: boolean } | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const maxImportance = Math.max(...map.nodes.map((n) => n.importance), 1)

  if (!map.nodes.length) {
    return <p className="text-sm text-muted-foreground">Not enough key concepts found — add a few more sentences to your notes.</p>
  }

  const toSvg = (event: React.PointerEvent): Point => {
    const rect = svgRef.current!.getBoundingClientRect()
    return { x: ((event.clientX - rect.left) / rect.width) * W, y: ((event.clientY - rect.top) / rect.height) * H }
  }
  const radius = (n: ConceptNode) => 16 + (n.importance / maxImportance) * 20
  const connected = (id: string) => map.edges.some((e) => (e.source === hover && e.target === id) || (e.target === hover && e.source === id))

  return (
    <div className="flex flex-col gap-5">
      <div className="relative overflow-hidden rounded-2xl border bg-muted/20">
        <div className="absolute right-2 top-2 z-10">
          <Button variant="ghost" size="sm" onClick={() => setPositions(initial)}>
            <RotateCcw data-icon="inline-start" /> Reset layout
          </Button>
        </div>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full touch-none select-none"
          role="img"
          aria-label="Concept map of your notes"
          onPointerMove={(e) => {
            if (!dragging.current) return
            dragging.current.moved = true
            const p = toSvg(e)
            setPositions((all) => ({ ...all, [dragging.current!.id]: { x: Math.min(W - 20, Math.max(20, p.x)), y: Math.min(H - 20, Math.max(20, p.y)) } }))
          }}
          onPointerUp={() => (dragging.current = null)}
          onPointerLeave={() => (dragging.current = null)}
        >
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" className="fill-muted-foreground/60" />
            </marker>
          </defs>
          {map.edges.map((e, i) => {
            const a = positions[e.source], b = positions[e.target]
            const na = map.nodes.find((n) => n.id === e.source), nb = map.nodes.find((n) => n.id === e.target)
            if (!a || !b || !na || !nb) return null
            const dx = b.x - a.x, dy = b.y - a.y
            const d = Math.sqrt(dx * dx + dy * dy) || 1
            const x1 = a.x + (dx / d) * radius(na), y1 = a.y + (dy / d) * radius(na)
            const x2 = b.x - (dx / d) * (radius(nb) + 4), y2 = b.y - (dy / d) * (radius(nb) + 4)
            const active = hover && (e.source === hover || e.target === hover)
            const faded = hover && !active
            return (
              <g key={i} opacity={faded ? 0.15 : 1}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} className={active ? 'stroke-primary' : 'stroke-muted-foreground/40'} strokeWidth={active ? 2 : 1.4} markerEnd="url(#arrow)" />
                {e.relation !== 'related to' && e.relation !== 'संबंधित' && (
                  <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 5} textAnchor="middle" className="fill-muted-foreground text-[11px]" paintOrder="stroke" stroke="var(--background)" strokeWidth={4}>
                    {e.relation}
                  </text>
                )}
              </g>
            )
          })}
          {map.nodes.map((n) => {
            const p = positions[n.id]
            if (!p) return null
            const r = radius(n)
            const faded = hover && hover !== n.id && !connected(n.id)
            const isSelected = selected?.id === n.id
            return (
              <g
                key={n.id}
                transform={`translate(${p.x},${p.y})`}
                className="cursor-grab active:cursor-grabbing"
                opacity={faded ? 0.25 : 1}
                tabIndex={0}
                role="button"
                aria-label={`${n.id}, importance ${n.importance} of 10`}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSelected(n)}
                onPointerEnter={() => setHover(n.id)}
                onPointerLeave={() => setHover(null)}
                onPointerDown={(e) => {
                  try {
                    ;(e.target as Element).setPointerCapture?.(e.pointerId)
                  } catch {
                    /* synthetic events have no active pointer */
                  }
                  dragging.current = { id: n.id, moved: false }
                }}
                onPointerUp={() => {
                  if (dragging.current && !dragging.current.moved) setSelected(n)
                  dragging.current = null
                }}
              >
                <circle r={r} className={isSelected ? 'fill-primary stroke-primary' : 'fill-primary/15 stroke-primary'} strokeWidth={isSelected ? 3 : 1.5} />
                <text y={r + 15} textAnchor="middle" className="fill-foreground text-[13px] font-medium" paintOrder="stroke" stroke="var(--background)" strokeWidth={4}>
                  {n.id}
                </text>
                <text y={4} textAnchor="middle" className={isSelected ? 'fill-primary-foreground text-[11px] font-semibold' : 'fill-primary text-[11px] font-semibold'}>
                  {n.importance}
                </text>
              </g>
            )
          })}
        </svg>
        <p className="px-4 pb-3 text-[11px] text-muted-foreground">Drag concepts to rearrange · hover to highlight links · click for details · circle size = importance</p>
        {selected && (
          <div className="absolute bottom-10 left-3 right-3 z-10 rounded-xl border bg-popover p-4 text-sm shadow-lg sm:left-auto sm:w-80">
            <div className="mb-2 flex items-center justify-between">
              <p className="font-semibold">{selected.id}</p>
              <Button variant="ghost" size="icon-xs" onClick={() => setSelected(null)} aria-label="Close">
                <X />
              </Button>
            </div>
            <p className="mb-2 text-xs text-muted-foreground">
              Importance {selected.importance}/10{selected.count ? ` · mentioned ${selected.count}×` : ''}
            </p>
            <ul className="flex max-h-40 flex-col gap-1.5 overflow-y-auto text-xs leading-5">
              {(selected.sentences ?? []).map((s, i) => (
                <li key={i} className="border-l-2 border-primary/40 pl-2">
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {map.keywords && map.keywords.length > 0 && (
        <div>
          <p className="mb-3 text-sm font-semibold">Top keywords by importance</p>
          <div className="flex flex-col gap-2">
            {map.keywords.map((k) => (
              <div key={k.word} className="grid grid-cols-[120px_1fr_40px] items-center gap-3 text-sm sm:grid-cols-[160px_1fr_44px]">
                <span className="truncate">{k.word}</span>
                <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${Math.round(k.score * 100)}%` }} />
                </div>
                <span className="text-right text-xs tabular-nums text-muted-foreground">{Math.round(k.score * 100)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
