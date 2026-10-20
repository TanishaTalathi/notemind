'use client'

import { useState, type ReactNode } from 'react'
import { PenLine } from 'lucide-react'
import type { CleanupResult, Difficulty, GlossaryResult } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { CopyButton, EngineNote, ListenButton, Segmented } from './shared'

const DIFF_STYLE: Record<Difficulty, string> = {
  Easy: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  Medium: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  Hard: 'bg-rose-500/15 text-rose-700 dark:text-rose-300',
}

function Gauge({ value }: { value: number }) {
  // semicircle gauge: 0 (hard to read) … 100 (very easy)
  const angle = Math.PI * (1 - value / 100)
  const x = 60 + 48 * Math.cos(angle), y = 60 - 48 * Math.sin(angle)
  const color = value >= 70 ? 'stroke-emerald-500' : value >= 50 ? 'stroke-amber-500' : 'stroke-rose-500'
  return (
    <svg viewBox="0 0 120 70" className="w-36" role="img" aria-label={`Readability score ${value} out of 100`}>
      <path d="M 12 60 A 48 48 0 0 1 108 60" fill="none" className="stroke-muted" strokeWidth={10} strokeLinecap="round" />
      <path d={`M 12 60 A 48 48 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)}`} fill="none" className={color} strokeWidth={10} strokeLinecap="round" />
      <text x={60} y={58} textAnchor="middle" className="fill-foreground text-[20px] font-bold">
        {value}
      </text>
    </svg>
  )
}

export function GlossaryTab({ data }: { data: GlossaryResult }) {
  const [filter, setFilter] = useState<'All' | Difficulty>('All')
  const [showSimple, setShowSimple] = useState(false)
  const terms = data.terms.filter((t) => filter === 'All' || t.difficulty === filter)

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-4 rounded-xl border bg-muted/20 p-4 sm:flex-row sm:items-center">
        <Gauge value={data.readability} />
        <div className="flex-1">
          <p className="text-sm font-semibold">Readability: {data.level ?? (data.readability >= 70 ? 'Easy' : data.readability >= 50 ? 'Moderate' : 'Difficult')}</p>
          <p className="text-sm text-muted-foreground">Reading level: {data.grade}. Higher scores are easier to read (Flesch Reading Ease).</p>
        </div>
        <Button variant={showSimple ? 'secondary' : 'default'} onClick={() => setShowSimple(!showSimple)} disabled={!data.simplified}>
          <PenLine data-icon="inline-start" /> {showSimple ? 'Hide simplified notes' : 'Simplify my notes'}
        </Button>
      </div>

      {showSimple && data.simplified && (
        <div className="rounded-xl border border-primary/30 bg-primary/[0.04] p-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Simplified version</p>
            <div className="flex">
              <CopyButton text={data.simplified} />
              <ListenButton text={data.simplified} />
            </div>
          </div>
          <p className="text-[15px] leading-7">{data.simplified}</p>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-semibold">Difficult words ({data.terms.length})</p>
        <Segmented label="Filter by difficulty" value={filter} options={['All', 'Hard', 'Medium', 'Easy'] as const} onChange={setFilter} />
      </div>
      <div className="overflow-x-auto rounded-xl border">
        <table className="w-full min-w-[560px] text-left text-sm">
          <thead className="bg-muted/40 text-xs">
            <tr>
              <th className="p-3">Term</th>
              <th className="p-3">Simple meaning (English)</th>
              <th className="p-3">अर्थ (Hindi)</th>
              <th className="p-3">Difficulty</th>
            </tr>
          </thead>
          <tbody>
            {terms.map((t) => (
              <tr key={t.term} className="border-t align-top">
                <td className="p-3 font-medium">{t.term}</td>
                <td className="p-3 text-muted-foreground">
                  {t.meaning || '—'}
                  {t.source === 'context' && <span className="ml-1 text-[10px] uppercase text-muted-foreground/70">(from context)</span>}
                </td>
                <td className="p-3 text-muted-foreground">{t.hindi || '—'}</td>
                <td className="p-3">
                  <span className={cn('rounded-full px-2 py-0.5 text-xs font-medium', DIFF_STYLE[t.difficulty])}>{t.difficulty}</span>
                </td>
              </tr>
            ))}
            {terms.length === 0 && (
              <tr>
                <td colSpan={4} className="p-4 text-center text-muted-foreground">
                  No {filter.toLowerCase()} words.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** Show the Devanagari version with the converted words highlighted. */
function highlightChanged(text: string, changed: string[]): ReactNode[] {
  const set = new Set(changed)
  return text.split(/([ऀ-ॿ]+)/).map((part, i) =>
    set.has(part) ? (
      <mark key={i} className="rounded bg-primary/15 px-0.5 text-foreground">
        {part}
      </mark>
    ) : (
      part
    ),
  )
}

export function CleanupTab({ data, notes }: { data: CleanupResult; notes: string }) {
  if (data.input_language === 'en' || data.message) {
    return (
      <div className="flex min-h-[260px] flex-col items-center justify-center gap-2 text-center">
        <Badge variant="secondary">English notes</Badge>
        <p className="max-w-sm text-sm text-muted-foreground">
          {data.message ?? 'Nothing to clean up.'} Try “Load sample → Hinglish” to see how code-mixed notes like
          “Photosynthesis me plants energy convert karte hain” are cleaned up.
        </p>
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-xl border bg-muted/20 p-4">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Your original notes</p>
        <p className="line-clamp-4 text-sm leading-6 text-muted-foreground">{notes}</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border p-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Devanagari (Hindi words converted)</p>
            <CopyButton text={data.devanagari} />
          </div>
          <p className="text-[15px] leading-8">{highlightChanged(data.devanagari, data.changed ?? [])}</p>
          {(data.changed?.length ?? 0) > 0 && <p className="mt-2 text-[11px] text-muted-foreground">{data.changed!.length} words transliterated · English terms kept as-is</p>}
        </div>
        <div className="rounded-xl border bg-primary/[0.04] p-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Clean standard English</p>
            <div className="flex">
              <CopyButton text={data.english} />
              <ListenButton text={data.english} />
            </div>
          </div>
          <p className="text-[15px] leading-7">{data.english}</p>
        </div>
      </div>
      <EngineNote engine={data.engine} />
    </div>
  )
}
