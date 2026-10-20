'use client'

import { useState } from 'react'
import { ArrowLeftRight, Loader2 } from 'lucide-react'
import { summarize, translate, type Language, type SummaryMode, type SummaryResult, type TranslateResult } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { CopyButton, EngineNote, ErrorNote, ListenButton, Segmented } from './shared'

const MODES = [
  { value: 'short' as SummaryMode, label: 'Short' },
  { value: 'detailed' as SummaryMode, label: 'Detailed' },
  { value: 'bullets' as SummaryMode, label: 'Bullet points' },
]

export function SummaryTab({ notes, initial }: { notes: string; initial: SummaryResult }) {
  const [mode, setMode] = useState<SummaryMode>('detailed')
  const [cache, setCache] = useState<Partial<Record<SummaryMode, SummaryResult>>>({ detailed: initial })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const current = cache[mode]

  const change = async (next: SummaryMode) => {
    setMode(next)
    if (cache[next]) return
    setLoading(true)
    setError('')
    try {
      const result = await summarize(notes, next)
      setCache((c) => ({ ...c, [next]: result }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const saved = current ? Math.max(0, Math.round((1 - current.summary_words / Math.max(current.original_words, 1)) * 100)) : 0

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Segmented label="Summary length" value={mode} options={MODES} onChange={change} />
        {current && (
          <div className="flex gap-1">
            <CopyButton text={current.summary} />
            <ListenButton text={current.summary} />
          </div>
        )}
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {loading || !current ? (
        <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
          <Loader2 className="mr-2 animate-spin" /> Summarising…
        </div>
      ) : (
        <>
          <div className="whitespace-pre-line rounded-2xl border bg-primary/[0.04] p-5 text-[15px] leading-7">{current.summary}</div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span>
              Compression <span className="font-medium text-foreground">{current.original_words} → {current.summary_words} words</span>
            </span>
            <Progress value={saved} className="w-32" aria-label="Words saved" />
            <span>{saved}% shorter</span>
            <span className="ml-auto">
              <EngineNote engine={current.engine} />
            </span>
          </div>
        </>
      )}
    </div>
  )
}

const LANG_NAME: Record<Language, string> = { en: 'English', hi: 'Hindi', hinglish: 'Hinglish' }

export function TranslateTab({
  notes,
  initial,
}: {
  notes: string
  initial: TranslateResult & { source: Language; target: Language }
}) {
  const [result, setResult] = useState(initial)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const swap = async () => {
    // Swap the direction, e.g. if the language was detected wrongly or you want the other way round.
    const source: Language = result.target === 'hi' ? 'hi' : 'en'
    const target: Language = source === 'hi' ? 'en' : 'hi'
    setLoading(true)
    setError('')
    try {
      setResult({ ...(await translate(notes, source, target)), source, target })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary" className="rounded-full px-3 py-1 text-xs">
          {LANG_NAME[result.source]} → {LANG_NAME[result.target]}
        </Badge>
        <Button variant="outline" size="sm" onClick={swap} disabled={loading}>
          {loading ? <Loader2 className="animate-spin" data-icon="inline-start" /> : <ArrowLeftRight data-icon="inline-start" />}
          Swap direction
        </Button>
        <span className="ml-auto">
          <EngineNote engine={result.engine} />
        </span>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {result.available === false && <ErrorNote>{result.message ?? 'Translation is unavailable right now.'}</ErrorNote>}
      <div className="grid gap-4 md:grid-cols-2">
        <Panel title={`Original (${LANG_NAME[result.source]})`} text={notes} />
        <Panel title={`Translation (${LANG_NAME[result.target]})`} text={result.translation} highlight />
      </div>
    </div>
  )
}

function Panel({ title, text, highlight }: { title: string; text: string; highlight?: boolean }) {
  return (
    <div className={`flex flex-col rounded-xl border p-4 ${highlight ? 'bg-primary/[0.04]' : ''}`}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
        {text && (
          <div className="flex">
            <CopyButton text={text} />
            <ListenButton text={text} />
          </div>
        )}
      </div>
      <p className="max-h-[420px] overflow-y-auto whitespace-pre-line text-[15px] leading-7">{text || '—'}</p>
    </div>
  )
}
