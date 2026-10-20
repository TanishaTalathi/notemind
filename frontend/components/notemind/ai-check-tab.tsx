'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { BarChart3, Loader2, RotateCcw, ShieldAlert } from 'lucide-react'
import { aiCheck, type AiCheckResult } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { EngineNote, ErrorNote } from './shared'

const MAX_CONTRIBUTION = 4

/** "Was this written by AI?" — an explainable estimate with the evidence behind it. */
export function AiCheckTab({ notes }: { notes: string }) {
  const [result, setResult] = useState<AiCheckResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const run = async () => {
    setLoading(true)
    setError('')
    try {
      setResult(await aiCheck(notes))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    run()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notes])

  if (loading) {
    return (
      <div className="flex min-h-[300px] flex-col items-center justify-center gap-3 text-center text-sm text-muted-foreground">
        <Loader2 className="size-6 animate-spin text-primary" />
        Checking your notes… the first check takes about 20 seconds while GPT-2 loads.
      </div>
    )
  }
  if (error) return <ErrorNote>{error}</ErrorNote>
  if (!result) return null

  const p = result.ai_probability
  const tone = p === null ? 'muted' : p >= 0.7 ? 'ai' : p <= 0.3 ? 'human' : 'unclear'

  return (
    <div className="flex flex-col gap-5">
      {/* verdict */}
      <div className="grid gap-4 rounded-xl border p-5 sm:grid-cols-[auto_1fr] sm:items-center">
        <div className="text-center sm:pr-4">
          <p className={cn('text-5xl font-bold tabular-nums',
            tone === 'ai' && 'text-rose-600 dark:text-rose-400', tone === 'human' && 'text-emerald-600 dark:text-emerald-400',
            tone === 'unclear' && 'text-amber-600 dark:text-amber-400', tone === 'muted' && 'text-muted-foreground')}>
            {p === null ? '—' : `${Math.round(p * 100)}%`}
          </p>
          <p className="text-xs text-muted-foreground">chance of AI writing</p>
        </div>
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-lg font-semibold">{result.label}</span>
            {result.confidence && result.confidence !== 'none' && (
              <Badge variant="outline" className="rounded-full">{result.confidence} confidence</Badge>
            )}
            <Button variant="ghost" size="sm" className="ml-auto" onClick={run}>
              <RotateCcw data-icon="inline-start" /> Check again
            </Button>
          </div>
          {p !== null && (
            <div>
              <div className="relative h-2.5 rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500">
                <span className="absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-background bg-foreground shadow"
                  style={{ left: `${p * 100}%` }} aria-hidden />
              </div>
              <div className="mt-1 flex justify-between text-[11px] text-muted-foreground">
                <span>Human</span><span>Unclear</span><span>AI</span>
              </div>
            </div>
          )}
          <div className="flex flex-wrap gap-x-3 text-[11px] text-muted-foreground">
            <span>{result.words} words</span>
            <EngineNote engine={result.engine} />
          </div>
        </div>
      </div>

      <div className="flex gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-900 dark:text-amber-200">
        <ShieldAlert className="mt-0.5 size-4 shrink-0" />
        <p>{result.message}</p>
      </div>

      {/* evidence */}
      {result.signals.length > 0 && (
        <section>
          <h3 className="mb-1 text-sm font-semibold">Why: the evidence</h3>
          <p className="mb-3 text-xs text-muted-foreground">
            Each bar shows how much a signal pushed the verdict towards <span className="text-emerald-600 dark:text-emerald-400">human</span> or{' '}
            <span className="text-rose-600 dark:text-rose-400">AI</span>.
          </p>
          <div className="flex flex-col gap-3">
            {result.signals.map((s) => {
              const width = Math.min(Math.abs(s.contribution) / MAX_CONTRIBUTION, 1) * 50
              return (
                <div key={s.name} className="grid gap-1 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] sm:items-center sm:gap-4">
                  <div>
                    <p className="text-sm font-medium">
                      {s.label} <span className="font-normal text-muted-foreground">· {s.display}</span>
                    </p>
                    <p className="text-xs text-muted-foreground">{s.explanation} {s.typical}</p>
                  </div>
                  <div className="relative h-3 rounded-full bg-muted" title={`contribution ${s.contribution > 0 ? '+' : ''}${s.contribution}`}>
                    <span className="absolute inset-y-0 left-1/2 w-px bg-foreground/30" />
                    <span className={cn('absolute inset-y-0 rounded-full', s.contribution >= 0 ? 'left-1/2 bg-rose-500' : 'right-1/2 bg-emerald-500')}
                      style={{ width: `${width}%` }} />
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* sentence view */}
      {result.sentences.length > 1 && (
        <section>
          <h3 className="mb-1 text-sm font-semibold">Sentence by sentence</h3>
          <p className="mb-3 text-xs text-muted-foreground">
            <mark className="rounded bg-rose-500/15 px-1 text-inherit">Highlighted</mark> sentences are very predictable for GPT-2 or use chatbot phrases.
            Hover a sentence to see its perplexity (lower = more predictable).
          </p>
          <p className="rounded-xl border p-4 text-[15px] leading-8">
            {result.sentences.map((s, i) => (
              <span key={i}
                title={[s.perplexity !== null && `perplexity ${s.perplexity}`, s.phrases.length > 0 && `phrases: ${s.phrases.join(', ')}`].filter(Boolean).join(' · ')}
                className={cn('rounded px-0.5', s.flag && 'bg-rose-500/15')}>
                {s.text}{' '}
              </span>
            ))}
          </p>
        </section>
      )}

      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <BarChart3 className="size-3.5" />
        {result.model_accuracy ? `Accuracy on held-out test data: ${Math.round(result.model_accuracy * 1000) / 10}%. ` : ''}
        <Link href="/evaluation" className="text-primary underline-offset-4 hover:underline">See how it was tested</Link>
      </p>
    </div>
  )
}
