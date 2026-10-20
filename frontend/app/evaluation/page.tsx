'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, BarChart3, Brain, Loader2, Moon, Play, PlayCircle, Sun } from 'lucide-react'
import { getEvaluation, runEvaluation, type EvalBar, type EvalOverview, type EvalResult } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ErrorNote } from '@/components/notemind/shared'
import { cn } from '@/lib/utils'

const timeAgo = (iso: string) => {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`
  return new Date(iso).toLocaleDateString()
}

export default function EvaluationPage() {
  const [data, setData] = useState<EvalOverview | null>(null)
  const [error, setError] = useState('')
  const [running, setRunning] = useState<string[]>([])
  const [runErrors, setRunErrors] = useState<Record<string, string>>({})
  const [dark, setDark] = useState(false)

  useEffect(() => {
    setDark(window.matchMedia('(prefers-color-scheme: dark)').matches)
    getEvaluation().then(setData).catch(() => setError('Could not reach the backend. Start it with: uvicorn main:app --port 8000 (in the backend folder).'))
  }, [])
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    document.documentElement.classList.toggle('light', !dark)
  }, [dark])

  const run = async (suite: string) => {
    setRunning((r) => [...r, suite])
    setRunErrors(({ [suite]: _removed, ...rest }) => rest)
    try {
      const result = await runEvaluation(suite)
      setData((d) => (d ? { ...d, results: { ...d.results, [suite]: result } } : d))
    } catch (e) {
      setRunErrors((all) => ({ ...all, [suite]: (e as Error).message }))
    } finally {
      setRunning((r) => r.filter((s) => s !== suite))
    }
  }

  const runAll = async () => {
    for (const suite of Object.keys(data?.suites ?? {})) await run(suite)
  }

  const suites = Object.entries(data?.suites ?? {})
  const done = suites.filter(([id]) => data?.results[id]).length

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-20 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 lg:px-8">
          <Link href="/" className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <Brain />
            </div>
            <div>
              <span className="font-bold tracking-tight">NoteMind</span>
              <p className="flex items-center gap-1 text-xs text-muted-foreground"><ArrowLeft className="size-3" /> Back to the app</p>
            </div>
          </Link>
          <Button variant="outline" size="icon" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">
            {dark ? <Sun /> : <Moon />}
          </Button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8 lg:px-8">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <Badge className="mb-3 rounded-full bg-primary/10 text-primary hover:bg-primary/10">
              <BarChart3 data-icon="inline-start" /> Evaluation
            </Badge>
            <h1 className="text-3xl font-bold tracking-tight">How well does NoteMind work?</h1>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              Every NLP feature measured on test data with standard metrics. Press Run to re-measure a feature on this computer.
            </p>
          </div>
          {data && (
            <div className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground">{done}/{suites.length} measured</span>
              <Button onClick={runAll} disabled={running.length > 0}>
                {running.length ? <Loader2 className="animate-spin" data-icon="inline-start" /> : <PlayCircle data-icon="inline-start" />}
                {running.length ? 'Running…' : 'Run all'}
              </Button>
            </div>
          )}
        </div>

        {error && <ErrorNote>{error}</ErrorNote>}
        {!data && !error && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Loading results…</p>
        )}

        {data && (
          <>
            <nav className="mb-6 flex flex-wrap gap-2" aria-label="Suites">
              {suites.map(([id, info]) => (
                <a key={id} href={`#${id}`} className="rounded-full border px-3 py-1 text-xs hover:border-primary hover:bg-primary/5">
                  {info.title}
                </a>
              ))}
            </nav>
            <div className="flex flex-col gap-6">
              {suites.map(([id, info]) => (
                <SuiteCard key={id} id={id} info={info} result={data.results[id]} running={running.includes(id)}
                  error={runErrors[id]} onRun={() => run(id)} />
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  )
}

function SuiteCard({ id, info, result, running, error, onRun }: {
  id: string; info: { title: string; metric: string; description: string }; result?: EvalResult
  running: boolean; error?: string; onRun: () => void
}) {
  return (
    <Card id={id} className="scroll-mt-20">
      <CardHeader className="border-b bg-muted/20">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle className="flex flex-wrap items-center gap-2 text-base">
              {info.title}
              <Badge variant="outline" className="rounded-full font-normal">{info.metric}</Badge>
            </CardTitle>
            <CardDescription className="mt-1">{info.description}</CardDescription>
          </div>
          <div className="flex items-center gap-3">
            {result && !running && (
              <span className="text-[11px] text-muted-foreground">{timeAgo(result.ran_at)} · took {result.seconds} s</span>
            )}
            <Button variant="outline" size="sm" onClick={onRun} disabled={running}>
              {running ? <Loader2 className="animate-spin" data-icon="inline-start" /> : <Play data-icon="inline-start" />}
              {running ? 'Running…' : result ? 'Run again' : 'Run'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-6 p-5">
        {error && <ErrorNote>{error}</ErrorNote>}
        {running && <p className="text-xs text-muted-foreground">Measuring… this can take a minute or two (models may need to load first).</p>}
        {!result && !running && <p className="text-sm text-muted-foreground">Not measured yet — press Run.</p>}
        {result && (
          <>
            <p className="text-xs text-muted-foreground">Test data: {result.dataset}</p>
            <div className="grid gap-3 sm:grid-cols-3">
              {result.headline.map((h) => (
                <div key={h.label} className="rounded-xl border bg-card px-4 py-3">
                  <p className="text-2xl font-bold tabular-nums">{h.value}</p>
                  <p className="text-sm font-medium">{h.label}</p>
                  {h.hint && <p className="text-xs text-muted-foreground">{h.hint}</p>}
                </div>
              ))}
            </div>
            {(result.bars.length > 0 || result.matrix) && (
              <div className="grid gap-6 lg:grid-cols-2">
                {result.bars.map((bar) => <Bars key={bar.title} bar={bar} />)}
                {result.matrix && <Matrix matrix={result.matrix} />}
              </div>
            )}
            {result.tables.map((table, i) =>
              i === 0 ? (
                <DataTable key={table.title} table={table} />
              ) : (
                <details key={table.title} className="group rounded-xl border">
                  <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium">
                    {table.title} <span className="text-xs font-normal text-muted-foreground">({table.rows.length})</span>
                  </summary>
                  <div className="border-t p-3"><DataTable table={table} bare /></div>
                </details>
              ),
            )}
            {result.notes.length > 0 && (
              <ul className="flex list-disc flex-col gap-1 pl-5 text-xs leading-5 text-muted-foreground">
                {result.notes.map((n) => <li key={n}>{n}</li>)}
              </ul>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

function Bars({ bar }: { bar: EvalBar }) {
  const max = bar.max || Math.max(...bar.items.map((i) => i.value), 1)
  return (
    <figure>
      <figcaption className="mb-2 text-sm font-medium">{bar.title}</figcaption>
      <div className="flex flex-col gap-2">
        {bar.items.map((item) => (
          <div key={item.label} className="grid grid-cols-[minmax(0,15rem)_1fr_3.5rem] items-center gap-2 text-xs">
            <span className="truncate text-muted-foreground" title={item.label}>{item.label}</span>
            <div className="h-3 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-primary" style={{ width: `${Math.min(100, (item.value / max) * 100)}%` }} />
            </div>
            <span className="text-right font-medium tabular-nums">{item.value}{bar.unit}</span>
          </div>
        ))}
        {bar.items.length === 0 && <p className="text-xs text-muted-foreground">No data.</p>}
      </div>
    </figure>
  )
}

function Matrix({ matrix }: { matrix: NonNullable<EvalResult['matrix']> }) {
  return (
    <figure>
      <figcaption className="mb-2 text-sm font-medium">{matrix.title}</figcaption>
      <table className="text-xs">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left font-normal text-muted-foreground">true ↓ / predicted →</th>
            {matrix.labels.map((l) => <th key={l} className="px-2 py-1 font-medium">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.values.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1
            return (
              <tr key={matrix.labels[i]}>
                <th className="px-2 py-1 text-left font-medium">{matrix.labels[i]}</th>
                {row.map((v, j) => (
                  <td key={j} className="p-0.5">
                    <div
                      className={cn('flex size-12 items-center justify-center rounded-md font-semibold tabular-nums',
                        i === j ? 'text-primary-foreground' : v ? 'text-rose-900 dark:text-rose-100' : 'text-muted-foreground')}
                      style={{ background: i === j ? `color-mix(in oklch, var(--primary) ${25 + 75 * (v / total)}%, transparent)` : v ? `rgb(244 63 94 / ${0.15 + 0.5 * (v / total)})` : 'var(--muted)' }}
                    >
                      {v}
                    </div>
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </figure>
  )
}

function DataTable({ table, bare }: { table: EvalResult['tables'][number]; bare?: boolean }) {
  return (
    <div className={cn('overflow-x-auto', !bare && 'rounded-xl border')}>
      {!bare && <p className="border-b px-4 py-2.5 text-sm font-medium">{table.title}</p>}
      <table className="w-full text-left text-xs">
        <thead className="bg-muted/40">
          <tr>{table.columns.map((c) => <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">{c}</th>)}</tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-t align-top">
              {row.map((cell, j) => (
                <td key={j} className={cn('px-3 py-2', typeof cell === 'number' && 'tabular-nums', String(cell).length > 40 ? 'min-w-56' : 'whitespace-nowrap')}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
