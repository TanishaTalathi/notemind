'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import {
  BarChart3, BookOpen, Bot, Brain, Check, Clock, Download, FileText, FlaskConical, Gauge, Languages, Lightbulb, Loader2, Moon,
  Network, Printer, ScanSearch, Sparkles, Sun, Upload, WandSparkles, Wifi, WifiOff,
} from 'lucide-react'
import {
  FEATURES, analyzeAll, checkBackend, countWords, detectLanguage, isUsingMock, sampleHindi, sampleHinglish, sampleNotes,
  type AnalysisResult, type Feature, type Language,
} from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { NoteEditor } from '@/components/notemind/note-editor'
import { SummaryTab, TranslateTab } from '@/components/notemind/summary-tab'
import { QuizTab } from '@/components/notemind/quiz-tab'
import { AskTab } from '@/components/notemind/ask-tab'
import { ConceptMapTab } from '@/components/notemind/concept-map-tab'
import { CleanupTab, GlossaryTab } from '@/components/notemind/glossary-tab'
import { AiCheckTab } from '@/components/notemind/ai-check-tab'
import { VoiceRecorder } from '@/components/notemind/voice-recorder'
import { EmptyState, ErrorNote } from '@/components/notemind/shared'

const LANG_LABEL: Record<Language, string> = { en: 'English', hi: 'Hindi (हिंदी)', hinglish: 'Hinglish (code-mixed)' }

const TABS: { value: string; label: string; feature?: Feature }[] = [
  { value: 'summary', label: 'Summary', feature: 'Summary' },
  { value: 'translate', label: 'Translate', feature: 'Translate' },
  { value: 'quiz', label: 'Quiz', feature: 'Quiz' },
  { value: 'ask', label: 'Ask Notes' },
  { value: 'map', label: 'Concept Map', feature: 'Concept Map' },
  { value: 'glossary', label: 'Glossary', feature: 'Glossary' },
  { value: 'cleanup', label: 'Hinglish Cleanup', feature: 'Hinglish Cleanup' },
  { value: 'aicheck', label: 'AI Check' },
]

export default function Page() {
  const [notes, setNotes] = useState('')
  const [analyzedNotes, setAnalyzedNotes] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [runId, setRunId] = useState(0)
  const [loading, setLoading] = useState(false)
  const [dark, setDark] = useState(false)
  const [tab, setTab] = useState('summary')
  const [liveLang, setLiveLang] = useState<{ language: Language; confidence: number } | null>(null)
  const [backend, setBackend] = useState<'checking' | 'online' | 'demo'>('checking')
  const [quizCount, setQuizCount] = useState<number | null>(null)
  const [enabled, setEnabled] = useState<Record<Feature, boolean>>(() => Object.fromEntries(FEATURES.map((f) => [f, true])) as Record<Feature, boolean>)
  const fileRef = useRef<HTMLInputElement>(null)
  const words = countWords(notes)

  // theme: follow the system on first load, then the toggle
  useEffect(() => {
    setDark(window.matchMedia('(prefers-color-scheme: dark)').matches)
  }, [])
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    document.documentElement.classList.toggle('light', !dark)
  }, [dark])

  useEffect(() => {
    checkBackend().then((ok) => setBackend(ok ? 'online' : 'demo'))
  }, [])

  // live language detection while typing
  useEffect(() => {
    if (!notes.trim()) {
      setLiveLang(null)
      return
    }
    const timer = setTimeout(() => detectLanguage(notes).then(setLiveLang).catch(() => undefined), 700)
    return () => clearTimeout(timer)
  }, [notes])

  const analyze = async () => {
    if (!notes.trim()) return
    setLoading(true)
    try {
      if (backend !== 'online') setBackend((await checkBackend()) ? 'online' : 'demo')
      const data = await analyzeAll(notes, enabled)
      setResult(data)
      setAnalyzedNotes(notes)
      setQuizCount(data.quiz?.questions.length ?? null)
      setRunId((n) => n + 1)
      const first = TABS.find((t) => !t.feature || enabled[t.feature])
      if (TABS.find((t) => t.value === tab)?.feature && !enabled[TABS.find((t) => t.value === tab)!.feature!]) setTab(first!.value)
    } finally {
      setLoading(false)
    }
  }

  const upload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) file.text().then(setNotes)
    event.target.value = ''
  }

  const exportTxt = () => {
    if (!result) return
    const lines: string[] = ['NoteMind study pack', '='.repeat(20), '']
    if (result.summary) lines.push('SUMMARY', result.summary.summary, '')
    if (result.translation?.translation) lines.push('TRANSLATION', result.translation.translation, '')
    if (result.quiz?.questions.length) {
      lines.push('QUIZ')
      result.quiz.questions.forEach((q, i) => {
        lines.push(`${i + 1}. ${q.question}`)
        q.options?.forEach((o, j) => lines.push(`   ${String.fromCharCode(97 + j)}) ${o}`))
        lines.push(`   Answer: ${q.answer}`, '')
      })
    }
    if (result.glossary?.terms.length) {
      lines.push('GLOSSARY')
      result.glossary.terms.forEach((t) => lines.push(`- ${t.term} [${t.difficulty}]: ${t.meaning}${t.hindi ? ` | ${t.hindi}` : ''}`))
      lines.push('', `Readability: ${result.glossary.readability}/100 (${result.glossary.grade})`)
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'notemind-study-pack.txt'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const concepts = result?.map?.nodes.map((n) => n.id) ?? []
  const stats = [
    { label: 'Words in notes', value: String(words), icon: BookOpen },
    { label: 'Key concepts found', value: result?.map ? String(result.map.nodes.length) : '—', icon: Network },
    { label: 'Questions generated', value: quizCount !== null ? String(quizCount) : '—', icon: Lightbulb },
    { label: 'Reading time', value: words ? `${Math.max(1, Math.round(words / 200))} min` : '—', icon: Clock },
    { label: 'Readability', value: result?.glossary ? `${result.glossary.readability}/100` : '—', icon: Gauge },
  ]

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-20 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-3 px-4 py-3 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <Brain />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold tracking-tight">NoteMind</span>
                <Badge variant="secondary" className="hidden rounded-full text-[10px] sm:inline-flex">AI STUDY ASSISTANT</Badge>
              </div>
              <p className="hidden text-xs text-muted-foreground sm:block">Your notes, any language, smarter study</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={backend === 'online' ? 'border-emerald-500/40 text-emerald-700 dark:text-emerald-300' : backend === 'demo' ? 'border-amber-500/40 text-amber-700 dark:text-amber-300' : ''}
              title={backend === 'demo' ? 'Backend not reachable — showing demo data. Start it with: uvicorn main:app --port 8000' : 'Connected to the NoteMind API'}
            >
              {backend === 'online' ? <Wifi data-icon="inline-start" /> : <WifiOff data-icon="inline-start" />}
              {backend === 'online' ? 'Backend connected' : backend === 'demo' ? 'Demo mode' : 'Connecting…'}
            </Badge>
            <Link href="/evaluation" className="inline-flex h-8 items-center gap-1.5 rounded-lg border px-2.5 text-sm font-medium hover:bg-muted">
              <BarChart3 className="size-4" /> <span className="hidden sm:inline">Evaluation</span>
            </Link>
            <Button variant="outline" size="icon" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">
              {dark ? <Sun /> : <Moon />}
            </Button>
          </div>
        </div>
      </header>

      <main className="no-print mx-auto max-w-[1500px] px-4 py-8 lg:px-8">
        <div className="mb-8">
          <Badge className="mb-3 rounded-full bg-primary/10 text-primary hover:bg-primary/10">
            <Sparkles data-icon="inline-start" /> English · हिंदी · Hinglish
          </Badge>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Turn notes into <span className="text-primary">momentum.</span>
          </h1>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            Type, paste or speak your notes once and get a summary, translation, quiz with answer checking, concept map, glossary, an AI-writing check and a chatbot that answers from your own notes.
          </p>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.5fr)]">
          {/* LEFT: notes */}
          <Card className="h-fit">
            <CardHeader className="border-b bg-muted/20">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <FileText className="text-primary" /> Your Notes
                  </CardTitle>
                  <CardDescription>Write in English, Hindi, or Hinglish</CardDescription>
                </div>
                {liveLang && (
                  <Badge variant="secondary" className="shrink-0 rounded-full">
                    <Languages data-icon="inline-start" /> Detected: {LANG_LABEL[liveLang.language]}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="flex flex-col gap-4 p-5">
              <NoteEditor value={notes} onChange={setNotes} />
              <VoiceRecorder onTranscript={(text) => setNotes((current) => (current.trim() ? `${current.trimEnd()}
${text}` : text))} />
              <div className="flex flex-wrap gap-2">
                <input ref={fileRef} type="file" accept=".txt,.md,text/plain" className="hidden" onChange={upload} />
                <Button variant="outline" size="sm" onClick={() => fileRef.current?.click()}>
                  <Upload data-icon="inline-start" /> Upload .txt
                </Button>
                <span className="flex items-center gap-1 rounded-lg border px-1">
                  <FlaskConical className="ml-1 size-3.5 text-muted-foreground" />
                  <Button variant="ghost" size="xs" onClick={() => setNotes(sampleNotes)}>English</Button>
                  <Button variant="ghost" size="xs" onClick={() => setNotes(sampleHindi)}>हिंदी</Button>
                  <Button variant="ghost" size="xs" onClick={() => setNotes(sampleHinglish)}>Hinglish</Button>
                </span>
                <Button variant="ghost" size="sm" onClick={() => { setNotes(''); setResult(null); setQuizCount(null) }}>
                  Clear
                </Button>
              </div>

              <fieldset className="rounded-xl border p-4">
                <legend className="px-1 text-sm font-semibold">Features to run</legend>
                <div className="grid grid-cols-2 gap-3 pt-1">
                  {FEATURES.map((feature) => (
                    <label key={feature} className="flex cursor-pointer items-center gap-2 text-sm">
                      <Checkbox checked={enabled[feature]} onCheckedChange={(checked) => setEnabled({ ...enabled, [feature]: !!checked })} />
                      {feature}
                    </label>
                  ))}
                </div>
                <p className="mt-3 text-[11px] text-muted-foreground">Ask Notes, AI Check and Autocomplete are always on.</p>
              </fieldset>

              <Button size="lg" className="h-11 text-base" onClick={analyze} disabled={loading || !notes.trim()}>
                {loading ? <Loader2 className="animate-spin" data-icon="inline-start" /> : <WandSparkles data-icon="inline-start" />}
                {loading ? 'Analyzing…' : 'Analyze Notes'}
              </Button>
              {result && notes !== analyzedNotes && <p className="-mt-2 text-center text-xs text-muted-foreground">Notes changed — analyze again to refresh the results.</p>}
            </CardContent>
          </Card>

          {/* RIGHT: results */}
          <Card className="min-h-[680px] min-w-0">
            <Tabs value={tab} onValueChange={(v) => setTab(String(v))}>
              <CardHeader className="border-b bg-muted/20 pb-0">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-4">
                  <div>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Lightbulb className="text-primary" /> Results
                    </CardTitle>
                    <CardDescription>Your personalised study workspace</CardDescription>
                  </div>
                  {result && (
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">
                        <Check data-icon="inline-start" /> Ready to study
                      </Badge>
                      <Button variant="outline" size="sm" onClick={exportTxt}>
                        <Download data-icon="inline-start" /> .txt
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => window.print()}>
                        <Printer data-icon="inline-start" /> PDF
                      </Button>
                    </div>
                  )}
                </div>
                <TabsList className="w-full justify-start overflow-x-auto bg-transparent">
                  {TABS.map((t) => (
                    <TabsTrigger key={t.value} value={t.value} className="shrink-0">
                      {t.value === 'ask' && <Bot className="size-3.5" />}
                      {t.value === 'aicheck' && <ScanSearch className="size-3.5" />}
                      {t.label}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </CardHeader>
              <CardContent className="p-5 sm:p-6">
                {loading ? (
                  <div className="flex flex-col gap-3">
                    <Skeleton className="h-8 w-56" />
                    <Skeleton className="h-36" />
                    <Skeleton className="h-5 w-3/4" />
                    <Skeleton className="h-5 w-1/2" />
                  </div>
                ) : (
                  <>
                    {result && result.errors.length > 0 && (
                      <div className="mb-4">
                        <ErrorNote>Some features failed: {result.errors.join(' · ')}</ErrorNote>
                      </div>
                    )}
                    {isUsingMock() && result && (
                      <p className="mb-4 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
                        Demo mode: the backend isn&apos;t running, so these are sample results. Start it with <code>uvicorn main:app --port 8000</code> in the backend folder.
                      </p>
                    )}
                    {TABS.map((t) => (
                      <TabsContent key={t.value} value={t.value} className="m-0">
                        {!result ? (
                          <EmptyState />
                        ) : t.feature && !enabled[t.feature] && !result[featureKey(t.feature)] ? (
                          <EmptyState title={`${t.label} is turned off`} hint={`Tick “${t.feature}” under Features to run, then analyze again.`} />
                        ) : (
                          <TabBody key={runId} tab={t.value} result={result} notes={analyzedNotes} concepts={concepts} onQuizCount={setQuizCount} />
                        )}
                      </TabsContent>
                    ))}
                  </>
                )}
              </CardContent>
            </Tabs>
          </Card>
        </div>

        <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {stats.map((s) => (
            <div key={s.label} className="flex items-center gap-3 rounded-xl border bg-card px-4 py-3">
              <s.icon className="shrink-0 text-primary" />
              <div>
                <p className="font-semibold tabular-nums">{s.value}</p>
                <p className="text-xs text-muted-foreground">{s.label}</p>
              </div>
            </div>
          ))}
        </div>
      </main>

      {result && <PrintPack result={result} />}

      <footer className="no-print mx-auto max-w-[1500px] px-4 pb-8 text-xs text-muted-foreground lg:px-8">
        Built for curious minds · NoteMind — NLP project (TextRank · Kneser-Ney language model · TF-IDF retrieval · MarianMT · Whisper · NLI · GPT-2 perplexity)
      </footer>
    </div>
  )
}

function featureKey(feature: Feature): keyof AnalysisResult {
  return ({ Summary: 'summary', Translate: 'translation', Quiz: 'quiz', 'Concept Map': 'map', Glossary: 'glossary', 'Hinglish Cleanup': 'cleanup' } as const)[feature]
}

function TabBody({ tab, result, notes, concepts, onQuizCount }: { tab: string; result: AnalysisResult; notes: string; concepts: string[]; onQuizCount: (n: number) => void }) {
  const missing = <EmptyState title="No result" hint="This feature didn't return anything — try analyzing again." />
  switch (tab) {
    case 'summary':
      return result.summary ? <SummaryTab notes={notes} initial={result.summary} /> : missing
    case 'translate':
      return result.translation ? <TranslateTab notes={notes} initial={result.translation} /> : missing
    case 'quiz':
      return result.quiz ? <QuizTab notes={notes} initial={result.quiz} onCountChange={onQuizCount} /> : missing
    case 'ask':
      return <AskTab notes={notes} concepts={concepts} />
    case 'map':
      return result.map ? <ConceptMapTab map={result.map} /> : missing
    case 'glossary':
      return result.glossary ? <GlossaryTab data={result.glossary} /> : missing
    case 'cleanup':
      return result.cleanup ? <CleanupTab data={result.cleanup} notes={notes} /> : missing
    case 'aicheck':
      return <AiCheckTab notes={notes} />
    default:
      return null
  }
}

/** Printable study pack (only visible when printing / saving as PDF). */
function PrintPack({ result }: { result: AnalysisResult }) {
  return (
    <section className="print-only p-8 text-black">
      <h1 className="mb-4 text-2xl font-bold">NoteMind study pack</h1>
      {result.summary && (
        <>
          <h2 className="mt-4 text-lg font-semibold">Summary</h2>
          <p className="whitespace-pre-line leading-7">{result.summary.summary}</p>
        </>
      )}
      {result.translation?.translation && (
        <>
          <h2 className="mt-4 text-lg font-semibold">Translation</h2>
          <p className="leading-7">{result.translation.translation}</p>
        </>
      )}
      {result.quiz && result.quiz.questions.length > 0 && (
        <>
          <h2 className="mt-4 text-lg font-semibold">Quiz</h2>
          <ol className="list-decimal pl-5">
            {result.quiz.questions.map((q, i) => (
              <li key={i} className="mb-2">
                {q.question}
                {q.options && <div className="text-sm">{q.options.map((o, j) => `${String.fromCharCode(97 + j)}) ${o}`).join('   ')}</div>}
                <div className="text-sm italic">Answer: {q.answer}</div>
              </li>
            ))}
          </ol>
        </>
      )}
      {result.glossary && (
        <>
          <h2 className="mt-4 text-lg font-semibold">Glossary</h2>
          <ul className="list-disc pl-5">
            {result.glossary.terms.map((t) => (
              <li key={t.term}>
                <b>{t.term}</b> ({t.difficulty}): {t.meaning} {t.hindi && `— ${t.hindi}`}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
