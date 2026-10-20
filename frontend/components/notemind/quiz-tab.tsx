'use client'

import { useState } from 'react'
import { Check, Eye, Layers, ListChecks, Loader2, RotateCcw, Sparkles, X } from 'lucide-react'
import {
  checkAnswer, generateQuestions, type AnswerCheckResult, type Difficulty, type Question, type QuestionType, type QuestionsResult,
} from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Slider } from '@/components/ui/slider'
import { cn } from '@/lib/utils'
import { ErrorNote, Segmented } from './shared'

type Verdict = { picked?: string; correct: boolean; partial?: boolean; check?: AnswerCheckResult }

const normalize = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, '').replace(/\s+/g, ' ').trim()
const kindOf = (q: Question) => q.type ?? (q.options?.length ? 'mcq' : q.question.includes('_____') ? 'fill' : 'short')

export function QuizTab({ notes, initial, onCountChange }: { notes: string; initial: QuestionsResult; onCountChange?: (n: number) => void }) {
  const [questions, setQuestions] = useState<Question[]>(initial.questions)
  const [count, setCount] = useState(5)
  const [difficulty, setDifficulty] = useState<Difficulty>('Medium')
  const [qtype, setQtype] = useState<QuestionType>('MCQ')
  const [verdicts, setVerdicts] = useState<Record<number, Verdict>>({})
  const [flashcards, setFlashcards] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [seed, setSeed] = useState(0)

  const generate = async (nextSeed = seed) => {
    setLoading(true)
    setError('')
    try {
      const result = await generateQuestions(notes, count, difficulty, qtype, nextSeed || undefined)
      setQuestions(result.questions)
      setVerdicts({})
      onCountChange?.(result.questions.length)
      if (result.questions.length < count) setError(`Your notes had enough material for ${result.questions.length} question(s). Add more notes for a longer quiz.`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const answered = Object.keys(verdicts).length
  // a partly correct short answer earns half a point
  const score = Object.values(verdicts).reduce((sum, v) => sum + (v.correct ? 1 : v.partial ? 0.5 : 0), 0)
  const mark = (i: number, v: Verdict) => setVerdicts((all) => (all[i] ? all : { ...all, [i]: v }))
  const unmark = (i: number) => setVerdicts(({ [i]: _removed, ...rest }) => rest)

  return (
    <div className="flex flex-col gap-5">
      {/* controls */}
      <div className="grid gap-4 rounded-xl border bg-muted/20 p-4 sm:grid-cols-2">
        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium">Number of questions: {count}</label>
          <Slider min={3} max={15} step={1} value={[count]} onValueChange={(v) => setCount(Array.isArray(v) ? v[0] : (v as number))} aria-label="Number of questions" />
        </div>
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium">Difficulty</span>
          <Segmented label="Difficulty" value={difficulty} options={['Easy', 'Medium', 'Hard'] as const} onChange={setDifficulty} />
        </div>
        <div className="flex flex-col gap-2 sm:col-span-2">
          <span className="text-xs font-medium">Question type</span>
          <div className="flex flex-wrap items-center gap-2">
            <Segmented label="Question type" value={qtype} options={['MCQ', 'Fill in the blank', 'Short answer', 'Mixed'] as const} onChange={setQtype} />
            <Button size="sm" className="ml-auto" onClick={() => generate()} disabled={loading}>
              {loading ? <Loader2 className="animate-spin" data-icon="inline-start" /> : <Sparkles data-icon="inline-start" />}
              Generate quiz
            </Button>
          </div>
        </div>
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}

      {/* score bar */}
      {questions.length > 0 && (
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm font-semibold">
            Score: {score}/{questions.length}
          </span>
          <Progress value={(answered / questions.length) * 100} className="w-40" aria-label="Quiz progress" />
          <span className="text-xs text-muted-foreground">{answered} answered</span>
          <div className="ml-auto flex gap-1">
            <Button variant="ghost" size="sm" onClick={() => setFlashcards(!flashcards)}>
              {flashcards ? <ListChecks data-icon="inline-start" /> : <Layers data-icon="inline-start" />}
              {flashcards ? 'Quiz mode' : 'Flashcard mode'}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setVerdicts({})}>
              <RotateCcw data-icon="inline-start" /> Retry
            </Button>
            <Button variant="ghost" size="sm" onClick={() => { const s = seed + 1; setSeed(s); generate(s) }} disabled={loading}>
              New questions
            </Button>
          </div>
        </div>
      )}

      {questions.length === 0 && !loading && (
        <p className="text-sm text-muted-foreground">No questions could be generated — try adding a few more sentences to your notes.</p>
      )}

      {flashcards ? (
        <div className="grid gap-4 sm:grid-cols-2">
          {questions.map((q, i) => (
            <Flashcard key={`${i}-${q.question}`} question={q} />
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {questions.map((q, i) => (
            <QuestionCard key={`${seed}-${i}-${q.question}`} index={i} question={q} verdict={verdicts[i]} onVerdict={(v) => mark(i, v)} onRetry={() => unmark(i)} />
          ))}
        </div>
      )}
      {answered === questions.length && questions.length > 0 && !flashcards && (
        <div className="rounded-xl border bg-primary/[0.06] p-4 text-center text-sm">
          🎉 You scored <b>{score}</b> out of <b>{questions.length}</b>.{' '}
          {score === questions.length ? 'Perfect — you know this topic!' : 'Press Retry to try again, or review the explanations above.'}
        </div>
      )}
    </div>
  )
}

function QuestionCard({ index, question, verdict, onVerdict, onRetry }: {
  index: number; question: Question; verdict?: Verdict; onVerdict: (v: Verdict) => void; onRetry: () => void
}) {
  const [typed, setTyped] = useState('')
  const [revealed, setRevealed] = useState(false)
  const [checking, setChecking] = useState(false)
  const [checkError, setCheckError] = useState('')
  const kind = kindOf(question)

  const check = async () => {
    setChecking(true)
    setCheckError('')
    try {
      const result = await checkAnswer(question.answer, typed, question.question)
      onVerdict({ picked: typed, correct: result.verdict === 'correct', partial: result.verdict === 'partial', check: result })
    } catch (e) {
      setCheckError((e as Error).message)
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className={cn('rounded-xl border p-4 transition-colors',
      verdict && (verdict.correct ? 'border-emerald-500/40' : verdict.partial ? 'border-amber-500/40' : 'border-rose-500/40'))}>
      <p className="font-medium leading-7">
        <span className="mr-1 text-muted-foreground">{index + 1}.</span>
        {question.question}
      </p>

      {kind === 'mcq' && (
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {question.options?.map((option) => {
            const isAnswer = option === question.answer
            const isPicked = verdict?.picked === option
            return (
              <button
                key={option}
                disabled={!!verdict}
                onClick={() => onVerdict({ picked: option, correct: isAnswer })}
                className={cn(
                  'flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition-colors',
                  !verdict && 'hover:border-primary hover:bg-primary/5',
                  verdict && isAnswer && 'border-emerald-500 bg-emerald-500/10 text-emerald-800 dark:text-emerald-300',
                  verdict && isPicked && !isAnswer && 'border-rose-500 bg-rose-500/10 text-rose-800 dark:text-rose-300',
                )}
              >
                {option}
                {verdict && isAnswer && <Check className="size-4" />}
                {verdict && isPicked && !isAnswer && <X className="size-4" />}
              </button>
            )
          })}
        </div>
      )}

      {kind === 'fill' && (
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            if (typed.trim()) onVerdict({ picked: typed, correct: normalize(typed) === normalize(question.answer) || normalize(typed).replace(/e?s$/, '') === normalize(question.answer).replace(/e?s$/, '') })
          }}
        >
          <input
            value={typed}
            disabled={!!verdict}
            onChange={(e) => setTyped(e.target.value)}
            placeholder="Type your answer…"
            aria-label={`Answer for question ${index + 1}`}
            className="min-w-0 flex-1 rounded-lg border bg-background px-3 py-1.5 text-sm outline-none focus:ring-3 focus:ring-ring/40"
          />
          <Button type="submit" size="sm" disabled={!!verdict || !typed.trim()}>
            Check
          </Button>
        </form>
      )}

      {kind === 'short' && !verdict && (
        <div className="mt-3 flex flex-wrap gap-2">
          {!revealed && (
            <form className="flex w-full flex-col gap-2" onSubmit={(e) => { e.preventDefault(); if (typed.trim()) check() }}>
              <textarea
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && typed.trim()) check() }}
                rows={2}
                placeholder="Type your answer in your own words — English or Hindi…"
                aria-label={`Answer for question ${index + 1}`}
                className="w-full resize-y rounded-lg border bg-background px-3 py-2 text-sm leading-6 outline-none focus:ring-3 focus:ring-ring/40"
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button type="submit" size="sm" disabled={!typed.trim() || checking}>
                  {checking ? <Loader2 className="animate-spin" data-icon="inline-start" /> : <Check data-icon="inline-start" />}
                  Check answer
                </Button>
                <Button type="button" variant="ghost" size="sm" onClick={() => setRevealed(true)}>
                  <Eye data-icon="inline-start" /> Just show the answer
                </Button>
                {checkError && <span className="text-xs text-rose-600 dark:text-rose-400">{checkError}</span>}
              </div>
            </form>
          )}
          {revealed && (
            <>
              <p className="w-full rounded-lg bg-muted/50 p-3 text-sm leading-6">{question.answer}</p>
              <span className="self-center text-xs text-muted-foreground">Did you get it right?</span>
              <Button variant="outline" size="sm" onClick={() => onVerdict({ correct: true })}>
                <Check data-icon="inline-start" /> Yes
              </Button>
              <Button variant="outline" size="sm" onClick={() => onVerdict({ correct: false })}>
                <X data-icon="inline-start" /> Not quite
              </Button>
            </>
          )}
        </div>
      )}

      {verdict?.check && <AnswerFeedback check={verdict.check} answer={question.answer} onRetry={onRetry} />}

      {verdict && !verdict.check && (
        <div className="mt-3 text-sm leading-6">
          <span className={cn('font-semibold', verdict.correct ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400')}>
            {verdict.correct ? 'Correct! ' : 'Not quite. '}
          </span>
          {kind !== 'mcq' && (
            <span>
              Answer: <b>{question.answer}</b>.{' '}
            </span>
          )}
          <span className="text-muted-foreground">{question.explanation}</span>
        </div>
      )}
    </div>
  )
}

function AnswerFeedback({ check, answer, onRetry }: { check: AnswerCheckResult; answer: string; onRetry: () => void }) {
  const tone = check.verdict === 'correct' ? 'emerald' : check.verdict === 'partial' ? 'amber' : 'rose'
  return (
    <div className="mt-3 flex flex-col gap-2 text-sm leading-6">
      <div className="flex flex-wrap items-center gap-3">
        <span className={cn('font-semibold', tone === 'emerald' && 'text-emerald-600 dark:text-emerald-400',
          tone === 'amber' && 'text-amber-600 dark:text-amber-400', tone === 'rose' && 'text-rose-600 dark:text-rose-400')}>
          {check.verdict === 'correct' ? 'Correct!' : check.verdict === 'partial' ? 'Partly correct' : 'Not quite'}
        </span>
        <div className="flex items-center gap-2">
          <div className="h-2 w-28 overflow-hidden rounded-full bg-muted" aria-label={`Score ${check.score} out of 100`}>
            <div className={cn('h-full rounded-full', tone === 'emerald' && 'bg-emerald-500', tone === 'amber' && 'bg-amber-500', tone === 'rose' && 'bg-rose-500')}
              style={{ width: `${check.score}%` }} />
          </div>
          <span className="text-xs tabular-nums text-muted-foreground">{check.score}/100</span>
        </div>
        <Button variant="ghost" size="xs" className="ml-auto" onClick={onRetry}>
          <RotateCcw data-icon="inline-start" /> Try again
        </Button>
      </div>
      <p>{check.feedback}</p>
      {(check.matched_terms.length > 0 || check.missing_terms.length > 0) && (
        <div className="flex flex-wrap gap-1.5 text-xs">
          {check.matched_terms.map((t) => (
            <span key={t} className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-emerald-800 dark:text-emerald-300">✓ {t}</span>
          ))}
          {check.missing_terms.map((t) => (
            <span key={t} className="rounded-full bg-muted px-2 py-0.5 text-muted-foreground">+ {t}</span>
          ))}
        </div>
      )}
      <p className="rounded-lg bg-muted/50 p-3">
        <span className="text-xs font-medium text-muted-foreground">Expected answer: </span>{answer}
      </p>
      <p className="text-[11px] text-muted-foreground">
        Meaning match {Math.round(check.similarity * 100)}%
        {check.entailment != null && ` · supported by the expected answer ${Math.round(check.entailment * 100)}%`}
        {check.contradiction != null && ` · contradiction ${Math.round(check.contradiction * 100)}%`}
        {` · ${check.engine}`}
      </p>
    </div>
  )
}

function Flashcard({ question }: { question: Question }) {
  const [flipped, setFlipped] = useState(false)
  return (
    <button
      onClick={() => setFlipped(!flipped)}
      aria-label={flipped ? 'Show question' : 'Show answer'}
      className="group h-48 w-full [perspective:1000px]"
    >
      <div className={cn('relative h-full w-full rounded-xl transition-transform duration-500 [transform-style:preserve-3d]', flipped && '[transform:rotateY(180deg)]')}>
        <div className="absolute inset-0 flex flex-col justify-between overflow-hidden rounded-xl border bg-card p-4 text-left [backface-visibility:hidden]">
          <p className="line-clamp-5 text-sm font-medium leading-6">{question.question.replace(/^(Choose the term that best completes the sentence:|Fill in the blank:|रिक्त स्थान भरिए:|रिक्त स्थान के लिए सही विकल्प चुनिए:)\s*/, '')}</p>
          <span className="text-[11px] text-muted-foreground">Click to flip</span>
        </div>
        <div className="absolute inset-0 flex flex-col justify-center overflow-hidden rounded-xl border border-primary/40 bg-primary/[0.07] p-4 text-left [backface-visibility:hidden] [transform:rotateY(180deg)]">
          <p className="line-clamp-6 text-sm font-semibold leading-6">{question.answer}</p>
        </div>
      </div>
    </button>
  )
}
