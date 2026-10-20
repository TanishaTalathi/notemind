'use client'

import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Check, Clipboard, Loader2, Sparkles, Square, Volume2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { speakOffline } from '@/lib/api'
import { cn } from '@/lib/utils'

export const isHindi = (text: string) => /[ऀ-ॿ]/.test(text)

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => {
        navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1200)
      }}
    >
      {copied ? <Check data-icon="inline-start" /> : <Clipboard data-icon="inline-start" />}
      {copied ? 'Copied!' : label}
    </Button>
  )
}

/** Best voice for a language. Installed (local) voices come first: Chrome's online "Google" voices need
 *  Google's servers and fail with "synthesis-failed" on networks that block Google. */
function pickVoice(voices: SpeechSynthesisVoice[], lang: string) {
  const base = lang.slice(0, 2)
  const score = (v: SpeechSynthesisVoice) =>
    (v.localService ? 4 : 0) + (v.lang.replace('_', '-') === lang ? 2 : 0) + (/natural/i.test(v.name) ? 1 : 0)
  return voices
    .filter((v) => v.lang.toLowerCase().startsWith(base))
    .sort((a, b) => score(b) - score(a))[0]
}

/** Chrome loads voices asynchronously and returns [] at first, so wait briefly for them. */
function loadVoices(synth: SpeechSynthesis): Promise<SpeechSynthesisVoice[]> {
  const now = synth.getVoices()
  if (now.length) return Promise.resolve(now)
  return new Promise((resolve) => {
    const done = () => {
      synth.removeEventListener('voiceschanged', done)
      resolve(synth.getVoices())
    }
    synth.addEventListener('voiceschanged', done)
    setTimeout(done, 1500)
  })
}

/** Sentence-sized pieces: Chrome's online voices stop speaking after ~15 s of a single utterance. */
function speechChunks(text: string, limit = 180) {
  const sentences = text.replace(/[•_]+/g, ' ').match(/[^.!?।\n]+[.!?।]*/g) ?? []
  const chunks: string[] = []
  for (const sentence of sentences.map((s) => s.trim()).filter(Boolean)) {
    const last = chunks[chunks.length - 1]
    if (last && last.length + sentence.length < limit) chunks[chunks.length - 1] = `${last} ${sentence}`
    else chunks.push(sentence)
  }
  return chunks
}

// a tiny silent clip played inside the click so the browser lets this <audio> element play later,
// after the slow fetch to the backend (autoplay rules want a recent click).
const SILENCE = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQQAAAAAAAAA'
const START_TIMEOUT_MS = 5000

// Once a browser voice fails for a language, use the backend's offline voice for the rest of the session.
const browserFailed: Record<'hi' | 'en', boolean> = { hi: false, en: false }

/** Reads text aloud. Uses the browser's speech synthesis (installed voices first) and falls back to the
 *  backend's offline voice (MMS-TTS) when there is no voice, it errors, or it never starts. */
export function ListenButton({ text }: { text: string }) {
  const [state, setState] = useState<'idle' | 'loading' | 'speaking'>('idle')
  const [problem, setProblem] = useState('')
  const stopRef = useRef<() => void>(() => undefined)
  useEffect(() => () => stopRef.current(), [])

  const finish = () => setState('idle')

  /** Plays chunks through the backend, fetching the next one while the current one plays. */
  const playFromServer = (chunks: string[], lang: 'hi' | 'en', audio: HTMLAudioElement) => {
    const controller = new AbortController()
    let url = ''
    stopRef.current = () => {
      controller.abort()
      audio.pause()
      if (url) URL.revokeObjectURL(url)
    }
    setState('loading')
    // a prefetch that Stop cancels is never awaited, so keep its rejection from going unhandled
    const fetchChunk = (chunk: string) => {
      const request = speakOffline(chunk, lang, controller.signal)
      request.catch(() => undefined)
      return request
    }
    const run = async () => {
      let next = fetchChunk(chunks[0])
      for (let i = 0; i < chunks.length; i++) {
        const blob = await next
        if (i + 1 < chunks.length) next = fetchChunk(chunks[i + 1])
        if (url) URL.revokeObjectURL(url)
        url = URL.createObjectURL(blob)
        audio.src = url
        await audio.play()
        setState('speaking')
        await new Promise<void>((resolve, reject) => {
          audio.onended = () => resolve()
          audio.onerror = () => reject(new Error('the audio could not be decoded'))
          controller.signal.addEventListener('abort', () => resolve(), { once: true })
        })
        if (controller.signal.aborted) return
      }
    }
    run()
      .catch((e: Error) => {
        if (controller.signal.aborted) return
        setProblem(e.name === 'NotAllowedError' ? 'The browser blocked the audio. Click Listen again.' : `Could not play audio: ${e.message}.`)
      })
      .finally(() => {
        if (url) URL.revokeObjectURL(url)
        finish()
      })
  }

  const toggle = async () => {
    if (state !== 'idle') {
      stopRef.current()
      finish()
      return
    }
    setProblem('')
    const audio = new Audio(SILENCE)
    audio.play().catch(() => undefined) // unlock while we still have the click
    const lang: 'hi' | 'en' = isHindi(text) ? 'hi' : 'en'
    const chunks = speechChunks(text)
    if (!chunks.length) return
    const synth = typeof window !== 'undefined' ? window.speechSynthesis : undefined
    const voice = synth && !browserFailed[lang] ? pickVoice(await loadVoices(synth), lang === 'hi' ? 'hi-IN' : 'en-US') : undefined
    if (!synth || !voice) return playFromServer(chunks, lang, audio)

    let stopped = false
    let current = 0 // index of the chunk being spoken
    let started = false
    const fallBack = () => {
      if (stopped) return
      stopped = true
      synth.cancel()
      browserFailed[lang] = true
      playFromServer(chunks.slice(current), lang, audio)
    }
    stopRef.current = () => {
      stopped = true
      synth.cancel()
    }
    synth.cancel()
    chunks.forEach((chunk, i) => {
      const utterance = new SpeechSynthesisUtterance(chunk)
      utterance.lang = voice.lang
      utterance.voice = voice
      utterance.rate = 0.95
      utterance.onstart = () => {
        started = true
        current = i
      }
      if (i === chunks.length - 1) utterance.onend = () => !stopped && finish()
      utterance.onerror = (e) => {
        if (stopped || e.error === 'interrupted' || e.error === 'canceled') return
        current = i
        fallBack()
      }
      synth.speak(utterance)
    })
    synth.resume() // Chrome can leave the queue paused after cancel()
    setState('speaking')
    // some voices fail silently: no error, no sound. If nothing starts, use the offline voice instead.
    setTimeout(() => {
      if (!started) fallBack()
    }, START_TIMEOUT_MS)
  }

  return (
    <span className="inline-flex items-center gap-1">
      <Button variant="ghost" size="sm" onClick={toggle} aria-pressed={state !== 'idle'}>
        {state === 'loading' ? (
          <Loader2 className="animate-spin" data-icon="inline-start" />
        ) : state === 'speaking' ? (
          <Square data-icon="inline-start" />
        ) : (
          <Volume2 data-icon="inline-start" />
        )}
        {state === 'loading' ? 'Loading…' : state === 'speaking' ? 'Stop' : 'Listen'}
      </Button>
      {problem && <span className="max-w-56 text-[11px] leading-tight text-amber-700 dark:text-amber-300">{problem}</span>}
    </span>
  )
}

export function EmptyState({ title = 'Your insights will appear here', hint }: { title?: string; hint?: string }) {
  return (
    <div className="flex min-h-[340px] flex-col items-center justify-center text-center">
      <div className="mb-4 flex size-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <Sparkles />
      </div>
      <p className="font-medium">{title}</p>
      <p className="mt-1 max-w-xs text-sm text-muted-foreground">
        {hint ?? 'Add notes on the left, then press Analyze Notes to unlock your study workspace.'}
      </p>
    </div>
  )
}

/** A small segmented control (row of toggle buttons). */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T
  options: readonly T[] | { value: T; label: ReactNode }[]
  onChange: (value: T) => void
  label: string
}) {
  const items = (options as (T | { value: T; label: ReactNode })[]).map((o) =>
    typeof o === 'string' ? { value: o, label: o } : o,
  )
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap rounded-lg border bg-muted/40 p-0.5">
      {items.map((item) => (
        <button
          key={item.value}
          role="radio"
          aria-checked={value === item.value}
          onClick={() => onChange(item.value)}
          className={cn(
            'rounded-md px-3 py-1 text-xs font-medium transition-colors',
            value === item.value ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}

export function EngineNote({ engine }: { engine?: string }) {
  if (!engine) return null
  return <span className="text-[11px] text-muted-foreground">Engine: {engine}</span>
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-900 dark:text-amber-200">
      {children}
    </div>
  )
}
