'use client'

import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Bot, Loader2, Send, User } from 'lucide-react'
import { askNotes, type AskResult } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { ListenButton } from './shared'

type Message = { role: 'user'; text: string } | { role: 'bot'; result: AskResult; question: string }

const STOP = new Set('what is are the a an of in on to why how does do explain which who where when me about simply क्या है हैं के की का में को'.split(' '))

/** Highlight the question's words inside a source sentence. */
function highlight(sentence: string, question: string): ReactNode[] {
  const words = question.toLowerCase().match(/[\p{L}\p{M}\p{N}]+/gu)?.filter((w) => w.length > 2 && !STOP.has(w)) ?? []
  if (!words.length) return [sentence]
  const pattern = new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'giu')
  return sentence.split(pattern).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="rounded bg-primary/20 px-0.5 text-foreground">
        {part}
      </mark>
    ) : (
      part
    ),
  )
}

export function AskTab({ notes, concepts }: { notes: string; concepts: string[] }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, loading])

  const suggestions = [
    'What is the main idea?',
    'Explain simply',
    ...(concepts[0] ? [`What is ${concepts[0].toLowerCase()}?`] : []),
    ...(concepts[1] ? [`${concepts[1]} क्या है?`] : []),
  ]

  const ask = async (q: string) => {
    const text = q.trim()
    if (!text || loading) return
    setQuestion('')
    setMessages((m) => [...m, { role: 'user', text }])
    setLoading(true)
    try {
      const result = await askNotes(notes, text)
      setMessages((m) => [...m, { role: 'bot', result, question: text }])
    } catch (e) {
      setMessages((m) => [...m, { role: 'bot', question: text, result: { answer: `Something went wrong: ${(e as Error).message}`, sources: [] } }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2">
        {suggestions.map((s) => (
          <button key={s} onClick={() => ask(s)} className="rounded-full border bg-background px-3 py-1 text-xs transition-colors hover:border-primary hover:bg-primary/5">
            {s}
          </button>
        ))}
      </div>
      <div className="flex h-[440px] flex-col gap-4 overflow-y-auto rounded-2xl border bg-muted/20 p-4">
        {messages.length === 0 && (
          <div className="m-auto max-w-sm text-center text-sm text-muted-foreground">
            <Bot className="mx-auto mb-2 text-primary" />
            Ask anything about your notes — in English or Hindi. Answers come only from what you wrote, with the matching lines shown as sources.
          </div>
        )}
        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="flex items-start justify-end gap-2">
              <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground">{m.text}</div>
              <User className="mt-1 size-5 shrink-0 text-muted-foreground" />
            </div>
          ) : (
            <div key={i} className="flex items-start gap-2">
              <Bot className="mt-1 size-5 shrink-0 text-primary" />
              <div className="max-w-[85%] rounded-2xl rounded-tl-sm border bg-background px-4 py-3 text-sm leading-6">
                <p>{m.result.answer}</p>
                <div className="mt-1 flex items-center gap-2">
                  <ListenButton text={m.result.answer} />
                  {m.result.confidence !== undefined && m.result.sources.length > 0 && (
                    <span className="text-[11px] text-muted-foreground">Match: {Math.round(m.result.confidence * 100)}%</span>
                  )}
                </div>
                {m.result.sources.length > 0 && (
                  <details className="mt-1 text-xs">
                    <summary className="cursor-pointer select-none font-medium text-primary">Source from your notes ({m.result.sources.length})</summary>
                    <ul className="mt-2 flex flex-col gap-1.5 border-l-2 border-primary/30 pl-3 text-muted-foreground">
                      {m.result.sources.map((s, j) => (
                        <li key={j}>{highlight(s, m.question)}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </div>
          ),
        )}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Searching your notes…
          </div>
        )}
        <div ref={endRef} />
      </div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          ask(question)
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask in English or Hindi… e.g. What does chlorophyll do?"
          aria-label="Ask a question about your notes"
          className="min-w-0 flex-1 rounded-xl border bg-background px-4 py-2 text-sm outline-none focus:ring-3 focus:ring-ring/40"
        />
        <Button type="submit" size="icon-lg" disabled={!question.trim() || loading} aria-label="Send question">
          <Send />
        </Button>
      </form>
    </div>
  )
}
