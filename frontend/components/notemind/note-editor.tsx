'use client'

import { useEffect, useRef, useState } from 'react'
import { autocomplete } from '@/lib/api'

const SHARED = 'w-full whitespace-pre-wrap break-words px-4 py-3 text-[15px] leading-7 font-sans'

/**
 * Textarea with "ghost text" autocomplete (the language-modelling feature).
 * A mirror div sits behind a transparent-background textarea and renders the same text plus the
 * greyed suggestion, so the suggestion appears right after the caret. Tab accepts, Esc dismisses.
 */
export function NoteEditor({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const [suggestion, setSuggestion] = useState('')
  const [caretAtEnd, setCaretAtEnd] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const mirrorRef = useRef<HTMLDivElement>(null)
  const requestId = useRef(0)

  useEffect(() => {
    setSuggestion('')
    if (!value.trim() || !caretAtEnd) return
    const id = ++requestId.current
    const timer = setTimeout(() => {
      autocomplete(value)
        .then((data) => {
          if (id === requestId.current) setSuggestion(data.suggestion ?? '')
        })
        .catch(() => setSuggestion(''))
    }, 600)
    return () => clearTimeout(timer)
  }, [value, caretAtEnd])

  const syncScroll = () => {
    if (mirrorRef.current && textareaRef.current) mirrorRef.current.scrollTop = textareaRef.current.scrollTop
  }

  const updateCaret = () => {
    const el = textareaRef.current
    if (el) setCaretAtEnd(el.selectionStart === el.value.length && el.selectionEnd === el.value.length)
  }

  const accept = () => {
    onChange(value + suggestion)
    setSuggestion('')
    requestAnimationFrame(() => {
      const el = textareaRef.current
      if (el) {
        el.selectionStart = el.selectionEnd = el.value.length
        syncScroll()
      }
    })
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="relative min-h-[350px] overflow-hidden rounded-xl border bg-muted/30 focus-within:ring-3 focus-within:ring-ring/40">
        <div ref={mirrorRef} aria-hidden className={`${SHARED} pointer-events-none absolute inset-0 overflow-y-scroll text-transparent [scrollbar-color:transparent_transparent]`}>
          {value}
          {suggestion && caretAtEnd && <span className="text-muted-foreground/70">{suggestion}</span>}
        </div>
        <textarea
          ref={textareaRef}
          value={value}
          aria-label="Your notes"
          spellCheck
          onChange={(event) => {
            onChange(event.target.value)
            setCaretAtEnd(event.target.selectionStart === event.target.value.length)
          }}
          onKeyDown={(event) => {
            if (event.key === 'Tab' && suggestion && caretAtEnd) {
              event.preventDefault()
              accept()
            } else if (event.key === 'Escape') {
              setSuggestion('')
            }
          }}
          onSelect={updateCaret}
          onClick={updateCaret}
          onScroll={syncScroll}
          placeholder="Type or paste your notes in English, Hindi, or Hinglish…"
          className={`${SHARED} relative block h-[350px] resize-y overflow-y-scroll bg-transparent outline-none placeholder:text-muted-foreground`}
        />
      </div>
      <div className="flex min-h-5 items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          {value.length} characters · {value.trim() ? value.trim().split(/\s+/).length : 0} words
        </span>
        {suggestion && caretAtEnd ? (
          <span className="flex items-center gap-1">
            <kbd className="rounded border bg-background px-1.5 py-0.5 font-mono text-[10px]">Tab</kbd> to accept suggestion
          </span>
        ) : (
          <span className="hidden sm:inline">Smart autocomplete learns from your notes as you type</span>
        )}
      </div>
    </div>
  )
}
