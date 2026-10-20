'use client'

import { useEffect, useRef, useState } from 'react'
import { FileAudio, Loader2, Mic, Square } from 'lucide-react'
import { transcribeAudio } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Segmented } from './shared'

const MAX_SECONDS = 10 * 60
const LANGS = [
  { value: 'auto', label: 'Auto' },
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'हिंदी' },
] as const
type SpokenLanguage = (typeof LANGS)[number]['value']

const formatTime = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

/** Voice notes: record from the microphone (or upload an audio file) and add Whisper's transcript to the notes. */
export function VoiceRecorder({ onTranscript }: { onTranscript: (text: string) => void }) {
  const [state, setState] = useState<'idle' | 'recording' | 'transcribing'>('idle')
  const [language, setLanguage] = useState<SpokenLanguage>('auto')
  const [seconds, setSeconds] = useState(0)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (state !== 'recording') return
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => clearInterval(timer)
  }, [state])

  useEffect(() => {
    if (state === 'recording' && seconds >= MAX_SECONDS) recorderRef.current?.stop()
  }, [seconds, state])

  // release the microphone if the page is closed mid-recording
  useEffect(() => () => recorderRef.current?.stream.getTracks().forEach((t) => t.stop()), [])

  const send = async (audio: Blob, filename: string) => {
    setState('transcribing')
    setError('')
    setStatus('')
    try {
      const result = await transcribeAudio(audio, language, filename)
      if (!result.text) {
        setError(result.message || 'No speech was recognised.')
        return
      }
      onTranscript(result.text)
      const lang = result.language === 'hi' ? 'Hindi' : result.language === 'en' ? 'English' : result.language
      setStatus(`Added ${result.text.split(/\s+/).length} words · ${lang} · ${Math.round(result.duration ?? 0)} s of audio · ${result.engine}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setState('idle')
    }
  }

  const start = async () => {
    setError('')
    setStatus('')
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('This browser cannot record audio. Try Chrome or Edge, or upload an audio file.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'].find((t) => MediaRecorder.isTypeSupported(t))
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      const chunks: Blob[] = []
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data)
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const type = recorder.mimeType || 'audio/webm'
        const ext = type.includes('ogg') ? 'ogg' : type.includes('mp4') ? 'm4a' : 'webm'
        send(new Blob(chunks, { type }), `voice-note.${ext}`)
      }
      recorderRef.current = recorder
      recorder.start(1000)
      setSeconds(0)
      setState('recording')
    } catch (e) {
      const name = (e as DOMException).name
      setError(name === 'NotAllowedError' ? 'Microphone access was blocked. Allow it from the address bar and try again.'
        : name === 'NotFoundError' ? 'No microphone was found.' : (e as Error).message)
    }
  }

  const upload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (file) send(file, file.name)
  }

  return (
    <div className="flex flex-col gap-2 rounded-xl border p-3">
      <div className="flex flex-wrap items-center gap-2">
        {state === 'recording' ? (
          <Button size="sm" variant="destructive" onClick={() => recorderRef.current?.stop()}>
            <Square data-icon="inline-start" /> Stop · {formatTime(seconds)}
          </Button>
        ) : (
          <Button size="sm" variant="outline" onClick={start} disabled={state === 'transcribing'}>
            {state === 'transcribing' ? <Loader2 className="animate-spin" data-icon="inline-start" /> : <Mic data-icon="inline-start" />}
            {state === 'transcribing' ? 'Transcribing…' : 'Voice note'}
          </Button>
        )}
        <input ref={fileRef} type="file" accept="audio/*,.m4a,.mp3,.wav,.ogg,.webm" className="hidden" onChange={upload} />
        <Button size="sm" variant="ghost" onClick={() => fileRef.current?.click()} disabled={state !== 'idle'}>
          <FileAudio data-icon="inline-start" /> Audio file
        </Button>
        <span className="ml-auto">
          <Segmented label="Spoken language" value={language} options={[...LANGS]} onChange={setLanguage} />
        </span>
      </div>
      {state === 'recording' && (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="size-2 animate-pulse rounded-full bg-rose-500" /> Recording… speak your notes, then press Stop.
        </p>
      )}
      {state === 'transcribing' && <p className="text-xs text-muted-foreground">Whisper is turning your speech into text. The first time takes longer while the model loads.</p>}
      {status && <p className="text-xs text-emerald-700 dark:text-emerald-300">{status}</p>}
      {error && <p className="text-xs text-amber-700 dark:text-amber-300">{error}</p>}
    </div>
  )
}
