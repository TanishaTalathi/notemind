// NoteMind API client.
// Talks to the FastAPI backend (backend/main.py). If the backend is not running, every call falls back to
// realistic demo data so the UI still works; the header badge shows "Demo mode" in that case.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
// Set NEXT_PUBLIC_USE_MOCK=true to force demo data even when the backend is running.
export const FORCE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === 'true'

export type Language = 'en' | 'hi' | 'hinglish'
export type SummaryMode = 'short' | 'detailed' | 'bullets'
export type Difficulty = 'Easy' | 'Medium' | 'Hard'
export type QuestionType = 'MCQ' | 'Fill in the blank' | 'Short answer' | 'Mixed'

export type DetectResult = { language: Language; confidence: number }
export type AutocompleteResult = { suggestion: string; confidence?: number; engine?: string }
export type SummaryResult = { summary: string; original_words: number; summary_words: number; engine?: string; key_sentences?: string[] }
export type TranslateResult = { translation: string; engine?: string; available?: boolean; message?: string }
export type Question = { type?: 'mcq' | 'fill' | 'short'; question: string; options?: string[]; answer: string; explanation: string }
export type QuestionsResult = { questions: Question[]; available?: number }
export type AskResult = { answer: string; sources: string[]; confidence?: number; engine?: string }
export type ConceptNode = { id: string; importance: number; count?: number; sentences?: string[] }
export type ConceptEdge = { source: string; target: string; relation: string; weight?: number }
export type ConceptMapResult = { nodes: ConceptNode[]; edges: ConceptEdge[]; keywords?: { word: string; score: number }[] }
export type GlossaryTerm = { term: string; meaning: string; hindi: string; difficulty: Difficulty; source?: string }
export type GlossaryResult = { terms: GlossaryTerm[]; readability: number; grade: string; level?: string; simplified?: string }
export type CleanupResult = { devanagari: string; english: string; changed?: string[]; engine?: string; message?: string; input_language?: string }
export type TranscribeResult = {
  available: boolean; text: string; segments: { start: number; end: number; text: string }[]
  language?: string; language_probability?: number; duration?: number; engine?: string; message?: string
}
export type AnswerCheckResult = {
  score: number; verdict: 'correct' | 'partial' | 'incorrect'; similarity: number; coverage: number
  entailment?: number | null; contradiction?: number | null; contradicts?: boolean
  matched_terms: string[]; missing_terms: string[]; feedback: string; engine: string
}
export type AiSignal = { name: string; label: string; display: string; contribution: number; leans: 'ai' | 'human' | 'neutral'; explanation: string; typical?: string }
export type AiCheckResult = {
  available: boolean; ai_probability: number | null; label: string; confidence?: 'high' | 'medium' | 'low' | 'none'
  signals: AiSignal[]; sentences: { text: string; perplexity: number | null; phrases: string[]; flag?: boolean }[]
  words: number; language: Language; engine: string; message: string; model_accuracy?: number
}
export type EvalBar = { title: string; unit: string; max: number; items: { label: string; value: number }[] }
export type EvalResult = {
  suite: string; title: string; metric: string; description: string; dataset: string; ran_at: string; seconds: number
  headline: { label: string; value: string | number; hint?: string }[]
  tables: { title: string; columns: string[]; rows: (string | number)[][] }[]
  bars: EvalBar[]
  matrix?: { title: string; labels: string[]; values: number[][] } | null
  notes: string[]
}
export type EvalOverview = { suites: Record<string, { title: string; metric: string; description: string }>; results: Record<string, EvalResult> }

let usingMock = FORCE_MOCK
export const isUsingMock = () => usingMock

export async function checkBackend(): Promise<boolean> {
  if (FORCE_MOCK) return false
  try {
    const response = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(2500) })
    usingMock = !response.ok
  } catch {
    usingMock = true
  }
  return !usingMock
}

const wait = (ms = 600) => new Promise((resolve) => setTimeout(resolve, ms))

async function request<T>(path: string, body: unknown, mock: () => T): Promise<T> {
  if (usingMock) {
    await wait()
    return mock()
  }
  const response = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(`${path} failed (${response.status})`)
  return response.json()
}

// ---------- API calls ----------
export const detectLanguage = (text: string) => request<DetectResult>('/detect', { text }, () => mock.detect(text))
export const autocomplete = (text: string) => request<AutocompleteResult>('/autocomplete', { text }, () => mock.autocomplete(text))
export const summarize = (text: string, mode: SummaryMode) => request<SummaryResult>('/summarize', { text, mode }, () => mock.summary(text, mode))
export const translate = (text: string, source: Language, target: Language) =>
  request<TranslateResult>('/translate', { text, source, target }, () => mock.translate(target))
export const generateQuestions = (text: string, count: number, difficulty: Difficulty, qtype: QuestionType, seed?: number) =>
  request<QuestionsResult>('/questions', { text, count, difficulty, qtype, seed }, () => mock.questions(count, qtype))
export const askNotes = (text: string, question: string) => request<AskResult>('/ask', { text, question }, () => mock.ask(question))
export const conceptMap = (text: string) => request<ConceptMapResult>('/concept-map', { text }, () => mock.conceptMap)
export const glossary = (text: string) => request<GlossaryResult>('/glossary', { text }, () => mock.glossary)
export const cleanupHinglish = (text: string) => request<CleanupResult>('/cleanup', { text }, () => mock.cleanup)

/** Speech as a WAV file from the backend's offline voices (used when the browser voice is missing or fails). */
export async function speakOffline(text: string, lang: 'hi' | 'en', signal?: AbortSignal): Promise<Blob> {
  if (usingMock) throw new Error('offline audio needs the backend to be running')
  const response = await fetch(`${API_URL}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang }),
    signal,
  })
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.message ?? `/tts failed (${response.status})`)
  }
  return response.blob()
}

/** Voice notes: send a recording or audio file to Whisper on the backend. */
export async function transcribeAudio(audio: Blob, language: 'auto' | 'en' | 'hi', filename = 'voice-note.webm'): Promise<TranscribeResult> {
  if (usingMock) throw new Error('Voice notes need the backend to be running')
  const form = new FormData()
  form.append('audio', audio, filename)
  form.append('language', language)
  const response = await fetch(`${API_URL}/transcribe`, { method: 'POST', body: form })
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new Error(data?.message ?? `/transcribe failed (${response.status})`)
  return data
}

export const checkAnswer = (expected: string, answer: string, question: string) =>
  request<AnswerCheckResult>('/check-answer', { expected, answer, question }, () => mock.checkAnswer(expected, answer))
export const aiCheck = (text: string) => request<AiCheckResult>('/ai-check', { text }, () => mock.aiCheck(text))

export async function getEvaluation(): Promise<EvalOverview> {
  const response = await fetch(`${API_URL}/evaluate`)
  if (!response.ok) throw new Error(`/evaluate failed (${response.status})`)
  return response.json()
}

export async function runEvaluation(suite: string): Promise<EvalResult> {
  const response = await fetch(`${API_URL}/evaluate/${suite}`, { method: 'POST' })
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new Error(data?.message ?? `/evaluate/${suite} failed (${response.status})`)
  return data
}

export const FEATURES = ['Summary', 'Translate', 'Quiz', 'Concept Map', 'Glossary', 'Hinglish Cleanup'] as const
export type Feature = (typeof FEATURES)[number]

export type AnalysisResult = {
  language: DetectResult
  summary?: SummaryResult
  translation?: TranslateResult & { source: Language; target: Language }
  quiz?: QuestionsResult
  map?: ConceptMapResult
  glossary?: GlossaryResult
  cleanup?: CleanupResult
  errors: string[]
}

/** Run every enabled feature in parallel. A failing feature is reported in `errors` instead of breaking the rest. */
export async function analyzeAll(text: string, enabled: Record<Feature, boolean>): Promise<AnalysisResult> {
  const language = await detectLanguage(text)
  const source: Language = language.language === 'hi' ? 'hi' : language.language
  const target: Language = language.language === 'en' ? 'hi' : 'en'
  const errors: string[] = []
  const run = async <T,>(feature: Feature, fn: () => Promise<T>): Promise<T | undefined> => {
    if (!enabled[feature]) return undefined
    try {
      return await fn()
    } catch (error) {
      errors.push(`${feature}: ${(error as Error).message}`)
      return undefined
    }
  }
  const [summary, translation, quiz, map, glossaryData, cleanup] = await Promise.all([
    run('Summary', () => summarize(text, 'detailed')),
    run('Translate', async () => ({ ...(await translate(text, source, target)), source, target })),
    run('Quiz', () => generateQuestions(text, 5, 'Medium', 'MCQ')),
    run('Concept Map', () => conceptMap(text)),
    run('Glossary', () => glossary(text)),
    run('Hinglish Cleanup', () => cleanupHinglish(text)),
  ])
  return { language, summary, translation, quiz, map, glossary: glossaryData, cleanup, errors }
}

export const countWords = (text: string) => (text.trim() ? text.trim().split(/\s+/).length : 0)

export const sampleNotes = `Photosynthesis is the process by which green plants convert light energy into chemical energy. Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts. Plants use carbon dioxide from the air and water from the soil to produce glucose and oxygen. Glucose provides energy for growth, while oxygen is released into the atmosphere. The chloroplasts contain chlorophyll, which gives leaves their green colour. Photosynthesis is important because it forms the base of almost every food chain and maintains the balance of oxygen and carbon dioxide in the atmosphere.`

export const sampleHindi = `प्रकाश संश्लेषण वह प्रक्रिया है जिसमें हरे पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलते हैं। क्लोरोफिल एक हरा वर्णक है जो सूर्य के प्रकाश को अवशोषित करता है। पौधे कार्बन डाइऑक्साइड और पानी से ग्लूकोज और ऑक्सीजन बनाते हैं। ग्लूकोज पौधों को वृद्धि के लिए ऊर्जा देता है। ऑक्सीजन वायुमंडल में छोड़ी जाती है।`

export const sampleHinglish = `Photosynthesis me plants light energy ko chemical energy me convert karte hain. Chlorophyll ek green pigment hai jo sunlight ko absorb karta hai. Paudhe carbon dioxide aur paani se glucose aur oxygen banate hain. Oxygen atmosphere me release hoti hai.`

// ---------- demo data (used only when the backend is offline) ----------
const mock = {
  detect: (text: string): DetectResult => ({
    language: /[ऀ-ॿ]/.test(text) ? 'hi' : /\b(hai|hain|ka|ki|ko|mein|me|aur|ke|karta|karte)\b/i.test(text) ? 'hinglish' : 'en',
    confidence: 0.96,
  }),
  autocomplete: (text: string): AutocompleteResult => ({ suggestion: /\s$/.test(text) && text.trim() ? 'which is essential for life' : '' }),
  summary: (text: string, mode: SummaryMode): SummaryResult => {
    const base = 'Photosynthesis is the process by which green plants convert light energy into chemical energy. Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts. Photosynthesis is important because it forms the base of almost every food chain.'
    const summary = mode === 'bullets'
      ? '• Plants convert light energy into chemical energy\n• Chlorophyll absorbs sunlight in the chloroplasts\n• Carbon dioxide and water produce glucose and oxygen\n• Photosynthesis is the base of food chains'
      : mode === 'short' ? base.split('. ').slice(0, 1).join('. ') + '.' : base
    return { summary, original_words: countWords(text) || 90, summary_words: countWords(summary) }
  },
  translate: (target: Language): TranslateResult => ({
    translation: target === 'hi'
      ? 'प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलते हैं। क्लोरोफिल हरा वर्णक है जो क्लोरोप्लास्ट में सूर्य के प्रकाश को अवशोषित करता है।'
      : 'Photosynthesis is the process by which green plants convert light energy into chemical energy. Chlorophyll is a green pigment that absorbs sunlight.',
    engine: 'demo',
    available: true,
  }),
  questions: (count: number, qtype: QuestionType): QuestionsResult => {
    const bank: Question[] = [
      { type: 'mcq', question: 'Choose the term that best completes the sentence: _____ is the green pigment that absorbs sunlight in the chloroplasts.', options: ['Glucose', 'Chlorophyll', 'Oxygen', 'Carbon dioxide'], answer: 'Chlorophyll', explanation: 'From your notes: “Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts.”' },
      { type: 'mcq', question: 'Choose the term that best completes the sentence: Glucose provides energy for growth, while _____ is released into the atmosphere.', options: ['Oxygen', 'Chlorophyll', 'Glucose', 'Soil'], answer: 'Oxygen', explanation: 'From your notes: “Glucose provides energy for growth, while oxygen is released into the atmosphere.”' },
      { type: 'mcq', question: 'Choose the term that best completes the sentence: Plants use _____ from the air and water from the soil.', options: ['Oxygen', 'Glucose', 'Carbon dioxide', 'Chlorophyll'], answer: 'Carbon dioxide', explanation: 'From your notes: “Plants use carbon dioxide from the air and water from the soil to produce glucose and oxygen.”' },
    ]
    const kind = qtype === 'Fill in the blank' ? 'fill' : qtype === 'Short answer' ? 'short' : 'mcq'
    return {
      questions: Array.from({ length: Math.min(count, 3) }, (_, i) => {
        const q = bank[i]
        if (kind === 'fill') return { ...q, type: 'fill', options: undefined, question: q.question.replace('Choose the term that best completes the sentence:', 'Fill in the blank:') }
        if (kind === 'short') return { type: 'short', question: ['What is chlorophyll?', 'What is released into the atmosphere?', 'What do plants take from the air?'][i], answer: q.explanation.replace('From your notes: ', ''), explanation: q.explanation }
        return q
      }),
    }
  },
  checkAnswer: (expected: string, answer: string): AnswerCheckResult => {
    const words = (s: string) => new Set(s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [])
    const exp = words(expected)
    const got = words(answer)
    const overlap = [...exp].filter((w) => got.has(w)).length / Math.max(exp.size, 1)
    const score = Math.round(100 * Math.min(1, overlap * 1.6))
    return {
      score, verdict: score >= 70 ? 'correct' : score >= 40 ? 'partial' : 'incorrect', similarity: overlap, coverage: overlap,
      matched_terms: [], missing_terms: [], engine: 'demo (word overlap)',
      feedback: 'Demo mode: this is a simple word-overlap check. Start the backend for meaning-based checking.',
    }
  },
  aiCheck: (text: string): AiCheckResult => ({
    available: true, ai_probability: 0.42, label: 'Unclear — could be either', confidence: 'low', words: countWords(text),
    language: 'en', engine: 'demo',
    message: 'Demo mode: these are sample results. Start the backend for a real check.',
    signals: [
      { name: 'log_ppl', label: 'Predictability (GPT-2 perplexity)', display: 'perplexity 24', contribution: 0.6, leans: 'ai', explanation: 'How surprised GPT-2 is by each word.' },
      { name: 'shorthand', label: 'Note-taking shorthand', display: '2.1 per 100 words', contribution: -0.5, leans: 'human', explanation: 'Arrows, &, w/, e.g., etc.' },
    ],
    sentences: [],
  }),
  ask: (question: string): AskResult => ({
    answer: /[ऀ-ॿ]/.test(question)
      ? 'क्लोरोफिल हरा वर्णक है जो क्लोरोप्लास्ट में सूर्य के प्रकाश को अवशोषित करता है।'
      : 'Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts.',
    sources: ['Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts.'],
    confidence: 0.8,
    engine: 'demo',
  }),
  conceptMap: {
    nodes: [
      { id: 'Photosynthesis', importance: 10, sentences: ['Photosynthesis is the process by which green plants convert light energy into chemical energy.'] },
      { id: 'Chlorophyll', importance: 8, sentences: ['Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts.'] },
      { id: 'Glucose', importance: 7, sentences: ['Glucose provides energy for growth, while oxygen is released into the atmosphere.'] },
      { id: 'Oxygen', importance: 7, sentences: ['Glucose provides energy for growth, while oxygen is released into the atmosphere.'] },
      { id: 'Carbon dioxide', importance: 6, sentences: ['Plants use carbon dioxide from the air and water from the soil to produce glucose and oxygen.'] },
      { id: 'Sunlight', importance: 5, sentences: ['Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts.'] },
      { id: 'Plants', importance: 5, sentences: ['Plants use carbon dioxide from the air and water from the soil to produce glucose and oxygen.'] },
      { id: 'Atmosphere', importance: 3, sentences: ['Glucose provides energy for growth, while oxygen is released into the atmosphere.'] },
    ],
    edges: [
      { source: 'Photosynthesis', target: 'Plants', relation: 'is' },
      { source: 'Chlorophyll', target: 'Sunlight', relation: 'absorbs' },
      { source: 'Plants', target: 'Carbon dioxide', relation: 'use' },
      { source: 'Carbon dioxide', target: 'Glucose', relation: 'produce' },
      { source: 'Glucose', target: 'Oxygen', relation: 'related to' },
      { source: 'Oxygen', target: 'Atmosphere', relation: 'is released into' },
      { source: 'Photosynthesis', target: 'Chlorophyll', relation: 'related to' },
    ],
    keywords: [
      { word: 'Photosynthesis', score: 1 }, { word: 'Chlorophyll', score: 0.82 }, { word: 'Oxygen', score: 0.74 },
      { word: 'Glucose', score: 0.7 }, { word: 'Carbon dioxide', score: 0.66 }, { word: 'Sunlight', score: 0.4 },
      { word: 'Plants', score: 0.38 }, { word: 'Atmosphere', score: 0.3 },
    ],
  } as ConceptMapResult,
  glossary: {
    readability: 48,
    grade: 'Grade 11',
    level: 'Difficult',
    terms: [
      { term: 'Chlorophyll', meaning: 'The green pigment that absorbs sunlight in the chloroplasts', hindi: 'हरा वर्णक जो सूर्य के प्रकाश को अवशोषित करता है', difficulty: 'Hard' },
      { term: 'Chloroplasts', meaning: 'Parts of a plant cell where photosynthesis happens', hindi: 'पौधे की कोशिका के भाग जहाँ प्रकाश संश्लेषण होता है', difficulty: 'Hard' },
      { term: 'Photosynthesis', meaning: 'The process by which green plants convert light energy into chemical energy', hindi: 'वह प्रक्रिया जिसमें हरे पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलते हैं', difficulty: 'Medium' },
      { term: 'Glucose', meaning: 'A sugar that plants make for energy', hindi: 'एक शर्करा जो पौधे ऊर्जा के लिए बनाते हैं', difficulty: 'Medium' },
      { term: 'Oxygen', meaning: 'A gas released into the atmosphere', hindi: 'वायुमंडल में छोड़ी जाने वाली गैस', difficulty: 'Easy' },
    ],
    simplified: 'Photosynthesis is how green plants change light energy into chemical energy. Chlorophyll is the green pigment that takes in sunlight. Plants use carbon dioxide and water to make glucose and oxygen.',
  } as GlossaryResult,
  cleanup: {
    devanagari: 'Photosynthesis में plants light energy को chemical energy में convert करते हैं। Chlorophyll एक green pigment है जो sunlight को absorb करता है।',
    english: 'In photosynthesis, plants convert light energy into chemical energy. Chlorophyll is a green pigment that absorbs sunlight.',
    changed: ['में', 'को', 'करते', 'हैं', 'एक', 'है', 'जो', 'करता'],
    engine: 'demo',
  } as CleanupResult,
}
