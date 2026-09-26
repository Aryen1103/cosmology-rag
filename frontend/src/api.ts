export type ProviderName = 'claude' | 'deepseek'
export type ProviderChoice = ProviderName | 'both'

export interface Stats {
  papers: number
  text_chunks: number
  figures: number
  providers: Record<ProviderName, boolean>
}

export interface Source {
  id: string // "S1" / "F1"
  kind: 'text' | 'figure'
  arxiv_id: string
  arxiv_url: string
  paper_title: string
  section: string
  text: string
  score: number
  images: string[]
}

export interface Citation {
  source_id: string
  cited_text: string | null
}

export interface AnswerOk {
  provider: ProviderName
  model: string
  text: string
  citations: Citation[]
  unknown_markers: string[]
  input_tokens: number
  output_tokens: number
  latency_s: number
  stop_reason: string | null
}

export interface AnswerError {
  provider: ProviderName
  error: string
}

export type Answer = AnswerOk | AnswerError

export interface AskResponse {
  question: string
  sources: Source[]
  answers: Answer[]
}

export interface Paper {
  arxiv_id: string
  title: string
  authors: string[]
  published: string
  abstract: string
  figures: number
  source_type: string
}

export interface RetrievalOptions {
  k_text: number
  k_figures: number
}

async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const data = await response.json()
      if (typeof data.detail === 'string') detail = data.detail
      else if (Array.isArray(data.detail)) detail = data.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch {
      // non-JSON error body (e.g. the proxy can't reach the backend)
    }
    if (response.status === 502 || response.status === 504) {
      detail = 'Cannot reach the backend. Is it running? (uvicorn app.main:app --port 8000 in backend/)'
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  stats: () => request<Stats>('/api/stats'),
  papers: () => request<Paper[]>('/api/papers'),
  search: (question: string, options: RetrievalOptions, signal?: AbortSignal) =>
    request<{ sources: Source[] }>('/api/search', { question, ...options }, signal),
  ask: (question: string, provider: ProviderChoice, options: RetrievalOptions, signal?: AbortSignal) =>
    request<AskResponse>('/api/ask', { question, provider, ...options }, signal),
}

export const PROVIDER_LABELS: Record<ProviderName, string> = { claude: 'Claude', deepseek: 'DeepSeek' }

export function isError(answer: Answer): answer is AnswerError {
  return 'error' in answer
}
