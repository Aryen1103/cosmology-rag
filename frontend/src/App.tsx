import { useCallback, useEffect, useRef, useState } from 'react'
import {
  type AgentStep,
  type Answer,
  type ProviderChoice,
  type ProviderName,
  type Source,
  type Stats,
  PROVIDER_LABELS,
  api,
  isError,
} from './api'
import { AgentSteps } from './components/AgentSteps'
import { AnswerCard } from './components/AnswerCard'
import { PapersView } from './components/PapersView'
import { SourceList } from './components/SourceList'

const EXAMPLES = [
  'What value of H0 is obtained using only the tip of the red giant branch?',
  'What limits does LUX-ZEPLIN place on primordial black hole evaporation?',
  'Which figure shows how a peak in the correlation function evolves?',
]

interface Result {
  question: string
  sources: Source[]
  answers: Answer[] | null // null = retrieval only
  steps?: AgentStep[] // agent mode only
}

type Tab = 'ask' | 'papers'

function useElapsed(running: boolean): number {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!running) return
    const started = performance.now()
    const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 100)
    return () => {
      clearInterval(timer)
      setElapsed(0)
    }
  }, [running])
  return elapsed
}

export default function App() {
  const [tab, setTab] = useState<Tab>('ask')
  const [stats, setStats] = useState<Stats | null>(null)
  const [statsError, setStatsError] = useState<string | null>(null)

  const [question, setQuestion] = useState('')
  const [provider, setProvider] = useState<ProviderChoice>('deepseek')
  const [kText, setKText] = useState(6)
  const [kFigures, setKFigures] = useState(3)

  const [loading, setLoading] = useState<'ask' | 'search' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [activeSource, setActiveSource] = useState<string | null>(null)
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const elapsed = useElapsed(loading !== null)

  useEffect(() => {
    api.stats().then(
      (s) => {
        setStats(s)
        const available = (Object.keys(s.providers) as ProviderName[]).filter((p) => s.providers[p])
        if (available.length === 1) setProvider(available[0])
        else if (available.length > 1) setProvider('claude')
      },
      (e: Error) => setStatsError(e.message),
    )
  }, [])

  useEffect(() => {
    if (!lightbox) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setLightbox(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightbox])

  const providerAvailable = (choice: ProviderChoice): boolean => {
    if (!stats) return true
    if (choice === 'both') return stats.providers.claude && stats.providers.deepseek
    return stats.providers[choice === 'agent' ? 'claude' : choice]
  }

  const run = async (mode: 'ask' | 'search', text = question) => {
    const q = text.trim()
    if (!q) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(mode)
    setError(null)
    setActiveSource(null)
    const options = { k_text: kText, k_figures: kFigures }
    try {
      if (mode === 'ask' && provider === 'agent') {
        const response = await api.agent(q, controller.signal)
        setResult({ question: q, sources: response.sources, answers: response.answers, steps: response.steps })
      } else if (mode === 'ask') {
        const response = await api.ask(q, provider, options, controller.signal)
        setResult({ question: q, sources: response.sources, answers: response.answers })
      } else {
        const response = await api.search(q, options, controller.signal)
        setResult({ question: q, sources: response.sources, answers: null })
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      if (abortRef.current === controller) setLoading(null)
    }
  }

  const cancel = () => {
    abortRef.current?.abort()
    setLoading(null)
  }

  const showSource = useCallback((id: string) => {
    setActiveSource(id)
    document.getElementById(`source-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [])

  const citedIds = result?.answers
    ? new Set(result.answers.flatMap((a) => (isError(a) ? [] : a.citations.map((c) => c.source_id))))
    : null

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Cosmology RAG</h1>
          <span className="meta">
            {stats
              ? `${stats.papers} papers · ${stats.text_chunks.toLocaleString()} passages · ${stats.figures} figures indexed`
              : statsError
                ? 'Backend unreachable'
                : 'Connecting…'}
          </span>
        </div>
        <nav className="tabs" aria-label="Views">
          {(['ask', 'papers'] as Tab[]).map((t) => (
            <button key={t} type="button" className={tab === t ? 'is-active' : ''} onClick={() => setTab(t)}>
              {t === 'ask' ? 'Ask' : 'Papers'}
            </button>
          ))}
        </nav>
      </header>

      {statsError && <p className="banner error-text">{statsError}</p>}

      {tab === 'papers' ? (
        <main className="page">
          <PapersView indexedPapers={stats?.papers ?? null} />
        </main>
      ) : (
        <main className="page">
          <form
            className="card ask-form"
            onSubmit={(e) => {
              e.preventDefault()
              run('ask')
            }}
          >
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault()
                  run('ask')
                }
              }}
              placeholder="Ask about recent astro-ph.CO papers: results, methods, or what a figure shows…"
              rows={3}
              maxLength={2000}
              aria-label="Question"
            />
            <div className="form-row">
              <div className="segmented" role="radiogroup" aria-label="Answer provider">
                {(['claude', 'deepseek', 'both', 'agent'] as ProviderChoice[]).map((p) => {
                  const available = providerAvailable(p)
                  return (
                    <button
                      key={p}
                      type="button"
                      role="radio"
                      aria-checked={provider === p}
                      className={provider === p ? 'is-active' : ''}
                      disabled={!available}
                      title={
                        !available
                          ? 'API key not set in backend/.env'
                          : p === 'agent'
                            ? 'Claude runs its own searches, views figures and computes distances'
                            : undefined
                      }
                      onClick={() => setProvider(p)}
                    >
                      {p === 'both' ? 'Both' : p === 'agent' ? 'Agent' : PROVIDER_LABELS[p]}
                    </button>
                  )
                })}
              </div>
              <label className="k-input" title={provider === 'agent' ? 'The agent picks its own search sizes' : undefined}>
                Passages
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={kText}
                  disabled={provider === 'agent'}
                  onChange={(e) => setKText(Number(e.target.value))}
                />
              </label>
              <label className="k-input" title={provider === 'agent' ? 'The agent picks its own search sizes' : undefined}>
                Figures
                <input
                  type="number"
                  min={0}
                  max={6}
                  value={kFigures}
                  disabled={provider === 'agent'}
                  onChange={(e) => setKFigures(Number(e.target.value))}
                />
              </label>
              <span className="spacer" />
              {loading ? (
                <button type="button" className="btn" onClick={cancel}>
                  Cancel
                </button>
              ) : (
                <button
                  type="button"
                  className="btn"
                  disabled={!question.trim()}
                  onClick={() => run('search')}
                  title="Show what retrieval finds, without calling a model"
                >
                  Search only
                </button>
              )}
              <button
                type="submit"
                className="btn primary"
                disabled={!question.trim() || loading !== null || !providerAvailable(provider)}
              >
                {loading === 'ask' ? `Answering… ${elapsed.toFixed(0)} s` : 'Ask'}
              </button>
            </div>
          </form>

          {error && <p className="banner error-text">{error}</p>}

          {!result && !loading && (
            <section className="examples">
              <h2 className="section-label">Try</h2>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  className="example"
                  onClick={() => {
                    setQuestion(ex)
                    run('ask', ex)
                  }}
                  disabled={!providerAvailable(provider)}
                >
                  {ex}
                </button>
              ))}
            </section>
          )}

          {loading && !result && (
            <p className="muted loading">
              {loading === 'search'
                ? 'Searching…'
                : provider === 'agent'
                  ? 'The agent is searching the papers…'
                  : 'Retrieving sources and asking the model…'}
            </p>
          )}

          {result && (
            <div className={`results${loading ? ' is-stale' : ''}`}>
              <div className="answers-col">
                <h2 className="question">{result.question}</h2>
                {result.answers === null ? (
                  <p className="muted">Retrieval only: these are the sources a model would receive.</p>
                ) : (
                  <div className={`answers${result.answers.length > 1 ? ' is-pair' : ''}`}>
                    {result.answers.map((a) => (
                      <AnswerCard key={a.provider} answer={a} sources={result.sources} onCite={showSource} />
                    ))}
                  </div>
                )}
                {result.steps && <AgentSteps steps={result.steps} />}
              </div>
              <aside className="sources-col" aria-label="Retrieved sources">
                <SourceList
                  sources={result.sources}
                  activeId={activeSource}
                  citedIds={citedIds}
                  onOpenImage={(src, caption) => setLightbox({ src, caption })}
                />
              </aside>
            </div>
          )}
        </main>
      )}

      {lightbox && (
        <div className="lightbox" role="dialog" aria-modal="true" onClick={() => setLightbox(null)}>
          <figure onClick={(e) => e.stopPropagation()}>
            <img src={lightbox.src} alt="" />
            <figcaption>{lightbox.caption}</figcaption>
          </figure>
          <button type="button" className="lightbox-close" onClick={() => setLightbox(null)} aria-label="Close">
            ×
          </button>
        </div>
      )}
    </div>
  )
}
