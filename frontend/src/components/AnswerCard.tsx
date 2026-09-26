import { type Answer, type Source, PROVIDER_LABELS, isError } from '../api'
import { Markdown } from './Markdown'

interface Props {
  answer: Answer
  sources: Source[]
  onCite: (id: string) => void
}

export function AnswerCard({ answer, sources, onCite }: Props) {
  const label = PROVIDER_LABELS[answer.provider]
  if (isError(answer)) {
    return (
      <article className="card answer">
        <header className="answer-head">
          <h3>{label}</h3>
        </header>
        <p className="error-text">{answer.error}</p>
      </article>
    )
  }

  const knownIds = new Set(sources.map((s) => s.id))
  const quotes = answer.citations.filter((c) => c.cited_text)

  return (
    <article className="card answer">
      <header className="answer-head">
        <h3>{label}</h3>
        <span className="meta">
          {answer.model} · {answer.latency_s.toFixed(1)} s · {answer.input_tokens.toLocaleString()} in /{' '}
          {answer.output_tokens.toLocaleString()} out
        </span>
      </header>

      {answer.stop_reason === 'refusal' ? (
        <p className="muted">{answer.text}</p>
      ) : (
        <Markdown text={answer.text} knownIds={knownIds} onCite={onCite} />
      )}

      {answer.unknown_markers.length > 0 && (
        <p className="warning">
          Cites sources that were not retrieved: {answer.unknown_markers.join(', ')}. Treat those claims with care.
        </p>
      )}
      {answer.stop_reason === 'max_tokens' && <p className="warning">The answer was cut off at the token limit.</p>}

      {quotes.length > 0 && (
        <details className="quotes">
          <summary>Quoted evidence ({quotes.length})</summary>
          <ul>
            {quotes.map((c, i) => (
              <li key={i}>
                <button type="button" className="cite-chip" onClick={() => onCite(c.source_id)}>
                  {c.source_id}
                </button>
                <q>{c.cited_text}</q>
              </li>
            ))}
          </ul>
        </details>
      )}
    </article>
  )
}
