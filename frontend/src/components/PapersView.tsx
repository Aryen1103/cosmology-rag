import { useEffect, useMemo, useState } from 'react'
import { type Paper, api } from '../api'
import { MathText } from './MathText'

export function PapersView({ indexedPapers }: { indexedPapers: number | null }) {
  const [papers, setPapers] = useState<Paper[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    api.papers().then(setPapers, (e: Error) => setError(e.message))
  }, [])

  const shown = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!papers || !q) return papers ?? []
    return papers.filter((p) =>
      [p.title, p.arxiv_id, p.abstract, ...p.authors].some((field) => field.toLowerCase().includes(q)),
    )
  }, [papers, filter])

  if (error) return <p className="error-text">{error}</p>
  if (!papers) return <p className="muted">Loading papers…</p>

  return (
    <div className="papers">
      <div className="papers-bar">
        <input
          type="search"
          placeholder="Filter by title, author, arXiv ID or abstract"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <span className="meta">
          {shown.length} of {papers.length} ingested
        </span>
      </div>
      {indexedPapers !== null && indexedPapers < papers.length && (
        <p className="warning">
          Only {indexedPapers} of {papers.length} ingested papers are in the search index. Run{' '}
          <code>python scripts/build_index.py</code> in backend/, then restart the server.
        </p>
      )}
      <ul className="paper-list">
        {shown.map((p) => (
          <li key={p.arxiv_id} className="card paper">
            <a href={`https://arxiv.org/abs/${p.arxiv_id}`} target="_blank" rel="noreferrer" className="paper-title">
              <MathText text={p.title} />
            </a>
            <span className="meta">
              {p.authors.slice(0, 4).join(', ')}
              {p.authors.length > 4 && ` +${p.authors.length - 4} more`} · {p.published.slice(0, 10)} · arXiv:
              {p.arxiv_id} · {p.figures} figure{p.figures === 1 ? '' : 's'}
              {p.source_type !== 'latex' && ' · PDF text only'}
            </span>
            <details>
              <summary>Abstract</summary>
              <p>{p.abstract}</p>
            </details>
          </li>
        ))}
      </ul>
    </div>
  )
}
