import type { Source } from '../api'
import { MathText } from './MathText'

interface Props {
  sources: Source[]
  activeId: string | null
  citedIds: Set<string> | null // null = no answer yet (search only), so nothing is "not cited"
  onOpenImage: (src: string, caption: string) => void
}

interface CardProps {
  source: Source
  active: boolean
  cited: boolean
  onOpenImage: Props['onOpenImage']
}

function SourceCard({ source, active, cited, onOpenImage }: CardProps) {
  return (
    <li id={`source-${source.id}`} className={`card source${active ? ' is-active' : ''}${cited ? '' : ' is-uncited'}`}>
      <div className="source-head">
        <span className={`source-id${source.kind === 'figure' ? ' is-figure' : ''}`}>{source.id}</span>
        <div className="source-title">
          <a href={source.arxiv_url} target="_blank" rel="noreferrer">
            <MathText text={source.paper_title} />
          </a>
          <span className="meta">
            arXiv:{source.arxiv_id} · {source.section}
            {!cited && ' · not cited'}
          </span>
        </div>
        <span className="score" title="Cosine similarity to the question">
          {source.score.toFixed(3)}
        </span>
      </div>
      {source.images.length > 0 && (
        <div className="thumbs">
          {source.images.map((src) => (
            <button key={src} type="button" className="thumb" onClick={() => onOpenImage(src, source.text)}>
              <img src={src} alt={`Figure from ${source.paper_title}`} loading="lazy" />
            </button>
          ))}
        </div>
      )}
      {source.kind === 'figure' && source.images.length === 0 && (
        <p className="muted small">No image: this figure is drawn in LaTeX, so only its caption is available.</p>
      )}
      <details>
        <summary>{source.kind === 'figure' ? 'Caption and citing text' : 'Passage'}</summary>
        <p className="source-text">{source.text}</p>
      </details>
    </li>
  )
}

export function SourceList({ sources, activeId, citedIds, onOpenImage }: Props) {
  const groups = [
    { title: 'Figures', items: sources.filter((s) => s.kind === 'figure') },
    { title: 'Passages', items: sources.filter((s) => s.kind === 'text') },
  ]
  return (
    <div className="sources">
      {groups.map(
        (group) =>
          group.items.length > 0 && (
            <section key={group.title}>
              <h3 className="section-label">{group.title}</h3>
              <ul className="source-list">
                {group.items.map((s) => (
                  <SourceCard
                    key={s.id}
                    source={s}
                    active={s.id === activeId}
                    cited={citedIds === null || citedIds.has(s.id)}
                    onOpenImage={onOpenImage}
                  />
                ))}
              </ul>
            </section>
          ),
      )}
      {sources.length === 0 && <p className="muted">Nothing retrieved. Is the index built?</p>}
    </div>
  )
}
