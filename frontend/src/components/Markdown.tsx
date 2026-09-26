import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'
import { CITE_PREFIX, linkMarkers, normalizeMath } from '../markers'

interface Props {
  text: string
  knownIds: Set<string>
  onCite: (id: string) => void
}

export function Markdown({ text, knownIds, onCite }: Props) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{
          a({ href, children }) {
            if (href?.startsWith(CITE_PREFIX)) {
              const id = href.slice(CITE_PREFIX.length)
              const known = knownIds.has(id)
              return (
                <button
                  type="button"
                  className={`cite-chip${id.startsWith('F') ? ' is-figure' : ''}${known ? '' : ' is-unknown'}`}
                  onClick={() => known && onCite(id)}
                  title={known ? `Show source ${id}` : `${id} is not one of the retrieved sources`}
                >
                  {id}
                </button>
              )
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            )
          },
        }}
      >
        {linkMarkers(normalizeMath(text))}
      </ReactMarkdown>
    </div>
  )
}
