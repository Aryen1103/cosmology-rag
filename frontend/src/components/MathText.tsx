import katex from 'katex'
import { useMemo } from 'react'

/** Plain text with inline $...$ math, for paper titles like "Two-rung ladder: $H_0$ from ...". */
export function MathText({ text }: { text: string }) {
  const parts = useMemo(
    () =>
      text.split(/(\$[^$]+\$)/g).map((part) =>
        part.length > 2 && part.startsWith('$') && part.endsWith('$')
          ? { math: katex.renderToString(part.slice(1, -1), { throwOnError: false, strict: false }) }
          : { text: part },
      ),
    [text],
  )
  return (
    <>
      {parts.map((p, i) =>
        'math' in p ? <span key={i} dangerouslySetInnerHTML={{ __html: p.math! }} /> : <span key={i}>{p.text}</span>,
      )}
    </>
  )
}
