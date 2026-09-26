// Mirrors backend app/answer/base.py: single "[S1]" and grouped "[S2, F1]" / "[S3; S6]" markers.
const GROUP_RE = /\[([SF]\d+(?:\s*[,;]\s*[SF]\d+)*)\]/g
const ID_RE = /[SF]\d+/g

export const CITE_PREFIX = '#cite-'

/** Rewrite markers as markdown links so the renderer can turn them into clickable chips. */
export function linkMarkers(text: string): string {
  return text.replace(GROUP_RE, (_, group: string) =>
    (group.match(ID_RE) ?? []).map((id) => `[${id}](${CITE_PREFIX}${id})`).join(''),
  )
}

/**
 * Models switch between $...$ and \(...\) / \[...\] from one answer to the next. remark-math only
 * understands dollars, and markdown would otherwise eat the backslashes, so convert up front.
 */
export function normalizeMath(text: string): string {
  return text
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, body: string) => `\n$$\n${body.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, body: string) => `$${body.trim()}$`)
}
