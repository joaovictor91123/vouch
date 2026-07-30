import { CITATION_RE } from './citations'
import type { ArtifactKind } from './citations'

/**
 * Href scheme the Markdown renderer recognises as an in-KB artifact link.
 * Not a real URL — it never reaches the network. The remark plugin below
 * mints these, `parseVouchHref` reads them back, and `Markdown`'s anchor
 * override turns them into drawer buttons.
 */
export const VOUCH_SCHEME = 'vouch:'

export function vouchHref(id: string, kind?: ArtifactKind): string {
  return `${VOUCH_SCHEME}${kind ?? ''}/${encodeURIComponent(id)}`
}

export function parseVouchHref(href: string): { kind?: ArtifactKind; id: string } | null {
  if (!href.startsWith(VOUCH_SCHEME)) return null
  const rest = href.slice(VOUCH_SCHEME.length)
  const slash = rest.indexOf('/')
  if (slash < 0) return null
  const id = decodeURIComponent(rest.slice(slash + 1))
  if (id === '') return null
  const kind = rest.slice(0, slash)
  return kind === '' ? { id } : { kind: kind as ArtifactKind, id }
}

/** Minimal mdast shape — enough to walk and rewrite phrasing content. */
type MdNode = { type: string; value?: string; url?: string; children?: MdNode[] }

// Nodes whose text is literal rather than prose. An id inside a code span is
// being shown, not referenced, and rewriting inside an existing link would
// nest anchors.
const OPAQUE = new Set(['code', 'inlineCode', 'link', 'linkReference', 'definition', 'html'])

/**
 * remark plugin: rewrite bracketed vouch ids in prose into `vouch:` links.
 *
 * This is what makes ids inside page bodies, session summaries and Claude
 * answers clickable — those render through `Markdown`, which previously
 * emitted every `[claim: id]` marker as dead text.
 */
export function remarkVouchIds() {
  return (tree: MdNode): void => {
    rewrite(tree)
  }
}

function rewrite(node: MdNode): void {
  if (!node.children) return
  const out: MdNode[] = []
  for (const child of node.children) {
    if (OPAQUE.has(child.type)) {
      out.push(child)
    } else if (child.type === 'text' && typeof child.value === 'string') {
      out.push(...splitIds(child.value))
    } else {
      rewrite(child)
      out.push(child)
    }
  }
  node.children = out
}

function splitIds(value: string): MdNode[] {
  const out: MdNode[] = []
  let last = 0
  for (const m of value.matchAll(CITATION_RE)) {
    const at = m.index ?? 0
    if (at > last) out.push({ type: 'text', value: value.slice(last, at) })
    const id = m[2]
    out.push({
      type: 'link',
      url: vouchHref(id, m[1] as ArtifactKind | undefined),
      children: [{ type: 'text', value: id }],
    })
    last = at + m[0].length
  }
  if (out.length === 0) return [{ type: 'text', value }]
  if (last < value.length) out.push({ type: 'text', value: value.slice(last) })
  return out
}
