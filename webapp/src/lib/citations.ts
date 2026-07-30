/** Every artifact kind a citation can name. Mirrors the drawer's OpenKind. */
export type ArtifactKind = 'claim' | 'page' | 'entity' | 'relation' | 'evidence' | 'source'

export type AnswerSegment =
  | { kind: 'text'; text: string }
  | { kind: 'citation'; claimId: string; refKind?: ArtifactKind }

/**
 * A vouch id in brackets, with an optional `kind:` prefix.
 *
 * Ids are lowercase alphanumerics and dashes, 3+ chars. The final character
 * may be a dash: `_slugify` truncates to 60 chars *after* stripping them, so
 * ids like `…-where-the-roadmap-already-` are real (~8% of claims here) and
 * must link like any other. Requiring 3+ chars keeps markdown task boxes
 * (`[x]`) and footnote refs (`[1]`) as prose.
 *
 * The `[claim: id]` form is vouch's canonical inline marker in page bodies
 * (see compile.py `_CLAIM_MARKER_RE`); the bare form is what synthesize emits.
 *
 * The bracket guards keep `[[wikilinks]]` whole: those name a page *title*,
 * not an id, and resolving them needs a title lookup this doesn't do.
 */
export const CITATION_RE =
  /(?<!\[)\[(?:(claim|page|entity|relation|evidence|source):\s*)?([a-z0-9][a-z0-9-]{2,})\](?!\])/g

export function parseAnswer(answer: string): AnswerSegment[] {
  const segments: AnswerSegment[] = []
  let last = 0
  for (const m of answer.matchAll(CITATION_RE)) {
    const at = m.index ?? 0
    if (at > last) segments.push({ kind: 'text', text: answer.slice(last, at) })
    segments.push(
      m[1]
        ? { kind: 'citation', claimId: m[2], refKind: m[1] as ArtifactKind }
        : { kind: 'citation', claimId: m[2] },
    )
    last = at + m[0].length
  }
  if (last < answer.length) segments.push({ kind: 'text', text: answer.slice(last) })
  return segments
}

export type SnippetSegment = { kind: 'plain' | 'match'; text: string }

const HIGHLIGHT = /«([^»]*)»/g

export function parseSnippet(snippet: string): SnippetSegment[] {
  const segments: SnippetSegment[] = []
  let last = 0
  for (const m of snippet.matchAll(HIGHLIGHT)) {
    const at = m.index ?? 0
    if (at > last) segments.push({ kind: 'plain', text: snippet.slice(last, at) })
    segments.push({ kind: 'match', text: m[1] })
    last = at + m[0].length
  }
  if (last < snippet.length) segments.push({ kind: 'plain', text: snippet.slice(last) })
  return segments
}
