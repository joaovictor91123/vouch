import type { ProjectState } from '../connection/ConnectionContext'
import type { ArtifactKind } from './citations'
import { rpc } from './rpc'

/** The by-id read behind each artifact kind. */
export const READ_METHOD: Record<ArtifactKind, { method: string; param: string }> = {
  claim: { method: 'kb.read_claim', param: 'claim_id' },
  page: { method: 'kb.read_page', param: 'page_id' },
  entity: { method: 'kb.read_entity', param: 'entity_id' },
  relation: { method: 'kb.read_relation', param: 'relation_id' },
  evidence: { method: 'kb.read_evidence', param: 'evidence_id' },
  source: { method: 'kb.read_source', param: 'source_id' },
}

/** Does the endpoint advertise the read this kind needs? */
export function canRead(project: ProjectState, kind: string): kind is ArtifactKind {
  return (
    kind in READ_METHOD &&
    (project.caps?.methods.includes(READ_METHOD[kind as ArtifactKind].method) ?? false)
  )
}

const RESOLVE_ORDER: ArtifactKind[] = ['claim', 'page', 'entity', 'evidence', 'source', 'relation']

// Ids carry their kind in their shape often enough to be worth a hint —
// evidence is `ev-<hex>`, a source id is the content hash itself. Probing in
// the right order usually costs one read instead of six.
function probeOrder(id: string): ArtifactKind[] {
  const first: ArtifactKind | null = /^ev-[0-9a-f]+$/.test(id)
    ? 'evidence'
    : /^[0-9a-f]{64}$/.test(id)
      ? 'source'
      : null
  return first ? [first, ...RESOLVE_ORDER.filter((k) => k !== first)] : RESOLVE_ORDER
}

/**
 * Find which kind an unqualified id names, by reading it.
 *
 * Citations in prose, audit `object_ids` and relation endpoints all arrive
 * without a kind. Returns null when nothing resolves — a proposal id, or a
 * deleted artifact.
 */
export async function resolveKind(
  project: ProjectState,
  id: string,
): Promise<ArtifactKind | null> {
  for (const kind of probeOrder(id)) {
    if (!canRead(project, kind)) continue
    try {
      await rpc(project.conn, READ_METHOD[kind].method, { [READ_METHOD[kind].param]: id })
      return kind
    } catch {
      // not this kind — try the next
    }
  }
  return null
}
