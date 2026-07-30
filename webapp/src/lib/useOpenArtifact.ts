import { useNavigate } from 'react-router-dom'
import type { OpenIdHandler } from '../components/IdChip'
import { useToast } from '../components/Toast'
import { useConnection } from '../connection/ConnectionContext'
import type { ProjectState } from '../connection/ConnectionContext'
import type { ArtifactKind } from './citations'
import { resolveKind } from './resolveArtifact'

/**
 * Open any KB id from a view that has no drawer of its own.
 *
 * `/browse/<kind>/<id>` already deep-links to an open drawer, so ids in the
 * claims list, the review queue and proposal payloads route there rather than
 * each view growing its own drawer. Returns undefined when there's no project
 * to read from — `IdChip` then renders a static chip instead of a link.
 */
export function useOpenArtifact(project: ProjectState | null): OpenIdHandler | undefined {
  const navigate = useNavigate()
  const { aggregated } = useConnection()
  const { toast } = useToast()
  if (!project) return undefined

  const go = (kind: ArtifactKind, id: string) => {
    const p = aggregated ? `?p=${encodeURIComponent(project.conn.endpoint)}` : ''
    navigate(`/browse/${kind}/${encodeURIComponent(id)}${p}`)
  }

  return (id, kind) => {
    if (kind) {
      go(kind, id)
      return
    }
    void resolveKind(project, id).then((resolved) => {
      if (resolved) go(resolved, id)
      else toast('error', `${id.slice(0, 24)} is not a readable artifact (a proposal, or deleted)`)
    })
  }
}

/**
 * Proposal-payload keys whose values are ids of readable artifacts.
 *
 * Deliberately excludes `session_id` and `proposal_ids` — neither resolves to
 * an artifact, so linking them would only ever produce a "not readable" toast.
 */
const ID_KEYS = new Set([
  'claim_id',
  'claims',
  'contradicts',
  'entity_id',
  'evidence',
  'evidence_id',
  'object',
  'page_id',
  'relation_id',
  'source',
  'source_id',
  'sources',
  'subject',
  'supersedes',
  'target',
])

/** The ids held under a payload key, or null when the key holds no ids. */
export function payloadIds(key: string, value: unknown): string[] | null {
  if (!ID_KEYS.has(key)) return null
  if (typeof value === 'string') return value === '' ? null : [value]
  if (Array.isArray(value) && value.every((v) => typeof v === 'string')) {
    return value.length > 0 ? (value as string[]) : null
  }
  return null
}
