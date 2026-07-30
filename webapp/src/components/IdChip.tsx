import type { ReactNode } from 'react'
import type { ArtifactKind } from '../lib/citations'

/** Opens a KB artifact. `kind` is absent when the id arrived unqualified. */
export type OpenIdHandler = (id: string, kind?: ArtifactKind) => void

const BASE =
  'mx-0.5 inline-block max-w-72 truncate rounded border border-accent/40 bg-accent/10 px-1.5 align-middle font-mono text-[11px] leading-5 text-accent-2 transition'

/**
 * A vouch id rendered inline — a button when something can open it, a static
 * chip otherwise. Every id in the console renders through this so a citation
 * looks the same whether it came from a synthesize answer, a page body or a
 * proposal payload.
 */
export function IdChip({
  id,
  kind,
  onOpen,
  children,
}: {
  id: string
  kind?: ArtifactKind
  onOpen?: OpenIdHandler
  children?: ReactNode
}) {
  const label = children ?? id
  const title = kind ? `${kind} ${id}` : id
  if (!onOpen) {
    return (
      <span title={title} className={BASE}>
        {label}
      </span>
    )
  }
  return (
    <button
      type="button"
      title={title}
      onClick={() => onOpen(id, kind)}
      className={`${BASE} hover:bg-accent/20`}
    >
      {label}
    </button>
  )
}
