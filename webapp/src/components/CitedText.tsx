import { parseAnswer } from '../lib/citations'
import type { ArtifactKind } from '../lib/citations'
import { IdChip } from './IdChip'
import type { OpenIdHandler } from './IdChip'

/**
 * Plain prose — not markdown — with bracketed vouch ids rendered as chips.
 *
 * Synthesize answers and claim text both carry inline citations but are shown
 * as prose, so they can't go through `Markdown`. `kindFor` lets a caller that
 * already knows an id's kind (the cited-claims/pages lists on a synthesize
 * result) skip the resolve probe.
 */
export function CitedText({
  text,
  onOpenId,
  kindFor,
}: {
  text: string
  onOpenId?: OpenIdHandler
  kindFor?: (id: string) => ArtifactKind | undefined
}) {
  return (
    <>
      {parseAnswer(text).map((seg, i) =>
        seg.kind === 'text' ? (
          <span key={i}>{seg.text}</span>
        ) : (
          <IdChip
            key={i}
            id={seg.claimId}
            kind={seg.refKind ?? kindFor?.(seg.claimId)}
            onOpen={onOpenId}
          />
        ),
      )}
    </>
  )
}
