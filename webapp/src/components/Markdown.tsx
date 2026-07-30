import ReactMarkdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { parseVouchHref, remarkVouchIds, VOUCH_SCHEME } from '../lib/vouchIds'
import { IdChip } from './IdChip'
import type { OpenIdHandler } from './IdChip'

// Renders KB markdown (page bodies, session summaries) with the console
// theme. react-markdown emits no raw HTML, so untrusted proposal payloads
// stay inert. Bracketed vouch ids in the prose are rewritten to `vouch:`
// links by remarkVouchIds and rendered here as drawer chips — pass
// `onOpenId` to make them clickable.
export function Markdown({
  children,
  onOpenId,
}: {
  children: string
  onOpenId?: OpenIdHandler
}) {
  return (
    <div className="markdown-body min-w-0 text-sm leading-relaxed text-ink-2">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkVouchIds]}
        // `vouch:` hrefs are minted by remarkVouchIds, never by the author's
        // markdown, and are consumed by the anchor override below rather than
        // reaching the DOM. Everything else keeps the default sanitiser.
        urlTransform={(url) => (url.startsWith(VOUCH_SCHEME) ? url : defaultUrlTransform(url))}
        components={{
          a({ node: _node, href, children: label, ...rest }) {
            const ref = href ? parseVouchHref(href) : null
            if (!ref) {
              return (
                <a href={href} {...rest}>
                  {label}
                </a>
              )
            }
            return (
              <IdChip id={ref.id} kind={ref.kind} onOpen={onOpenId}>
                {label}
              </IdChip>
            )
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
