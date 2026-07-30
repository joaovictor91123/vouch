import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, test, vi } from 'vitest'
import { Markdown } from './Markdown'
import { parseVouchHref, vouchHref } from '../lib/vouchIds'

test('a [claim: id] marker in a page body opens the artifact', async () => {
  const onOpenId = vi.fn()
  render(
    <Markdown onOpenId={onOpenId}>
      {'The gate is mandatory [claim: review-gate-is-the-ingest-review].'}
    </Markdown>,
  )
  await userEvent.click(screen.getByRole('button', { name: 'review-gate-is-the-ingest-review' }))
  expect(onOpenId).toHaveBeenCalledWith('review-gate-is-the-ingest-review', 'claim')
})

test('a bare id ending in a dash still opens', async () => {
  const onOpenId = vi.fn()
  const id = '00-point-in-time-and-zero-exactly-where-the-roadmap-already-'
  render(<Markdown onOpenId={onOpenId}>{`grounded [${id}] here`}</Markdown>)
  await userEvent.click(screen.getByRole('button', { name: id }))
  expect(onOpenId).toHaveBeenCalledWith(id, undefined)
})

test('ids render as static chips when nothing can open them', () => {
  render(<Markdown>{'see [claim: some-claim-id] ok'}</Markdown>)
  expect(screen.queryByRole('button')).toBeNull()
  expect(screen.getByText('some-claim-id')).toBeInTheDocument()
})

test('ids inside code spans stay literal', () => {
  render(<Markdown onOpenId={vi.fn()}>{'run `[claim: not-a-link]` verbatim'}</Markdown>)
  expect(screen.queryByRole('button')).toBeNull()
})

test('ordinary markdown links are untouched', () => {
  render(<Markdown onOpenId={vi.fn()}>{'[docs](https://example.com)'}</Markdown>)
  expect(screen.getByRole('link', { name: 'docs' })).toHaveAttribute('href', 'https://example.com')
})

test('markdown task boxes are not mistaken for ids', () => {
  render(<Markdown onOpenId={vi.fn()}>{'- [x] shipped\n- [ ] pending'}</Markdown>)
  expect(screen.queryByRole('button')).toBeNull()
})

test('vouch hrefs round-trip, including ids with a trailing dash', () => {
  const id = 'ends-with-a-dash-'
  expect(parseVouchHref(vouchHref(id, 'page'))).toEqual({ kind: 'page', id })
  expect(parseVouchHref(vouchHref(id))).toEqual({ id })
  expect(parseVouchHref('https://example.com')).toBeNull()
})
