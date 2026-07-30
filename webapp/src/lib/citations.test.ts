import { expect, test } from 'vitest'
import { parseAnswer, parseSnippet } from './citations'

test('parses a real synthesize answer into text + citation segments', () => {
  const answer =
    'The vouch HTTP server binds 127.0.0.1:8731 by default [the-vouch-http-server-binds-127-0-0-1-8731-by-default]. Vouch stores reviewed knowledge [vouch-starter-reviewed-knowledge].'
  const segs = parseAnswer(answer)
  expect(segs).toEqual([
    { kind: 'text', text: 'The vouch HTTP server binds 127.0.0.1:8731 by default ' },
    { kind: 'citation', claimId: 'the-vouch-http-server-binds-127-0-0-1-8731-by-default' },
    { kind: 'text', text: '. Vouch stores reviewed knowledge ' },
    { kind: 'citation', claimId: 'vouch-starter-reviewed-knowledge' },
    { kind: 'text', text: '.' },
  ])
})

test('answer with no citations is a single text segment', () => {
  expect(parseAnswer('nothing cited here')).toEqual([{ kind: 'text', text: 'nothing cited here' }])
})

test('adjacent citations produce no empty text segments', () => {
  const segs = parseAnswer('[claim-a][claim-b]')
  expect(segs).toEqual([
    { kind: 'citation', claimId: 'claim-a' },
    { kind: 'citation', claimId: 'claim-b' },
  ])
})

test('bracketed text that is not a slug stays text', () => {
  // uppercase / spaces / underscores are not claim slugs
  expect(parseAnswer('see [NOT A SLUG] ok')).toEqual([{ kind: 'text', text: 'see [NOT A SLUG] ok' }])
})

test('an id truncated onto a trailing dash still cites', () => {
  // _slugify truncates to 60 chars after stripping dashes, so ~8% of real
  // claim ids end in "-". Those must link like any other.
  const id = '00-point-in-time-and-zero-exactly-where-the-roadmap-already-'
  expect(parseAnswer(`grounded [${id}].`)).toEqual([
    { kind: 'text', text: 'grounded ' },
    { kind: 'citation', claimId: id },
    { kind: 'text', text: '.' },
  ])
})

test('kind-prefixed markers carry their kind', () => {
  // `[claim: id]` is vouch's canonical inline marker in page bodies.
  expect(parseAnswer('This gate is the ingest review [claim: review-gate-is-the-ingest-review].')).toEqual([
    { kind: 'text', text: 'This gate is the ingest review ' },
    { kind: 'citation', claimId: 'review-gate-is-the-ingest-review', refKind: 'claim' },
    { kind: 'text', text: '.' },
  ])
  expect(parseAnswer('see [page: vouch-north-star]')).toEqual([
    { kind: 'text', text: 'see ' },
    { kind: 'citation', claimId: 'vouch-north-star', refKind: 'page' },
  ])
  expect(parseAnswer('see [evidence: ev-00053fcb82b37ffe]')).toEqual([
    { kind: 'text', text: 'see ' },
    { kind: 'citation', claimId: 'ev-00053fcb82b37ffe', refKind: 'evidence' },
  ])
})

test('short brackets that are markdown, not ids, stay text', () => {
  // task-list boxes and footnote refs share the bracket syntax
  expect(parseAnswer('- [x] done')).toEqual([{ kind: 'text', text: '- [x] done' }])
  expect(parseAnswer('cited [1] there')).toEqual([{ kind: 'text', text: 'cited [1] there' }])
})

test('wikilinks stay whole — they name a page title, not an id', () => {
  expect(parseAnswer('see [[the-koth-competition-model]] for more')).toEqual([
    { kind: 'text', text: 'see [[the-koth-competition-model]] for more' },
  ])
})

test('a bare 64-char source hash cites', () => {
  const id = '001bc7c58dde7c9dbe3e836b78432a13917236ea371a43c9a8513998c836c920'
  expect(parseAnswer(`[${id}]`)).toEqual([{ kind: 'citation', claimId: id }])
})

test('empty answer parses to empty list', () => {
  expect(parseAnswer('')).toEqual([])
})

test('parseSnippet splits guillemet highlights', () => {
  expect(parseSnippet('The vouch «HTTP» «server» binds')).toEqual([
    { kind: 'plain', text: 'The vouch ' },
    { kind: 'match', text: 'HTTP' },
    { kind: 'plain', text: ' ' },
    { kind: 'match', text: 'server' },
    { kind: 'plain', text: ' binds' },
  ])
})

test('parseSnippet without highlights returns one plain segment', () => {
  expect(parseSnippet('plain text')).toEqual([{ kind: 'plain', text: 'plain text' }])
})
