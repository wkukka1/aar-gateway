import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { ENABLE_TEXT, vsAsked, why } from '../hooks/register'
import type { TurnRow } from '../types'

// Beneath the mod: the session, a gateway answering (or not), and the model
function world(on: On, gateway: 'up' | 'down', picked: string | null = 'claude-haiku-5-5', failing?: string, requested?: string) {
  const asked: string[] = []
  const sent: string[] = []
  const toasts: string[] = []
  const ran: string[] = []
  const reports: any[] = []
  const routeBodies: any[] = []
  on('session.id', async () => ({ value: 'session-under-test' }))
  on('session.model', async () => ({ value: 'claude-opus-5-5' }))
  on('classic.SessionStart', async ($, e) => e as any)  // the settings hooks beneath: none answer it
  on('turn.start', async ($, e) => ({ turnId: e.turnId }))
  on('turn.complete', async ($, e) => ({ text: (e as any).answer ?? '' }))  // as core: the answer itself
  on('ui.status', async () => ({ value: undefined }))
  mock.clock(on)
  on('session.start', async ($, e) => ({ cwd: (e as any).cwd }))
  on('command.register', async () => ({ value: undefined }))
  on('ui.open', async () => ({ value: { isPlaced: false } }))
  on('ui.toast', async ($, e) => { toasts.push(String((e as any).text ?? e)); return { value: undefined } })
  on('process.run', async ($, e) => { ran.push(e.argv.join(' ')); throw new Error('spawn uvx ENOENT') })
  on('fs.exists', async () => ({ value: true }))
  on('http.fetch', async ($, e) => {
    asked.push(String(e.url).replace('http://127.0.0.1:8787', ''))
    if (gateway === 'down') throw new Error('connect ECONNREFUSED')
    const path = String(e.url)
    if (path.includes('/api/mod-record')) reports.push(JSON.parse(String(e.init?.body)))
    if (path.includes('/api/route')) routeBodies.push(JSON.parse(String(e.init?.body)))
    const body = path.includes('/api/route') ? { model: picked, reason: 'cheapest', requested }
      : path.includes('/api/subagent-spawn') ? { alias: 'haiku', role: 'lookup' }
      : path.includes('/stats') ? { totals: { requests: 10, cost_usd: 7, cost_if_requested_usd: 10 } }
      : { saved_pct: 30, turns: [] }
    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify(body) } }
  })
  on('turn.step', async function* ($, e) {
    sent.push(e.model)
    if (e.model === failing) throw new Error('404 model not found')
    return {
      turnId: e.turnId, index: e.index, answer: 'done', toolUses: [],
      usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 9000, cache_creation_input_tokens: 990, model: e.model },
    } as any
  })
  return { asked, sent, toasts, ran, reports, routeBodies }
}

async function oneTurn($: any, text: string, steps = 2) {
  await $.turn.start({ text, turnId: 't1' })
  for (let index = 0; index < steps; index++) {
    const stream = $.turn.step({ turnId: 't1', index, model: 'claude-opus-5-5', messageCount: 3 })
    for await (const _ of stream) { /* drain */ }
  }
  return $.turn.complete({ turnId: 't1', answer: 'done', durationMs: 10, isAborted: false } as any)
}

test('with routing on, every request of the turn goes to the model the gateway picked', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  const done: any = await oneTurn($, 'rename this variable')
  expect(w.sent).toEqual(['claude-haiku-5-5', 'claude-haiku-5-5'])
  expect(w.asked).toContain('/api/route')
  expect(done.text).toMatch(/haiku 5\.5 · cache 90% hit/)
})

test('with routing off, the model is left alone and the router is not asked', { options: { route: false } }, async ($, on) => {
  const w = world(on, 'up')
  await oneTurn($, 'rename this variable')
  expect(w.sent).toEqual(['claude-opus-5-5', 'claude-opus-5-5'])
  expect(w.asked).not.toContain('/api/route')
})

test('with the gateway down, routing on keeps the model asked for and the turn still ends', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'down')
  const done: any = await oneTurn($, 'rename this variable')
  expect(w.sent).toEqual(['claude-opus-5-5', 'claude-opus-5-5'])
  expect(done.text).toMatch(/opus 5\.5 · cache 90% hit \(18k read, 2k written\) · 2 requests/)
})

test('a subagent request is neither switched nor counted in the line', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  await $.turn.start({ text: 'look this up', turnId: 't1' })
  const sub = $.turn.step({ turnId: 't9', index: 0, model: 'claude-opus-5-5', messageCount: 1, agentId: 'a1' })
  for await (const _ of sub) { /* drain */ }
  expect(w.sent).toEqual(['claude-opus-5-5'])
})

test('a session straight to Anthropic reports each request to the gateway; one through the proxy does not', async ($, on) => {
  const w = world(on, 'up')
  on('env.get', async ($, e) => ({ value: undefined }))
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  await oneTurn($, 'rename this variable')
  expect(w.asked.filter(p => p === '/api/mod-record').length).toBe(2)
})

test('through the proxy, nothing is reported twice', async ($, on) => {
  const w = world(on, 'up')
  on('env.get', async ($, e) => ({ value: String((e as any).name ?? e).includes('ANTHROPIC_BASE_URL') ? 'http://127.0.0.1:8787' : undefined }))
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  await oneTurn($, 'rename this variable')
  expect(w.asked).not.toContain('/api/mod-record')
})

test('with the gateway down and no uv, the plugin says how to get uv and Claude Code carries on', async ($, on) => {
  const w = world(on, 'down')
  on('env.get', async () => ({ value: '/home/someone' }))
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  for (let i = 0; i < 50 && w.toasts.length === 0; i++) await new Promise(r => setTimeout(r, 10))
  expect(w.ran.length).toBeGreaterThan(0)
  expect(w.toasts.join(' ')).toMatch(/needs uv/)
  const done: any = await oneTurn($, 'rename this variable')
  expect(w.sent).toEqual(['claude-opus-5-5', 'claude-opus-5-5'])
  expect(done.text).toMatch(/opus 5\.5/)
})

test('with the router up, the plugin says once how to route plain sessions too', async ($, on) => {
  const w = world(on, 'up')
  const kept: Record<string, unknown> = {}
  on('store.get', async ($, e: any) => ({ value: kept[e.key] }))
  on('store.set', async ($, e: any) => { kept[e.key] = e.value; return { value: undefined } })
  on('env.get', async () => ({ value: undefined }))
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  for (let i = 0; i < 50 && w.toasts.length === 0; i++) await new Promise(r => setTimeout(r, 10))
  expect(w.toasts).toEqual([ENABLE_TEXT])
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  await new Promise(r => setTimeout(r, 50))
  expect(w.toasts).toEqual([ENABLE_TEXT])  // never twice
})

test('a session already through the proxy is not told to enable it', async ($, on) => {
  const w = world(on, 'up')
  on('store.get', async () => ({ value: undefined }))
  on('store.set', async () => ({ value: undefined }))
  on('env.get', async ($, e) => ({ value: String((e as any).name ?? e).includes('ANTHROPIC_BASE_URL') ? 'http://127.0.0.1:8787' : undefined }))
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  await new Promise(r => setTimeout(r, 50))
  expect(w.toasts).toEqual([])
})

test('the pane says in words why each turn ran where it did', async () => {
  const turn = (models: string[], reason: string): TurnRow => ({
    time: '21:11', requested_model: 'claude-opus-5-5', models, reason, requests: 3, side_requests: 0, errors: 0,
    cache_read: 0, cache_written: 0, fresh: 0, saved_pct: null, subagents: [],
  })
  const routed = 'routed in Claude Code: cheapest within 0.08 utility of the best'
  expect(why(turn(['claude-opus-5-5'], routed))).toBe('router: needs opus 5.5')
  expect(why(turn(['claude-sonnet-5-5'], routed))).toBe('router: sonnet 5.5 should do as well')
  expect(why(turn(['claude-opus-5-5'], "client's choice"))).toBe('your model, as asked')
  expect(why(turn(['claude-opus-5-5'], 'no eligible candidates'))).toBe('no model fit this request: kept yours')
  expect([vsAsked(48), vsAsked(0), vsAsked(-5), vsAsked(null)]).toEqual(['48% less', 'same usage', '5% more', ''])
})

test("a subagent's requests are reported with its kind of work, for the pane", { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  on('env.get', async () => ({ value: undefined }))
  on('agent.spawn', async ($, e) => ({ model: (e as any).model, agentId: 'agent-7' }) as any)
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  await $.turn.start({ text: 'find widget_checksum', turnId: 't1' })
  const spawned: any = await $.agent.spawn({ subagentType: 'Explore', prompt: 'find widget_checksum', description: 'find' } as any)
  expect(spawned.model).toBe('haiku')
  for await (const _ of $.turn.step({ turnId: 't9', index: 0, model: 'claude-haiku-5-5', messageCount: 1, agentId: 'agent-7' })) { /* drain */ }
  expect(w.reports.at(-1)).toMatchObject({ agent_id: 'agent-7', agent_role: 'lookup' })
})

test('the line under an answer is the summary alone, never the answer again', { options: { route: true } }, async ($, on) => {
  world(on, 'up')
  const done: any = await oneTurn($, 'rename this variable')
  expect(done.text).toMatch(/^Aptic: haiku 5\.5 · cache 90% hit/)
  expect(done.text).not.toContain('done')
})

test('a routed model that fails is retried on the model asked for, and not routed to again', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up', 'claude-haiku-5-5', 'claude-haiku-5-5')
  on('env.get', async () => ({ value: undefined }))
  await oneTurn($, 'rename this variable', 1)
  expect(w.sent).toEqual(['claude-haiku-5-5', 'claude-opus-5-5'])
  await $.turn.start({ text: 'and this one', turnId: 't2' })
  expect(w.routeBodies.at(-1).blocked).toEqual(['claude-haiku-5-5'])
})

test('an API error that ends a routed turn says so and keeps the session on the model asked for', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  on('env.get', async () => ({ value: undefined }))
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  await $.turn.start({ text: 'rename this variable', turnId: 't1' })
  for await (const _ of $.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 3 })) { /* drain */ }
  await $.turn.complete({ turnId: 't1', answer: '', durationMs: 10, isAborted: false, reason: 'error' } as any)
  expect(w.toasts.join(' ')).toMatch(/haiku 5\.5 failed, so this session stays on opus 5\.5/)
  await $.turn.start({ text: 'again', turnId: 't2' })
  expect(w.routeBodies.at(-1).blocked).toEqual(['claude-haiku-5-5'])
})

test('a resumed session goes back to the model first asked for when the gateway says so', { options: { route: true } }, async ($, on) => {
  // the session reports the model it resumed on; the gateway says the person first asked for another
  const w = world(on, 'up', null, undefined, 'claude-sonnet-5')
  on('env.get', async () => ({ value: undefined }))
  await $.classic.SessionStart({ source: 'resume' } as any)
  await oneTurn($, 'next step', 1)
  expect(w.routeBodies[0].resumed).toBe(true)
  expect(w.sent).toEqual(['claude-sonnet-5'])
})

test('only the first turn of a resumed session asks the gateway about the resume', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  on('env.get', async () => ({ value: undefined }))
  await $.classic.SessionStart({ source: 'resume' } as any)
  await oneTurn($, 'first', 1)
  await oneTurn($, 'second', 1)
  expect(w.routeBodies.map(b => b.resumed)).toEqual([true, false])
})

test('a new session, or a /model choice, is never read as a resume', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  on('env.get', async () => ({ value: undefined }))
  await $.classic.SessionStart({ source: 'startup' } as any)
  await oneTurn($, 'rename this variable', 1)
  expect(w.routeBodies[0].resumed).toBe(false)
})
