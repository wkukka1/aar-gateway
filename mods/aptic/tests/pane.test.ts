import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// The words a drawn tree shows, one line per Text (a Text's runs are joined)
const words = (node: any): string =>
  typeof node === 'string' ? node
    : node.type === 'Text' ? (node.children ?? []).map(words).join('') + '\n'
    : (node.children ?? []).map(words).join('')

// The pane as the engine draws it: a session with two turns, one routed to Sonnet
// with a Haiku subagent, one kept on Opus
export function session(on: On) {
  const turns = [
    { time: '21:28', requested_model: 'claude-opus-5-5', models: ['claude-sonnet-5-5'], reason: 'routed in Claude Code: cheapest within 0.05 utility of the best',
      requests: 2, side_requests: 0, errors: 0, cache_read: 9000, cache_written: 990, fresh: 10, saved_pct: 50,
      subagents: [{ role: 'lookup', model: 'claude-haiku-5-5', requests: 2 }] },
    { time: '21:30', requested_model: 'claude-opus-5-5', models: ['claude-opus-5-5'], reason: "client's choice", requests: 1, side_requests: 0,
      errors: 0, cache_read: 0, cache_written: 0, fresh: 10, saved_pct: 0, subagents: [] },
  ]
  on('ui.render', async () => ({ type: 'Box' }) as any)  // the engine's own answer beneath the plugin
  on('http.fetch', async ($, e) => {
    const url = String(e.url)
    const body = url.includes('/stats') ? { totals: { requests: 12, cost_usd: 7, cost_if_requested_usd: 10 } }
      : url.includes('/api/session-requests') ? { saved_pct: 50, turns } : {}
    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify(body) } }
  })
}

test('the pane draws each turn with its model, saving, reason and subagents', async ($, on) => {
  const kept: Record<string, unknown> = {}
  on("store.get", async ($, e: any) => ({ value: kept[e.key] }))
  on("store.set", async ($, e: any) => { kept[e.key] = e.value; return { value: undefined } })
  session(on)
  on('session.id', async () => ({ value: 'session-under-test' }))
  on('session.model', async () => ({ value: 'claude-opus-5-5' }))
  on('env.get', async () => ({ value: undefined }))
  on('session.start', async ($, e, next) => ({ cwd: (e as any).cwd }))
  on('command.register', async () => ({ value: undefined }))
  on('ui.open', async () => ({ value: { isPlaced: false } }))
  on('ui.status', async () => ({ value: undefined }))
  mock.clock(on)
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)  // the first refresh fills the view
  // env and http come from the test kit; the mounted pane reads the gateway's view
  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ plugin: 'aptic', surface, component: 'Pane', requestId: 'aptic', props: {}, viewport: { columns: 60, rows: 48 } } as any)
    const tree = words(await ui.drawn())
    const has = (pattern: RegExp) => expect(tree).toMatch(pattern)
    has(/THIS SESSION/)
    has(/50%/)
    has(/saved vs\. Opus 5\.5/)
    has(/1 of 2 turns routed to Sonnet 5\.5/)
    has(/Opus 5\.5 · 1 req/)
    has(/Routed down · 2/)
    has(/12 requests routed/)
    has(/−30%/)
    has(/Routed down from Opus 5\.5/)
    has(/Kept · as asked/)
    has(/↳ lookup on Haiku 5\.5 · 2/)
    has(/Baseline: always Opus 5\.5/)
    // newest first by default
    expect(tree.indexOf('21:30')).toBeLessThan(tree.indexOf('21:28'))
    await ui.unmount()
  }
})

test('the pane collapses to a summary and a chart, and remembers it', async ($, on) => {
  const kept: Record<string, unknown> = {}
  on("store.get", async ($, e: any) => ({ value: kept[e.key] }))
  on("store.set", async ($, e: any) => { kept[e.key] = e.value; return { value: undefined } })
  session(on)
  on('session.id', async () => ({ value: 'session-under-test' }))
  on('session.model', async () => ({ value: 'claude-opus-5-5' }))
  on('env.get', async () => ({ value: undefined }))
  on('session.start', async ($, e, next) => ({ cwd: (e as any).cwd }))
  on('command.register', async () => ({ value: undefined }))
  on('ui.open', async () => ({ value: { isPlaced: false } }))
  on('ui.status', async () => ({ value: undefined }))
  mock.clock(on)
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  const ui = await $.ui.mount({ plugin: 'aptic', surface: 'terminal', component: 'Pane', requestId: 'aptic', props: {}, viewport: { columns: 60, rows: 48 } } as any)
  await ui.press({ key: 'settings' } as any)
  await ui.press({ key: 'collapse' } as any)
  const tree = words(await ui.drawn())
  expect(tree).toMatch(/· 2 turns · 3 req/)
  expect(tree).toMatch(/Requests per turn/)
  expect(tree).toMatch(/plan usage · 12 req/)
  expect(tree).not.toMatch(/THIS SESSION/)
  expect(kept['pane.prefs']).toMatchObject({ collapsed: true })
  await ui.press({ key: 'expand' } as any)
  expect(words(await ui.drawn())).toMatch(/THIS SESSION/)
  await ui.unmount()
})

test('View all sessions opens the dashboard in the browser', async ($, on) => {
  const kept: Record<string, unknown> = {}
  on("store.get", async ($, e: any) => ({ value: kept[e.key] }))
  on("store.set", async ($, e: any) => { kept[e.key] = e.value; return { value: undefined } })
  session(on)
  const ran: string[][] = []
  on('process.run', async ($, e: any) => { ran.push([...e.argv]); return { value: { exitCode: 0, stdout: '', stderr: '' } } })
  on('session.id', async () => ({ value: 'session-under-test' }))
  on('session.model', async () => ({ value: 'claude-opus-5-5' }))
  on('env.get', async ($, e: any) => ({ value: e.name === 'OS' ? 'Windows_NT' : undefined }))
  on('session.start', async ($, e, next) => ({ cwd: (e as any).cwd }))
  on('command.register', async () => ({ value: undefined }))
  on('ui.open', async () => ({ value: { isPlaced: false } }))
  on('ui.status', async () => ({ value: undefined }))
  mock.clock(on)
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  const ui = await $.ui.mount({ plugin: 'aptic', surface: 'desktop', component: 'Pane', requestId: 'aptic', props: {}, viewport: { columns: 60, rows: 48 } } as any)
  ran.length = 0
  await ui.press({ key: 'dashboard' } as any)
  expect(ran).toEqual([['rundll32', 'url.dll,FileProtocolHandler', 'http://127.0.0.1:8787/dashboard']])
  await ui.unmount()
})
