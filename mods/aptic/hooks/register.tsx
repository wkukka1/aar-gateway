// Aptic inside Claude Code: each turn's model switched from inside Claude Code
// by the local router (the `route` option, on by default); a pane with each
// turn's model, why, cache use and savings; and a line under each answer.
// The router being down never blocks anything: the mod then keeps the model
// Claude Code asked for and shows what it can count itself.

import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { CacheTotals, GatewayView, PanePrefs, TurnRow } from '../types'

const GATEWAY = 'http://127.0.0.1:8787'
const PANE = 'aptic'
const EVERY_MS = 60_000

const empty: GatewayView = { isRunning: false, sessionSavedPct: null, allSavedPct: null, allRequests: 0, turns: [] }
const gateway = atom({ plugin: 'aptic', key: 'gateway' } as const, empty)
const cache = atom({ plugin: 'aptic', key: 'cache' } as const, { read: 0, written: 0, fresh: 0 })
const DOWN = "Aptic's router isn't running: /aptic starts it."

// Share of the prompt read from the cache, of everything the requests sent
export const hitRate = (u: CacheTotals): number | null => {
  const total = u.read + u.written + u.fresh
  return total ? Math.round((u.read / total) * 100) : null
}

export const pretty = (m: string | null | undefined): string =>
  (m || '?').replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2')

const k = (n: number): string => (n >= 1000 ? Math.round(n / 1000) + 'k' : String(n))

// A turn's usage against the model asked for, as the pane prints it
export const vsAsked = (pct: number | null): string =>
  pct == null ? '' : pct === 0 ? 'same usage' : pct > 0 ? `${pct}% less` : `${-pct}% more`

const kept = (t: TurnRow): boolean => t.models.every(m => m === t.requested_model)

// Why a turn ran where it did, in words (the gateway logs the router's own terms)
export const why = (t: TurnRow): string => {
  const r = (t.reason || '').replace(/^routed in Claude Code: /, '')
  if (/^cheapest within/.test(r) || r === 'router') {
    return kept(t) ? `router: needs ${pretty(t.requested_model)}` : `router: ${pretty(t.models.at(-1))} should do as well`
  }
  if (r.startsWith('continues a server-side thread')) return 'continues a thread, so stays on its model'
  if (r === "client's choice" || r === 'as asked' || r === 'same turn as the previous request') return 'your model, as asked'
  if (r === 'no eligible candidates') return 'no model fit this request: kept yours'
  if (r.startsWith('side request')) return "Claude Code's own small requests"
  if (r.startsWith('fell back')) return 'routed model failed: yours answered'
  return r || 'your model, as asked'
}

// The model(s) that answered a turn and its cache use, for the line under the answer
export const turnLine = (t: CacheTotals & { requests: number; models: string[] }): string => {
  const hit = hitRate(t)
  return `${t.models.map(pretty).join(', ') || '?'} · cache ${hit === null ? '–' : hit + '% hit'} ` +
    `(${k(t.read)} read, ${k(t.written)} written) · ${t.requests} request${t.requests === 1 ? '' : 's'}`
}

// A model as the pane names it: `claude-opus-5-5` -> `Opus 5.5`
export const modelName = (m: string | null | undefined): string => {
  const p = pretty(m)
  return p.charAt(0).toUpperCase() + p.slice(1)
}

// A turn's decision, the line under its model
export const decision = (t: TurnRow): string => {
  if (!kept(t)) return `Routed ${t.saved_pct != null && t.saved_pct < 0 ? 'up' : 'down'} from ${modelName(t.requested_model)}`
  const w = why(t)
  if (w.startsWith('router: needs')) return 'Kept · router agreed'
  return 'Kept · ' + (w === 'your model, as asked' ? 'as asked' : w)
}

// The model a session is measured against: the one its turns asked for most
export const baselineOf = (turns: TurnRow[]): string => {
  const n: Record<string, number> = {}
  for (const t of turns) if (t.requested_model) n[t.requested_model] = (n[t.requested_model] ?? 0) + 1
  return Object.entries(n).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'claude-opus-5-5'
}

// Requests by where they ran: kept on the model asked for, routed down, routed up.
// The gateway counts a turn's requests, not its requests per model, so a turn
// that ran on several models counts wholly as routed.
export const mixOf = (turns: TurnRow[]) => {
  const mix = { kept: 0, down: 0, up: 0 }
  for (const t of turns) mix[kept(t) ? 'kept' : t.saved_pct != null && t.saved_pct < 0 ? 'up' : 'down'] += t.requests
  return mix
}

// Why this session saved what it did, in one line
export const sessionReason = (turns: TurnRow[], short = false): string => {
  const base = modelName(baselineOf(turns))
  const moved = turns.filter(t => !kept(t))
  const n = turns.length
  if (!n) return 'No turns yet in this session.'
  if (!moved.length) {
    if (short) return `all turns on ${base}`
    const agreed = turns.every(t => why(t).startsWith('router: needs'))
    const all = n === 1 ? 'The one turn' : `All ${n} turns`
    return agreed ? `${all} needed ${base}, so nothing was routed down.` : `${all} stayed on ${base}.`
  }
  if (short) return `${moved.length} of ${n} turns routed`
  const to = [...new Set(moved.flatMap(t => t.models.filter(m => m !== t.requested_model)))].map(modelName).join(', ')
  return `${moved.length} of ${n} turn${n === 1 ? '' : 's'} routed to ${to}.`
}

const MINUS = '−'
const signed = (pct: number): string => (pct < 0 ? MINUS + -pct : pct > 0 ? '+' + pct : '0') + '%'
const num = (n: number): string => n.toLocaleString('en-US')

// The pane's colors (it draws its own dark card, so they don't follow the host theme)
const C = {
  bg: '#1e2227', border: '#30363d', text: '#e6e9ec', secondary: '#c3c9cf', muted: '#9aa3ad',
  divider: '#2b3036', row: '#262b31', tile: '#252a30', card: '#1d2a29', cardBorder: '#2f4a46',
  accent: '#5fb8a8', opus: '#8b95a1', up: '#e0b45c', warn: '#e0b45c', error: '#f08a7e', faint: '#8f98a2', cardMuted: '#9fb8b3',
}
const MONO = "'JetBrains Mono', ui-monospace, monospace"
const DASHBOARD = GATEWAY + '/dashboard'

// Open the gateway's dashboard in the browser. A Link to an http:// address is
// drawn as plain text on VS Code and desktop, so the pane's button runs the
// system's own URL opener instead.
export async function openDashboard($: any): Promise<boolean> {
  const windows = String((await $.env.get('OS')) || '') === 'Windows_NT'
  const openers = windows ? [['rundll32', 'url.dll,FileProtocolHandler', DASHBOARD]] : [['open', DASHBOARD], ['xdg-open', DASHBOARD]]
  for (const argv of openers) {
    try {
      const r = await $.process.run(argv, { timeoutMs: 10_000 })
      if (!r.exitCode) return true
    } catch {
      // this opener isn't there: try the next
    }
  }
  $.ui.toast(`Open ${DASHBOARD} in your browser to see all sessions.`)
  return false
}

// The pane's per-user choices: drawn from $.state, kept across sessions in $.store
const PREFS = 'pane.prefs'
const noPrefs: PanePrefs = { collapsed: false, oldestFirst: false, settingsOpen: false }
const prefs = atom({ plugin: 'aptic', key: 'prefs' } as const, noPrefs)

const ROUTE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="${C.accent}" stroke-width="1.6" stroke-linecap="round"><path d="M2 8h4M10 4h4M10 12h4M6 8l4-4M6 8l4 4"/></svg>`

const ruleSvg = (color: string) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="372" height="1" viewBox="0 0 372 1" preserveAspectRatio="none"><rect width="372" height="1" fill="${color}"/></svg>`

// The model-mix bar: one rounded track, segments by requests
const mixSvg = (mix: { kept: number; down: number; up: number }): string => {
  const W = 372
  const total = mix.kept + mix.down + mix.up
  let x = 0
  const seg = (n: number, fill: string) => {
    if (!total || !n) return ''
    const w = (n / total) * W
    const r = `<rect x="${x.toFixed(1)}" width="${w.toFixed(1)}" height="8" fill="${fill}"/>`
    x += w
    return r
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="8" viewBox="0 0 ${W} 8" preserveAspectRatio="none">` +
    `<defs><clipPath id="t"><rect width="${W}" height="8" rx="4"/></clipPath></defs><g clip-path="url(#t)">` +
    `<rect width="${W}" height="8" fill="${C.tile}"/>${seg(mix.kept, C.opus)}${seg(mix.down, C.accent)}${seg(mix.up, C.up)}</g></svg>`
}

const barColor = (t: TurnRow) => (kept(t) ? C.opus : t.saved_pct != null && t.saved_pct < 0 ? C.up : C.accent)

// Requests per turn, oldest to newest: count above each bar, time below
const chartSvg = (turns: TurnRow[]): string => {
  const W = 360, H = 130, top = 18, base = 104
  const max = Math.max(1, ...turns.map(t => t.requests))
  const slot = W / Math.max(turns.length, 1)
  const bw = Math.min(40, slot * 0.6)
  const bars = turns.map((t, i) => {
    const barH = Math.max(2, ((base - top) * t.requests) / max)
    const cx = slot * i + slot / 2
    return `<rect x="${(cx - bw / 2).toFixed(1)}" y="${(base - barH).toFixed(1)}" width="${bw.toFixed(1)}" height="${barH.toFixed(1)}" rx="3" fill="${barColor(t)}"/>` +
      `<text x="${cx.toFixed(1)}" y="${(base - barH - 5).toFixed(1)}" text-anchor="middle" fill="${C.muted}">${t.requests}</text>` +
      `<text x="${cx.toFixed(1)}" y="${base + 16}" text-anchor="middle" fill="${C.faint}">${t.time}</text>`
  }).join('')
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="${MONO}" font-size="10">` +
    `<rect x="0" y="${base}" width="${W}" height="1" fill="${C.border}"/>${bars}</svg>`
}

async function getJson($: any, path: string, init?: any): Promise<any | null> {
  try {
    const res = await $.http.fetch(GATEWAY + path, init)
    return res.ok ? JSON.parse(res.text) : null
  } catch {
    return null
  }
}

async function refresh($: any): Promise<void> {
  const id = String(await $.session.id())
  const [stats, mine] = await Promise.all([
    getJson($, '/stats'),
    getJson($, '/api/session-requests?limit=40&session_id=' + encodeURIComponent(id)),
  ])
  const t = stats && stats.totals
  const view: GatewayView = {
    isRunning: Boolean(stats),
    sessionSavedPct: mine ? mine.saved_pct : null,
    allSavedPct: t && t.cost_if_requested_usd ? Math.round((1 - t.cost_usd / t.cost_if_requested_usd) * 100) : null,
    allRequests: t ? t.requests : 0,
    turns: mine && mine.turns ? mine.turns : [],
  }
  await update($, gateway, () => view)
  const hit = hitRate(await read($, cache))
  $.ui.status(
    (view.isRunning
      ? `Aptic: ${view.allSavedPct ?? '–'}% less plan usage · ${view.allRequests} requests`
      : DOWN) +
      (hit === null ? '' : ` · this session's cache: ${hit}% hit`),
  )
}

// How the plugin runs the gateway when it isn't running: uv fetches and caches it,
// so the user installs nothing by hand. Pinned to the release this plugin ships with.
export const GATEWAY_SPEC = 'aar-gateway @ git+https://github.com/wkukka1/aar-gateway@main'

// Start the gateway if nothing answers: first-time setup (router file, config,
// nothing else changed), then a background start. Resolves to what happened.
async function ensureGateway($: any): Promise<'running' | 'started' | 'failed' | 'no-uv'> {
  if (await getJson($, '/health')) return 'running'
  const home = String((await $.env.get('HOME')) || (await $.env.get('USERPROFILE')) || '')
  const sep = home.includes('\\') ? '\\' : '/'
  // VS Code started from the dock often lacks ~/.local/bin on PATH: try uv's own install folder too
  for (const uvx of ['uvx', home + sep + '.local' + sep + 'bin' + sep + 'uvx']) {
    const run = (args: string[]) => $.process.run([uvx, '--from', GATEWAY_SPEC, 'aar-gateway', ...args], { timeoutMs: 600_000 })
    try {
      if (home && !(await $.fs.exists(home + sep + '.aar-gateway' + sep + 'config.yaml'))) {
        await run(['setup', '--no-enable', '--no-share', '--no-mod'])
      }
      await run(['start', '--quiet'])
    } catch {
      continue  // this uvx isn't there (or couldn't start): try the next
    }
    return (await getJson($, '/health')) ? 'started' : 'failed'
  }
  return 'no-uv'
}

// Once ever, with the router up: plain `claude` sessions can go through the gateway too,
// which also tells the main agent it can hand work to subagents. The plugin itself
// changes no settings; `aar-gateway enable` does, when the person runs it.
export const ENABLE_NOTE = 'notice.enable'
export const ENABLE_TEXT = 'Aptic: to route plain `claude` sessions too, and let the main agent hand searches, checks '
  + `and changes to subagents, run \`uv tool install "${GATEWAY_SPEC}"\` then \`aar-gateway enable\` in a terminal `
  + '(`aar-gateway stop` undoes it).'

async function offerEnable($: any): Promise<void> {
  const shown = await $.store.get(ENABLE_NOTE).catch(() => true)  // no store: stay quiet, not every session
  if (shown) return
  await $.store.set(ENABLE_NOTE, true).catch(() => {})
  $.ui.toast(ENABLE_TEXT)
}

export const register: Register = (on, options) => {
  const routeInClaudeCode = options?.route !== false  // on unless turned off: a plugin-only session has no proxy
  // set at session start: whether this session's requests go through the gateway's proxy
  // (it logs them itself) or straight to Anthropic (the mod reports them)
  let viaProxy = false
  let paneOpen = false
  const roles: Record<string, string> = {}  // subagent id -> its kind of work, for the pane
  const failed: string[] = []  // routed models that failed in this session: not routed to again
  let resumed = false  // this session was resumed (claude -c): its first turn says so to the gateway
  let sessionModel = ''
  // the current main-loop turn: what it ran on and its cache use; its routed model when routing here
  let turn: (CacheTotals & { requests: number; models: string[]; model: string | null }) | null = null

  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'aptic', description: "Show Aptic's savings now (and start its router if it isn't running)" })
    await $.command.register({ name: 'aptic-pane', description: 'Open the Aptic pane' })
    // the pane's view choices from an earlier session; no store, the defaults
    const stored = await $.store.get(PREFS).catch(() => null)
    if (stored && typeof stored === 'object') await update($, prefs, () => ({ ...noPrefs, ...(stored as Partial<PanePrefs>) }))
    void $.ui.open({ id: PANE, title: 'Aptic' })
    paneOpen = true
    viaProxy = String((await $.env.get('ANTHROPIC_BASE_URL')) || '').includes('127.0.0.1:8787')
    sessionModel = String(await $.session.model())
    ensureGateway($).then(async state => {
      if (state === 'no-uv') {
        $.ui.toast('Aptic needs uv to run its router: install it from https://docs.astral.sh/uv/ then restart Claude Code.')
      } else if (state !== 'failed' && !viaProxy) await offerEnable($)
      return refresh($)
    }).catch(() => {})
    refresh($).catch(() => {})  // display only: a failed refresh shows the old numbers
    $.clock.every(EVERY_MS, () => refresh($).catch(() => {}))
    return next(e)
  })

  on('command.run', { command: 'aptic-pane' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Aptic', focus: true })
    return { text: 'Aptic pane opened.' }
  })

  on('command.run', { command: 'aptic' }, async $ => {
    const state = await ensureGateway($)
    await refresh($)
    const view = await read($, gateway)
    const hit = hitRate(await read($, cache))
    if (!view.isRunning) {
      return { text: state === 'no-uv' ? 'Aptic needs uv to run its router: https://docs.astral.sh/uv/' : "Aptic's router couldn't start." }
    }
    return {
      text: (state === 'started' ? "Started Aptic's router. " : '') +
        `Aptic: ${view.allSavedPct ?? '–'}% less plan usage · ${view.allRequests} requests` +
        (view.sessionSavedPct == null ? '' : ` · this session: ${view.sessionSavedPct}% less`) +
        (hit === null ? '' : ` · cache ${hit}% hit`),
    }
  })

  on('classic.SessionStart', async ($, e, next) => {
    resumed = e.source === 'resume'
    return next(e)
  })

  on('turn.start', async ($, e, next) => {
    if (e.agentId) return next(e)
    const isResumed = resumed
    resumed = false
    if (!paneOpen) {
      paneOpen = true
      void $.ui.open({ id: PANE, title: 'Aptic' })
    }
    turn = { read: 0, written: 0, fresh: 0, requests: 0, models: [], model: null }
    sessionModel = String(await $.session.model())
    if (routeInClaudeCode && e.text) {
      // ask the gateway's router; no answer keeps the model Claude Code asked for
      const asked = sessionModel
      const picked = await getJson($, '/api/route', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text: e.text, model: asked, session_id: String(await $.session.id()), blocked: failed,
          resumed: isResumed }),
      })
      // the gateway may correct the model asked for: a resumed session restarts on the model
      // this plugin last switched to, and the person asked for the one before
      if (picked && picked.requested && picked.requested !== asked) sessionModel = String(picked.requested)
      turn.model = picked && picked.model ? picked.model : sessionModel !== asked ? sessionModel : null
    }
    return next(e)
  })

  // Each request: on the turn's routed model when routing here; count its cache use
  on('turn.step', async function* ($, e, next) {
    const mine = !e.agentId && turn
    const routed = mine && turn!.model && turn!.model !== e.model ? turn!.model : null
    let result
    try {
      result = yield* next(routed ? { ...e, model: routed } : e)
    } catch (err) {
      if (!routed) throw err
      // the routed model failed outright: this request, and the session's next turns, on the model asked for
      if (!failed.includes(routed)) failed.push(routed)
      turn!.model = null
      result = yield* next(e)
    }
    const u: any = result && (result as any).usage
    if (mine && u && turn) {
      const add = { read: u.cache_read_input_tokens || 0, written: u.cache_creation_input_tokens || 0, fresh: u.input_tokens || 0 }
      turn.read += add.read
      turn.written += add.written
      turn.fresh += add.fresh
      turn.requests += 1
      if (u.model && !turn.models.includes(u.model)) turn.models.push(u.model)
      await update($, cache, c => ({ read: c.read + add.read, written: c.written + add.written, fresh: c.fresh + add.fresh }))
    }
    if (u && !viaProxy) {
      // straight to Anthropic: the gateway never saw this request, so report its usage for the savings and the pane
      getJson($, '/api/mod-record', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          session_id: String(await $.session.id()), agent_id: e.agentId, new_turn: !e.agentId && e.index === 0,
          agent_role: e.agentId ? roles[e.agentId] ?? null : null,
          requested_model: sessionModel || e.model, routed_model: u.model || (mine && turn ? turn.model : null) || e.model,
          input_tokens: u.input_tokens, output_tokens: u.output_tokens,
          cache_read_input_tokens: u.cache_read_input_tokens, cache_creation_input_tokens: u.cache_creation_input_tokens,
        }),
      }).catch(() => {})
    }
    return result
  })

  // A subagent's model by the kind of work, from the gateway (when routing here)
  on('agent.spawn', async ($, e, next) => {
    if (!routeInClaudeCode) return next(e)
    const picked = await getJson($, '/api/subagent-spawn', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session_id: String(await $.session.id()), subagent_type: e.subagentType, prompt: e.prompt, model: (e as any).model }),
    })
    const result: any = await next(picked && picked.alias ? { ...e, model: picked.alias } : e)
    if (result && result.agentId && picked && picked.role) roles[result.agentId] = String(picked.role)
    return result
  })

  on('turn.complete', async ($, e, next) => {
    if (!e.agentId && turn && turn.model && turn.model !== sessionModel && e.reason === 'error' && !failed.includes(turn.model)) {
      // an API error ended a routed turn: don't route to that model again this session
      failed.push(turn.model)
      $.ui.toast(`Aptic: ${pretty(turn.model)} failed, so this session stays on ${pretty(sessionModel)}. Send your message again.`)
    }
    if (e.agentId || !turn || !turn.requests) return next(e)
    const text = turnLine(turn)
    turn = null
    const result: any = await next(e)  // let Claude Code and other mods finish the turn first
    refresh($).catch(() => {})  // display only: a failed refresh shows the old numbers
    // `text` from beneath is the answer itself unless another plugin put a note there;
    // whatever differs from the answer is shown under it, as one line
    const note = result && result.text && result.text !== e.answer ? result.text + ' · ' : ''
    return { ...(result || {}), text: 'Aptic: ' + note + text }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const els: any = $.ui.resolve(e)
    const { Box, Text, Button } = els
    const Svg = e.surface === 'terminal' ? null : els.Svg  // the terminal draws no Svg: text bars there
    const [view, counted, chosen] = await Promise.all([read($, gateway), read($, cache), read($, prefs)])
    const { collapsed, oldestFirst, settingsOpen } = { ...noPrefs, ...chosen }
    const toggle = (key: keyof PanePrefs) => async () => {
      const next = await update($, prefs, p => ({ ...noPrefs, ...p, [key]: !p[key] }))
      await $.store.set(PREFS, next).catch(() => {})
    }

    const all = view.turns
    const base = modelName(baselineOf(all))
    const mix = mixOf(all)
    const requests = all.reduce((n, t) => n + t.requests, 0)
    // the mod's own count of this session's cache use, else the gateway's per-turn one
    const hit = hitRate(counted) ?? hitRate(all.reduce((c, t) => ({ read: c.read + t.cache_read, written: c.written + t.cache_written,
      fresh: c.fresh + t.fresh }), { read: 0, written: 0, fresh: 0 }))
    const saved = view.sessionSavedPct == null ? null : Math.round(view.sessionSavedPct)
    const allUsage = view.allSavedPct == null ? '–' : signed(-view.allSavedPct)

    // Type steps follow the design: caption 11px, footnote 12px, body 13px, heading 16-20px, title 30px
    const label = (s: string) => <Text size="caption" weight="medium" color={C.muted}>{s.toUpperCase()}</Text>
    const rule = (color: string) => Svg ? <Svg source={ruleSvg(color)} alt="" height={1} />
      : <Text color={color} wrap="truncate">{'─'.repeat(120)}</Text>
    const dot = (color: string) => <Text color={color}>■ </Text>
    const legend = (counts: boolean) => (
      <Box columnGap="md" flexWrap="wrap">
        <Text size="caption" color={C.muted}>{dot(C.opus)}{base}{counts ? ` · ${num(mix.kept)} req` : ''}</Text>
        <Text size="caption" color={C.muted}>{dot(C.accent)}Routed down{counts ? ` · ${num(mix.down)}` : ''}</Text>
        {mix.up ? <Text size="caption" color={C.muted}>{dot(C.up)}Routed up{counts ? ` · ${num(mix.up)}` : ''}</Text> : null}
      </Box>
    )
    const icon = Svg ? <Svg source={ROUTE_ICON} alt="" width={16} height={16} /> : <Text color={C.accent}>⇄</Text>
    const down = !view.isRunning && <Text size="footnote" color={C.warn}>{DOWN}</Text>
    const tile = (name: string, value: string, sub?: string, accent = false) => (
      <Box flexDirection="column" flexGrow={1} flexShrink={1} width="25%" paddingX="sm" paddingY="sm"
        backgroundColor={accent ? C.card : C.tile} borderStyle="round" borderColor={accent ? C.cardBorder : C.tile}>
        <Text size="caption" color={accent ? C.cardMuted : C.muted}>{name}</Text>
        <Text size="heading" mono tabular color={accent ? C.accent : C.text}>{value}</Text>
        {sub ? <Text size="caption" color={accent ? C.cardMuted : C.muted}>{sub}</Text> : null}
      </Box>
    )

    // the pane's whole height, so the card fills it and the footer sits at the bottom
    const fill = e.props.scroll?.bodyRows ?? e.viewport?.rows

    if (collapsed) {
      const chartTurns = all.slice(-12)
      const maxReq = Math.max(1, ...chartTurns.map(t => t.requests))
      return (
        <Box flexDirection="column" flexGrow={1} minHeight={fill} backgroundColor={C.bg} borderStyle="round" borderColor={C.border} paddingX="md" paddingY="md" rowGap="md">
          <Box justifyContent="space-between" alignItems="center">
            <Box columnGap="sm" alignItems="center">
              {icon}
              <Text size="body" weight="semibold" color={C.text}>aptic</Text>
              <Text size="footnote" color={C.muted}>· {all.length} turn{all.length === 1 ? '' : 's'} · {num(requests)} req</Text>
            </Box>
            <Button key="expand" variant="secondary" onPress={toggle('collapsed')}><Text size="footnote" color={C.secondary}>▾</Text></Button>
          </Box>
          {down}
          <Box columnGap="sm" flexWrap="nowrap">
            {tile('This session', saved == null ? '–' : `${saved}%`, sessionReason(all, true))}
            {tile('All sessions', allUsage, `plan usage · ${num(view.allRequests)} req`, true)}
          </Box>
          <Box flexDirection="column" rowGap="sm">
            <Box justifyContent="space-between">
              <Text size="caption" color={C.muted}>Requests per turn</Text>
              <Text size="caption" color={C.muted}>cache {hit == null ? '–' : hit + '%'}</Text>
            </Box>
            {chartTurns.length === 0 ? <Text size="footnote" color={C.muted}>No turns yet in this session.</Text>
              : Svg ? <Svg source={chartSvg(chartTurns)} alt={'Requests per turn: ' + chartTurns.map(t => `${t.time} ${t.requests}`).join(', ')} />
              : chartTurns.map(t => (
                <Text>
                  <Text color={C.muted}>{t.time} </Text>
                  <Text color={barColor(t)}>{'█'.repeat(Math.max(1, Math.round((t.requests / maxReq) * 20)))}</Text>
                  <Text color={C.secondary}> {t.requests}</Text>
                </Text>
              ))}
          </Box>
          <Box flexGrow={1} />
          {legend(false)}
        </Box>)
    }

    // a turn takes three rows with its divider (more with subagents or errors); the rest of the pane about 22
    const rows = e.props.scroll?.bodyRows ?? (e.viewport?.rows ?? 24) - 4
    const shown = all.slice(-Math.max(3, Math.floor((rows - 22) / 3)))
    const turns = oldestFirst ? shown : [...shown].reverse()
    const cacheOf = (t: TurnRow) => {
      const total = t.cache_read + t.cache_written + t.fresh
      return total ? Math.round((t.cache_read / total) * 100) + '%' : '–'
    }
    const cols = { time: 6, req: 5, cache: 5 }

    // header and footer on divider rules, the footer held at the bottom
    return (
      <Box flexDirection="column" flexGrow={1} minHeight={fill} backgroundColor={C.bg} borderStyle="round" borderColor={C.border}>
        <Box justifyContent="space-between" alignItems="center" paddingX="md" paddingY="sm">
          <Box columnGap="sm" alignItems="center">
            {icon}
            <Text size="body" weight="semibold" color={C.text}>aptic</Text>
            <Text size="footnote" color={C.muted}>router</Text>
          </Box>
          <Button key="settings" variant="secondary" plain onPress={toggle('settingsOpen')}><Text size="body" color={C.muted}>⚙︎</Text></Button>
        </Box>
        {rule(C.divider)}
        <Box flexDirection="column" paddingX="md" paddingY="md" rowGap="md">
          {settingsOpen && (
            <Box flexDirection="column" rowGap="xs" paddingX="sm" paddingY="sm" backgroundColor={C.tile} borderStyle="round" borderColor={C.border}>
              <Text size="footnote" color={C.secondary}>Turn order and the collapsed view are remembered for you.</Text>
              <Text size="footnote" color={C.muted}>Routing on or off: /plugin, then aptic, then Configure.</Text>
              <Button key="collapse" variant="secondary" onPress={toggle('collapsed')}><Text size="caption" color={C.secondary}>Collapse the pane ▴</Text></Button>
            </Box>
          )}
          {down}
          <Box flexDirection="column" rowGap="xs">
            {label('This session')}
            <Box columnGap="sm" alignItems="flex-end">
              <Text size="title" mono tabular weight="medium" color={saved != null && saved < 0 ? C.warn : C.text}>
                {saved == null ? '–' : (saved < 0 ? MINUS + -saved : saved) + '%'}
              </Text>
              <Text size="body" color={C.secondary}>{saved != null && saved < 0 ? 'more than' : 'saved vs.'} {base}</Text>
            </Box>
            <Text size="footnote" color={C.muted}>{sessionReason(all)}</Text>
          </Box>
          <Box flexDirection="column" rowGap="xs">
            {Svg ? <Svg source={mixSvg(mix)} alt={`${mix.kept} requests on ${base}, ${mix.down} routed down`} height={8} />
              : <Text>
                <Text color={C.opus}>{'█'.repeat(Math.round((mix.kept / Math.max(requests, 1)) * 30))}</Text>
                <Text color={C.accent}>{'█'.repeat(Math.round((mix.down / Math.max(requests, 1)) * 30))}</Text>
                <Text color={C.up}>{'█'.repeat(Math.round((mix.up / Math.max(requests, 1)) * 30))}</Text>
              </Text>}
            {legend(true)}
          </Box>
          <Box columnGap="sm" flexWrap="nowrap">
            {tile('Requests', num(requests))}
            {tile('Cache hit', hit == null ? '–' : hit + '%')}
            {tile('Turns', num(all.length))}
          </Box>
          <Box justifyContent="space-between" alignItems="center" paddingX="md" paddingY="sm"
            backgroundColor={C.card} borderStyle="round" borderColor={C.cardBorder}>
            <Box flexDirection="column">
              <Text size="caption" color={C.cardMuted}>All sessions</Text>
              <Text size="body" color={C.text}>{num(view.allRequests)} requests routed</Text>
            </Box>
            <Box flexDirection="column" alignItems="flex-end">
              <Text size="heading" mono tabular color={C.accent}>{allUsage}</Text>
              <Text size="caption" color={C.cardMuted}>plan usage</Text>
            </Box>
          </Box>
        </Box>
        <Box justifyContent="space-between" alignItems="center" paddingX="md" paddingBottom="xs">
          {label('Turns')}
          <Button key="sort" variant="secondary" onPress={toggle('oldestFirst')}>
            <Text size="caption" color={C.secondary}>{oldestFirst ? 'Oldest first' : 'Newest first'} ▾</Text>
          </Button>
        </Box>
        <Box columnGap="sm" paddingX="md" paddingY="xs">
          <Box width={cols.time}><Text size="caption" color={C.faint}>Time</Text></Box>
          <Box flexGrow={1}><Text size="caption" color={C.faint}>Model · decision</Text></Box>
          <Box width={cols.req} justifyContent="flex-end"><Text size="caption" color={C.faint}>Req</Text></Box>
          <Box width={cols.cache} justifyContent="flex-end"><Text size="caption" color={C.faint}>Cache</Text></Box>
        </Box>
        {rule(C.divider)}
        {turns.length === 0 && <Box paddingX="md" paddingY="sm"><Text size="footnote" color={C.muted}>No turns yet in this session.</Text></Box>}
        {turns.map((t: TurnRow) => (
          <Box key={`turn-${t.time}-${all.indexOf(t)}`} flexDirection="column">
            <Box columnGap="sm" paddingX="md" paddingY="sm" alignItems="center">
              <Box width={cols.time}><Text size="footnote" mono color={C.muted}>{t.time}</Text></Box>
              <Box flexGrow={1} flexDirection="column">
                <Text size="body" weight="medium" color={C.text}>{t.models.map(modelName).join(', ') || modelName(t.requested_model)}</Text>
                <Text size="caption" color={kept(t) ? C.muted : C.accent}>{decision(t)}</Text>
                {t.errors ? <Text size="caption" color={C.error}>{t.errors} failed and retried on your model</Text> : null}
                {t.subagents.map(a => (
                  <Text size="caption" color={C.muted}>↳ {a.role} on {modelName(a.model)} · {a.requests}</Text>
                ))}
              </Box>
              <Box width={cols.req} justifyContent="flex-end"><Text size="footnote" mono tabular color={C.text}>{num(t.requests)}</Text></Box>
              <Box width={cols.cache} justifyContent="flex-end"><Text size="footnote" mono tabular color={C.muted}>{cacheOf(t)}</Text></Box>
            </Box>
            {rule(C.row)}
          </Box>
        ))}
        <Box flexGrow={1} />
        {rule(C.divider)}
        <Box justifyContent="space-between" alignItems="center" paddingX="md" paddingY="sm">
          <Text size="caption" color={C.faint}>Baseline: always {base}</Text>
          <Button key="dashboard" variant="secondary" plain onPress={() => openDashboard($)}><Text size="caption" color={C.accent} underline>View all sessions</Text></Button>
        </Box>
      </Box>)
  })
}
