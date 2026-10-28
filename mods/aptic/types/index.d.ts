// One turn as the pane shows it: what you typed, grouped with its tool loop and
// subagents (from the gateway's request log; no prompt text)
export type TurnRow = {
  time: string  // HH:MM, local time
  requested_model: string | null
  models: string[]
  reason: string | null
  requests: number
  side_requests: number
  errors: number
  cache_read: number
  cache_written: number
  fresh: number
  saved_pct: number | null
  subagents: { role: string; model: string | null; requests: number }[]
}

export type GatewayView = {
  isRunning: boolean
  sessionSavedPct: number | null
  allSavedPct: number | null
  allRequests: number
  turns: TurnRow[]
}

// This session's cache use, counted from the usage Claude Code reports per request
export type CacheTotals = { read: number; written: number; fresh: number }

// The pane's view choices, per user (copied to $.store so they outlast the session)
export type PanePrefs = { collapsed: boolean; oldestFirst: boolean; settingsOpen: boolean }

declare module 'claude-code' {
  interface PluginState {
    'aptic': { gateway: GatewayView; cache: CacheTotals; prefs: PanePrefs }
  }
}
