// REST + type definitions for the backend at FastAPI /api/*
//
// In dev, vite proxies /api → http://localhost:8000.

export type EventType =
  | 'ticker_received'
  | 'routed'
  | 'bounced'
  | 'decision_started'
  | 'decision_complete'

export interface PipelineEvent {
  id: string
  type: EventType
  ts: number
  data: Record<string, any>
}

export interface DecisionSummary {
  id: string
  file: string
  ts: number
  ticker: string
  market_type: string
  status: string  // APPROVED | VETOED | PASS | UNKNOWN
  action?: string
  price?: number
  confidence?: string
  edge?: number
  calibration_gap?: number
  risk_score?: number
  veto_reason?: string
}

export interface FullDecision {
  id: string
  ts: number
  ticker: string
  market_type: string
  status: string
  trade_packet: Record<string, any>
  decision: Record<string, any>
}

export interface Trade {
  id: number
  logged_at: string
  ticker: string
  market_title?: string
  market_type: string
  player_name?: string
  prop_threshold?: number
  action: string
  side: string
  yes_price: number
  entry_cents: number
  contracts: number
  cost_usd: number
  kelly?: number
  confidence?: string
  calibration_gap?: number
  sample_size?: number
  verdict?: string
  risk_score?: number
  concerns?: string
  status: string
  result?: string
  payout_usd?: number
  pnl_usd?: number
  evaluated_at?: string
}

export interface TradesSummary {
  n_trades: number
  n_wins: number
  total_pnl: number
  total_staked: number
  win_rate: number
  roi: number
}

async function jsonGet<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export const api = {
  health: () => jsonGet<{ ok: boolean; pipeline_running: boolean; pipeline_error: string | null }>('/api/health'),
  decisions: (limit = 200) => jsonGet<DecisionSummary[]>(`/api/decisions?limit=${limit}`),
  decision: (id: string) => jsonGet<FullDecision>(`/api/decisions/${id}`),
  trades: (status: 'open' | 'evaluated' | 'all' = 'all') => jsonGet<Trade[]>(`/api/trades?status=${status}`),
  tradesSummary: () => jsonGet<TradesSummary>('/api/trades/summary'),
  settle: async (): Promise<string> => {
    const r = await fetch('/api/settle', { method: 'POST' })
    if (!r.ok) throw new Error(`settle: ${r.status}`)
    return r.text()
  },
}
