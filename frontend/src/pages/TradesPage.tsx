import { useEffect, useMemo, useState } from 'react'
import { api, type Trade } from '../lib/api'

type Filter = 'all' | 'open' | 'evaluated'

export default function TradesPage() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [filter, setFilter] = useState<Filter>('all')
  const [marketFilter, setMarketFilter] = useState<'all' | 'GAME_WINNER' | 'PLAYER_PROP'>('all')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = () => {
    setLoading(true)
    api.trades(filter).then(setTrades).finally(() => setLoading(false))
  }

  useEffect(refresh, [filter])

  const visible = useMemo(
    () => marketFilter === 'all' ? trades : trades.filter((t) => t.market_type === marketFilter),
    [trades, marketFilter],
  )

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-3 border-b border-zinc-800 flex items-center gap-3 bg-zinc-950/40">
        <h2 className="text-sm font-semibold text-zinc-200">Logged Trades</h2>
        <FilterPill value="all" current={filter} onChange={setFilter}>All</FilterPill>
        <FilterPill value="open" current={filter} onChange={setFilter}>Open</FilterPill>
        <FilterPill value="evaluated" current={filter} onChange={setFilter}>Evaluated</FilterPill>
        <span className="mx-2 text-zinc-700">|</span>
        <FilterPill value="all" current={marketFilter} onChange={setMarketFilter}>Any market</FilterPill>
        <FilterPill value="GAME_WINNER" current={marketFilter} onChange={setMarketFilter}>Game</FilterPill>
        <FilterPill value="PLAYER_PROP" current={marketFilter} onChange={setMarketFilter}>Prop</FilterPill>
        <button onClick={refresh} className="ml-auto text-xs text-zinc-400 hover:text-zinc-200">↻ refresh</button>
        <span className="text-xs text-zinc-500">{visible.length} rows</span>
      </div>

      <div className="flex-1 overflow-auto">
        <table className="w-full text-[12px]">
          <thead className="text-[10px] uppercase tracking-wider text-zinc-500 sticky top-0 bg-zinc-950 z-10">
            <tr>
              <Th>#</Th><Th>Logged</Th><Th>Type</Th><Th>Ticker</Th><Th>Action</Th>
              <Th>Yes¢</Th><Th>Entry¢</Th><Th>Contracts</Th><Th>Cost</Th>
              <Th>Conf</Th><Th>Risk</Th><Th>Status</Th><Th>Result</Th><Th right>P&amp;L</Th>
            </tr>
          </thead>
          <tbody>
            {loading && <tr><td colSpan={14} className="text-zinc-500 text-center py-4">Loading…</td></tr>}
            {!loading && visible.length === 0 && (
              <tr><td colSpan={14} className="text-zinc-500 text-center py-6 italic">No trades match.</td></tr>
            )}
            {visible.map((t) => (
              <Row
                key={t.id}
                t={t}
                expanded={expanded === t.id}
                onToggle={() => setExpanded(expanded === t.id ? null : t.id)}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function FilterPill<T extends string>({
  value, current, onChange, children,
}: { value: T; current: T; onChange: (v: T) => void; children: React.ReactNode }) {
  const active = value === current
  return (
    <button
      onClick={() => onChange(value)}
      className={`text-xs px-2 py-1 rounded transition-colors ${
        active ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
      }`}
    >
      {children}
    </button>
  )
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={`px-3 py-2 font-medium ${right ? 'text-right' : 'text-left'}`}>{children}</th>
}

function Row({ t, expanded, onToggle }: { t: Trade; expanded: boolean; onToggle: () => void }) {
  const pnlColor = t.pnl_usd == null ? 'text-zinc-500' : t.pnl_usd >= 0 ? 'text-emerald-400' : 'text-rose-400'
  const statusBadge =
    t.status === 'EVALUATED' ? 'bg-zinc-700/40 text-zinc-200'
    : 'bg-amber-900/40 text-amber-300'

  return (
    <>
      <tr
        onClick={onToggle}
        className="border-b border-zinc-900 hover:bg-zinc-900/40 cursor-pointer"
      >
        <td className="px-3 py-2 text-zinc-500">{t.id}</td>
        <td className="px-3 py-2 text-zinc-400 whitespace-nowrap">{shortDate(t.logged_at)}</td>
        <td className="px-3 py-2 text-zinc-400">{t.market_type === 'GAME_WINNER' ? 'GAME' : 'PROP'}</td>
        <td className="px-3 py-2 font-mono text-zinc-200">{t.ticker}</td>
        <td className="px-3 py-2 text-zinc-300">{t.action}</td>
        <td className="px-3 py-2 text-zinc-300">{t.yes_price}</td>
        <td className="px-3 py-2 text-zinc-300">{t.entry_cents}</td>
        <td className="px-3 py-2 text-zinc-300">{t.contracts}</td>
        <td className="px-3 py-2 text-zinc-300">${t.cost_usd?.toFixed(2)}</td>
        <td className="px-3 py-2 text-zinc-400">{t.confidence ?? '—'}</td>
        <td className="px-3 py-2 text-zinc-400">{t.risk_score ?? '—'}</td>
        <td className="px-3 py-2">
          <span className={`px-1.5 py-0.5 text-[10px] rounded ${statusBadge}`}>
            {t.status === 'EVALUATED' ? 'EVAL' : 'PEND'}
          </span>
        </td>
        <td className="px-3 py-2 text-zinc-400">{t.result ?? '—'}</td>
        <td className={`px-3 py-2 text-right font-mono ${pnlColor}`}>
          {t.pnl_usd == null ? '—' : `${t.pnl_usd >= 0 ? '+' : ''}$${t.pnl_usd.toFixed(2)}`}
        </td>
      </tr>
      {expanded && (
        <tr className="bg-zinc-950/60 border-b border-zinc-900">
          <td colSpan={14} className="p-4">
            <div className="grid grid-cols-2 gap-4 text-[12px]">
              <div>
                <h4 className="text-zinc-300 font-semibold mb-1">Market</h4>
                <Field k="Title" v={t.market_title} />
                <Field k="Player" v={t.player_name} />
                <Field k="Prop line" v={t.prop_threshold} />
                <Field k="Side" v={t.side} />
              </div>
              <div>
                <h4 className="text-zinc-300 font-semibold mb-1">Decision</h4>
                <Field k="Kelly" v={t.kelly} />
                <Field k="Calibration gap" v={t.calibration_gap} />
                <Field k="Sample size" v={t.sample_size} />
                <Field k="Verdict" v={t.verdict} />
                <Field k="Concerns" v={t.concerns} />
                <Field k="Evaluated at" v={t.evaluated_at} />
                <Field k="Payout" v={t.payout_usd != null ? `$${t.payout_usd.toFixed(2)}` : null} />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

function Field({ k, v }: { k: string; v: any }) {
  if (v == null || v === '') return null
  return (
    <div className="flex gap-2">
      <span className="text-zinc-500 w-32 shrink-0">{k}</span>
      <span className="text-zinc-200 break-all">{String(v)}</span>
    </div>
  )
}

function shortDate(iso: string): string {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
