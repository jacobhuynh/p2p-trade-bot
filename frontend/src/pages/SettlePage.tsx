import { useEffect, useMemo, useState } from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar, Cell,
} from 'recharts'
import { api, type Trade, type TradesSummary } from '../lib/api'

export default function SettlePage() {
  const [summary, setSummary] = useState<TradesSummary | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const [running, setRunning] = useState(false)
  const [output, setOutput] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    try {
      const [s, t] = await Promise.all([api.tradesSummary(), api.trades('evaluated')])
      setSummary(s); setTrades(t)
    } catch (e) {
      setError(String(e))
    }
  }

  useEffect(() => { refresh() }, [])

  const handleSettle = async () => {
    setRunning(true); setOutput(''); setError(null)
    try {
      const text = await api.settle()
      setOutput(text)
      await refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  const equity = useMemo(() => buildEquityCurve(trades), [trades])
  const byConf = useMemo(() => bucketByConfidence(trades), [trades])

  return (
    <div className="h-full overflow-auto p-6 space-y-6">
      <header className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-zinc-100">Settlement &amp; Stats</h2>
        <button
          onClick={handleSettle}
          disabled={running}
          className={`ml-auto px-4 py-2 rounded text-sm font-medium ${
            running ? 'bg-zinc-700 text-zinc-400 cursor-wait' : 'bg-emerald-600 hover:bg-emerald-500 text-white'
          }`}
        >
          {running ? '⟳ Resolving…' : '▶ Run settle'}
        </button>
      </header>

      <div className="grid grid-cols-4 gap-3">
        <Stat label="Trades evaluated" value={summary?.n_trades ?? '—'} />
        <Stat label="Win rate" value={summary ? `${(summary.win_rate * 100).toFixed(1)}%` : '—'}
              hint={summary ? `${summary.n_wins}/${summary.n_trades}` : undefined} />
        <Stat
          label="Total P&L"
          value={summary ? `${summary.total_pnl >= 0 ? '+' : ''}$${summary.total_pnl.toFixed(2)}` : '—'}
          tone={summary ? (summary.total_pnl >= 0 ? 'pos' : 'neg') : undefined}
        />
        <Stat
          label="ROI"
          value={summary ? `${summary.roi >= 0 ? '+' : ''}${(summary.roi * 100).toFixed(1)}%` : '—'}
          hint={summary ? `staked $${summary.total_staked.toFixed(2)}` : undefined}
          tone={summary ? (summary.roi >= 0 ? 'pos' : 'neg') : undefined}
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Card title="Equity curve" subtitle="cumulative P&L by evaluation time" className="col-span-2">
          {equity.length === 0 ? (
            <Empty>No evaluated trades yet — run settle once trades resolve.</Empty>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={equity}>
                <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
                <XAxis dataKey="i" stroke="#71717a" tick={{ fontSize: 11 }} />
                <YAxis stroke="#71717a" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }}
                  formatter={(v: any) => [`$${Number(v).toFixed(2)}`, 'cum P&L']}
                />
                <Line type="monotone" dataKey="cum" stroke="#34d399" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="P&L by confidence" subtitle="aggregated by orchestrator confidence">
          {byConf.length === 0 ? (
            <Empty>No data.</Empty>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={byConf}>
                <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
                <XAxis dataKey="confidence" stroke="#71717a" tick={{ fontSize: 11 }} />
                <YAxis stroke="#71717a" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }}
                  formatter={(v: any, _n, p: any) => [`$${Number(v).toFixed(2)} (${p.payload.n} trades)`, 'P&L']}
                />
                <Bar dataKey="pnl">
                  {byConf.map((b, i) => (
                    <Cell key={i} fill={b.pnl >= 0 ? '#34d399' : '#fb7185'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <Card title="Settle output" subtitle="latest run of src/settle.py">
        {error && <div className="text-rose-400 text-xs mb-2">{error}</div>}
        <pre className="text-[12px] text-zinc-300 font-mono whitespace-pre-wrap break-words bg-zinc-950 border border-zinc-800 rounded p-3 max-h-96 overflow-auto">
          {output || (running ? 'Running…' : '(click Run settle)')}
        </pre>
      </Card>
    </div>
  )
}

function Stat({ label, value, hint, tone }: {
  label: string; value: React.ReactNode; hint?: string; tone?: 'pos' | 'neg'
}) {
  const valueColor = tone === 'pos' ? 'text-emerald-400' : tone === 'neg' ? 'text-rose-400' : 'text-zinc-100'
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
      <div className="text-[11px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${valueColor}`}>{value}</div>
      {hint && <div className="text-xs text-zinc-500 mt-0.5">{hint}</div>}
    </div>
  )
}

function Card({
  title, subtitle, children, className,
}: { title: string; subtitle?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 ${className ?? ''}`}>
      <div className="mb-2">
        <h3 className="text-sm font-semibold text-zinc-200">{title}</h3>
        {subtitle && <div className="text-[11px] text-zinc-500">{subtitle}</div>}
      </div>
      {children}
    </section>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-zinc-500 italic text-sm py-12 text-center">{children}</div>
}

function buildEquityCurve(trades: Trade[]): Array<{ i: number; cum: number }> {
  const sorted = [...trades]
    .filter((t) => t.evaluated_at && t.pnl_usd != null)
    .sort((a, b) => (a.evaluated_at! < b.evaluated_at! ? -1 : 1))
  let cum = 0
  return sorted.map((t, i) => {
    cum += t.pnl_usd ?? 0
    return { i: i + 1, cum: Number(cum.toFixed(2)) }
  })
}

function bucketByConfidence(trades: Trade[]): Array<{ confidence: string; pnl: number; n: number }> {
  const buckets: Record<string, { pnl: number; n: number }> = {}
  for (const t of trades) {
    if (t.pnl_usd == null) continue
    const k = t.confidence || 'UNKNOWN'
    buckets[k] ??= { pnl: 0, n: 0 }
    buckets[k].pnl += t.pnl_usd
    buckets[k].n += 1
  }
  const order = ['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']
  return Object.entries(buckets)
    .map(([confidence, v]) => ({ confidence, pnl: Number(v.pnl.toFixed(2)), n: v.n }))
    .sort((a, b) => order.indexOf(a.confidence) - order.indexOf(b.confidence))
}
