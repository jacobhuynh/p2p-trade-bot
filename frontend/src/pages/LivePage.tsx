import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useEvents } from '../lib/ws'
import { api, type DecisionSummary, type PipelineEvent } from '../lib/api'

const RESOLVED_CAP = 40

export default function LivePage() {
  const { events, connected } = useEvents(800)
  const [seed, setSeed] = useState<PipelineEvent[]>([])

  // Hydrate the resolved column from disk on mount so refreshes don't clear it.
  useEffect(() => {
    let alive = true
    api.decisions(RESOLVED_CAP)
      .then((rows) => { if (alive) setSeed(rows.map(decisionToEvent)) })
      .catch(() => { /* leave seed empty; live events will still arrive */ })
    return () => { alive = false }
  }, [])

  const { stream, processing, resolved } = useMemo(() => {
    const stream: PipelineEvent[] = []
    const processingMap = new Map<string, PipelineEvent>()
    const resolvedMap = new Map<string, PipelineEvent>()

    // Seed first — live events will overwrite by decision_id when they arrive.
    for (const ev of seed) {
      if (ev.data.decision_id) resolvedMap.set(ev.data.decision_id, ev)
    }

    for (const ev of events) {
      if (ev.type === 'routed' || ev.type === 'bounced') {
        stream.push(ev)
      } else if (ev.type === 'decision_started') {
        processingMap.set(ev.data.ticker, ev)
      } else if (ev.type === 'decision_complete') {
        processingMap.delete(ev.data.ticker)
        const key = ev.data.decision_id ?? ev.id
        resolvedMap.set(key, ev)
      }
    }

    const resolved = Array.from(resolvedMap.values())
      .sort((a, b) => b.ts - a.ts)
      .slice(0, RESOLVED_CAP)

    return {
      stream: stream.slice(-80).reverse(),
      processing: Array.from(processingMap.values()).reverse(),
      resolved,
    }
  }, [events, seed])

  function decisionToEvent(d: DecisionSummary): PipelineEvent {
    return {
      id: `seed-${d.id}`,
      type: 'decision_complete',
      ts: d.ts,
      data: {
        decision_id: d.id,
        ticker: d.ticker,
        market_type: d.market_type,
        status: d.status,
        action: d.action,
        yes_price: d.price,
        confidence: d.confidence,
        edge: d.edge,
        trade_id: undefined,
      },
    }
  }

  return (
    <div className="h-full grid grid-cols-12 gap-3 p-3">
      <Column title="Stream" subtitle="incoming tickers + bouncer outcome" connected={connected} className="col-span-4">
        {stream.length === 0 && <Empty hint={connected ? 'Waiting for trades…' : 'WebSocket disconnected'} />}
        {stream.map((ev) => <StreamRow key={ev.id} ev={ev} />)}
      </Column>

      <Column title="Processing" subtitle="agents running" className="col-span-3">
        {processing.length === 0 && <Empty hint="Idle" />}
        {processing.map((ev) => <ProcessingCard key={ev.id} ev={ev} />)}
      </Column>

      <Column title="Resolved" subtitle="click for full workflow" className="col-span-5">
        {resolved.length === 0 && <Empty hint="No decisions yet" />}
        {resolved.map((ev) => <ResolvedCard key={ev.id} ev={ev} />)}
      </Column>
    </div>
  )
}

function Column({
  title, subtitle, children, className, connected,
}: {
  title: string; subtitle: string; children: React.ReactNode; className?: string; connected?: boolean
}) {
  return (
    <section className={`flex flex-col bg-zinc-950/50 border border-zinc-800 rounded-lg overflow-hidden ${className ?? ''}`}>
      <div className="px-3 py-2 border-b border-zinc-800 flex items-baseline gap-2">
        <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
        <span className="text-[11px] text-zinc-500">{subtitle}</span>
        {connected !== undefined && (
          <span className={`ml-auto text-[10px] uppercase tracking-wider ${connected ? 'text-emerald-400' : 'text-zinc-500'}`}>
            {connected ? '● live' : '○ off'}
          </span>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1">{children}</div>
    </section>
  )
}

function Empty({ hint }: { hint: string }) {
  return <div className="text-zinc-600 text-xs italic px-2 py-4">{hint}</div>
}

function marketColor(m?: string): string {
  switch (m) {
    case 'GAME_WINNER': return 'text-sky-300'
    case 'PLAYER_PROP': return 'text-amber-300'
    case 'TOTALS':      return 'text-fuchsia-300'
    default:            return 'text-zinc-500'
  }
}

function StreamRow({ ev }: { ev: PipelineEvent }) {
  const isBounced = ev.type === 'bounced'
  const market = ev.data.market_type
  const ticker = ev.data.ticker || '—'
  const yes = ev.data.yes_price
  const accepted = ev.type === 'routed' && ev.data.accepted

  return (
    <div className={`text-[12px] font-mono flex items-center gap-2 px-2 py-1 rounded ${
      isBounced ? 'bg-rose-950/30 border border-rose-900/40'
      : accepted ? 'bg-emerald-950/30 border border-emerald-900/40'
      : 'bg-zinc-900/40 border border-zinc-800/40'
    }`}>
      <span className={`w-16 shrink-0 text-[10px] uppercase tracking-wider ${marketColor(market)}`}>
        {market || '—'}
      </span>
      <span className="truncate flex-1 text-zinc-300">{ticker}</span>
      <span className="text-zinc-500 w-10 text-right">{yes != null ? `${formatPrice(yes)}¢` : ''}</span>
      <span className={`text-[10px] w-12 text-right ${
        isBounced ? 'text-rose-400'
        : accepted ? 'text-emerald-400'
        : 'text-zinc-500'
      }`}>
        {isBounced ? 'BOUNCED' : accepted ? 'ACCEPT' : 'drop'}
      </span>
    </div>
  )
}

function ProcessingCard({ ev }: { ev: PipelineEvent }) {
  const stages = ev.data.market_type === 'PLAYER_PROP'
    ? ['PropAgent', 'Sentiment', 'Orchestrator', 'Critic']
    : ['Quant', 'Sentiment', 'Orchestrator', 'Critic']
  return (
    <div className="rounded-md border border-amber-900/50 bg-amber-950/20 p-2 animate-pulse-slow">
      <div className="flex items-center gap-2 text-[12px]">
        <span className={`text-[10px] uppercase tracking-wider ${marketColor(ev.data.market_type)}`}>
          {ev.data.market_type}
        </span>
        <span className="ml-auto text-amber-300 text-[10px]">⟳ working</span>
      </div>
      <div className="font-mono text-xs text-zinc-200 mt-1 truncate">{ev.data.ticker}</div>
      {ev.data.market_title && (
        <div className="text-[11px] text-zinc-400 truncate mt-0.5">{ev.data.market_title}</div>
      )}
      {ev.data.player_name && (
        <div className="text-[11px] text-zinc-300 mt-0.5">
          {ev.data.player_name} · {ev.data.prop_type} {ev.data.prop_threshold}+
        </div>
      )}
      <div className="flex gap-1 mt-2">
        {stages.map((s) => (
          <span key={s} className="px-1.5 py-0.5 text-[10px] rounded bg-zinc-900/60 text-zinc-400">{s}</span>
        ))}
      </div>
    </div>
  )
}

function ResolvedCard({ ev }: { ev: PipelineEvent }) {
  const { status, ticker, action, yes_price, confidence, edge, decision_id, market_type, trade_id } = ev.data
  const badge =
    status === 'APPROVED' ? { color: 'bg-emerald-500/15 text-emerald-300 border-emerald-700/40', icon: '✓' }
    : status === 'VETOED' ? { color: 'bg-rose-500/15 text-rose-300 border-rose-700/40', icon: '✕' }
    : status === 'PASS'   ? { color: 'bg-zinc-500/15 text-zinc-300 border-zinc-700/40', icon: '–' }
    :                       { color: 'bg-zinc-500/10 text-zinc-400 border-zinc-700/40', icon: '?' }

  return (
    <Link
      to={`/decision/${decision_id}`}
      className="block rounded-md border border-zinc-800 hover:border-zinc-600 hover:bg-zinc-900/40 transition-colors p-2"
    >
      <div className="flex items-center gap-2">
        <span className={`px-1.5 py-0.5 text-[10px] font-semibold border rounded ${badge.color}`}>
          {badge.icon} {status}
        </span>
        <span className={`text-[10px] uppercase tracking-wider ${marketColor(market_type)}`}>{market_type}</span>
        {trade_id != null && (
          <span className="text-[10px] text-emerald-400 ml-auto">#trade {trade_id}</span>
        )}
      </div>
      <div className="font-mono text-xs text-zinc-200 mt-1 truncate">{ticker}</div>
      <div className="text-[11px] text-zinc-400 mt-1 flex gap-3">
        {action && <span>{action}</span>}
        {yes_price != null && <span>@ {yes_price}¢</span>}
        {confidence && <span>conf={confidence}</span>}
        {edge != null && <span>edge={(edge * 100).toFixed(2)}%</span>}
      </div>
    </Link>
  )
}

function formatPrice(p: number | string): string {
  const n = typeof p === 'string' ? parseFloat(p) : p
  if (n <= 1) return Math.round(n * 100).toString()
  return Math.round(n).toString()
}
