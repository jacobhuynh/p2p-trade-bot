import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Background, Controls, ReactFlow, Handle, Position,
  type Node, type Edge, type NodeProps,
} from '@xyflow/react'
import { api, type FullDecision } from '../lib/api'

type StageKey =
  | 'router' | 'bouncer' | 'propparse'
  | 'quant' | 'propa' | 'sentiment'
  | 'orchestrator' | 'critic' | 'logger'

interface StageView {
  key: StageKey
  label: string
  status: 'ok' | 'skipped' | 'veto' | 'approved' | 'pass'
  oneLiner: string
  payload: Record<string, any>
}

export default function TradeDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [data, setData] = useState<FullDecision | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [openStage, setOpenStage] = useState<StageKey | null>(null)

  useEffect(() => {
    if (!id) return
    setData(null); setError(null)
    api.decision(id).then(setData).catch((e) => setError(String(e)))
  }, [id])

  const stages = useMemo(() => (data ? buildStages(data) : []), [data])

  const { nodes, edges } = useMemo(
    () => buildGraph(stages, openStage, data?.market_type ?? 'GAME_WINNER'),
    [stages, openStage, data?.market_type],
  )

  if (error) {
    return <div className="p-6 text-rose-400">Error: {error} — <Link to="/live" className="underline">back to live</Link></div>
  }
  if (!data) {
    return <div className="p-6 text-zinc-400">Loading…</div>
  }

  const status = data.status
  const statusColor =
    status === 'APPROVED' ? 'bg-emerald-500/15 text-emerald-300 border-emerald-700/40'
    : status === 'VETOED' ? 'bg-rose-500/15 text-rose-300 border-rose-700/40'
    : 'bg-zinc-500/15 text-zinc-300 border-zinc-700/40'

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-3 border-b border-zinc-800 flex items-center gap-3 bg-zinc-950/40">
        <Link to="/live" className="text-zinc-500 hover:text-zinc-300 text-sm">← live</Link>
        <span className={`px-2 py-0.5 text-xs font-semibold border rounded ${statusColor}`}>{status}</span>
        <span className="text-[11px] uppercase tracking-wider text-zinc-500">{data.market_type}</span>
        <code className="text-sm text-zinc-200">{data.ticker}</code>
        <div className="ml-auto text-xs text-zinc-500">
          {data.decision.action ?? '—'} @ {data.decision.price ?? '?'}¢ ·
          conf={data.decision.confidence ?? '?'} ·
          edge={data.decision.edge != null ? (data.decision.edge * 100).toFixed(2) + '%' : '?'} ·
          kelly={data.decision.kelly_fraction ?? '?'}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 relative min-h-0">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodeClick={(_, n) => setOpenStage(n.id as StageKey)}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={24} color="#1f2128" />
            <Controls className="!bg-zinc-900 !border-zinc-800" />
          </ReactFlow>
        </div>
        {openStage && (
          <Drawer
            stage={stages.find((s) => s.key === openStage)!}
            onClose={() => setOpenStage(null)}
          />
        )}
      </div>
    </div>
  )
}

// ── Stage extraction ────────────────────────────────────────────────────────

function buildStages(d: FullDecision): StageView[] {
  return d.market_type === 'PLAYER_PROP' ? buildPropStages(d) : buildGameStages(d)
}

function routerStage(d: FullDecision, tp: any): StageView {
  return {
    key: 'router',
    label: 'Router',
    status: 'ok',
    oneLiner: `classified ${d.market_type}`,
    payload: { ticker: d.ticker, market_type: d.market_type, yes_price: tp.market_price },
  }
}

function sentimentStage(tp: any, dec: any): StageView {
  const sentiment = tp.sentiment_context || dec.sentiment_context
  return {
    key: 'sentiment',
    label: 'Sentiment',
    status: sentiment ? 'ok' : 'skipped',
    oneLiner: sentiment ? firstLine(sentiment) : '(no news / skipped)',
    payload: { sentiment_context: sentiment ?? null },
  }
}

function orchestratorStage(dec: any): StageView {
  return {
    key: 'orchestrator',
    label: 'Orchestrator',
    status: dec.status === 'PASS' ? 'pass' : 'ok',
    oneLiner: `${dec.confidence ?? '?'} · edge=${fmtPct(dec.edge)} · ${dec.reason ?? ''}`.slice(0, 90),
    payload: {
      confidence: dec.confidence,
      edge: dec.edge,
      kelly_fraction: dec.kelly_fraction,
      action: dec.action,
      price: dec.price,
      reason: dec.reason,
      status: dec.status,
    },
  }
}

function criticStage(dec: any): StageView {
  const critic = dec.critic || {}
  if (Object.keys(critic).length === 0) {
    return {
      key: 'critic', label: 'Critic', status: 'skipped',
      oneLiner: '(short-circuited at orchestrator)', payload: {},
    }
  }
  return {
    key: 'critic',
    label: 'Critic',
    status: critic.decision === 'VETO' ? 'veto' : critic.decision === 'APPROVE' ? 'approved' : 'ok',
    oneLiner: `${critic.decision ?? '?'} · risk ${critic.risk_score ?? '?'}/10${critic.veto_reason ? ` — ${critic.veto_reason}` : ''}`.slice(0, 100),
    payload: critic,
  }
}

function loggerStage(d: FullDecision, dec: any): StageView {
  return {
    key: 'logger',
    label: 'Logger',
    status: dec.trade_id != null ? 'approved' : 'skipped',
    oneLiner: dec.trade_id != null ? `Logged as trade #${dec.trade_id}` : 'Not logged',
    payload: { trade_id: dec.trade_id ?? null, status: d.status },
  }
}

function buildGameStages(d: FullDecision): StageView[] {
  const tp = d.trade_packet || {}
  const dec = d.decision || {}
  const quant = dec.quant_summary || {}

  return [
    routerStage(d, tp),
    {
      key: 'bouncer',
      label: 'Bouncer',
      status: 'ok',
      oneLiner: `${tp.action ?? '—'} @ ${tp.market_price ?? '?'}¢ — ${tp.reason ?? 'longshot'}`,
      payload: {
        action: tp.action,
        market_price: tp.market_price,
        reason: tp.reason,
        market_title: tp.market_title,
        rules_primary: tp.rules_primary,
        live_open_interest: tp.live_open_interest,
        live_volume_24h: tp.live_volume_24h,
      },
    },
    {
      key: 'quant',
      label: 'GameQuantAgent',
      status: 'ok',
      oneLiner: `gap=${fmtPct(quant.calibration_gap)} · n=${quant.sample_size ?? '?'} · ${quant.verdict ?? '?'}`,
      payload: quant,
    },
    sentimentStage(tp, dec),
    orchestratorStage(dec),
    criticStage(dec),
    loggerStage(d, dec),
  ]
}

function buildPropStages(d: FullDecision): StageView[] {
  const tp = d.trade_packet || {}
  const dec = d.decision || {}
  const quant = dec.quant_summary || {}

  const propParseLabel = `${tp.player_name ?? '?'} · ${tp.prop_type ?? '?'} ${tp.prop_threshold ?? '?'}+ · ${tp.action ?? '—'} @ ${tp.market_price ?? '?'}¢`

  return [
    routerStage(d, tp),
    {
      key: 'propparse',
      label: 'Bouncer',
      status: 'ok',
      oneLiner: propParseLabel,
      payload: {
        player_name: tp.player_name,
        prop_type: tp.prop_type,
        prop_threshold: tp.prop_threshold,
        action: tp.action,
        market_price: tp.market_price,
        market_title: tp.market_title,
        live_open_interest: tp.live_open_interest,
      },
    },
    {
      key: 'propa',
      label: 'PropAgent',
      status: 'ok',
      oneLiner: `hit=${fmtPct(quant.actual_win_rate ?? quant.hit_rate)} · avg=${quant.recent_avg ?? '?'} vs ${quant.prop_threshold ?? '?'} · n=${quant.sample_size ?? quant.n_games_sampled ?? '?'}`,
      payload: quant,
    },
    sentimentStage(tp, dec),
    orchestratorStage(dec),
    criticStage(dec),
    loggerStage(d, dec),
  ]
}

function fmtPct(v: any): string {
  if (v == null) return '?'
  const n = typeof v === 'string' ? parseFloat(v) : v
  if (Number.isNaN(n)) return '?'
  return `${(n * 100).toFixed(2)}%`
}
function firstLine(s: string): string {
  return s.split('\n').find((l) => l.trim().length > 0)?.trim().slice(0, 90) ?? s.slice(0, 90)
}

// ── React Flow graph layout ────────────────────────────────────────────────

const GAME_POSITIONS: Record<string, { x: number; y: number }> = {
  router:       { x:    0, y: 200 },
  bouncer:      { x:  220, y: 200 },
  quant:        { x:  470, y:  90 },
  sentiment:    { x:  470, y: 310 },
  orchestrator: { x:  730, y: 200 },
  critic:       { x:  980, y: 200 },
  logger:       { x: 1230, y: 200 },
}

const PROP_POSITIONS: Record<string, { x: number; y: number }> = {
  router:       { x:    0, y: 200 },
  propparse:    { x:  220, y: 200 },
  propa:        { x:  470, y:  90 },
  sentiment:    { x:  470, y: 310 },
  orchestrator: { x:  730, y: 200 },
  critic:       { x:  980, y: 200 },
  logger:       { x: 1230, y: 200 },
}

function buildGraph(
  stages: StageView[],
  openStage: StageKey | null,
  marketType: string,
): { nodes: Node[]; edges: Edge[] } {
  const isProp = marketType === 'PLAYER_PROP'
  const positions = isProp ? PROP_POSITIONS : GAME_POSITIONS
  const find = (k: StageKey) => stages.find((s) => s.key === k)

  const nodes: Node[] = stages.map((s) => ({
    id: s.key,
    type: 'stage',
    position: positions[s.key] ?? { x: 0, y: 0 },
    data: { stage: s, selected: openStage === s.key },
    draggable: false,
  }))

  const edges: Edge[] = []
  const e = (from: StageKey, to: StageKey, animated = false) => {
    if (find(from) && find(to)) {
      edges.push({
        id: `${from}-${to}`,
        source: from,
        target: to,
        animated,
        style: { stroke: '#3f3f46', strokeWidth: 1.5 },
      })
    }
  }

  if (isProp) {
    e('router', 'propparse', true)
    e('propparse', 'propa', true)
    e('propparse', 'sentiment', true)
    e('propa', 'orchestrator', true)
    e('sentiment', 'orchestrator', true)
    e('orchestrator', 'critic', true)
    e('critic', 'logger', true)
  } else {
    e('router', 'bouncer', true)
    e('bouncer', 'quant', true)
    e('bouncer', 'sentiment', true)
    e('quant', 'orchestrator', true)
    e('sentiment', 'orchestrator', true)
    e('orchestrator', 'critic', true)
    e('critic', 'logger', true)
  }

  return { nodes, edges }
}

// ── Stage node renderer ─────────────────────────────────────────────────────

const NODE_TYPES = { stage: StageNode }

function StageNode({ data }: NodeProps) {
  const stage = (data as any).stage as StageView
  const selected = (data as any).selected as boolean
  const tone = stageTone(stage.status)
  return (
    <div
      className={`rounded-lg border p-2 w-56 shadow-md cursor-pointer transition-colors ${tone} ${
        selected ? 'ring-2 ring-amber-400/60' : ''
      }`}
    >
      <Handle type="target" position={Position.Left} className="!bg-zinc-700" />
      <Handle type="source" position={Position.Right} className="!bg-zinc-700" />
      <div className="flex items-baseline gap-2">
        <span className="font-semibold text-sm text-zinc-100">{stage.label}</span>
        <span className="ml-auto text-[10px] uppercase tracking-wider opacity-70">{stage.status}</span>
      </div>
      <div className="text-[11px] mt-1 text-zinc-300/90 leading-snug">{stage.oneLiner}</div>
    </div>
  )
}

function stageTone(s: StageView['status']): string {
  switch (s) {
    case 'veto':     return 'bg-rose-950/40 border-rose-800'
    case 'approved': return 'bg-emerald-950/40 border-emerald-800'
    case 'pass':     return 'bg-zinc-900 border-zinc-700'
    case 'skipped':  return 'bg-zinc-900/40 border-zinc-800 opacity-70'
    default:         return 'bg-zinc-900 border-zinc-700'
  }
}

// ── Drawer ──────────────────────────────────────────────────────────────────

function Drawer({ stage, onClose }: { stage: StageView; onClose: () => void }) {
  return (
    <aside className="w-[420px] shrink-0 border-l border-zinc-800 bg-zinc-950 flex flex-col min-h-0 h-full">
      <div className="px-4 py-3 border-b border-zinc-800 flex items-center gap-2 shrink-0">
        <h3 className="font-semibold text-zinc-100">{stage.label}</h3>
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">{stage.status}</span>
        <button onClick={onClose} className="ml-auto text-zinc-500 hover:text-zinc-200">✕</button>
      </div>
      <div className="px-4 py-2 text-xs text-zinc-400 shrink-0">{stage.oneLiner}</div>
      <div className="flex-1 min-h-0 overflow-auto">
        <pre className="px-4 py-2 text-[12px] leading-snug text-zinc-300 font-mono whitespace-pre-wrap break-words">
{JSON.stringify(stage.payload, null, 2)}
        </pre>
      </div>
    </aside>
  )
}
