import { NavLink, Outlet } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'

export default function Layout() {
  const [health, setHealth] = useState<{ pipeline_running: boolean; pipeline_error: string | null } | null>(null)

  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const h = await api.health()
        if (alive) setHealth(h)
      } catch {
        if (alive) setHealth(null)
      }
    }
    tick()
    const t = setInterval(tick, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-10">
        <div className="px-6 py-3 flex items-center gap-6">
          <div className="text-sm font-semibold tracking-wider text-zinc-200">P2P TRADE BOT</div>
          <nav className="flex gap-1 text-sm">
            <Tab to="/live">Live Feed</Tab>
            <Tab to="/trades">Logged Trades</Tab>
            <Tab to="/settle">Settle &amp; Stats</Tab>
          </nav>
          <div className="ml-auto flex items-center gap-2 text-xs">
            <MockButton type="GAME_WINNER">Mock Game</MockButton>
            <MockButton type="PLAYER_PROP">Mock Prop</MockButton>
            <span className="mx-2 text-zinc-700">|</span>
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                health?.pipeline_running ? 'bg-emerald-400' : 'bg-zinc-500'
              }`}
              title={health?.pipeline_error ?? ''}
            />
            <span className="text-zinc-400">
              {health?.pipeline_running
                ? 'Pipeline live'
                : health?.pipeline_error
                ? `Pipeline offline — ${health.pipeline_error}`
                : 'Pipeline offline'}
            </span>
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}

function Tab({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-1.5 rounded-md transition-colors ${
          isActive ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
        }`
      }
    >
      {children}
    </NavLink>
  )
}

function MockButton({ type, children }: { type: 'GAME_WINNER' | 'PLAYER_PROP'; children: React.ReactNode }) {
  const [busy, setBusy] = useState(false)
  const [hint, setHint] = useState<string | null>(null)
  const onClick = async () => {
    setBusy(true); setHint(null)
    try {
      const r = await api.mock(type)
      setHint(r.source.startsWith('replay') ? 'replayed real' : 'fixture')
    } catch (e) {
      setHint('error')
    } finally {
      setBusy(false)
      setTimeout(() => setHint(null), 2500)
    }
  }
  return (
    <button
      onClick={onClick}
      disabled={busy}
      title="Replay newest decision of this type, or hardcoded fixture"
      className={`px-2 py-1 rounded text-xs border transition-colors ${
        busy
          ? 'bg-zinc-800 text-zinc-500 border-zinc-800 cursor-wait'
          : 'bg-zinc-900 text-zinc-300 border-zinc-700 hover:bg-zinc-800 hover:text-white'
      }`}
    >
      {busy ? '⟳ ' : '▶ '}{children}{hint && <span className="ml-1 text-[10px] text-zinc-500">({hint})</span>}
    </button>
  )
}
