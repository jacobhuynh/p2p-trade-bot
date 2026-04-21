// useEvents — subscribe to /ws and accumulate the last N events.

import { useEffect, useRef, useState } from 'react'
import type { PipelineEvent } from './api'

export function useEvents(maxEvents = 500): {
  events: PipelineEvent[]
  connected: boolean
} {
  const [events, setEvents] = useState<PipelineEvent[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let cancelled = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      if (cancelled) return
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const url = `${proto}://${location.host}/ws`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) reconnectTimer = setTimeout(connect, 2000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (e) => {
        try {
          const evt = JSON.parse(e.data) as PipelineEvent
          setEvents((prev) => {
            const next = [...prev, evt]
            return next.length > maxEvents ? next.slice(next.length - maxEvents) : next
          })
        } catch {
          // ignore malformed
        }
      }
    }

    connect()
    return () => {
      cancelled = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [maxEvents])

  return { events, connected }
}
