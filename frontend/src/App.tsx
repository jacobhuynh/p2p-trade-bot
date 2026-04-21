import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './shell/Layout'
import LivePage from './pages/LivePage'
import TradeDetailPage from './pages/TradeDetailPage'
import TradesPage from './pages/TradesPage'
import SettlePage from './pages/SettlePage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/live" replace />} />
          <Route path="/live" element={<LivePage />} />
          <Route path="/decision/:id" element={<TradeDetailPage />} />
          <Route path="/trades" element={<TradesPage />} />
          <Route path="/settle" element={<SettlePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
