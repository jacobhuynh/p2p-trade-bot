import asyncio
import json
import os
import time
import base64
import websockets
from datetime import datetime
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from src.pipeline import router
from src.agents.orchestrator import LeadAnalyst
from src.execution.trade_logger import TradeLogger

load_dotenv()

class KalshiWebsocketClient:
    def __init__(self):
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self.uri = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self.private_key = self._load_private_key()
        self.analyst = LeadAnalyst()
        self.logger = TradeLogger()

    def _load_private_key(self):
        with open(self.private_key_path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(), password=None
            )

    def _generate_auth_headers(self):
        timestamp = str(int(time.time() * 1000))
        msg_string = f"{timestamp}GET/trade-api/ws/v2"
        signature = self.private_key.sign(
            msg_string.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode('utf-8'),
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }

    async def connect(self):
        headers = self._generate_auth_headers()
        async with websockets.connect(
            self.uri,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=30,
        ) as websocket:
            print("✅ Connected to Kalshi Stream.")
            await websocket.send(json.dumps({
                "id": 1, "cmd": "subscribe", "params": {"channels": ["trade"]}
            }))
            print("📡 Listening for NBA trades (Game Winners → full pipeline | Totals/Props → placeholder)...")

            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(data)

    async def handle_message(self, data):
        if data.get("type") == "trade":
            payload = data.get("msg", {})
            market_type, trade_packet = router.route(payload)

            if market_type == "GAME_WINNER" and trade_packet:
                # ── Full longshot pipeline: Quant → Orchestrator → Critic ────
                timestamp = datetime.now().strftime("%H:%M:%S")
                decision  = await asyncio.to_thread(self.analyst.analyze_signal, trade_packet)
                quant     = decision.get("quant_summary", {})
                critic    = decision.get("critic", {})
                status    = decision.get("status")

                status_emoji = {
                    "APPROVED": "✅",
                    "VETOED":   "🚫",
                    "PASS":     "⏭️",
                }.get(status, "❓")

                print(f"\n{'='*60}")
                print(f"🚨 GAME WINNER | {timestamp}  {status_emoji} {status}")
                print(f"{'='*60}")
                print(f"📌 Ticker:      {trade_packet['ticker']}")
                print(f"📋 Title:       {trade_packet.get('market_title', 'N/A')}")
                print(f"📝 Rules:       {trade_packet.get('rules_primary', 'N/A')}")
                print(f"💰 Price:       {decision['price']}¢")
                print(f"🎯 Action:      {decision['action']}")
                print(f"{'─'*60}")
                sc = trade_packet.get("sentiment_context")
                print(f"📰 Sentiment (live ESPN context)")
                if sc:
                    for line in (sc or "").strip().splitlines():
                        print(f"   {line}")
                else:
                    print(f"   (no data or skipped)")
                print(f"{'─'*60}")
                print(f"📊 Quant")
                print(f"   Gap:           {quant.get('calibration_gap')}")
                print(f"   Win Rate:      {quant.get('actual_win_rate')}")
                print(f"   Implied Prob:  {quant.get('implied_prob')}")
                print(f"   Sample:        {quant.get('sample_size')}  [{quant.get('data_quality')}]")
                print(f"   Verdict:       {quant.get('verdict')}")
                print(f"   NO Win Rate:   {quant.get('no_win_rate')}  (longshot bias)")
                print(f"   Asymmetry:     {quant.get('yes_no_asymmetry')}")
                pb  = quant.get("price_bucket_edge") or {}
                lb  = quant.get("longshot_bias") or {}
                tw  = quant.get("taker_win_rate") or {}
                inv = quant.get("inverse_bucket") or {}
                print(f"   Price Bucket:  win={pb.get('actual_win_rate')}  edge={pb.get('edge')}  n={pb.get('sample_size')}")
                print(f"   Taker WR:      win={tw.get('win_rate')}  n={tw.get('sample_size')}")
                print(f"   Longshot Bias: no_wr={lb.get('no_win_rate')}  avg_price={lb.get('avg_price')}  n={lb.get('sample_size')}")
                print(f"   Inverse Bkt:   win={inv.get('actual_win_rate')}  edge={inv.get('edge')}  n={inv.get('sample_size')}")
                gc = quant.get("game_context")
                if gc:
                    score_str = f"{gc.get('away_abbr')} {gc.get('away_score')} @ {gc.get('home_abbr')} {gc.get('home_score')}"
                    status_str = gc.get("status", "").replace("STATUS_", "")
                    winner_str = f"  Winner: {gc.get('winner_abbr')}" if gc.get("winner_abbr") else ""
                    print(f"   ESPN:          {score_str}  [{status_str}]{winner_str}")
                else:
                    print(f"   ESPN:          no data")
                ts = quant.get("team_stats")
                if ts:
                    h = ts.get("home") or {}
                    a = ts.get("away") or {}
                    print(f"   NBA Records:   {h.get('abbr')} last10={h.get('last10')} home={h.get('home_record')} away={h.get('away_record')}")
                    print(f"                  {a.get('abbr')} last10={a.get('last10')} home={a.get('home_record')} away={a.get('away_record')}")
                else:
                    print(f"   NBA Records:   no data")
                print(f"   Summary:       {quant.get('summary')}")
                print(f"{'─'*60}")
                print(f"🧠 Orchestrator")
                print(f"   Confidence:  {decision['confidence']}")
                print(f"   Edge:        {decision['edge']}")
                print(f"   Kelly:       {decision['kelly_fraction']}")
                print(f"   Reason:      {decision.get('reason')}")

                if critic:
                    print(f"{'─'*60}")
                    print(f"🔍 Critic")
                    print(f"   Decision:    {critic.get('decision')}")
                    print(f"   Risk Score:  {critic.get('risk_score')}/10")
                    if critic.get("veto_reason"):
                        print(f"   Veto:        {critic.get('veto_reason')}")
                    if critic.get("concerns"):
                        for c in critic["concerns"]:
                            print(f"   ⚠️  {c}")
                    print(f"   Summary:     {critic.get('summary')}")
                    print(f"   Sentiment:   {critic.get('sentiment_note', '')}")

                if status == "APPROVED":
                    trade_id = self.logger.log_trade(decision, trade_packet)
                    print(f"{'─'*60}")
                    print(f"💾 Logged as trade #{trade_id} — run `python -m src.settle` to check P&L")

                print(f"{'='*60}")

            elif market_type in ("TOTALS", "PLAYER_PROP"):
                # Placeholder handlers print their own one-liner inside router.py.
                # Nothing more to do here until real strategies are implemented.
                pass

            else:
                # NON_NBA, UNKNOWN, or mid-price GAME_WINNER (filtered by bouncer)
                print(".", end="", flush=True)

    async def run_forever(self):
        backoff = 1
        while True:
            try:
                await self.connect()
            except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
                print(f"\n⚠️  Connection lost: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                backoff = 1

if __name__ == "__main__":
    client = KalshiWebsocketClient()
    try:
        asyncio.run(client.run_forever())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")