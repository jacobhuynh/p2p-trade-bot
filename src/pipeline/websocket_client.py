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
from src.pipeline import bouncer
from src.agents.orchestrator import LeadAnalyst

load_dotenv()

class KalshiWebsocketClient:
    def __init__(self):
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self.uri = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self.private_key = self._load_private_key()
        self.analyst = LeadAnalyst()

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
        async with websockets.connect(self.uri, additional_headers=headers) as websocket:
            print("✅ Connected to Kalshi Stream.")
            await websocket.send(json.dumps({
                "id": 1, "cmd": "subscribe", "params": {"channels": ["trade"]}
            }))
            print("📡 Listening for NBA Longshots (~20% contracts)...")

            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(data)

    async def handle_message(self, data):
        if data.get("type") == "trade":
            payload = data.get("msg", {})
            trade_packet = bouncer.process_trade(payload)

            if trade_packet:
                timestamp = datetime.now().strftime("%H:%M:%S")
                decision  = self.analyst.analyze_signal(trade_packet)
                quant     = decision.get("quant_summary", {})
                critic    = decision.get("critic", {})
                status    = decision.get("status")

                # Status emoji
                status_emoji = {
                    "APPROVED": "✅",
                    "VETOED":   "🚫",
                    "PASS":     "⏭️",
                }.get(status, "❓")

                print(f"\n{'='*60}")
                print(f"🚨 SIGNAL | {timestamp}  {status_emoji} {status}")
                print(f"{'='*60}")
                print(f"📌 Ticker:      {trade_packet['ticker']}")
                print(f"📋 Title:       {trade_packet.get('market_title', 'N/A')}")
                print(f"📝 Rules:       {trade_packet.get('rules_primary', 'N/A')}")
                print(f"💰 Price:       {decision['price']}¢")
                print(f"🎯 Action:      {decision['action']}")
                print(f"{'─'*60}")
                print(f"📊 Quant")
                print(f"   Gap:         {quant.get('calibration_gap')}")
                print(f"   Win Rate:    {quant.get('actual_win_rate')}")
                print(f"   Sample:      {quant.get('sample_size')}")
                print(f"   Verdict:     {quant.get('verdict')}")
                print(f"   Summary:     {quant.get('summary')}")
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

                print(f"{'='*60}")
            else:
                print(".", end="", flush=True)

if __name__ == "__main__":
    client = KalshiWebsocketClient()
    try:
        asyncio.run(client.connect())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")