import os
import time
import base64
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def _get_auth_headers(method: str, path: str) -> dict:
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    timestamp = str(int(time.time() * 1000))
    msg_string = f"{timestamp}{method}{path}"

    signature = private_key.sign(
        msg_string.encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )

    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode('utf-8'),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }

def get_market_details(ticker: str) -> dict | None:
    """Fetch full market details for a given ticker."""
    path = f"/markets/{ticker}"
    headers = _get_auth_headers("GET", path)

    try:
        response = requests.get(f"{BASE_URL}{path}", headers=headers)
        response.raise_for_status()
        return response.json().get("market")
    except Exception as e:
        print(f"⚠️  REST error for {ticker}: {e}")
        return None