import os
import time
import base64
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def _get_auth_headers(method: str, path: str) -> dict:
    api_key_id       = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    timestamp  = str(int(time.time() * 1000))
    msg_string = f"{timestamp}{method}{path}"

    signature = private_key.sign(
        msg_string.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

    return {
        "KALSHI-ACCESS-KEY":       api_key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type":            "application/json",
    }


def get_orderbook(ticker: str) -> dict | None:
    """
    Fetch the live order book for a ticker.

    Returns a dict with "yes" and "no" keys, each a list of [price, size] pairs
    representing available contracts at each price level.
    Example: {"yes": [[14, 50], [13, 120]], "no": [[86, 50], [87, 120]]}

    Returns None (without raising) when:
      - Credentials are not set
      - Any network / HTTP error occurs
    When None, callers should treat depth as unknown and skip liquidity enforcement.
    """
    api_key_id       = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    if not api_key_id or not private_key_path:
        return None

    try:
        path     = f"/markets/{ticker}/orderbook"
        headers  = _get_auth_headers("GET", path)
        response = requests.get(f"{BASE_URL}{path}", headers=headers)
        response.raise_for_status()
        data = response.json()
        # Kalshi returns {"orderbook": {"yes": [...], "no": [...]}}
        return data.get("orderbook", data)
    except Exception as e:
        print(f"[kalshi_rest] Orderbook error for {ticker}: {e}")
        return None


def get_market_details(ticker: str) -> dict | None:
    """
    Fetch full market details for a given ticker.

    Returns None (without raising) when:
      - KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH are not set
      - The key file does not exist
      - Any network / HTTP error occurs

    Callers (e.g. bouncer.py) already handle a None return by substituting
    'Unknown' fields, so this is safe for offline / test use.
    """
    api_key_id       = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    if not api_key_id or not private_key_path:
        return None  # no credentials configured — offline / rule mode

    try:
        path     = f"/markets/{ticker}"
        headers  = _get_auth_headers("GET", path)
        response = requests.get(f"{BASE_URL}{path}", headers=headers)
        response.raise_for_status()
        return response.json().get("market")
    except Exception as e:
        print(f"[kalshi_rest] REST error for {ticker}: {e}")
        return None
