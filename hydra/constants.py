REST = "https://api.kraken.com/0/public"
WS = "wss://ws.kraken.com/v2"

TOP_N = 300
BOOK_N = 30
BOOK_DEPTH = 10
# Fallback only; live checks use ``Config.book_stale_sec``.
BOOK_STALE_SEC = 15.0

FALLBACK_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD",
    "ADA/USD", "AVAX/USD", "LINK/USD", "LTC/USD",
]
FORCE_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
BTC_SYMBOL = "BTC/USD"
STATE_LEN = 180
STATUS_EVERY = 30

DASH_CAPITAL_LEN = 1200
DASH_PRICE_LEN = 600
DASH_TAPE_LEN = 600
DASH_FLOW_LEN = 480
DASH_PERF_LEN = 1200
