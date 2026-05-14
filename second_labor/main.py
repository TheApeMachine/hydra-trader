import os

os.environ.setdefault("WEBSOCKETS_MAX_LINE_LENGTH", "65536")

import asyncio
import logging

from .dash_bus import DashBus
from .numeric.dynamic import DynamicValue
from .numeric.state import State
from .observer import Observer
from .signals.hawkes import HawkesSignal
from .signals.ohlc import OhlcSignal
from .signals.trade_feed import TradeFeed


SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]


async def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    bus = DashBus()
    await bus.start()

    hawkes = HawkesSignal(
        focus_pair_key="BTC/USD",
        mu=DynamicValue(initial=State({"list": [0.5]})),
        alpha=DynamicValue(initial=State({"list": [0.3]})),
        beta=DynamicValue(initial=State({"list": [1.0]})),
        publisher=bus,
    )

    ohlc = OhlcSignal(publisher=bus)
    trades = TradeFeed(publisher=bus)

    observer = Observer()
    observer.subscribe(
        ["trade.symbol", "trade.timestamp"],
        hawkes.measure,
        symbols=["BTC/USD"],
    )
    observer.subscribe(
        ["trade.symbol", "trade.side", "trade.price", "trade.qty", "trade.timestamp"],
        trades.measure,
        symbols=SYMBOLS,
    )
    observer.subscribe(
        ["ohlc.symbol", "ohlc.open", "ohlc.high", "ohlc.low", "ohlc.close",
         "ohlc.volume", "ohlc.interval", "ohlc.interval_begin"],
        ohlc.measure,
        symbols=SYMBOLS,
        channel_params={"ohlc": {"interval": 1, "snapshot": True}},
    )

    try:
        await observer.run()
    finally:
        await bus.stop()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
