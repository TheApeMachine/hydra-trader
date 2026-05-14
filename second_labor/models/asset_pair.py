from __future__ import annotations

from typing import Any


class AssetPair:
    def __init__(self) -> None:
        # Pair names and their info
        # Trading Asset Pair
        self.name: str
        # Alternate pair name
        self.altname: str
        # WebSocket pair name (if available)
        self.wsname: str
        # Asset class of base component
        self.aclass_base: str
        # Asset ID of base component
        self.base: str
        # Asset class of quote component
        self.aclass_quote: str
        # Asset ID of quote component
        self.quote: str
        # Execution venue where the order book for this pair is listed
        # Possible values: international, bitnomial_exchange
        self.execution_venue: str
        # deprecated — volume lot size
        self.lot: str
        # Number of decimal places for prices in this pair
        self.pair_decimals: int
        # Number of decimal places for cost of trades in pair (quote asset terms)
        self.cost_decimals: int
        # Number of decimal places for volume (base asset terms)
        self.lot_decimals: int
        # Amount to multiply lot volume by to get currency volume
        self.lot_multiplier: int
        # Array of leverage amounts available when buying
        self.leverage_buy: list[int]
        # Array of leverage amounts available when selling
        self.leverage_sell: list[int]
        # Fee schedule array in [<volume>, <percent fee>] tuples
        self.fees: list[tuple[float, float]]
        # Maker fee schedule array in [<volume>, <percent fee>] tuples (if on maker/taker)
        self.fees_maker: list[tuple[float, float]]
        # Volume discount currency
        self.fee_volume_currency: str
        # Margin call level
        self.margin_call: int
        # Stop-out/liquidation margin level
        self.margin_stop: int
        # Minimum order size (in terms of base currency)
        self.ordermin: str
        # Minimum order cost (in terms of quote currency)
        self.costmin: str
        # Minimum increment between valid price levels
        self.tick_size: str
        # Status of asset: online, cancel_only, post_only, limit_only, reduce_only
        self.status: str
        # Maximum long margin position size (in terms of base currency)
        self.long_position_limit: int
        # Maximum short margin position size (in terms of base currency)
        self.short_position_limit: int
        # Error strings when applicable
        self.error: list[str]

    @classmethod
    def from_kraken(cls, name: str, d: dict[str, Any]) -> AssetPair:
        def fee_tiers(rows: Any) -> list[tuple[float, float]]:
            if not isinstance(rows, list):
                return []
            out: list[tuple[float, float]] = []
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    out.append((float(row[0]), float(row[1])))
            return out

        o = cls.__new__(cls)
        o.name = name
        o.altname = str(d["altname"])
        o.wsname = str(d.get("wsname") or "")
        o.aclass_base = str(d["aclass_base"])
        o.base = str(d["base"])
        o.aclass_quote = str(d["aclass_quote"])
        o.quote = str(d["quote"])
        o.execution_venue = str(d.get("execution_venue") or "")
        o.lot = str(d.get("lot") or "")
        o.pair_decimals = int(d["pair_decimals"])
        o.cost_decimals = int(d["cost_decimals"])
        o.lot_decimals = int(d["lot_decimals"])
        o.lot_multiplier = int(d["lot_multiplier"])
        o.leverage_buy = [int(x) for x in d.get("leverage_buy", [])]
        o.leverage_sell = [int(x) for x in d.get("leverage_sell", [])]
        o.fees = fee_tiers(d.get("fees"))
        o.fees_maker = fee_tiers(d.get("fees_maker"))
        o.fee_volume_currency = str(d["fee_volume_currency"])
        o.margin_call = int(d["margin_call"])
        o.margin_stop = int(d["margin_stop"])
        o.ordermin = str(d["ordermin"])
        o.costmin = str(d["costmin"])
        o.tick_size = str(d["tick_size"])
        o.status = str(d["status"])
        o.long_position_limit = int(d.get("long_position_limit", 0))
        o.short_position_limit = int(d.get("short_position_limit", 0))
        errs = d.get("error", [])
        o.error = [str(e) for e in errs] if isinstance(errs, list) else []
        return o
