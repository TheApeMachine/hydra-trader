from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InstrumentAsset:
    """Asset reference data as streamed on the `instrument` channel."""

    id: str
    status: str
    precision: int
    precision_display: int
    borrowable: bool
    collateral_value: float
    margin_rate: float | None
    multiplier: float | None

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> InstrumentAsset:
        return cls(
            id=str(d["id"]),
            status=str(d["status"]),
            precision=int(d["precision"]),
            precision_display=int(d["precision_display"]),
            borrowable=bool(d["borrowable"]),
            collateral_value=float(d["collateral_value"]),
            margin_rate=_opt_float(d.get("margin_rate")),
            multiplier=_opt_float(d.get("multiplier")),
        )


@dataclass(frozen=True, slots=True)
class InstrumentPair:
    """Trading-pair reference data as streamed on the `instrument` channel."""

    symbol: str
    base: str
    quote: str
    status: str
    qty_precision: int | None
    qty_increment: float | None
    price_precision: int | None
    cost_precision: int | None
    price_increment: float | None
    qty_min: float | None
    cost_min: str | None
    marginable: bool | None
    margin_initial: float | None
    position_limit_long: int | None
    position_limit_short: int | None
    has_index: bool | None
    tick_size: float | None

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> InstrumentPair:
        return cls(
            symbol=str(d["symbol"]),
            base=str(d["base"]),
            quote=str(d["quote"]),
            status=str(d["status"]),
            qty_precision=_opt_int(d.get("qty_precision")),
            qty_increment=_opt_float(d.get("qty_increment")),
            price_precision=_opt_int(d.get("price_precision")),
            cost_precision=_opt_int(d.get("cost_precision")),
            price_increment=_opt_float(d.get("price_increment")),
            qty_min=_opt_float(d.get("qty_min")),
            cost_min=_opt_str(d.get("cost_min")),
            marginable=_opt_bool(d.get("marginable")),
            margin_initial=_opt_float(d.get("margin_initial")),
            position_limit_long=_opt_int(d.get("position_limit_long")),
            position_limit_short=_opt_int(d.get("position_limit_short")),
            has_index=_opt_bool(d.get("has_index")),
            tick_size=_opt_float(d.get("tick_size")),
        )


@dataclass(frozen=True, slots=True)
class InstrumentMessage:
    """Envelope around the `instrument` channel; payload is an object with `assets` + `pairs` lists."""

    is_snapshot: bool
    assets: tuple[InstrumentAsset, ...]
    pairs: tuple[InstrumentPair, ...]

    @classmethod
    def from_kraken(cls, msg: dict[str, Any]) -> InstrumentMessage:
        if msg.get("channel") != "instrument":
            raise ValueError(f"InstrumentMessage: expected channel='instrument', got {msg.get('channel')!r}")

        mtype = msg.get("type")

        if mtype not in ("snapshot", "update"):
            raise ValueError(f"InstrumentMessage: unexpected type {mtype!r}")

        data = msg.get("data") or {}

        if not isinstance(data, dict):
            raise ValueError(f"InstrumentMessage: data must be an object, got {type(data).__name__}")

        return cls(
            is_snapshot=(mtype == "snapshot"),
            assets=tuple(InstrumentAsset.from_kraken(a) for a in data.get("assets") or ()),
            pairs=tuple(InstrumentPair.from_kraken(p) for p in data.get("pairs") or ()),
        )


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _opt_int(v: Any) -> int | None:
    return None if v is None else int(v)


def _opt_bool(v: Any) -> bool | None:
    return None if v is None else bool(v)


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)
