/** Shape of the JSON envelope broadcast by hydra/dash_ws.py. */
export interface DashEnvelope {
    ts: number;
    snap: DashSnapshot;
    error?: string;
}

export interface DashSnapshot {
    mode: string;
    params_source: string;
    market: MarketView;
    focus: string;
    chart_symbols: string[];
    cash: number;
    equity: number;
    position: Position | null;
    trades: Trade[];
    watch: [string, WatchEntry][];
    pump_events: PumpEvent[];
    candidates: Candidate[];
    btc_ok: boolean;
    btc_strict: boolean;
    btc_flush: boolean;
    auto: AutoStats;
    performance: PerformanceSnap;
    strategies: Strategy[];
    optuna: Record<string, any>;
    backtest: Record<string, any>;
    readiness: Record<string, number>;
}

export interface MarketView {
    ts: number;
    focus: string;
    price_series: [number, number][];
    tape_series: [number, number, number][];
    bids: [number, number][];
    asks: [number, number][];
    entry_marks: any[];
    exit_marks: any[];
    capital_history: [number, number][];
    micro: Record<string, number | null>;
    thrust: Record<string, number | null>;
    last_price: number;
    spread_bps: number | null;
    book_imbalance: number | null;
    flow_series: [number, number, number, number, number, number][];
    performance_history: [number, number, number, number, number, number, number, number, number, number][];
    multi_prices: Record<string, [number, number][]>;
    hawkes_history: [number, number, number, number][];
    hawkes_predictions: [number, number, number][];
    multi_hawkes_history: Record<string, [number, number, number, number][]>;
    multi_hawkes_predictions: Record<string, [number, number, number][]>;
    hawkes_learning: Record<string, HawkesLearning>;
    inertia_state: Record<string, { pressure: number; readiness: number }>;
    btc_backdrop_state: number;
}

export interface HawkesLearning {
    status?: string;
    hit_count?: number;
    n_with_move?: number;
    n_matured?: number;
    hit_rate?: number;
    k?: number;
    slope_weight?: number;
    bias?: number;
}

export interface Position {
    pair: string;
    regime: string;
    entry: number;
    entry_signal: number;
    entry_cost: number;
    entry_time: number;
    qty: number;
    stop_price?: number | null;
    trail_active?: boolean;
    note?: string;
}

export interface Trade {
    pair: string;
    regime: string;
    score: number;
    entry_time: string | number;
    exit_time: string | number;
    hold_sec: number;
    entry: number;
    exit: number;
    qty: number;
    entry_cost: number;
    reason: string;
    pnl_usd: number;
    return_pct: number;
    return_velocity_pct_per_min?: number;
    capital_after: number;
    note?: string;
}

export interface WatchEntry {
    at: number;
    spike_pct: number;
    burst_x: number;
    anchor: number;
}

export interface PumpEvent {
    symbol: string;
    at: number;
    spike_pct: number;
    burst_x: number;
    anchor: number;
}

export interface Candidate {
    symbol: string;
    score: number;
    [k: string]: any;
}

export interface AutoStats {
    dd_pct?: number;
    idle_min?: number;
    missed?: number;
    [k: string]: any;
}

export interface PerformanceSnap {
    [k: string]: any;
}

export interface Strategy {
    name: string;
    equity: number;
    start_capital: number;
    net_return_pct: number;
    trades: number;
    active: boolean;
}
