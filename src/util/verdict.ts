export const BUY_NOW_THRESHOLD = 60.0;
export const ALMOST_THRESHOLD = 20.0;
export const STAY_OUT_THRESHOLD = -40.0;
export const MARKET_FLAT_THRESHOLD_PCT = 0.1;

export type Verdict = "positive" | "neutral" | "negative";

export const labelForScore = (score: number): { label: string; verdict: Verdict } => {
    if (score >= BUY_NOW_THRESHOLD) return { label: "BUYING SOON", verdict: "positive" };
    if (score >= ALMOST_THRESHOLD) return { label: "ALMOST", verdict: "neutral" };
    if (score <= STAY_OUT_THRESHOLD) return { label: "STAY OUT", verdict: "negative" };
    return { label: "WAITING", verdict: "neutral" };
};

export const perSymbolChangeLabel = (changePct: number): "up" | "flat" | "down" => {
    if (changePct > MARKET_FLAT_THRESHOLD_PCT) return "up";
    if (changePct < -MARKET_FLAT_THRESHOLD_PCT) return "down";
    return "flat";
};

export const marketOverviewVerdict = (perSymbolChange: Record<string, number>): { mood: string; verdict: Verdict } => {
    let ups = 0;
    let downs = 0;
    for (const v of Object.values(perSymbolChange)) {
        const lbl = perSymbolChangeLabel(v);
        if (lbl === "up") ups += 1;
        else if (lbl === "down") downs += 1;
    }
    if (ups >= 2 && ups > downs) return { mood: "broad strength", verdict: "positive" };
    if (downs >= 2 && downs > ups) return { mood: "broad weakness", verdict: "negative" };
    return { mood: "mixed", verdict: "neutral" };
};

export const hawkesHitAggregate = (perSymbol: Record<string, [number, number]>): [number, number] => {
    let agree = 0;
    let total = 0;
    for (const [a, t] of Object.values(perSymbol)) {
        agree += a;
        total += t;
    }
    return [agree, total];
};

export const REASON_PHRASES: Record<string, string> = {
    take_profit: "hit take-profit",
    hard_stop_net: "loss guard tripped",
    hard_stop: "loss guard tripped",
    trail: "trailing stop locked profit",
    trail_loss: "trailing stop closed it",
    timeout: "held too long, time was up",
    macro_early_fail: "setup never confirmed",
    btc_flush: "BTC dumped, exited for safety",
    hawkes_sell_flip: "crowd flipped to selling",
    velocity_stall: "stopped going up",
    return_velocity_stall: "stopped going up",
    manual: "closed by you",
    eob: "end of session",
    spread_blowout: "spread got too expensive",
    book_flip: "order book flipped",
};

export const REGIME_PHRASES: Record<string, string> = {
    book_ignition: "order-book ignition",
    macro_reclaim_v2: "pullback reclaim",
    macro_thrust: "momentum thrust",
};

export const plainReason = (code: string): string => {
    if (!code) return "closed";
    const base = String(code).trim();
    for (const [key, phrase] of Object.entries(REASON_PHRASES)) {
        if (base.startsWith(key)) return phrase;
    }
    return base.replace(/_/g, " ");
};

export const plainRegime = (code: string): string => {
    const key = String(code ?? "").trim();
    return REGIME_PHRASES[key] ?? key.replace(/_/g, " ");
};
