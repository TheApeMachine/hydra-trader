import type { Record, StreamClient } from "../data/ws";
import { addHeader } from "./_panel";

interface TradeRecord {
    symbol: string;
    side: string;
    price: number;
    qty: number;
    t: number;
}

const isTrade = (rec: Record): rec is Record & TradeRecord =>
    rec.kind === "trade" &&
    typeof rec.symbol === "string" &&
    typeof rec.side === "string" &&
    typeof rec.price === "number" &&
    typeof rec.qty === "number" &&
    typeof rec.t === "number";

const MAX_ROWS = 200;

const fmtTime = (t: number): string => {
    const d = new Date(t * 1000);
    return d.toLocaleTimeString("en-GB", { hour12: false });
};

export const mountTrades = async (root: HTMLDivElement, stream: StreamClient) => {
    addHeader(root, "trades");

    const list = document.createElement("ul");
    list.className = "dash-trades-list";
    list.style.cssText = "flex:1 1 auto;min-height:0;overflow-y:auto;margin:0;padding:8px 12px;list-style:none;";
    root.appendChild(list);

    stream.on("trade", (rec) => {
        if (!isTrade(rec)) return;

        const li = document.createElement("li");

        const mark = document.createElement("span");
        mark.className = "dash-trade-mark";
        mark.textContent = rec.side === "buy" ? "▲" : "▼";
        mark.style.color = rec.side === "buy" ? "var(--pos)" : "var(--neg)";

        const time = document.createElement("span");
        time.className = "dash-trade-time";
        time.textContent = fmtTime(rec.t);

        const sym = document.createElement("span");
        sym.className = "dash-trade-sym";
        sym.textContent = rec.symbol;

        const body = document.createElement("span");
        body.className = "dash-trade-body";
        body.textContent = `${rec.price.toFixed(2)} × ${rec.qty}`;

        li.appendChild(mark);
        li.appendChild(time);
        li.appendChild(sym);
        li.appendChild(body);

        list.insertBefore(li, list.firstChild);

        while (list.children.length > MAX_ROWS && list.lastChild) {
            list.removeChild(list.lastChild);
        }
    });
};
