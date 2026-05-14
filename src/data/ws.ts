export type Record = { kind: string } & { [k: string]: unknown };
export type RecordListener = (rec: Record) => void;
export type StatusListener = (
    status: "connecting" | "open" | "closed" | "error",
    detail?: string,
) => void;

export interface StreamClientOptions {
    url?: string;
    reconnectMs?: number;
}

export class StreamClient {
    private url: string;
    private reconnectMs: number;
    private ws: WebSocket | null = null;
    private byKind: Map<string, RecordListener[]> = new Map();
    private statusListeners: StatusListener[] = [];
    private closedByUser = false;

    constructor(opts: StreamClientOptions = {}) {
        const defaultUrl =
            typeof window !== "undefined"
                ? `ws://${window.location.hostname || "localhost"}:8765`
                : "ws://127.0.0.1:8765";
        this.url = opts.url ?? defaultUrl;
        this.reconnectMs = opts.reconnectMs ?? 2000;
    }

    on(kind: string, fn: RecordListener): () => void {
        const list = this.byKind.get(kind) ?? [];
        list.push(fn);
        this.byKind.set(kind, list);
        return () => {
            const cur = this.byKind.get(kind);
            if (!cur) return;
            this.byKind.set(
                kind,
                cur.filter((f) => f !== fn),
            );
        };
    }

    onStatus(fn: StatusListener): () => void {
        this.statusListeners.push(fn);
        return () => {
            this.statusListeners = this.statusListeners.filter((f) => f !== fn);
        };
    }

    connect(): void {
        this.closedByUser = false;
        this.openSocket();
    }

    close(): void {
        this.closedByUser = true;
        if (this.ws) {
            try {
                this.ws.close();
            } catch {}
        }
        this.ws = null;
    }

    private openSocket(): void {
        this.emitStatus("connecting", this.url);
        try {
            this.ws = new WebSocket(this.url);
        } catch (e) {
            this.emitStatus("error", String(e));
            this.scheduleReconnect();
            return;
        }
        this.ws.onopen = () => this.emitStatus("open");
        this.ws.onmessage = (ev) => this.handleMessage(ev.data);
        this.ws.onerror = () => this.emitStatus("error");
        this.ws.onclose = () => {
            this.emitStatus("closed");
            this.scheduleReconnect();
        };
    }

    private scheduleReconnect(): void {
        if (this.closedByUser) return;
        setTimeout(() => {
            if (!this.closedByUser) this.openSocket();
        }, this.reconnectMs);
    }

    private handleMessage(raw: unknown): void {
        if (typeof raw !== "string") return;
        let parsed: unknown;
        try {
            parsed = JSON.parse(raw);
        } catch (e) {
            console.error("[STREAM] JSON parse failed", e, raw);
            return;
        }
        if (typeof parsed !== "object" || parsed === null) {
            console.warn("[STREAM] non-object record", parsed);
            return;
        }
        const rec = parsed as Record;
        const kind = rec.kind;
        if (typeof kind !== "string") {
            console.warn("[STREAM] record without kind", rec);
            return;
        }
        const listeners = this.byKind.get(kind);
        if (!listeners) return;
        for (const fn of listeners) {
            try {
                fn(rec);
            } catch (e) {
                console.error(`[STREAM] listener for kind=${kind} threw`, e);
            }
        }
    }

    private emitStatus(status: Parameters<StatusListener>[0], detail?: string): void {
        for (const fn of this.statusListeners) {
            try {
                fn(status, detail);
            } catch (e) {
                console.error("[STREAM] status listener threw", e);
            }
        }
    }
}
