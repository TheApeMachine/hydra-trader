import { SciChartSurface } from "scichart";
import { mountCandles } from "./charts/candles";
import { mountGauge } from "./charts/gauge";
import { mountHawkes } from "./charts/hawkes";
import { mountMultiSymbol } from "./charts/multiSymbol";
import { mountStatus } from "./charts/status";
import { mountStrategies } from "./charts/strategies";
import { mountTrades } from "./charts/trades";
import { StreamClient } from "./data/ws";

SciChartSurface.configure({
    wasmUrl: "/scichart2d.wasm",
    wasmNoSimdUrl: "/scichart2d-nosimd.wasm",
});
SciChartSurface.UseCommunityLicense();

const main = async () => {
    const el = (id: string) => document.getElementById(id) as HTMLDivElement;

    const stream = new StreamClient();

    await Promise.all([
        mountCandles(el("panel-candles"), stream),
        mountMultiSymbol(el("panel-multi"), stream),
        mountHawkes(el("panel-hawkes"), stream),
        mountGauge(el("panel-gauge"), stream),
        mountStrategies(el("panel-strategies"), stream),
        mountTrades(el("panel-trades"), stream),
        mountStatus(el("panel-status"), stream),
    ]);

    const statusBar = document.getElementById("status-bar");
    const setStatus = (text: string) => {
        if (statusBar) statusBar.textContent = text;
    };

    stream.onStatus((s) => setStatus(s));
    stream.connect();
};

main();
