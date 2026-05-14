import {
    CategoryAxis,
    EAutoRange,
    EAxisAlignment,
    FastCandlestickRenderableSeries,
    MouseWheelZoomModifier,
    NumericAxis,
    NumberRange,
    OhlcDataSeries,
    RolloverModifier,
    SciChartSurface,
    ZoomExtentsModifier,
    ZoomPanModifier,
} from "scichart";
import type { Record, StreamClient } from "../data/ws";
import { appTheme } from "../theme";
import { addChartHost, addHeader } from "./_panel";

interface OhlcRecord {
    symbol: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    interval: number;
    interval_begin: number;
}

const isOhlc = (rec: Record): rec is Record & OhlcRecord =>
    rec.kind === "ohlc" &&
    typeof rec.symbol === "string" &&
    typeof rec.open === "number" &&
    typeof rec.high === "number" &&
    typeof rec.low === "number" &&
    typeof rec.close === "number" &&
    typeof rec.interval_begin === "number";

export const mountCandles = async (root: HTMLDivElement, stream: StreamClient) => {
    addHeader(root, "BTC/USD candles");

    const { sciChartSurface, wasmContext } = await SciChartSurface.create(root, {
        theme: appTheme.SciChartJsTheme,
        disableAspect: true,
    });

    sciChartSurface.xAxes.add(
        new CategoryAxis(wasmContext, {
            axisAlignment: EAxisAlignment.Bottom,
            autoRange: EAutoRange.Always,
            growBy: new NumberRange(0.05, 0.05),
            visibleRange: new NumberRange(-1, 60),
        }),
    );
    sciChartSurface.yAxes.add(
        new NumericAxis(wasmContext, {
            autoRange: EAutoRange.Always,
            growBy: new NumberRange(0.05, 0.05),
        }),
    );

    const ds = new OhlcDataSeries(wasmContext, {
        dataSeriesName: "BTC/USD",
        isSorted: true,
        containsNaN: false,
    });

    sciChartSurface.renderableSeries.add(
        new FastCandlestickRenderableSeries(wasmContext, {
            dataSeries: ds,
            strokeThickness: 1,
            dataPointWidth: 0.6,
            brushUp: appTheme.VividGreen + "AA",
            brushDown: appTheme.VividRed + "AA",
            strokeUp: appTheme.VividGreen,
            strokeDown: appTheme.VividRed,
        }),
    );

    sciChartSurface.chartModifiers.add(
        new ZoomPanModifier(),
        new MouseWheelZoomModifier(),
        new ZoomExtentsModifier(),
        new RolloverModifier({ snapToDataPoint: true, showTooltip: true }),
    );

    const indexByBucket = new Map<number, number>();

    stream.on("ohlc", (rec) => {
        if (!isOhlc(rec)) {
            console.warn("[candles] failed isOhlc guard", rec);
            return;
        }
        if (rec.symbol !== "BTC/USD") return;

        const bucket = rec.interval_begin;
        const idx = indexByBucket.get(bucket);

        if (idx === undefined) {
            indexByBucket.set(bucket, ds.count());
            ds.append(bucket, rec.open, rec.high, rec.low, rec.close);
        } else {
            ds.update(idx, rec.open, rec.high, rec.low, rec.close);
        }
    });

    return { sciChartSurface };
};
