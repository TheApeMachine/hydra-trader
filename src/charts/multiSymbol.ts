import {
    DateTimeNumericAxis,
    EAutoRange,
    EAxisAlignment,
    FastLineRenderableSeries,
    NumberRange,
    NumericAxis,
    SciChartSurface,
    XyDataSeries,
} from "scichart";
import type { Record, StreamClient } from "../data/ws";
import { appTheme } from "../theme";
import { addChartHost, addHeader } from "./_panel";

interface TradeRecord {
    symbol: string;
    price: number;
    t: number;
}

const isTrade = (rec: Record): rec is Record & TradeRecord =>
    rec.kind === "trade" &&
    typeof rec.symbol === "string" &&
    typeof rec.price === "number" &&
    typeof rec.t === "number";

const PALETTE = [
    appTheme.VividSkyBlue,
    appTheme.VividOrange,
    appTheme.VividTeal,
    appTheme.VividPink,
    appTheme.VividPurple,
    appTheme.VividGreen,
];

export const mountMultiSymbol = async (root: HTMLDivElement, stream: StreamClient) => {
    addHeader(root, "prices");

    const { sciChartSurface, wasmContext } = await SciChartSurface.create(root, {
        theme: appTheme.SciChartJsTheme,
        disableAspect: true,
    });

    sciChartSurface.xAxes.add(
        new DateTimeNumericAxis(wasmContext, {
            autoRange: EAutoRange.Always,
            growBy: new NumberRange(0.0, 0.05),
        }),
    );

    interface SymbolSeries {
        ds: XyDataSeries;
        axisId: string;
    }

    const seriesBySymbol = new Map<string, SymbolSeries>();

    stream.on("trade", (rec) => {
        if (!isTrade(rec)) return;

        let entry = seriesBySymbol.get(rec.symbol);

        if (!entry) {
            const idx = seriesBySymbol.size;
            const color = PALETTE[idx % PALETTE.length];
            const axisId = `y-${rec.symbol}`;

            const axis = new NumericAxis(wasmContext, {
                id: axisId,
                axisAlignment: idx === 0 ? EAxisAlignment.Left : EAxisAlignment.Right,
                autoRange: EAutoRange.Always,
                growBy: new NumberRange(0.05, 0.05),
                axisTitle: rec.symbol,
                axisTitleStyle: { fontSize: 10, color },
                labelStyle: { color },
            });
            sciChartSurface.yAxes.add(axis);

            const ds = new XyDataSeries(wasmContext, {
                dataSeriesName: rec.symbol,
                isSorted: true,
                containsNaN: false,
            });
            sciChartSurface.renderableSeries.add(
                new FastLineRenderableSeries(wasmContext, {
                    dataSeries: ds,
                    stroke: color,
                    strokeThickness: 1.5,
                    yAxisId: axisId,
                }),
            );

            entry = { ds, axisId };
            seriesBySymbol.set(rec.symbol, entry);
        }

        entry.ds.append(rec.t, rec.price);
    });

    return { sciChartSurface };
};
