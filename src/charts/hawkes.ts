import {
    DateTimeNumericAxis,
    EAutoRange,
    FastLineRenderableSeries,
    NumberRange,
    NumericAxis,
    SciChartSurface,
    XyDataSeries,
} from "scichart";
import type { Record, StreamClient } from "../data/ws";
import { appTheme } from "../theme";
import { addChartHost, addHeader } from "./_panel";

interface HawkesRecord {
    symbol: string | null;
    t: number;
    intensity: number;
}

const isHawkes = (rec: Record): rec is Record & HawkesRecord =>
    rec.kind === "hawkes" &&
    typeof rec.t === "number" &&
    typeof rec.intensity === "number";

export const mountHawkes = async (root: HTMLDivElement, stream: StreamClient) => {
    addHeader(root, "hawkes λ(t)");
    const host = addChartHost(root);

    const { sciChartSurface, wasmContext } = await SciChartSurface.create(host, {
        theme: appTheme.SciChartJsTheme,
        disableAspect: true,
    });

    sciChartSurface.xAxes.add(
        new DateTimeNumericAxis(wasmContext, {
            autoRange: EAutoRange.Always,
            growBy: new NumberRange(0.0, 0.05),
        }),
    );
    sciChartSurface.yAxes.add(
        new NumericAxis(wasmContext, {
            autoRange: EAutoRange.Always,
            growBy: new NumberRange(0.1, 0.1),
        }),
    );

    const ds = new XyDataSeries(wasmContext, {
        dataSeriesName: "λ(t)",
        isSorted: true,
        containsNaN: false,
    });

    sciChartSurface.renderableSeries.add(
        new FastLineRenderableSeries(wasmContext, {
            dataSeries: ds,
            stroke: appTheme.VividTeal,
            strokeThickness: 2,
        }),
    );

    stream.on("hawkes", (rec) => {
        if (!isHawkes(rec)) return;
        ds.append(rec.t, rec.intensity);
    });

    return { sciChartSurface };
};
