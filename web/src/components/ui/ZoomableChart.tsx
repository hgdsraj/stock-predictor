import { useState, useMemo, useCallback } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  Brush,
  ReferenceArea,
  ReferenceLine,
} from "recharts";
import { ZoomOut, MousePointerClick } from "lucide-react";
import { cn } from "@/lib/cn";

export interface ChartSeries {
  type: "line" | "area" | "bar";
  dataKey: string;
  name: string;
  color: string; // any CSS color, e.g. "hsl(var(--primary))"
  yAxisId?: "left" | "right";
  strokeWidth?: number;
  strokeDasharray?: string;
  fillOpacity?: number;
  dot?: boolean;
}

export interface ChartRefLine {
  y: number;
  yAxisId?: "left" | "right";
  label: string;
  color?: string;
}

interface ZoomableChartProps {
  data: Record<string, any>[];
  xKey: string;
  series: ChartSeries[];
  refLines?: ChartRefLine[];
  height?: number;
  rightAxis?: boolean;
  /** Format y-axis ticks + tooltip values on the left axis. */
  leftFormatter?: (v: number) => string;
  rightFormatter?: (v: number) => string;
  /** Format the x tick label. */
  xTickFormatter?: (v: string) => string;
  legend?: boolean;
  syncId?: string;
}

const tooltipStyle = {
  background: "hsl(var(--card))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 6,
  fontSize: 12,
};

/**
 * A reusable chart with three navigation gestures:
 *   • drag across the plot to highlight a section and zoom to it
 *   • drag the brush handles at the bottom to scroll / pan / fine-tune
 *   • "Reset zoom" to return to the full range
 * Plus optional horizontal reference lines (e.g. support / resistance / 52w).
 */
export function ZoomableChart({
  data,
  xKey,
  series,
  refLines = [],
  height = 320,
  rightAxis = false,
  leftFormatter,
  rightFormatter,
  xTickFormatter,
  legend = false,
  syncId,
}: ZoomableChartProps) {
  const lastIdx = Math.max(0, data.length - 1);
  const [win, setWin] = useState<[number, number]>([0, lastIdx]);
  const [selL, setSelL] = useState<string | null>(null);
  const [selR, setSelR] = useState<string | null>(null);

  // Keep the window valid if the data length changes (e.g. async load).
  const clampedWin = useMemo<[number, number]>(() => {
    const end = Math.min(win[1], lastIdx);
    const start = Math.min(win[0], end);
    return [start, end];
  }, [win, lastIdx]);

  const isZoomed = clampedWin[0] > 0 || clampedWin[1] < lastIdx;

  const indexOf = useCallback(
    (label: string | null) => (label == null ? -1 : data.findIndex((d) => String(d[xKey]) === String(label))),
    [data, xKey],
  );

  function onMouseDown(e: any) {
    if (e && e.activeLabel != null) {
      setSelL(String(e.activeLabel));
      setSelR(String(e.activeLabel));
    }
  }
  function onMouseMove(e: any) {
    if (selL != null && e && e.activeLabel != null) setSelR(String(e.activeLabel));
  }
  function onMouseUp() {
    if (selL != null && selR != null && selL !== selR) {
      const a = indexOf(selL);
      const b = indexOf(selR);
      if (a >= 0 && b >= 0) setWin([Math.min(a, b), Math.max(a, b)]);
    }
    setSelL(null);
    setSelR(null);
  }

  function reset() {
    setWin([0, lastIdx]);
  }

  if (data.length === 0) {
    return (
      <div style={{ height }} className="flex items-center justify-center text-sm text-muted-foreground">
        No data.
      </div>
    );
  }

  return (
    <div className="relative">
      <div className="absolute right-1 top-1 z-10 flex items-center gap-2">
        {isZoomed ? (
          <button
            onClick={reset}
            className="inline-flex items-center gap-1 rounded-md border border-border bg-card/80 px-2 py-1 text-xs text-muted-foreground backdrop-blur transition-colors hover:text-foreground"
          >
            <ZoomOut className="h-3 w-3" /> Reset zoom
          </button>
        ) : (
          <span className="hidden items-center gap-1 rounded-md px-2 py-1 text-[10px] text-muted-foreground/70 sm:inline-flex">
            <MousePointerClick className="h-3 w-3" /> drag to zoom
          </span>
        )}
      </div>

      <div style={{ height, userSelect: "none" }}>
        <ResponsiveContainer>
          <ComposedChart
            data={data}
            syncId={syncId}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
          >
            <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey={xKey}
              tick={{ fontSize: 11 }}
              stroke="hsl(var(--muted-foreground))"
              tickFormatter={xTickFormatter}
              minTickGap={32}
            />
            <YAxis
              yAxisId="left"
              tick={{ fontSize: 11 }}
              stroke="hsl(var(--muted-foreground))"
              tickFormatter={leftFormatter}
              width={56}
              domain={["auto", "auto"]}
            />
            {rightAxis && (
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fontSize: 11 }}
                stroke="hsl(var(--muted-foreground))"
                tickFormatter={rightFormatter}
                width={48}
                domain={["auto", "auto"]}
              />
            )}
            <Tooltip contentStyle={tooltipStyle} />
            {legend && <Legend wrapperStyle={{ fontSize: 11 }} />}

            {refLines.map((rl, i) => (
              <ReferenceLine
                key={`ref-${i}`}
                y={rl.y}
                yAxisId={rl.yAxisId ?? "left"}
                stroke={rl.color ?? "hsl(var(--muted-foreground))"}
                strokeDasharray="4 4"
                strokeOpacity={0.7}
                label={{
                  value: rl.label,
                  position: "insideTopRight",
                  fontSize: 10,
                  fill: rl.color ?? "hsl(var(--muted-foreground))",
                }}
              />
            ))}

            {series.map((s) => {
              const common = {
                key: s.dataKey,
                dataKey: s.dataKey,
                name: s.name,
                yAxisId: s.yAxisId ?? "left",
                isAnimationActive: false,
              };
              if (s.type === "line")
                return (
                  <Line
                    {...common}
                    type="monotone"
                    stroke={s.color}
                    strokeWidth={s.strokeWidth ?? 1.5}
                    strokeDasharray={s.strokeDasharray}
                    dot={s.dot ?? false}
                  />
                );
              if (s.type === "area")
                return (
                  <Area
                    {...common}
                    type="monotone"
                    stroke={s.color}
                    fill={s.color}
                    fillOpacity={s.fillOpacity ?? 0.2}
                    strokeWidth={s.strokeWidth ?? 1.5}
                  />
                );
              return <Bar {...common} fill={s.color} fillOpacity={s.fillOpacity ?? 0.5} />;
            })}

            {selL != null && selR != null && selL !== selR && (
              <ReferenceArea
                x1={selL}
                x2={selR}
                strokeOpacity={0.3}
                fill="hsl(var(--primary))"
                fillOpacity={0.12}
              />
            )}

            {data.length > 2 && (
              <Brush
                dataKey={xKey}
                height={22}
                travellerWidth={8}
                startIndex={clampedWin[0]}
                endIndex={clampedWin[1]}
                onChange={(r: any) => {
                  if (r && typeof r.startIndex === "number" && typeof r.endIndex === "number") {
                    setWin([r.startIndex, r.endIndex]);
                  }
                }}
                stroke="hsl(var(--muted-foreground))"
                fill="hsl(var(--muted))"
                tickFormatter={xTickFormatter as any}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
