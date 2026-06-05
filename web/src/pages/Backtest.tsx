import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from "@/components/ui/Card";
import { ZoomableChart, ChartSeries } from "@/components/ui/ZoomableChart";
import { InfoTooltip, LabelWithInfo } from "@/components/ui/InfoTooltip";
import { GlossaryKey } from "@/lib/glossary";
import { formatPercent, formatPercentSigned, formatNumber, signClass } from "@/lib/format";
import { cn } from "@/lib/cn";

// ─── Rolling window computation ──────────────────────────────────────────────

function rollingWindow<T>(arr: T[], n: number, fn: (window: T[]) => number): (number | null)[] {
  return arr.map((_, i) => (i < n - 1 ? null : fn(arr.slice(i - n + 1, i + 1))));
}

function rollingMean(vals: number[]): number {
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function rollingStd(vals: number[]): number {
  const m = rollingMean(vals);
  return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / (vals.length - 1)) || 0;
}

// ─── Main page ───────────────────────────────────────────────────────────────

export function Backtest() {
  const [costBps, setCostBps] = useState(0);

  const { data, isLoading } = useQuery({
    queryKey: ["bt"],
    queryFn: () => api.backtestSummary().catch(() => null),
  });

  const hasSPY = useMemo(
    () => (data?.equity_curve ?? []).some((p) => p.benchmark_return != null && p.benchmark_return !== 0),
    [data],
  );

  // Build equity / drawdown / rolling-Sharpe rows, applying optional cost drag.
  const { equityRows, ddRows, sharpeRows } = useMemo(() => {
    const curve = data?.equity_curve ?? [];
    let cumStrat = 1, cumInv = 1, cumSPY = 1;
    const dailyRets: number[] = [];
    const eqRows: Record<string, number | string | null>[] = [];
    const ddRowsLocal: Record<string, number | string | null>[] = [];

    curve.forEach((p) => {
      const rawR = p.daily_return ?? 0;
      const tv = p.turnover ?? 0;
      // cost-adjust: subtract one-way bps on each side of turnover
      const costDrag = tv * (costBps / 10000);
      const r = rawR - costDrag;
      const br = p.benchmark_return ?? 0;

      cumStrat *= 1 + r;
      cumInv *= 1 - r;
      cumSPY *= 1 + br;
      dailyRets.push(r);

      eqRows.push({ date: p.date, strategy: +cumStrat.toFixed(6), spy: +cumSPY.toFixed(6), inverse: +cumInv.toFixed(6) });
      ddRowsLocal.push({ date: p.date, dd: p.drawdown ?? 0 });
    });

    // Rolling 252-day Sharpe
    const WINDOW = 252;
    const annualised = rollingWindow(dailyRets, WINDOW, (w) => {
      const std = rollingStd(w);
      return std > 0 ? (rollingMean(w) / std) * Math.sqrt(252) : null as any;
    });
    const sharpeRowsLocal = curve.map((p, i) => ({
      date: p.date,
      sharpe: annualised[i],
    }));

    return { equityRows: eqRows, ddRows: ddRowsLocal, sharpeRows: sharpeRowsLocal };
  }, [data, costBps]);

  // Annual breakdown (strategy + SPY side by side)
  const yearly = useMemo(() => {
    const map: Record<string, { rets: number[]; spyRets: number[]; sumLog: number; spySumLog: number }> = {};
    (data?.equity_curve ?? []).forEach((p) => {
      const y = (p.date || "").slice(0, 4);
      if (!y) return;
      if (!map[y]) map[y] = { rets: [], spyRets: [], sumLog: 0, spySumLog: 0 };
      const r = p.daily_return ?? 0;
      const tv = p.turnover ?? 0;
      const adjusted = r - tv * (costBps / 10000);
      const br = p.benchmark_return ?? 0;
      map[y].rets.push(adjusted);
      map[y].spyRets.push(br);
      if (adjusted) map[y].sumLog += Math.log1p(adjusted);
      if (br) map[y].spySumLog += Math.log1p(br);
    });
    return Object.entries(map)
      .map(([year, agg]) => {
        const ann = Math.exp(agg.sumLog) - 1;
        const spy = hasSPY ? Math.exp(agg.spySumLog) - 1 : null;
        const mean = agg.rets.reduce((a, b) => a + b, 0) / Math.max(agg.rets.length, 1);
        const std = rollingStd(agg.rets) || NaN;
        const sharpe = std ? (mean / std) * Math.sqrt(252) : NaN;
        return { year, ann, spy, sharpe, n: agg.rets.length };
      })
      .sort((a, b) => a.year.localeCompare(b.year));
  }, [data, costBps, hasSPY]);

  // Bar chart data for annual returns
  const annualBarData = yearly.map((y) => ({
    year: y.year,
    strategy: +(y.ann * 100).toFixed(2),
    spy: y.spy != null ? +(y.spy * 100).toFixed(2) : null,
  }));

  const equitySeries: ChartSeries[] = [
    { type: "area", dataKey: "strategy", name: "Strategy", color: "hsl(var(--primary))", fillOpacity: 0.15, strokeWidth: 1.8 },
    ...(hasSPY ? [{ type: "line" as const, dataKey: "spy", name: "S&P 500", color: "#f59e0b", strokeWidth: 1.5, strokeDasharray: "5 3" }] : []),
    { type: "line", dataKey: "inverse", name: "Inverse", color: "hsl(var(--muted-foreground))", strokeWidth: 1.0 },
  ];
  const ddSeries: ChartSeries[] = [
    { type: "area", dataKey: "dd", name: "Drawdown", color: "hsl(var(--negative, 0 84% 60%))", fillOpacity: 0.2, strokeWidth: 1.4 },
  ];
  const sharpeSeries: ChartSeries[] = [
    { type: "line", dataKey: "sharpe", name: "Rolling Sharpe (252d)", color: "hsl(var(--primary))", strokeWidth: 1.5 },
  ];
  const annualSeries: ChartSeries[] = [
    { type: "bar", dataKey: "strategy", name: "Strategy", color: "hsl(var(--primary))", fillOpacity: 0.85 },
    ...(hasSPY ? [{ type: "bar" as const, dataKey: "spy", name: "S&P 500", color: "#f59e0b", fillOpacity: 0.75 }] : []),
  ];

  if (isLoading) return <Card><CardContent>Loading backtest…</CardContent></Card>;
  if (!data) return (
    <Card><CardContent>No backtest yet. Queue a job on the Jobs page.</CardContent></Card>
  );

  const m = data.run.metrics;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Backtest #{data.run.id}</h1>
          <p className="text-sm text-muted-foreground">
            {data.run.status} · {data.run.tickers_count} tickers · {data.run.note}
          </p>
        </div>
        {/* Cost-adjustment slider */}
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Cost assumption</span>
          <input
            type="range" min={0} max={20} step={1} value={costBps}
            onChange={e => setCostBps(+e.target.value)}
            className="w-28 accent-primary"
          />
          <span className="w-16 text-right text-xs font-mono tabular">
            {costBps === 0 ? "0 bps (gross)" : `${costBps} bps`}
          </span>
        </div>
      </div>

      {/* KPI tiles */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Tile label="Ann. return" termKey="ann_return" value={formatPercentSigned(m.ann_return)} signed={m.ann_return} />
        <Tile label="Sharpe" termKey="sharpe" value={formatNumber(m.sharpe)} signed={m.sharpe} />
        <Tile label="Sortino" termKey="sortino" value={formatNumber(m.sortino)} signed={m.sortino} />
        <Tile label="Max DD" termKey="max_drawdown" value={formatPercent(m.max_drawdown)} />
        <Tile label="Calmar" termKey="calmar" value={formatNumber(m.calmar)} signed={m.calmar} />
        <Tile label="Vol" termKey="ann_vol" value={formatPercent(m.ann_vol)} />
        <Tile label="Hit ratio" termKey="hit_ratio" value={formatPercent(m.hit_ratio)} />
        <Tile label="Days" value={String(m.n_days ?? "—")} />
      </div>

      {costBps > 0 && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          ⚠ Chart and annual table reflect {costBps}bps round-trip cost drag applied to stored daily turnover.
          KPI tiles above are gross (from the run record).
        </p>
      )}

      {/* Equity curve */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Equity curve</CardTitle>
            <InfoTooltip termKey="inverse" title="Strategy vs S&P 500 vs Inverse" />
          </div>
          <CardSubtitle>
            Growth of $1, walk-forward.{hasSPY ? " Amber dashed = S&P 500 buy-and-hold. " : " "}
            Faint line = inverse of every trade. Drag to zoom.
          </CardSubtitle>
        </CardHeader>
        <CardContent>
          <ZoomableChart
            data={equityRows}
            xKey="date"
            series={equitySeries}
            height={300}
            legend
            leftFormatter={(v) => `$${(v as number).toFixed(2)}`}
            xTickFormatter={(v) => (v || "").slice(0, 7)}
          />
        </CardContent>
      </Card>

      {/* Rolling Sharpe */}
      <Card>
        <CardHeader>
          <CardTitle>Rolling 252-day Sharpe</CardTitle>
          <CardSubtitle>Annualised Sharpe over the trailing year. Above 0 = positive edge; below = destructive.</CardSubtitle>
        </CardHeader>
        <CardContent>
          <ZoomableChart
            data={sharpeRows}
            xKey="date"
            series={sharpeSeries}
            height={200}
            refLines={[{ y: 0, label: "0", color: "hsl(var(--muted-foreground))" }, { y: 1, label: "Sharpe 1", color: "#22c55e" }]}
            leftFormatter={(v) => (v as number).toFixed(2)}
            xTickFormatter={(v) => (v || "").slice(0, 7)}
          />
        </CardContent>
      </Card>

      {/* Annual return bar chart */}
      <Card>
        <CardHeader>
          <CardTitle>Annual returns</CardTitle>
          <CardSubtitle>
            {hasSPY ? "Strategy vs S&P 500 buy-and-hold, year by year." : "Strategy by year."}
            {costBps > 0 ? ` Includes ${costBps}bps cost drag.` : " Gross of costs."}
          </CardSubtitle>
        </CardHeader>
        <CardContent>
          <ZoomableChart
            data={annualBarData}
            xKey="year"
            series={annualSeries}
            height={240}
            legend
            leftFormatter={(v) => `${(v as number).toFixed(0)}%`}
          />
        </CardContent>
      </Card>

      {/* Drawdown */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Drawdown</CardTitle>
            <InfoTooltip termKey="max_drawdown" title="Drawdown from peak" />
          </div>
        </CardHeader>
        <CardContent>
          <ZoomableChart
            data={ddRows}
            xKey="date"
            series={ddSeries}
            height={200}
            leftFormatter={(v) => `${((v as number) * 100).toFixed(0)}%`}
            xTickFormatter={(v) => (v || "").slice(0, 7)}
          />
        </CardContent>
      </Card>

      {/* Per-horizon diagnostics */}
      <Card>
        <CardHeader>
          <CardTitle>Per-horizon diagnostics</CardTitle>
          <CardSubtitle>Out-of-sample, walk-forward, before portfolio cost drag.</CardSubtitle>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm tabular">
            <thead className="text-xs uppercase text-muted-foreground">
              <tr>
                <th className="py-2 text-left">Horizon</th>
                <th className="py-2 text-right">
                  <LabelWithInfo label="Hit rate" termKey="hit_rate" className="justify-end" side="top" align="end" />
                </th>
                <th className="py-2 text-right">
                  <LabelWithInfo label="IC mean" termKey="ic" className="justify-end" side="top" align="end" />
                </th>
                <th className="py-2 text-right">
                  <LabelWithInfo label="IC IR" termKey="ic_ir" className="justify-end" side="top" align="end" />
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {Object.entries(data.run.per_horizon_diagnostics ?? {}).map(([h, d]) => (
                <tr key={h}>
                  <td className="py-2">{h}d</td>
                  <td className={cn("py-2 text-right", signClass(d.hit_rate - 0.5))}>{formatPercent(d.hit_rate)}</td>
                  <td className={cn("py-2 text-right", signClass(d.ic_mean))}>{formatNumber(d.ic_mean, { maximumFractionDigits: 4 })}</td>
                  <td className={cn("py-2 text-right", signClass(d.ic_ir))}>{formatNumber(d.ic_ir, { maximumFractionDigits: 2 })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Annual breakdown table */}
      <Card>
        <CardHeader>
          <CardTitle>Yearly breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm tabular">
            <thead className="text-xs uppercase text-muted-foreground">
              <tr>
                <th className="py-2 text-left">Year</th>
                <th className="py-2 text-right">Strategy</th>
                {hasSPY && <th className="py-2 text-right">S&amp;P 500</th>}
                {hasSPY && <th className="py-2 text-right">Alpha</th>}
                <th className="py-2 text-right">Sharpe</th>
                <th className="py-2 text-right">Days</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {yearly.map((y) => {
                const alpha = hasSPY && y.spy != null ? y.ann - y.spy : null;
                return (
                  <tr key={y.year}>
                    <td className="py-2 font-medium">{y.year}</td>
                    <td className={cn("py-2 text-right", signClass(y.ann))}>{formatPercentSigned(y.ann)}</td>
                    {hasSPY && <td className={cn("py-2 text-right", signClass(y.spy ?? 0))}>{y.spy != null ? formatPercentSigned(y.spy) : "—"}</td>}
                    {hasSPY && <td className={cn("py-2 text-right", signClass(alpha ?? 0))}>{alpha != null ? formatPercentSigned(alpha) : "—"}</td>}
                    <td className={cn("py-2 text-right", signClass(y.sharpe))}>{formatNumber(y.sharpe)}</td>
                    <td className="py-2 text-right text-muted-foreground">{y.n}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}

function Tile({
  label,
  termKey,
  value,
  signed,
}: {
  label: string;
  termKey?: GlossaryKey;
  value: string;
  signed?: number;
}) {
  return (
    <Card>
      <CardContent className="space-y-1">
        <LabelWithInfo label={label} termKey={termKey} className="text-xs uppercase text-muted-foreground" />
        <div className={cn("text-xl font-semibold tabular", signed !== undefined && signClass(signed))}>{value}</div>
      </CardContent>
    </Card>
  );
}
