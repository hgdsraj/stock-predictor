import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from "@/components/ui/Card";
import { ZoomableChart, ChartSeries } from "@/components/ui/ZoomableChart";
import { InfoTooltip, LabelWithInfo } from "@/components/ui/InfoTooltip";
import { GlossaryKey } from "@/lib/glossary";
import { formatPercent, formatPercentSigned, formatNumber, signClass } from "@/lib/format";
import { cn } from "@/lib/cn";

export function Backtest() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["bt"],
    queryFn: () => api.backtestSummary().catch(() => null),
  });

  // Recompute strategy + inverse growth-of-$1 from daily returns so both curves
  // are directly comparable. Drawdown comes straight from the stored curve.
  let cumStrat = 1;
  let cumInv = 1;
  const equityRows = (data?.equity_curve ?? []).map((p) => {
    const r = p.daily_return ?? 0;
    cumStrat *= 1 + r;
    cumInv *= 1 - r;
    return { date: p.date, equity: cumStrat, inverse: cumInv, dd: p.drawdown };
  });

  const equitySeries: ChartSeries[] = [
    { type: "area", dataKey: "equity", name: "Strategy", color: "hsl(var(--primary))", fillOpacity: 0.18, strokeWidth: 1.8 },
    { type: "line", dataKey: "inverse", name: "Inverse", color: "hsl(var(--muted-foreground))", strokeWidth: 1.2 },
  ];
  const ddSeries: ChartSeries[] = [
    { type: "area", dataKey: "dd", name: "Drawdown", color: "hsl(var(--negative))", fillOpacity: 0.18, strokeWidth: 1.4 },
  ];

  const yearly = useMemo(() => {
    const map: Record<string, { rets: number[]; sumLog: number }> = {};
    (data?.equity_curve ?? []).forEach((p) => {
      const y = (p.date || "").slice(0, 4);
      if (!y) return;
      if (!map[y]) map[y] = { rets: [], sumLog: 0 };
      const r = p.daily_return ?? 0;
      map[y].rets.push(r);
      if (r) map[y].sumLog += Math.log1p(r);
    });
    return Object.entries(map)
      .map(([year, agg]) => {
        const ann = Math.exp(agg.sumLog) - 1;
        const mean = agg.rets.reduce((a, b) => a + b, 0) / Math.max(agg.rets.length, 1);
        const std =
          Math.sqrt(
            agg.rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(agg.rets.length - 1, 1),
          ) || NaN;
        const sharpe = std ? (mean / std) * Math.sqrt(252) : NaN;
        return { year, ann, sharpe, n: agg.rets.length };
      })
      .sort((a, b) => a.year.localeCompare(b.year));
  }, [data]);

  if (isLoading) {
    return <Card><CardContent>Loading backtest…</CardContent></Card>;
  }
  if (!data) {
    return (
      <Card>
        <CardContent>
          No backtest yet. Trigger a run via the Refresh button.
        </CardContent>
      </Card>
    );
  }

  const m = data.run.metrics;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Backtest #{data.run.id}</h1>
        <p className="text-sm text-muted-foreground">
          {data.run.status} · {data.run.tickers_count} tickers · {data.run.note}
        </p>
      </div>

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

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Equity curve</CardTitle>
            <InfoTooltip termKey="inverse" title="Strategy vs Inverse" />
          </div>
          <CardSubtitle>
            Growth of $1, walk-forward, net of costs. The faint line is the inverse (doing the opposite of
            every trade). Drag on the chart to zoom; drag the bar below to scroll.
          </CardSubtitle>
        </CardHeader>
        <CardContent>
          <ZoomableChart
            data={equityRows}
            xKey="date"
            series={equitySeries}
            height={300}
            legend
            leftFormatter={(v) => `$${v.toFixed(2)}`}
            xTickFormatter={(v) => (v || "").slice(0, 7)}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Drawdown</CardTitle>
            <InfoTooltip termKey="max_drawdown" title="Drawdown" />
          </div>
        </CardHeader>
        <CardContent>
          <ZoomableChart
            data={equityRows}
            xKey="date"
            series={ddSeries}
            height={210}
            leftFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            xTickFormatter={(v) => (v || "").slice(0, 7)}
          />
        </CardContent>
      </Card>

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

      <Card>
        <CardHeader>
          <CardTitle>Yearly performance</CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm tabular">
            <thead className="text-xs uppercase text-muted-foreground">
              <tr>
                <th className="py-2 text-left">Year</th>
                <th className="py-2 text-right">Return</th>
                <th className="py-2 text-right">Sharpe</th>
                <th className="py-2 text-right">Days</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {yearly.map((y) => (
                <tr key={y.year}>
                  <td className="py-2">{y.year}</td>
                  <td className={cn("py-2 text-right", signClass(y.ann))}>{formatPercentSigned(y.ann)}</td>
                  <td className={cn("py-2 text-right", signClass(y.sharpe))}>{formatNumber(y.sharpe)}</td>
                  <td className="py-2 text-right text-muted-foreground">{y.n}</td>
                </tr>
              ))}
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
