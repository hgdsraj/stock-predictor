import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  LineChart,
  Line,
} from "recharts";
import { api } from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from "@/components/ui/Card";
import { formatPercent, formatPercentSigned, formatNumber, signClass } from "@/lib/format";
import { cn } from "@/lib/cn";

export function Backtest() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["bt"],
    queryFn: () => api.backtestSummary().catch(() => null),
  });

  const equityRows = (data?.equity_curve ?? []).map((p) => ({
    date: p.date,
    equity: p.cumulative_return !== null ? p.cumulative_return + 1 : null,
    dd: p.drawdown,
  }));

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
        <Tile label="Ann. return" value={formatPercentSigned(m.ann_return)} signed={m.ann_return} />
        <Tile label="Sharpe" value={formatNumber(m.sharpe)} signed={m.sharpe} />
        <Tile label="Sortino" value={formatNumber(m.sortino)} signed={m.sortino} />
        <Tile label="Max DD" value={formatPercent(m.max_drawdown)} />
        <Tile label="Calmar" value={formatNumber(m.calmar)} signed={m.calmar} />
        <Tile label="Vol" value={formatPercent(m.ann_vol)} />
        <Tile label="Hit ratio" value={formatPercent(m.hit_ratio)} />
        <Tile label="Days" value={String(m.n_days ?? "—")} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Equity curve</CardTitle>
          <CardSubtitle>Walk-forward, net of costs. Honest — including the bad parts.</CardSubtitle>
        </CardHeader>
        <CardContent>
          <div className="h-72 w-full">
            <ResponsiveContainer>
              <AreaChart data={equityRows}>
                <defs>
                  <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                <Area type="monotone" dataKey="equity" stroke="hsl(var(--primary))" fill="url(#eq)" strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Drawdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48 w-full">
            <ResponsiveContainer>
              <LineChart data={equityRows}>
                <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                <Line type="monotone" dataKey="dd" stroke="hsl(var(--negative))" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
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
                <th className="py-2 text-right">Hit rate</th>
                <th className="py-2 text-right">IC mean</th>
                <th className="py-2 text-right">IC IR</th>
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

function Tile({ label, value, signed }: { label: string; value: string; signed?: number }) {
  return (
    <Card>
      <CardContent className="space-y-1">
        <div className="text-xs uppercase text-muted-foreground">{label}</div>
        <div className={cn("text-xl font-semibold tabular", signed !== undefined && signClass(signed))}>{value}</div>
      </CardContent>
    </Card>
  );
}
