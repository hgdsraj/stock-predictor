import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip, AreaChart, Area, CartesianGrid } from "recharts";
import { TrendingUp, TrendingDown, Activity, Calendar } from "lucide-react";
import { api } from "@/api/client";
import { Card, CardHeader, CardTitle, CardSubtitle, CardContent } from "@/components/ui/Card";
import { formatPercent, formatPercentSigned, formatDate, signClass, formatNumber } from "@/lib/format";
import { cn } from "@/lib/cn";

export function Home() {
  const movers = useQuery({ queryKey: ["movers"], queryFn: () => api.latestPredictions(10) });
  const summary = useQuery({ queryKey: ["backtest"], queryFn: () => api.backtestSummary().catch(() => null) });

  const metrics = summary.data?.run.metrics ?? {};
  const equity = (summary.data?.equity_curve ?? []).map((p) => ({
    date: p.date,
    value: p.cumulative_return !== null ? p.cumulative_return + 1 : null,
    dd: p.drawdown,
  }));

  return (
    <div className="space-y-6">
      {/* Header / status */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Today’s view</h1>
          <p className="text-sm text-muted-foreground">
            Long/short cross-sectional signal. As of {formatDate(movers.data?.date)}.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Calendar className="h-4 w-4" />
          <span>Last update: {formatDate(summary.data?.run.completed_at)}</span>
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <KPI label="Sharpe (net)" value={metrics.sharpe} kind="number" />
        <KPI label="Ann. return" value={metrics.ann_return} kind="percent" signed />
        <KPI label="Max drawdown" value={metrics.max_drawdown} kind="percent" />
        <KPI label="Ann. vol" value={metrics.ann_vol} kind="percent" />
      </div>

      {/* Equity curve */}
      <Card>
        <CardHeader>
          <CardTitle>Equity curve</CardTitle>
          <CardSubtitle>Cumulative growth of $1 (long/short ensemble). Honest, walk-forward.</CardSubtitle>
        </CardHeader>
        <CardContent>
          <div className="h-72 w-full">
            {summary.isLoading ? (
              <Skeleton />
            ) : equity.length === 0 ? (
              <EmptyChart message="No backtest yet. Trigger a refresh." />
            ) : (
              <ResponsiveContainer>
                <AreaChart data={equity}>
                  <defs>
                    <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
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
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--primary))"
                    fill="url(#g1)"
                    strokeWidth={1.5}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Top movers */}
      <div className="grid gap-4 md:grid-cols-2">
        <MoversCard
          title="Top longs"
          subtitle="Highest predicted-score names today"
          rows={movers.data?.long ?? []}
          icon={<TrendingUp className="h-4 w-4 text-positive" />}
          loading={movers.isLoading}
          side="long"
        />
        <MoversCard
          title="Top shorts"
          subtitle="Lowest predicted-score names today"
          rows={movers.data?.short ?? []}
          icon={<TrendingDown className="h-4 w-4 text-negative" />}
          loading={movers.isLoading}
          side="short"
        />
      </div>
    </div>
  );
}

function KPI({
  label,
  value,
  kind,
  signed = false,
}: {
  label: string;
  value: number | undefined;
  kind: "percent" | "number";
  signed?: boolean;
}) {
  const display =
    value === undefined
      ? "—"
      : kind === "percent"
        ? signed
          ? formatPercentSigned(value)
          : formatPercent(value)
        : formatNumber(value);
  return (
    <Card>
      <CardContent className="space-y-1">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        <div className={cn("text-2xl font-semibold tabular", signed && signClass(value))}>{display}</div>
      </CardContent>
    </Card>
  );
}

interface MoversProps {
  title: string;
  subtitle: string;
  rows: { ticker: string; score: number; weight: number | null; per_horizon: Record<string, number | null> }[];
  icon: JSX.Element;
  loading: boolean;
  side: "long" | "short";
}

function MoversCard({ title, subtitle, rows, icon, loading, side }: MoversProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          {icon}
          <CardTitle>{title}</CardTitle>
        </div>
        <CardSubtitle>{subtitle}</CardSubtitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton />
        ) : rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">No data yet.</p>
        ) : (
          <ul className="divide-y divide-border">
            {rows.map((r) => (
              <li key={r.ticker} className="flex items-center justify-between py-2">
                <Link
                  to={`/ticker/${encodeURIComponent(r.ticker)}`}
                  className="flex flex-col text-sm font-medium hover:underline"
                >
                  {r.ticker}
                  <span className="text-xs text-muted-foreground">weight {formatPercent(Math.abs(r.weight ?? 0), 1)}</span>
                </Link>
                <div className={cn("font-semibold tabular text-sm", side === "long" ? "text-positive" : "text-negative")}>
                  {formatNumber(r.score, { signDisplay: "always", maximumFractionDigits: 3 })}
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function Skeleton() {
  return <div className="h-full w-full animate-pulse rounded-md bg-muted" />;
}

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
      <Activity className="mr-2 h-4 w-4" /> {message}
    </div>
  );
}
