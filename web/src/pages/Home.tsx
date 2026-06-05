import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { TrendingUp, TrendingDown, Activity, Calendar, Pin } from "lucide-react";
import { api } from "@/api/client";
import { Card, CardHeader, CardTitle, CardSubtitle, CardContent } from "@/components/ui/Card";
import { ZoomableChart, ChartSeries } from "@/components/ui/ZoomableChart";
import { InfoTooltip, LabelWithInfo } from "@/components/ui/InfoTooltip";
import { SignalBanner } from "@/components/SignalBanner";
import { GlossaryKey } from "@/lib/glossary";
import { formatPercent, formatPercentSigned, formatDate, signClass, formatNumber } from "@/lib/format";
import { useActiveRun } from "@/hooks/useActiveRun";
import { cn } from "@/lib/cn";

export function Home() {
  // The active run determines what data we fetch. Pinning is reflected in the
  // URL (?run_id=…) and the header RunPicker, but we surface a small banner
  // here too so a user can't forget they're looking at a non-default run.
  const { runId, isPinned, queryKeyPart, setRunId } = useActiveRun();
  const movers = useQuery({
    queryKey: ["movers", queryKeyPart],
    queryFn: () => api.latestPredictions(10, runId),
  });
  const summary = useQuery({
    queryKey: ["backtest", queryKeyPart],
    queryFn: () => api.backtestSummary(runId).catch(() => null),
  });

  const metrics = summary.data?.run.metrics ?? {};

  // Recompute strategy, S&P 500, and inverse growth-of-$1 from daily returns
  // so all three curves share the same $1 basis.
  const rawCurve = summary.data?.equity_curve ?? [];
  const hasSPY = rawCurve.some((p) => p.benchmark_return != null && p.benchmark_return !== 0);
  let cumStrat = 1, cumInv = 1, cumSPY = 1;
  const equity = rawCurve.map((p) => {
    const r = p.daily_return ?? 0;
    const br = p.benchmark_return ?? 0;
    cumStrat *= 1 + r;
    cumInv *= 1 - r;
    cumSPY *= 1 + br;
    return { date: p.date, strategy: cumStrat, spy: cumSPY, inverse: cumInv };
  });

  const equitySeries: ChartSeries[] = [
    { type: "area", dataKey: "strategy", name: "Strategy", color: "hsl(var(--primary))", fillOpacity: 0.18, strokeWidth: 1.8 },
    ...(hasSPY ? [{ type: "line" as const, dataKey: "spy", name: "S&P 500", color: "#f59e0b", strokeWidth: 1.5, strokeDasharray: "5 3" }] : []),
    { type: "line", dataKey: "inverse", name: "Inverse", color: "hsl(var(--muted-foreground))", strokeWidth: 1.2 },
  ];

  const topLong = movers.data?.long?.[0];
  const topShort = movers.data?.short?.[0];

  return (
    <div className="space-y-6">
      {/* Pinned-run notice: shown only when the user has explicitly overridden
          the server default. Clicking "Clear" reverts to the default. */}
      {isPinned && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-primary/30 bg-primary/5 px-3 py-2 text-sm">
          <span className="flex items-center gap-2">
            <Pin className="h-3.5 w-3.5 text-primary" />
            Viewing pinned run{" "}
            <Link to={`/runs?expanded=${runId}`} className="font-mono font-semibold text-primary underline-offset-2 hover:underline">
              #{runId}
            </Link>{" "}
            — not the server default.
          </span>
          <button
            onClick={() => setRunId(null)}
            className="rounded-md border border-primary/30 px-2 py-0.5 text-xs text-primary hover:bg-primary/10"
          >
            Clear pin
          </button>
        </div>
      )}

      {/* Header / status */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Today’s view</h1>
          <p className="text-sm text-muted-foreground">
            Long/short cross-sectional signal. As of {formatDate(movers.data?.date)}
            {summary.data?.run.id != null && (
              <>
                {" · "}
                <Link to="/runs" className="hover:underline">
                  Run #{summary.data.run.id}
                </Link>
              </>
            )}
            .
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Calendar className="h-4 w-4" />
          <span>Last update: {formatDate(summary.data?.run.completed_at)}</span>
        </div>
      </div>

      {/* Headline signals (highest-conviction calls for next session) */}
      <div className="grid gap-3 md:grid-cols-2">
        <Link to={topLong ? `/ticker/${encodeURIComponent(topLong.ticker)}` : "#"} className="block">
          <SignalBanner score={topLong?.score ?? null} side="long" subject={topLong?.ticker} asOf={movers.data?.date} />
        </Link>
        <Link to={topShort ? `/ticker/${encodeURIComponent(topShort.ticker)}` : "#"} className="block">
          <SignalBanner score={topShort?.score ?? null} side="short" subject={topShort?.ticker} asOf={movers.data?.date} />
        </Link>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <KPI label="Sharpe (net)" termKey="sharpe" value={metrics.sharpe} kind="number" />
        <KPI label="Ann. return" termKey="ann_return" value={metrics.ann_return} kind="percent" signed />
        <KPI label="Max drawdown" termKey="max_drawdown" value={metrics.max_drawdown} kind="percent" />
        <KPI label="Ann. vol" termKey="ann_vol" value={metrics.ann_vol} kind="percent" />
      </div>

      {/* Equity curve (strategy vs inverse) */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Equity curve</CardTitle>
            <InfoTooltip termKey="inverse" title="Strategy vs Inverse" />
          </div>
          <CardSubtitle>
            Growth of $1, walk-forward. The faint line is the <strong>inverse</strong> — doing the exact
            opposite of every trade. For a real signal the inverse should lose; if it wins, the signal is
            backwards. Drag on the chart to zoom; drag the bar below to scroll.
          </CardSubtitle>
        </CardHeader>
        <CardContent>
          {summary.isLoading ? (
            <Skeleton />
          ) : equity.length === 0 ? (
            <EmptyChart message="No backtest yet. Queue a job on the Jobs page." />
          ) : (
            <ZoomableChart
              data={equity}
              xKey="date"
              series={equitySeries}
              height={300}
              legend
              leftFormatter={(v) => `$${v.toFixed(2)}`}
              xTickFormatter={(v) => (v || "").slice(0, 7)}
            />
          )}
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
  termKey,
  value,
  kind,
  signed = false,
}: {
  label: string;
  termKey: GlossaryKey;
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
        <LabelWithInfo
          label={label}
          termKey={termKey}
          className="text-xs uppercase tracking-wide text-muted-foreground"
        />
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
          <InfoTooltip termKey="score" title="What is the score?" />
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
  return <div className="h-72 w-full animate-pulse rounded-md bg-muted" />;
}

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="flex h-72 items-center justify-center text-sm text-muted-foreground">
      <Activity className="mr-2 h-4 w-4" /> {message}
    </div>
  );
}
