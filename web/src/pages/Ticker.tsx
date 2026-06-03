import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Line,
  Bar,
  Legend,
} from "recharts";
import { ArrowLeft, Info } from "lucide-react";
import { api } from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from "@/components/ui/Card";
import { formatNumber, formatPercent, formatDate, formatCompactNumber } from "@/lib/format";

export function Ticker() {
  const { ticker = "" } = useParams();
  const { data, isLoading, error } = useQuery({
    queryKey: ["ticker", ticker],
    queryFn: () => api.tickerDetail(ticker, 730),
    enabled: !!ticker,
  });

  if (error) {
    return (
      <div className="space-y-4">
        <Link to="/screener" className="inline-flex items-center gap-1 text-sm text-muted-foreground">
          <ArrowLeft className="h-4 w-4" /> back to screener
        </Link>
        <Card>
          <CardContent className="text-sm">
            <Info className="mr-1 inline h-4 w-4" /> Failed to load <code>{ticker}</code>: {(error as Error).message}
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isLoading || !data) {
    return <Card><CardContent>Loading {ticker}…</CardContent></Card>;
  }

  // Merge prices and predictions for the composed chart.
  const chartRows = data.prices.map((p) => {
    const pred = data.predictions.find((q) => q.date === p.date);
    return {
      date: p.date,
      close: p.adj_close,
      volume: p.volume,
      score: pred?.score ?? null,
    };
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link to="/screener" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:underline">
            <ArrowLeft className="h-4 w-4" /> back to screener
          </Link>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">{data.ticker}</h1>
          <p className="text-sm text-muted-foreground">
            {data.industry ?? "—"}{data.sector ? ` · ${data.sector}` : ""}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Market cap" value={formatCompactNumber(data.market_cap)} />
          <Stat label="Beta" value={formatNumber(data.beta)} />
          <Stat label="P/E (TTM)" value={formatNumber(data.trailing_pe)} />
          <Stat label="Div yield" value={data.dividend_yield !== null ? formatPercent(data.dividend_yield) : "—"} />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Price & predictions</CardTitle>
          <CardSubtitle>Adjusted close with our model score overlaid; signal at end-of-day.</CardSubtitle>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full">
            <ResponsiveContainer>
              <ComposedChart data={chartRows}>
                <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis yAxisId="left" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 11 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="close"
                  stroke="hsl(var(--primary))"
                  strokeWidth={1.5}
                  dot={false}
                  name="Close"
                />
                <Bar
                  yAxisId="right"
                  dataKey="score"
                  fill="hsl(var(--positive))"
                  fillOpacity={0.5}
                  name="Score"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Key fundamentals</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <Row label="Sector" value={data.sector ?? "—"} />
              <Row label="Industry" value={data.industry ?? "—"} />
              <Row label="52w high" value={formatNumber(data.fifty_two_week_high, { style: "currency", currency: "USD" })} />
              <Row label="52w low" value={formatNumber(data.fifty_two_week_low, { style: "currency", currency: "USD" })} />
              <Row label="Short ratio" value={formatNumber(data.short_ratio)} />
              <Row label="Short % float" value={data.short_percent_of_float !== null ? formatPercent(data.short_percent_of_float) : "—"} />
              <Row label="Fwd P/E" value={formatNumber(data.forward_pe)} />
              <Row label="Beta" value={formatNumber(data.beta)} />
            </dl>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>About</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="max-h-64 overflow-auto text-sm leading-relaxed text-muted-foreground">
              {data.long_business_summary || "No description available."}
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-card px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold tabular">{value}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="text-right font-medium tabular">{value}</dd>
    </>
  );
}
