import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ExternalLink, Info, Newspaper, Radio } from "lucide-react";
import { api } from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from "@/components/ui/Card";
import { ZoomableChart, ChartSeries, ChartRefLine } from "@/components/ui/ZoomableChart";
import { InfoTooltip, LabelWithInfo } from "@/components/ui/InfoTooltip";
import { SignalBanner } from "@/components/SignalBanner";
import { GlossaryKey } from "@/lib/glossary";
import {
  formatNumber,
  formatPercent,
  formatPercentSigned,
  formatDate,
  formatDateTime,
  formatCompactNumber,
  signClass,
} from "@/lib/format";
import { cn } from "@/lib/cn";

export function Ticker() {
  const { ticker = "" } = useParams();
  const { data, isLoading, error } = useQuery({
    queryKey: ["ticker", ticker],
    queryFn: () => api.tickerDetail(ticker, 730),
    enabled: !!ticker,
  });
  const news = useQuery({
    queryKey: ["news", ticker],
    queryFn: () => api.tickerNews(ticker, 15),
    enabled: !!ticker,
  });
  // Live (delayed) quote — polls every 10s. Server-cached so this is cheap.
  const quote = useQuery({
    queryKey: ["quote", ticker],
    queryFn: () => api.quote(ticker),
    enabled: !!ticker,
    refetchInterval: 10_000,
    refetchIntervalInBackground: false,
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

  // Price + score chart rows.
  const chartRows = data.prices.map((p) => {
    const pred = data.predictions.find((q) => q.date === p.date);
    return { date: p.date, close: p.adj_close, score: pred?.score ?? null };
  });

  // Latest prediction → the BUY/SELL signal for next session.
  const latestPred = [...data.predictions].sort((a, b) => a.date.localeCompare(b.date)).at(-1);

  // Per-stock "listen to the score" strategy: each day go long if score>0,
  // short if score<0, flat if no score; realise the next day's return; compound
  // from $1. Compared against simple buy-and-hold over the same window.
  const scoreByDate = new Map(data.predictions.map((q) => [q.date, q.score]));
  const sortedPrices = [...data.prices].filter((p) => p.adj_close != null);
  let stratEq = 1;
  let holdEq = 1;
  let traded = false;
  const strategyRows: { date: string; strategy: number; hold: number }[] = [];
  for (let i = 0; i < sortedPrices.length - 1; i++) {
    const c0 = sortedPrices[i].adj_close as number;
    const c1 = sortedPrices[i + 1].adj_close as number;
    if (!c0) continue;
    const ret = c1 / c0 - 1;
    const score = scoreByDate.get(sortedPrices[i].date);
    const pos = score == null ? 0 : score > 0 ? 1 : score < 0 ? -1 : 0;
    if (pos !== 0) traded = true;
    stratEq *= 1 + pos * ret;
    holdEq *= 1 + ret;
    strategyRows.push({ date: sortedPrices[i + 1].date, strategy: stratEq, hold: holdEq });
  }
  const stratFinal = strategyRows.at(-1)?.strategy ?? 1;
  const holdFinal = strategyRows.at(-1)?.hold ?? 1;

  const priceSeries: ChartSeries[] = [
    { type: "line", dataKey: "close", name: "Close", color: "hsl(var(--primary))", yAxisId: "left", strokeWidth: 1.6 },
    { type: "bar", dataKey: "score", name: "Score", color: "hsl(var(--positive))", yAxisId: "right", fillOpacity: 0.45 },
  ];
  const priceRefLines: ChartRefLine[] = [];
  if (data.fifty_two_week_high != null)
    priceRefLines.push({ y: data.fifty_two_week_high, yAxisId: "left", label: "52w high", color: "hsl(var(--negative))" });
  if (data.fifty_two_week_low != null)
    priceRefLines.push({ y: data.fifty_two_week_low, yAxisId: "left", label: "52w low", color: "hsl(var(--positive))" });

  const strategySeries: ChartSeries[] = [
    { type: "area", dataKey: "strategy", name: "Score strategy", color: "hsl(var(--primary))", fillOpacity: 0.18, strokeWidth: 1.8 },
    { type: "line", dataKey: "hold", name: "Buy & hold", color: "hsl(var(--muted-foreground))", strokeWidth: 1.2 },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link to="/screener" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:underline">
            <ArrowLeft className="h-4 w-4" /> back to screener
          </Link>
          <div className="mt-2 flex items-center gap-3">
            <h1 className="text-3xl font-semibold tracking-tight">{data.ticker}</h1>
            <LiveQuote
              price={quote.data?.price ?? null}
              change={quote.data?.change ?? null}
              changePct={quote.data?.change_pct ?? null}
              asOf={quote.data?.as_of}
              loading={quote.isLoading}
            />
          </div>
          <p className="text-sm text-muted-foreground">
            {data.industry ?? "—"}{data.sector ? ` · ${data.sector}` : ""}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Market cap" termKey="market_cap" value={formatCompactNumber(data.market_cap)} />
          <Stat label="Beta" termKey="beta" value={formatNumber(data.beta)} />
          <Stat label="P/E (TTM)" termKey="trailing_pe" value={formatNumber(data.trailing_pe)} />
          <Stat label="Div yield" termKey="dividend_yield" value={data.dividend_yield !== null ? formatPercent(data.dividend_yield) : "—"} />
        </div>
      </div>

      {/* BUY / SELL signal for next session */}
      <SignalBanner score={latestPred?.score ?? null} asOf={latestPred?.date ?? null} subject={`${data.ticker} — next session`} />

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Price &amp; predictions</CardTitle>
            <InfoTooltip termKey="score" title="What is the score?" />
          </div>
          <CardSubtitle>
            Adjusted close (left axis) with the model score overlaid as bars (right axis). Dashed lines mark
            the 52-week high/low — common support &amp; resistance levels. Drag on the chart to zoom; drag the
            bar below to scroll.
          </CardSubtitle>
        </CardHeader>
        <CardContent>
          <ZoomableChart
            data={chartRows}
            xKey="date"
            series={priceSeries}
            refLines={priceRefLines}
            rightAxis
            legend
            height={340}
            leftFormatter={(v) => `$${v.toFixed(0)}`}
            rightFormatter={(v) => v.toFixed(2)}
            xTickFormatter={(v) => (v || "").slice(0, 7)}
          />
        </CardContent>
      </Card>

      {/* Per-stock score strategy: growth of $1 */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>If you traded {data.ticker} on the score</CardTitle>
            <InfoTooltip termKey="per_stock_strategy" title="Score strategy ($1 growth)" />
          </div>
          <CardSubtitle>
            Go long when the score is positive, short when negative, flat otherwise — compounded from $1, vs.
            simply buying and holding. A single-name illustration; not the portfolio strategy.
          </CardSubtitle>
        </CardHeader>
        <CardContent>
          {!traded ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No model scores for {data.ticker} in this window yet — run a refresh to populate predictions.
            </p>
          ) : (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                <BigStat
                  label="$1 on the score →"
                  value={`$${stratFinal.toFixed(2)}`}
                  cls={signClass(stratFinal - 1)}
                  sub={formatPercentSigned(stratFinal - 1)}
                />
                <BigStat
                  label="$1 buy & hold →"
                  value={`$${holdFinal.toFixed(2)}`}
                  cls={signClass(holdFinal - 1)}
                  sub={formatPercentSigned(holdFinal - 1)}
                />
                <BigStat
                  label="Score edge"
                  value={formatPercentSigned(stratFinal - holdFinal)}
                  cls={signClass(stratFinal - holdFinal)}
                  sub="strategy − hold"
                />
              </div>
              <ZoomableChart
                data={strategyRows}
                xKey="date"
                series={strategySeries}
                height={260}
                legend
                leftFormatter={(v) => `$${v.toFixed(2)}`}
                xTickFormatter={(v) => (v || "").slice(0, 7)}
              />
            </>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Key fundamentals</CardTitle>
            <CardSubtitle>Hover any label for a plain-English explanation. Context only — never fed to the model.</CardSubtitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <Row label="Sector" value={data.sector ?? "—"} />
              <Row label="Industry" value={data.industry ?? "—"} />
              <Row label="52w high" termKey="fifty_two_week_high" value={formatNumber(data.fifty_two_week_high, { style: "currency", currency: "USD" })} />
              <Row label="52w low" termKey="fifty_two_week_low" value={formatNumber(data.fifty_two_week_low, { style: "currency", currency: "USD" })} />
              <Row label="Short ratio" termKey="short_ratio" value={formatNumber(data.short_ratio)} />
              <Row label="Short % float" termKey="short_percent_of_float" value={data.short_percent_of_float !== null ? formatPercent(data.short_percent_of_float) : "—"} />
              <Row label="Fwd P/E" termKey="forward_pe" value={formatNumber(data.forward_pe)} />
              <Row label="Beta" termKey="beta" value={formatNumber(data.beta)} />
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

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Newspaper className="h-4 w-4 text-muted-foreground" />
            <CardTitle>Recent news</CardTitle>
          </div>
          <CardSubtitle>
            Headlines from Yahoo Finance. We do not feed these into the model — see CONCEPTS.md §7e on
            news as features.
          </CardSubtitle>
        </CardHeader>
        <CardContent>
          {news.isLoading ? (
            <div className="text-sm text-muted-foreground">Loading…</div>
          ) : (news.data ?? []).length === 0 ? (
            <div className="text-sm text-muted-foreground">No news cached. Try the Refresh button.</div>
          ) : (
            <ul className="divide-y divide-border">
              {(news.data ?? []).map((n) => (
                <li key={n.uuid} className="py-3">
                  {n.link ? (
                    <a
                      href={n.link}
                      target="_blank"
                      rel="noopener noreferrer nofollow"
                      className="group flex items-start gap-2 hover:text-foreground"
                    >
                      <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground group-hover:text-foreground" />
                      <div className="flex-1">
                        <div className="text-sm font-medium leading-snug">{n.title || "(untitled)"}</div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {n.publisher ?? "Unknown publisher"} · {formatDateTime(n.published_at)}
                        </div>
                      </div>
                    </a>
                  ) : (
                    <div>
                      <div className="text-sm font-medium leading-snug">{n.title || "(untitled)"}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {n.publisher ?? "Unknown publisher"} · {formatDateTime(n.published_at)}
                      </div>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function LiveQuote({
  price,
  change,
  changePct,
  asOf,
  loading,
}: {
  price: number | null;
  change: number | null;
  changePct: number | null;
  asOf?: string;
  loading: boolean;
}) {
  if (loading && price == null) {
    return <span className="text-sm text-muted-foreground">loading quote…</span>;
  }
  if (price == null) return null;
  const up = (change ?? 0) >= 0;
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-2">
        <span className="text-2xl font-semibold tabular">${price.toFixed(2)}</span>
        {change != null && (
          <span className={cn("text-sm font-medium tabular", up ? "text-positive" : "text-negative")}>
            {up ? "▲" : "▼"} {formatNumber(Math.abs(change), { maximumFractionDigits: 2 })}
            {changePct != null && ` (${formatPercentSigned(changePct)})`}
          </span>
        )}
        {/* Pulsing badge — click/hover for full explanation */}
        <InfoTooltip
          title="About this price"
          body={
            <>
              <strong>~15-minute delayed.</strong> Source is yfinance, an unofficial scraper — not a
              real-time feed. Prices only update during market hours (NYSE: 9:30 AM–4:00 PM ET,
              Mon–Fri). Outside those hours the quote shows the last close.{" "}
              {asOf && <>Last fetched at {formatDateTime(asOf)} (server time).</>}
            </>
          }
        >
          <span className="inline-flex items-center gap-1 rounded-full bg-positive/10 px-2 py-0.5 text-[10px] font-medium text-positive">
            <Radio className="h-3 w-3 animate-pulse" /> LIVE
          </span>
        </InfoTooltip>
      </div>
      {/* Always-visible delay notice — no hover required */}
      <p className="text-[11px] text-muted-foreground">
        ~15 min delayed · market hours only · refreshes every 10s
        {asOf && <> · fetched {formatDateTime(asOf)}</>}
      </p>
    </div>
  );
}

function Stat({ label, termKey, value }: { label: string; termKey?: GlossaryKey; value: string }) {
  return (
    <div className="rounded-md border border-border bg-card px-3 py-2">
      <LabelWithInfo
        label={label}
        termKey={termKey}
        className="text-[10px] uppercase tracking-wide text-muted-foreground"
        side="bottom"
        align="end"
      />
      <div className="text-sm font-semibold tabular">{value}</div>
    </div>
  );
}

function BigStat({ label, value, sub, cls }: { label: string; value: string; sub?: string; cls?: string }) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={cn("text-2xl font-bold tabular", cls)}>{value}</div>
      {sub && <div className={cn("text-xs tabular", cls)}>{sub}</div>}
    </div>
  );
}

function Row({ label, termKey, value }: { label: string; termKey?: GlossaryKey; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">
        <LabelWithInfo label={label} termKey={termKey} side="top" align="start" />
      </dt>
      <dd className="text-right font-medium tabular">{value}</dd>
    </>
  );
}
