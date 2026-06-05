import { ReactNode } from "react";
import {
  Database,
  Cog,
  Layers,
  Wallet,
  ShieldCheck,
  TriangleAlert,
  GitBranch,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from "@/components/ui/Card";
import { InfoTooltip } from "@/components/ui/InfoTooltip";

export function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">How this works</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          A plain-English walkthrough of the full algorithm — from raw prices to the BUY/SELL
          signal you see on each page. The one sentence the whole project is built around:{" "}
          <em>
            most published backtests are wrong because the future subtly leaks into the past, and
            avoiding that leakage is harder than building the model itself.
          </em>
        </p>
      </div>

      <Section icon={<Layers className="h-5 w-5" />} title="1. The goal" >
        <p>
          Every trading day, for every stock in the S&P 500, we predict its{" "}
          <strong>return over the next few days</strong>. Then we build a{" "}
          <strong>dollar-neutral long/short portfolio</strong>: buy the names we score highest, short
          the names we score lowest, in equal dollar amounts.
        </p>
        <p>
          Going long and short the same dollar amount cancels out general market moves. If the whole
          market rises 1%, our longs and shorts both rise ~1% and roughly cancel. What's left is the{" "}
          <strong>cross-sectional</strong> question: did the stocks we picked beat the stocks we
          shorted? That's a much cleaner test of whether the model knows anything — and it means we{" "}
          <em>cannot</em> tell you whether the market as a whole goes up tomorrow, only which stocks
          look relatively strong or weak.
        </p>
      </Section>

      <Section icon={<Database className="h-5 w-5" />} title="2. Data">
        <p>
          All free, all public: daily OHLCV prices and basic fundamentals from{" "}
          <Mono>yfinance</Mono>, macro series (VIX, term spread, USD) from <Mono>FRED</Mono>, and
          point-in-time S&P 500 membership reconstructed from Wikipedia's change log. Prices are
          cached locally as parquet so repeat runs are fast.
        </p>
        <p>
          Using point-in-time membership matters: if we backtested only on stocks that are in the
          index <em>today</em>, we'd silently exclude every company that went bankrupt or got
          delisted — making the strategy look 3–5%/yr better than reality (survivorship bias).
        </p>
      </Section>

      <Section icon={<Cog className="h-5 w-5" />} title="3. Features & the model">
        <p>
          For each stock/day we compute features that only look <strong>backward</strong> in time:
          trailing technical signals, and in Phase 5 a richer "tier-2" set — 12-1 momentum, idiosyncratic
          volatility, market beta, max daily return, Amihud illiquidity — plus market-regime features
          (VIX, term spread, USD, cross-sectional dispersion).
        </p>
        <p>
          A <strong>LightGBM</strong> gradient-boosted tree model is trained per forecast horizon to
          predict forward returns. Its output is a single{" "}
          <span className="inline-flex items-center gap-1">
            <strong>score</strong>
            <InfoTooltip termKey="score" />
          </span>{" "}
          per stock per day — positive means we expect out-performance, negative under-performance,
          and bigger magnitude means more conviction. The score has no units; it's used only to{" "}
          <strong>rank</strong> stocks within each day.
        </p>
      </Section>

      <Section icon={<GitBranch className="h-5 w-5" />} title="4. The ensemble">
        <p>
          We don't train one model — we train several, one per horizon (e.g. 1-day and 5-day). Different
          horizons capture different effects (short-term mean-reversion vs. weekly momentum), and
          disagreement between them is a useful leakage cross-check.
        </p>
        <p>
          Phase 5 combines them with an{" "}
          <span className="inline-flex items-center gap-1">
            <strong>IC-IR weighting</strong>
            <InfoTooltip termKey="ic_ir" />
          </span>
          : each horizon is weighted by how <em>consistent</em> its out-of-sample signal is, and any
          horizon whose signal is ≤ 0 is dropped entirely so it can't drag down the good ones.
        </p>
      </Section>

      <Section icon={<Wallet className="h-5 w-5" />} title="5. From scores to a portfolio">
        <p>A score per stock isn't a strategy. We turn scores into position weights:</p>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <strong>Vol-scaled sizing</strong> — weight ∝ |score| ÷ volatility, so a high-conviction,
            low-volatility name gets more capital than a wild one.
          </li>
          <li>
            <strong>Sector caps</strong> — no single GICS sector exceeds a gross-exposure cap (default
            25–30%) so the book doesn't quietly become a sector bet.
          </li>
          <li>
            <strong>Minimum trade threshold</strong> — skip rebalances smaller than ~0.5% so we don't
            bleed fees trading noise.
          </li>
          <li>
            <strong>Beta-neutralisation</strong> (optional) — strip out market beta so returns reflect
            stock-picking, not market exposure.
          </li>
        </ul>
      </Section>

      <Section icon={<ShieldCheck className="h-5 w-5" />} title="6. Honest validation">
        <p>
          This is where most backtests cheat and where we spend the most effort:
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <strong>Walk-forward CV</strong> — train on the past, test on the future, never the
            reverse. The training window expands over time like real life.
          </li>
          <li>
            <strong>Purge &amp; embargo</strong> — drop training labels whose forward window overlaps
            the test window, plus a buffer measured in <em>trading</em> days, so future prices can't
            leak through labels.
          </li>
          <li>
            <strong>Untouched holdout</strong> — the last N years are never seen during training or
            tuning. The numbers we report are the holdout numbers.
          </li>
          <li>
            <strong>Realistic costs</strong> — ~12 bps round-trip by default. A cost-free backtest is
            fiction.
          </li>
          <li>
            <strong>Bootstrap confidence interval</strong> — if zero sits inside the Sharpe CI, the
            strategy isn't statistically distinguishable from random.
          </li>
        </ul>
      </Section>

      <Section icon={<TriangleAlert className="h-5 w-5" />} title="7. What it cannot do">
        <p>Said plainly, so there's no confusion:</p>
        <ul className="ml-4 list-disc space-y-1">
          <li>It will not place real orders — there is no broker integration. Backtest only.</li>
          <li>It cannot predict whether the market as a whole rises tomorrow.</li>
          <li>It does not day-trade — daily bars only.</li>
          <li>
            It very likely will not beat the market net of costs. A realistic target for a free-data,
            daily-bar, S&P-only long/short strategy is a net{" "}
            <span className="inline-flex items-center gap-1">
              Sharpe
              <InfoTooltip termKey="sharpe" />
            </span>{" "}
            of 0.4–0.8. Treat every number on this site as <strong>research output, not advice.</strong>
          </li>
        </ul>
      </Section>

      <p className="pb-4 text-center text-xs text-muted-foreground">
        Deeper reading lives in the repo: <Mono>docs/CONCEPTS.md</Mono>, <Mono>docs/USAGE.md</Mono>,{" "}
        and <Mono>docs/PROJECT_LOG.md</Mono>.
      </p>
    </div>
  );
}

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2 text-primary">
          {icon}
          <CardTitle>{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0 text-sm leading-relaxed text-muted-foreground">
        {children}
      </CardContent>
    </Card>
  );
}

function Mono({ children }: { children: ReactNode }) {
  return <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>;
}
