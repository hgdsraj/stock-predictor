import { TrendingUp, TrendingDown, MinusCircle } from "lucide-react";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import { formatDate, formatNumber } from "@/lib/format";
import { cn } from "@/lib/cn";

export type Side = "long" | "short" | null;

interface SignalBannerProps {
  /** Latest score; sign drives the call. */
  score: number | null | undefined;
  /** Explicit side override (else derived from score sign). */
  side?: Side;
  /** Date the latest signal is based on. */
  asOf?: string | null;
  /** Optional subject, e.g. a ticker. Omit for the portfolio-level banner. */
  subject?: string;
  className?: string;
}

/**
 * Bold BUY / SELL / NEUTRAL banner. Because the model predicts forward
 * returns, the most recent score is effectively the call for the next session.
 */
export function SignalBanner({ score, side, asOf, subject, className }: SignalBannerProps) {
  const resolved: Side =
    side ?? (score == null ? null : score > 0 ? "long" : score < 0 ? "short" : null);

  const hasData = score != null || side != null;

  const config = {
    long: {
      label: "BUY",
      sub: "Model expects out-performance next session",
      icon: TrendingUp,
      box: "border-positive/40 bg-positive/10",
      text: "text-positive",
    },
    short: {
      label: "SELL",
      sub: "Model expects under-performance next session",
      icon: TrendingDown,
      box: "border-negative/40 bg-negative/10",
      text: "text-negative",
    },
    neutral: {
      label: "NEUTRAL",
      sub: "No clear directional signal",
      icon: MinusCircle,
      box: "border-border bg-muted/40",
      text: "text-muted-foreground",
    },
  };

  const c = resolved ? config[resolved] : config.neutral;
  const Icon = c.icon;

  return (
    <div className={cn("flex items-center justify-between gap-4 rounded-xl border p-4", c.box, className)}>
      <div className="flex items-center gap-4">
        <Icon className={cn("h-8 w-8 shrink-0", c.text)} />
        <div>
          <div className="flex items-center gap-2">
            <span className={cn("text-2xl font-extrabold tracking-tight", c.text)}>
              {hasData ? c.label : "—"}
            </span>
            {subject && <span className="text-lg font-semibold text-foreground">{subject}</span>}
            <InfoTooltip termKey="signal" side="bottom" align="start" />
          </div>
          <p className="text-xs text-muted-foreground">
            {hasData ? c.sub : "No prediction available yet — run a refresh."}
          </p>
        </div>
      </div>
      <div className="text-right">
        {score != null && (
          <div className={cn("text-lg font-bold tabular", c.text)}>
            {formatNumber(score, { signDisplay: "always", maximumFractionDigits: 3 })}
          </div>
        )}
        <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
          signal for next session
        </div>
        {asOf && <div className="text-[11px] text-muted-foreground">as of {formatDate(asOf)}</div>}
      </div>
    </div>
  );
}
