// Runs history page.
//
// The single source of truth for "what model runs have we done, and which one
// should the live UI show?". Built around the existing /runs endpoint, which
// now returns the full PipelineConfig snapshot, the triggering job's id, and
// the active-pin flag.
//
// What the page does:
//   - Sortable, filterable table of every run we have (capped at 50 by
//     cleanup_old_runs on the backend).
//   - Per-row "View" → pins the run via the URL (?run_id=) and navigates Home
//     so the whole site immediately re-fetches against that run.
//   - Per-row "Activate" → server-side pin, requires X-Password, affects all
//     viewers (the dropdown shows "server default" for the new active row).
//   - Expandable row reveals the full config JSON + a mini equity sparkline.
//   - Side-by-side compare: pick two rows via checkbox → modal diffs configs
//     and metrics, plus overlaid equity curves.
//   - Each row links back to the triggering job in /jobs (`?job=<id>`).
//
// Why URL-driven instead of context: pinning is shareable, browser-history
// friendly, and lets a user "open in new tab" different runs without one
// fighting the other. We never read or write localStorage.

import { useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import {
  History,
  Pin,
  PinOff,
  Eye,
  Briefcase,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  XCircle,
  Clock,
  Loader2,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Copy,
  GitCompareArrows,
  X,
} from "lucide-react";
import { api } from "@/api/client";
import type { RunSummary, BacktestSummary } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useActiveRun } from "@/hooks/useActiveRun";
import { formatNumber, formatPercent, formatPercentSigned, formatDateTime, signClass } from "@/lib/format";
import { cn } from "@/lib/cn";

// ─── Types & helpers ─────────────────────────────────────────────────────────

type SortKey = "id" | "completed_at" | "status" | "phase" | "tickers" | "sharpe" | "annret" | "maxdd";
type SortDir = "asc" | "desc";

function getPhase(r: RunSummary): number | null {
  const p = r.config?.phase;
  if (typeof p === "number") return p;
  return null;
}

function getMetric(r: RunSummary, key: string): number | null {
  const v = r.metrics?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function statusBadge(status: string) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    ok:      { cls: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",  icon: <CheckCircle2 className="h-3 w-3" />, label: "ok" },
    failed:  { cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",          icon: <AlertTriangle className="h-3 w-3" />, label: "failed" },
    running: { cls: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",      icon: <Loader2 className="h-3 w-3 animate-spin" />, label: "running" },
    pending: { cls: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300", icon: <Clock className="h-3 w-3" />, label: "pending" },
  };
  const s = map[status] ?? { cls: "bg-gray-100 text-gray-600", icon: <XCircle className="h-3 w-3" />, label: status };
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", s.cls)}>
      {s.icon}
      {s.label}
    </span>
  );
}

// "Phase 5 · meta-on · EDGAR" style tag stream — a quick visual of what the
// run actually enabled, derived from the config dict.
function configBadges(cfg: Record<string, unknown>): string[] {
  const bits: string[] = [];
  if (cfg.use_meta_labelling) bits.push("meta");
  if (cfg.use_triple_barrier_labels) bits.push("triple-barrier");
  if (cfg.ranks_only) bits.push("ranks-only");
  if (cfg.use_sector_features) bits.push("sector");
  if (cfg.use_tier2_features) bits.push("tier2");
  if (cfg.use_regime_features) bits.push("regime");
  if (cfg.beta_neutralise) bits.push("β-neutral");
  if (cfg.use_edgar_features) bits.push("EDGAR-evt");
  if (cfg.use_edgar_item_features) bits.push("EDGAR-item");
  if (cfg.use_gdelt_features) bits.push("GDELT");
  if (typeof cfg.bayesian_shrinkage_alpha === "number" && cfg.bayesian_shrinkage_alpha > 0) {
    bits.push(`shrink ${(cfg.bayesian_shrinkage_alpha as number).toFixed(2)}`);
  }
  return bits;
}

// ─── Mini sparkline ──────────────────────────────────────────────────────────
//
// Tiny inline equity-curve preview. We fetch the run's equity samples lazily
// only when a row is expanded so the initial /runs page is cheap. SVG path
// drawn directly (no recharts) to keep the row visually compact.

function Sparkline({ runId }: { runId: number }) {
  const { data, isLoading } = useQuery({
    queryKey: ["runEquity", runId],
    queryFn: () => api.runEquity(runId),
    staleTime: 5 * 60_000,
  });

  if (isLoading) return <div className="h-10 w-32 animate-pulse rounded bg-muted" />;
  if (!data || data.length === 0) return <div className="text-xs text-muted-foreground">no equity data</div>;

  // Cumulative-return curve (already provided by the API). Pad / clip values.
  const points = data.map((p) => p.cumulative_return ?? 0);
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const W = 160;
  const H = 36;
  const path = points
    .map((v, i) => {
      const x = (i / Math.max(points.length - 1, 1)) * W;
      const y = H - ((v - min) / span) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const final = points[points.length - 1];
  const positive = final >= 0;

  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      className="overflow-visible"
      aria-label={`equity sparkline ending at ${(final * 100).toFixed(1)}%`}
    >
      {/* zero line if it's in range */}
      {min < 0 && max > 0 && (
        <line
          x1={0}
          x2={W}
          y1={H - (-min / span) * H}
          y2={H - (-min / span) * H}
          stroke="currentColor"
          strokeOpacity={0.15}
          strokeDasharray="2 2"
        />
      )}
      <path
        d={path}
        fill="none"
        stroke={positive ? "hsl(142 71% 45%)" : "hsl(0 84% 60%)"}
        strokeWidth={1.5}
      />
    </svg>
  );
}

// ─── Compare modal ───────────────────────────────────────────────────────────
//
// Two-run side-by-side diff. We pull both BacktestSummary payloads and:
//   - List config keys whose values differ.
//   - Render the two equity curves overlaid in SVG.
//   - Show side-by-side KPI tiles.

function CompareModal({ a, b, onClose }: { a: number; b: number; onClose: () => void }) {
  const qa = useQuery({ queryKey: ["runBt", a], queryFn: () => api.runBacktest(a) });
  const qb = useQuery({ queryKey: ["runBt", b], queryFn: () => api.runBacktest(b) });

  const both = qa.data && qb.data ? [qa.data, qb.data] as [BacktestSummary, BacktestSummary] : null;

  const diff = useMemo(() => {
    if (!both) return [];
    const [A, B] = both;
    const keys = new Set([...Object.keys(A.run.config), ...Object.keys(B.run.config)]);
    const rows: { key: string; a: unknown; b: unknown }[] = [];
    keys.forEach((k) => {
      const va = A.run.config[k];
      const vb = B.run.config[k];
      // Stable JSON compare so we don't false-flag object identity differences.
      if (JSON.stringify(va) !== JSON.stringify(vb)) rows.push({ key: k, a: va, b: vb });
    });
    return rows.sort((x, y) => x.key.localeCompare(y.key));
  }, [both]);

  // Overlaid equity curves: align by date string, draw two SVG paths.
  const overlay = useMemo(() => {
    if (!both) return null;
    const [A, B] = both;
    // Recompute strategy = product(1+r) so both curves start at 1.
    function recompute(rows: BacktestSummary["equity_curve"]) {
      let c = 1;
      return rows.map((p) => {
        c *= 1 + (p.daily_return ?? 0);
        return { date: p.date, v: c };
      });
    }
    const A2 = recompute(A.equity_curve);
    const B2 = recompute(B.equity_curve);
    const all = [...A2.map((p) => p.v), ...B2.map((p) => p.v)];
    if (all.length === 0) return null;
    const min = Math.min(...all);
    const max = Math.max(...all);
    const span = max - min || 1;
    const W = 520;
    const H = 160;
    function path(rows: { date: string; v: number }[]) {
      return rows
        .map((p, i) => {
          const x = (i / Math.max(rows.length - 1, 1)) * W;
          const y = H - ((p.v - min) / span) * H;
          return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
    }
    return { pathA: path(A2), pathB: path(B2), W, H };
  }, [both]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div className="flex items-center gap-2">
            <GitCompareArrows className="h-4 w-4 text-primary" />
            <h2 className="text-lg font-semibold">
              Compare run #{a} vs #{b}
            </h2>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} title="Close">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex-1 space-y-5 overflow-y-auto px-6 py-4">
          {qa.isLoading || qb.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading runs…</p>
          ) : !both ? (
            <p className="text-sm text-red-500">Failed to load one or both runs.</p>
          ) : (
            <>
              {/* KPI side-by-side */}
              <div className="grid grid-cols-2 gap-4">
                {(["sharpe", "ann_return", "max_drawdown", "ann_vol"] as const).map((key) => {
                  const va = getMetric(both[0].run, key);
                  const vb = getMetric(both[1].run, key);
                  const fmt = key === "sharpe" ? (v: number | null) => (v != null ? formatNumber(v) : "—") :
                              key === "ann_return" ? (v: number | null) => (v != null ? formatPercentSigned(v) : "—") :
                              (v: number | null) => (v != null ? formatPercent(v) : "—");
                  const winner = va != null && vb != null
                    ? (key === "max_drawdown" || key === "ann_vol" ? (va > vb ? "b" : "a") : (va > vb ? "a" : "b"))
                    : null;
                  return (
                    <div key={key} className="rounded-lg border border-border p-3">
                      <div className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">{key}</div>
                      <div className="grid grid-cols-2 gap-2">
                        <div className={cn("text-lg font-semibold tabular", winner === "a" && "text-primary")}>
                          #{a}: {fmt(va)}
                        </div>
                        <div className={cn("text-lg font-semibold tabular", winner === "b" && "text-primary")}>
                          #{b}: {fmt(vb)}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Overlaid equity curves */}
              {overlay && (
                <div className="rounded-lg border border-border p-3">
                  <div className="mb-2 flex items-center gap-3 text-xs">
                    <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded bg-primary" /> #{a}</span>
                    <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded bg-amber-500" /> #{b}</span>
                    <span className="text-muted-foreground">growth of $1, both rebased</span>
                  </div>
                  <svg width={overlay.W} height={overlay.H} viewBox={`0 0 ${overlay.W} ${overlay.H}`} className="w-full">
                    <path d={overlay.pathA} fill="none" stroke="hsl(var(--primary))" strokeWidth={1.5} />
                    <path d={overlay.pathB} fill="none" stroke="#f59e0b" strokeWidth={1.5} />
                  </svg>
                </div>
              )}

              {/* Config diff */}
              <div className="rounded-lg border border-border">
                <div className="border-b border-border px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Config differences ({diff.length})
                </div>
                {diff.length === 0 ? (
                  <p className="px-3 py-4 text-sm text-muted-foreground">Configs are identical.</p>
                ) : (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border text-left text-muted-foreground">
                        <th className="px-3 py-1.5">Key</th>
                        <th className="px-3 py-1.5">#{a}</th>
                        <th className="px-3 py-1.5">#{b}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diff.map(({ key, a: va, b: vb }) => (
                        <tr key={key} className="border-b border-border/50">
                          <td className="px-3 py-1.5 font-medium">{key}</td>
                          <td className="px-3 py-1.5 font-mono">{JSON.stringify(va) ?? "—"}</td>
                          <td className="px-3 py-1.5 font-mono">{JSON.stringify(vb) ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Password modal (reused shape from Jobs page) ────────────────────────────

function ServerPinModal({
  runId,
  onConfirm,
  onClose,
  isLoading,
  error,
}: {
  runId: number;
  onConfirm: (pw: string) => void;
  onClose: () => void;
  isLoading: boolean;
  error: string | null;
}) {
  const [pw, setPw] = useState("");
  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (pw.trim()) onConfirm(pw.trim());
  }
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card p-6 shadow-xl">
        <h2 className="mb-1 text-lg font-semibold">Activate run #{runId} server-wide</h2>
        <p className="mb-3 text-sm text-muted-foreground">
          This makes run #{runId} the default data source for every viewer.
          Requires STOCKPRED_PW.
        </p>
        <form onSubmit={submit} className="space-y-3">
          <input
            type="password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            autoFocus
            placeholder="Password"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          />
          {error && <p className="text-xs text-red-500">{error}</p>}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" type="button" onClick={onClose} disabled={isLoading}>
              Cancel
            </Button>
            <Button size="sm" type="submit" disabled={isLoading || !pw.trim()}>
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Activate"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Main page ───────────────────────────────────────────────────────────────

export function Runs() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { setRunId } = useActiveRun();
  const [params, setParams] = useSearchParams();

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>("all"); // all | ok | failed | other
  const [phaseFilter, setPhaseFilter] = useState<string>("all");   // all | 1 | 5
  const [search, setSearch] = useState("");

  // Sort
  const [sortKey, setSortKey] = useState<SortKey>("completed_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // Expansion + compare selection. Expanded persists in URL so deep-links work.
  const expanded = useMemo(() => {
    const raw = params.get("expanded");
    const n = raw != null ? Number(raw) : NaN;
    return Number.isFinite(n) && n > 0 ? new Set([n]) : new Set<number>();
  }, [params]);
  function toggleExpanded(id: number) {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (next.get("expanded") === String(id)) next.delete("expanded");
        else next.set("expanded", String(id));
        return next;
      },
      { replace: true },
    );
  }

  const [comparePair, setComparePair] = useState<[number?, number?]>([undefined, undefined]);
  const [showCompare, setShowCompare] = useState(false);
  const [pinPwFor, setPinPwFor] = useState<number | null>(null);
  const [pinPwError, setPinPwError] = useState<string | null>(null);
  const [pinPwLoading, setPinPwLoading] = useState(false);

  const { data: runs = [], isLoading } = useQuery({
    queryKey: ["runs", "history"],
    queryFn: () => api.runs(50),
    refetchInterval: 15_000,
  });

  const phaseChoices = useMemo(() => {
    const s = new Set<string>();
    runs.forEach((r) => {
      const p = getPhase(r);
      if (p != null) s.add(String(p));
    });
    return ["all", ...[...s].sort()];
  }, [runs]);

  const filtered = useMemo(() => {
    return runs.filter((r) => {
      if (statusFilter === "ok" && r.status !== "ok") return false;
      if (statusFilter === "failed" && r.status !== "failed") return false;
      if (statusFilter === "other" && (r.status === "ok" || r.status === "failed")) return false;
      if (phaseFilter !== "all") {
        const p = getPhase(r);
        if (p == null || String(p) !== phaseFilter) return false;
      }
      if (search.trim()) {
        const q = search.trim().toLowerCase();
        const haystack = [
          String(r.id),
          r.note ?? "",
          r.job_id ?? "",
          JSON.stringify(r.config).toLowerCase(),
        ].join(" ");
        if (!haystack.toLowerCase().includes(q)) return false;
      }
      return true;
    });
  }, [runs, statusFilter, phaseFilter, search]);

  const sorted = useMemo(() => {
    const rows = [...filtered];
    const dir = sortDir === "asc" ? 1 : -1;
    // Extract a comparable value (always non-null) per sort key. `-Infinity`
    // as a sentinel keeps missing values sorted last in descending order.
    function key(r: RunSummary): number | string {
      switch (sortKey) {
        case "id": return r.id;
        case "completed_at": return r.completed_at ?? r.started_at ?? "";
        case "status": return r.status;
        case "phase": return getPhase(r) ?? -1;
        case "tickers": return r.tickers_count;
        case "sharpe": return getMetric(r, "sharpe") ?? -Infinity;
        case "annret": return getMetric(r, "ann_return") ?? -Infinity;
        case "maxdd": return getMetric(r, "max_drawdown") ?? -Infinity;
      }
    }
    rows.sort((a, b) => {
      const av = key(a);
      const bv = key(b);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    });
    return rows;
  }, [filtered, sortKey, sortDir]);

  function setSort(k: SortKey) {
    if (k === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(k); setSortDir(k === "id" || k === "completed_at" ? "desc" : "desc"); }
  }

  function viewRun(id: number) {
    setRunId(id);
    navigate("/");
  }

  function toggleCompareSelect(id: number) {
    setComparePair(([a, b]) => {
      if (a === id) return [b, undefined];
      if (b === id) return [a, undefined];
      if (a == null) return [id, b];
      if (b == null) return [a, id];
      return [b, id]; // bump the oldest selection
    });
  }

  const activateMut = useMutation({
    mutationFn: ({ id, pw }: { id: number; pw: string }) => api.activateRun(id, pw),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      setPinPwFor(null);
      setPinPwError(null);
    },
    onError: (e: Error) => setPinPwError(e.message),
    onSettled: () => setPinPwLoading(false),
  });

  const deactivateMut = useMutation({
    mutationFn: (pw: string) => api.deactivateRun(pw),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
            <History className="h-6 w-6 text-primary" /> Runs
          </h1>
          <p className="text-sm text-muted-foreground">
            Every model run we've kept ({runs.length} shown, capped at 50 by retention).
            Click <strong>View</strong> to switch the site's data source for your tab, or{" "}
            <strong>Activate</strong> to make it the global default (requires password).
          </p>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="flex flex-wrap items-end gap-3 py-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Status</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded-md border border-border bg-background px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="all">All</option>
              <option value="ok">ok only</option>
              <option value="failed">failed only</option>
              <option value="other">other (running/crashed)</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Phase</label>
            <select
              value={phaseFilter}
              onChange={(e) => setPhaseFilter(e.target.value)}
              className="rounded-md border border-border bg-background px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            >
              {phaseChoices.map((p) => (
                <option key={p} value={p}>{p === "all" ? "All" : `Phase ${p}`}</option>
              ))}
            </select>
          </div>
          <div className="flex flex-1 flex-col gap-1">
            <label className="text-xs text-muted-foreground">Search (id, note, job, config)</label>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="e.g. use_meta_labelling, 200, job-abc…"
              className="rounded-md border border-border bg-background px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div className="ml-auto flex items-end gap-2">
            <Button
              size="sm"
              variant={comparePair[0] != null && comparePair[1] != null ? "default" : "ghost"}
              disabled={comparePair[0] == null || comparePair[1] == null}
              onClick={() => setShowCompare(true)}
              title={
                comparePair[0] == null || comparePair[1] == null
                  ? "Tick the compare box on two rows to enable"
                  : `Compare #${comparePair[0]} vs #${comparePair[1]}`
              }
            >
              <GitCompareArrows className="h-4 w-4" /> Compare ({[comparePair[0], comparePair[1]].filter((x) => x != null).length}/2)
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                const pw = prompt("Password (STOCKPRED_PW) to clear the server pin:");
                if (pw) deactivateMut.mutate(pw);
              }}
              title="Clear server-side active-run pin"
            >
              <PinOff className="h-4 w-4" /> Clear server pin
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{sorted.length} runs</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <p className="px-4 py-6 text-sm text-muted-foreground">Loading runs…</p>
          ) : sorted.length === 0 ? (
            <p className="px-4 py-6 text-sm text-muted-foreground">No runs match the filter.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="w-8 px-3 py-2"></th>
                    <th className="w-8 px-3 py-2"></th>
                    <Th label="ID"        k="id"           sortKey={sortKey} sortDir={sortDir} onSort={setSort} />
                    <Th label="Completed" k="completed_at" sortKey={sortKey} sortDir={sortDir} onSort={setSort} />
                    <Th label="Status"    k="status"       sortKey={sortKey} sortDir={sortDir} onSort={setSort} />
                    <Th label="Phase"     k="phase"        sortKey={sortKey} sortDir={sortDir} onSort={setSort} />
                    <Th label="Tickers"   k="tickers"      sortKey={sortKey} sortDir={sortDir} onSort={setSort} align="right" />
                    <Th label="Sharpe"    k="sharpe"       sortKey={sortKey} sortDir={sortDir} onSort={setSort} align="right" />
                    <Th label="Ann ret"   k="annret"       sortKey={sortKey} sortDir={sortDir} onSort={setSort} align="right" />
                    <Th label="Max DD"    k="maxdd"        sortKey={sortKey} sortDir={sortDir} onSort={setSort} align="right" />
                    <th className="px-3 py-2 text-left">Tags</th>
                    <th className="px-3 py-2 text-left">Equity</th>
                    <th className="px-3 py-2 text-left">Job</th>
                    <th className="px-3 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((r) => {
                    const isExpanded = expanded.has(r.id);
                    const isCompareChecked = comparePair[0] === r.id || comparePair[1] === r.id;
                    const sharpe = getMetric(r, "sharpe");
                    const annret = getMetric(r, "ann_return");
                    const maxdd = getMetric(r, "max_drawdown");
                    return (
                      <RowGroup key={r.id}>
                        <tr
                          className={cn(
                            "border-b border-border/50 transition-colors",
                            r.is_active && "bg-primary/5",
                            isExpanded && "bg-accent/30",
                          )}
                        >
                          <td className="px-3 py-2">
                            <button
                              onClick={() => toggleExpanded(r.id)}
                              className="text-muted-foreground hover:text-foreground"
                              title={isExpanded ? "Collapse" : "Expand"}
                            >
                              {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                            </button>
                          </td>
                          <td className="px-3 py-2">
                            <input
                              type="checkbox"
                              checked={isCompareChecked}
                              onChange={() => toggleCompareSelect(r.id)}
                              className="h-4 w-4 rounded border-border accent-primary"
                              title="Tick to add to compare (pick 2)"
                            />
                          </td>
                          <td className="px-3 py-2 font-mono text-xs">
                            #{r.id}
                            {r.is_active && (
                              <span
                                className="ml-1.5 inline-flex items-center gap-0.5 rounded-full bg-primary/15 px-1.5 text-[10px] font-semibold text-primary"
                                title="Server-side active data source"
                              >
                                <Pin className="h-2.5 w-2.5" /> active
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-xs">{formatDateTime(r.completed_at)}</td>
                          <td className="px-3 py-2">{statusBadge(r.status)}</td>
                          <td className="px-3 py-2 text-xs">{getPhase(r) ?? "—"}</td>
                          <td className="px-3 py-2 text-right tabular text-xs">{r.tickers_count}</td>
                          <td className={cn("px-3 py-2 text-right tabular text-xs", sharpe != null && signClass(sharpe))}>{sharpe != null ? formatNumber(sharpe) : "—"}</td>
                          <td className={cn("px-3 py-2 text-right tabular text-xs", annret != null && signClass(annret))}>{annret != null ? formatPercentSigned(annret) : "—"}</td>
                          <td className="px-3 py-2 text-right tabular text-xs">{maxdd != null ? formatPercent(maxdd) : "—"}</td>
                          <td className="px-3 py-2">
                            <div className="flex flex-wrap gap-1">
                              {configBadges(r.config).slice(0, 4).map((b) => (
                                <span key={b} className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">{b}</span>
                              ))}
                              {configBadges(r.config).length > 4 && (
                                <span className="text-[10px] text-muted-foreground">+{configBadges(r.config).length - 4}</span>
                              )}
                            </div>
                          </td>
                          <td className="px-3 py-2">
                            {r.status === "ok" && <Sparkline runId={r.id} />}
                          </td>
                          <td className="px-3 py-2 text-xs">
                            {r.job_id ? (
                              <Link
                                to={`/jobs?job=${r.job_id}`}
                                className="inline-flex items-center gap-1 font-mono text-muted-foreground hover:text-foreground"
                                title={`Open job ${r.job_id}`}
                              >
                                <Briefcase className="h-3 w-3" />
                                {r.job_id.slice(0, 8)}…
                              </Link>
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                          <td className="px-3 py-2">
                            <div className="flex items-center justify-end gap-1">
                              <Button size="sm" variant="ghost" onClick={() => viewRun(r.id)} title="Pin this run for your tab & go Home">
                                <Eye className="h-3.5 w-3.5" /> View
                              </Button>
                              {r.status === "ok" && (
                                <Button
                                  size="sm"
                                  variant={r.is_active ? "ghost" : "default"}
                                  disabled={r.is_active}
                                  onClick={() => { setPinPwError(null); setPinPwFor(r.id); }}
                                  title={r.is_active ? "Already the active source" : "Set as server-wide default (X-Password)"}
                                >
                                  <Pin className="h-3.5 w-3.5" /> {r.is_active ? "Active" : "Activate"}
                                </Button>
                              )}
                            </div>
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr className="border-b border-border bg-muted/20">
                            <td colSpan={14} className="px-6 py-4">
                              <ExpandedDetails run={r} />
                            </td>
                          </tr>
                        )}
                      </RowGroup>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {showCompare && comparePair[0] != null && comparePair[1] != null && (
        <CompareModal a={comparePair[0]} b={comparePair[1]} onClose={() => setShowCompare(false)} />
      )}

      {pinPwFor != null && (
        <ServerPinModal
          runId={pinPwFor}
          onConfirm={(pw) => { setPinPwLoading(true); activateMut.mutate({ id: pinPwFor, pw }); }}
          onClose={() => { setPinPwFor(null); setPinPwError(null); }}
          isLoading={pinPwLoading}
          error={pinPwError}
        />
      )}
    </div>
  );
}

// React's table rules forbid a <Fragment> directly inside <tbody> if it
// contains conditional sibling rows — wrap in a no-op fragment so we can
// emit two <tr> rows per run without violating row semantics.
function RowGroup({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

function Th({
  label, k, sortKey, sortDir, onSort, align,
}: {
  label: string;
  k: SortKey;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (k: SortKey) => void;
  align?: "left" | "right";
}) {
  const isActive = sortKey === k;
  const Icon = isActive ? (sortDir === "asc" ? ArrowUp : ArrowDown) : ArrowUpDown;
  return (
    <th className={cn("px-3 py-2", align === "right" ? "text-right" : "text-left")}>
      <button
        onClick={() => onSort(k)}
        className={cn(
          "inline-flex items-center gap-1 text-xs font-medium uppercase tracking-wide",
          isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground",
          align === "right" && "flex-row-reverse",
        )}
      >
        {label}
        <Icon className="h-3 w-3" />
      </button>
    </th>
  );
}

function ExpandedDetails({ run }: { run: RunSummary }) {
  const [copied, setCopied] = useState(false);
  const json = useMemo(() => JSON.stringify(run.config, null, 2), [run.config]);

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div>
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Full config</span>
          <button
            onClick={() => {
              navigator.clipboard.writeText(json);
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
            className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-0.5 text-xs text-muted-foreground hover:text-foreground"
            title="Copy as JSON"
          >
            <Copy className="h-3 w-3" />
            {copied ? "Copied!" : "Copy JSON"}
          </button>
        </div>
        <pre className="max-h-80 overflow-auto rounded-md bg-background p-3 font-mono text-xs leading-relaxed">
          {json}
        </pre>
      </div>
      <div className="space-y-3">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Per-horizon diagnostics</span>
          {Object.keys(run.per_horizon_diagnostics ?? {}).length === 0 ? (
            <p className="text-xs text-muted-foreground">no per-horizon stats</p>
          ) : (
            <table className="w-full text-xs">
              <thead className="text-muted-foreground">
                <tr><th className="text-left">h</th><th className="text-right">IC mean</th><th className="text-right">IC IR</th><th className="text-right">Hit</th></tr>
              </thead>
              <tbody>
                {Object.entries(run.per_horizon_diagnostics ?? {}).map(([h, d]) => (
                  <tr key={h}>
                    <td>{h}d</td>
                    <td className="text-right tabular">{formatNumber(d.ic_mean, { maximumFractionDigits: 4 })}</td>
                    <td className="text-right tabular">{formatNumber(d.ic_ir, { maximumFractionDigits: 2 })}</td>
                    <td className="text-right tabular">{formatPercent(d.hit_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div>
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Note</span>
          <p className="text-xs">{run.note || "—"}</p>
        </div>
        <div className="flex flex-wrap gap-1">
          {configBadges(run.config).map((b) => (
            <span key={b} className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">{b}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
