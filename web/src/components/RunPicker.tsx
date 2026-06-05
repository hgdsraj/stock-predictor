// Header-mounted dropdown that lets the user switch which model run drives
// every page's data fetches. Backed by ?run_id= in the URL via useActiveRun.
//
// Visual contract:
//   - Compact pill that fits in the header next to the theme toggle.
//   - When following the server default, shows "Run #N (active)" — same
//     content as a pinned run, just without the "pinned" indicator. This
//     keeps the picker informative even when not in use.
//   - When pinned to a non-default run, shows an explicit "⚲ Pinned" badge
//     so the user knows their view is overriding the server's choice.
//   - Dropdown is keyboard-navigable (native <select>) for simplicity and
//     accessibility; we don't need search since the runs list is bounded
//     (cleanup_old_runs caps at 50).

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Database, Pin, History } from "lucide-react";
import { api } from "@/api/client";
import { useActiveRun } from "@/hooks/useActiveRun";
import type { RunSummary } from "@/api/types";
import { cn } from "@/lib/cn";

function shortMetric(r: RunSummary): string {
  const sh = r.metrics?.sharpe;
  if (typeof sh === "number" && Number.isFinite(sh)) {
    return `Sharpe ${sh >= 0 ? "+" : ""}${sh.toFixed(2)}`;
  }
  return r.status;
}

function runLabel(r: RunSummary): string {
  const phase = (r.config?.phase as number | undefined) ?? "?";
  return `#${r.id} · Phase ${phase} · ${shortMetric(r)}`;
}

export function RunPicker() {
  const { runId, isPinned, setRunId } = useActiveRun();

  const { data: runs, isLoading } = useQuery({
    queryKey: ["runs", "picker"],
    queryFn: () => api.runs(50),
    // Refresh occasionally so new runs appear without a full reload.
    refetchInterval: 30_000,
  });

  const okRuns = useMemo(
    () => (runs ?? []).filter((r) => r.status === "ok"),
    [runs],
  );
  const activeServer = useMemo(
    () => okRuns.find((r) => r.is_active) ?? okRuns[0] ?? null,
    [okRuns],
  );

  // What's currently shown:
  //   - If pinned and the run is in our list, show that.
  //   - Otherwise fall back to the server default.
  const shown = useMemo(() => {
    if (isPinned && runId != null) {
      const hit = okRuns.find((r) => r.id === runId);
      if (hit) return hit;
      // Pinned to a run that's been deleted/aged out: leave value as the id;
      // we still let the user navigate via the dropdown.
    }
    return activeServer;
  }, [okRuns, isPinned, runId, activeServer]);

  function onChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const v = e.target.value;
    if (v === "__default__") {
      setRunId(null);
    } else {
      setRunId(Number(v));
    }
  }

  return (
    <div className="flex items-center gap-1.5">
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs",
          isPinned
            ? "border-primary/40 bg-primary/5 text-primary"
            : "border-border bg-background text-muted-foreground",
        )}
        title={
          isPinned
            ? "Viewing a pinned run — different from the server's default. Clear in the dropdown."
            : "Viewing the server's active run (latest ok by default)."
        }
      >
        {isPinned ? <Pin className="h-3 w-3" /> : <Database className="h-3 w-3" />}
        <select
          value={isPinned && runId != null ? String(runId) : "__default__"}
          onChange={onChange}
          disabled={isLoading || okRuns.length === 0}
          className={cn(
            "max-w-[16rem] truncate bg-transparent text-xs focus:outline-none",
            "disabled:opacity-50",
          )}
          title="Switch data source"
        >
          <option value="__default__">
            {activeServer
              ? `Default: ${runLabel(activeServer)}`
              : isLoading
                ? "Loading runs…"
                : "No runs yet"}
          </option>
          {/* If pinned to a run that's not in the list (e.g. aged out), keep
              the selection valid by adding a synthetic option. */}
          {isPinned && runId != null && !okRuns.some((r) => r.id === runId) && (
            <option value={String(runId)}>#{runId} (not in recent list)</option>
          )}
          {okRuns.map((r) => (
            <option key={r.id} value={String(r.id)}>
              {runLabel(r)}
              {r.is_active ? " · server default" : ""}
            </option>
          ))}
        </select>
        {shown && <span className="hidden sm:inline">·</span>}
        {shown && <span className="hidden sm:inline">{shown.tickers_count} tk</span>}
      </div>
      <Link
        to="/runs"
        title="See all runs"
        className="rounded-md border border-border bg-background p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
      >
        <History className="h-3.5 w-3.5" />
      </Link>
    </div>
  );
}
