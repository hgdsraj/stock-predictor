import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import {
  Play, Trash2, XCircle, Plus, RefreshCw, ChevronDown, ChevronUp,
  Clock, CheckCircle, AlertCircle, Loader2, Copy, ExternalLink, Zap,
  RotateCcw,
} from "lucide-react";
import { api } from "@/api/client";
import type { HypersearchRun, HypersearchTrial, QueuedJob } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

// ─── Shared helpers ───────────────────────────────────────────────────────────

function fmtDuration(s: number | null | undefined) {
  if (s == null) return "—";
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtElapsed(startedAt: string | null | undefined) {
  if (!startedAt) return "—";
  return fmtDuration((Date.now() - new Date(startedAt).getTime()) / 1000);
}
function fmtTs(ts: string | null | undefined) {
  return ts ? new Date(ts).toLocaleString() : "—";
}
function fmtSharpe(v: number | null | undefined) {
  if (v == null || !isFinite(v) || v <= -9) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(3);
}
function fmtPct(v: number | null | undefined) {
  if (v == null || !isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
}

// Estimate seconds remaining for a running hypersearch.
// ~3 min/trial × n_tickers/25 scaling factor, minus elapsed.
function estimateSecondsRemaining(run: HypersearchRun, elapsedS: number): number | null {
  const nTickers = (run.config.n_tickers as number | null) ?? 25;
  const secsPerTrial = 180 * (nTickers / 25);
  const remaining = run.n_trials_requested - run.n_trials_done;
  return Math.max(0, remaining * secsPerTrial - elapsedS);
}

const ACTIVE_STATUSES = new Set(["running", "queued", "cancelling"]);
const RESTARTABLE_STATUSES = new Set(["failed", "cancelled", "crashed"]);

// ─── Status badge ─────────────────────────────────────────────────────────────

function statusBadge(status: string) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    queued:     { cls: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300", icon: <Clock className="h-3 w-3" />, label: "Queued" },
    running:    { cls: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",         icon: <Loader2 className="h-3 w-3 animate-spin" />, label: "Running" },
    cancelling: { cls: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300", icon: <Loader2 className="h-3 w-3 animate-spin" />, label: "Cancelling…" },
    ok:         { cls: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",     icon: <CheckCircle className="h-3 w-3" />, label: "Completed" },
    failed:     { cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",             icon: <AlertCircle className="h-3 w-3" />, label: "Failed" },
    crashed:    { cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",             icon: <AlertCircle className="h-3 w-3" />, label: "Crashed" },
    cancelled:  { cls: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",            icon: <XCircle className="h-3 w-3" />, label: "Cancelled" },
    pending:    { cls: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300", icon: <Clock className="h-3 w-3" />, label: "Pending" },
  };
  const s = map[status] ?? { cls: "bg-gray-100 text-gray-600", icon: null, label: status };
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", s.cls)}>
      {s.icon}{s.label}
    </span>
  );
}

// ─── RunIdCell: copy + external link ─────────────────────────────────────────

function RunIdCell({ runId, jobId, onClick }: { runId: number; jobId: string | null; onClick?: () => void }) {
  const [copied, setCopied] = useState(false);
  const label = `run-${runId}`;
  function copy(e: React.MouseEvent) {
    e.stopPropagation();
    const text = jobId ?? label;
    navigator.clipboard.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
  }
  return (
    <span className="flex items-center gap-1 font-mono text-xs">
      <button onClick={onClick} className="hover:underline" title={jobId ?? label}>{label}</button>
      <button onClick={copy} title="Copy job ID" className="text-muted-foreground hover:text-foreground">
        {copied ? <CheckCircle className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
      </button>
      {jobId && (
        <a href={`/jobs/${jobId}`} target="_blank" rel="noopener noreferrer" title="Open raw job JSON"
          onClick={e => e.stopPropagation()} className="text-muted-foreground hover:text-foreground">
          <ExternalLink className="h-3 w-3" />
        </a>
      )}
    </span>
  );
}

// ─── Password modal ───────────────────────────────────────────────────────────

function PasswordModal({ title, description, confirmLabel, confirmVariant = "default", onConfirm, onClose, isLoading, error }: {
  title: string; description: string; confirmLabel: string; confirmVariant?: "default" | "destructive";
  onConfirm: (pw: string) => void; onClose: () => void; isLoading: boolean; error: string | null;
}) {
  const [pw, setPw] = useState("");
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => { ref.current?.focus(); }, []);
  function submit(e: React.FormEvent) { e.preventDefault(); if (pw.trim()) onConfirm(pw.trim()); }
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card p-6 shadow-xl">
        <h2 className="mb-1 text-lg font-semibold">{title}</h2>
        <p className="mb-4 text-sm text-muted-foreground">{description}</p>
        <form onSubmit={submit} className="space-y-3">
          <input ref={ref} type="password" value={pw} onChange={e => setPw(e.target.value)}
            placeholder="Enter password (STOCKPRED_PW)"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary" />
          {error && <p className="text-xs text-red-500">{error}</p>}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" type="button" onClick={onClose} disabled={isLoading}>Cancel</Button>
            <Button variant={confirmVariant} size="sm" type="submit" disabled={isLoading || !pw.trim()}>
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : confirmLabel}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── New hypersearch form ─────────────────────────────────────────────────────

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-muted-foreground">{label}</label>
      {children}
      {hint && <p className="mt-0.5 text-xs text-muted-foreground/70">{hint}</p>}
    </div>
  );
}

const inp = "w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary";
const sel = inp;

function NewHypersearchForm({ onClose, onQueued }: { onClose: () => void; onQueued: () => void }) {
  const qc = useQueryClient();
  const [cfg, setCfg] = useState({
    n_trials: 50, n_tickers: 25, start_date: "2015-01-01",
    end_date: "" as string, holdout_years: 2, bootstrap_n: 50,
    universe_sampling: "current" as "current" | "first" | "random", seed: 42,
  });
  const [error, setError] = useState<string | null>(null);

  const mutate = useMutation({
    mutationFn: () => api.queueHypersearch({ ...cfg, end_date: cfg.end_date || null }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["queued"] }); onQueued(); },
    onError: (e: Error) => setError(e.message),
  });

  function set<K extends keyof typeof cfg>(k: K, v: (typeof cfg)[K]) {
    setCfg(c => ({ ...c, [k]: v }));
  }

  const estHours = Math.max(0.5, Math.round(cfg.n_trials * (cfg.n_tickers / 25) * 3 / 60 * 10) / 10);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/50 p-4">
      <div className="w-full max-w-lg rounded-xl border border-border bg-card shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-semibold">
              <Zap className="h-5 w-5 text-primary" /> New Hypersearch
            </h2>
            <p className="text-xs text-muted-foreground">Bayesian TPE search over 20 parameters · objective: holdout Sharpe</p>
          </div>
        </div>
        <div className="space-y-4 px-6 py-4">
          <div className="rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            Estimated runtime: <strong>~{estHours}h</strong> ({cfg.n_trials} trials × {cfg.n_tickers} tickers).
            Password required to launch.
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Trials" hint="50 = good balance; 100+ = overnight">
              <input type="number" className={inp} value={cfg.n_trials} min={5} max={500}
                onChange={e => set("n_trials", Number(e.target.value))} />
            </Field>
            <Field label="Tickers" hint="25 ≈ 2-4 min/trial">
              <input type="number" className={inp} value={cfg.n_tickers} min={5} max={200}
                onChange={e => set("n_tickers", Number(e.target.value))} />
            </Field>
            <Field label="Start date">
              <input type="date" className={inp} value={cfg.start_date}
                onChange={e => set("start_date", e.target.value)} />
            </Field>
            <Field label="End date" hint="Leave blank = today">
              <input type="date" className={inp} value={cfg.end_date}
                onChange={e => set("end_date", e.target.value)} />
            </Field>
            <Field label="Holdout years" hint="Never seen during tuning">
              <input type="number" className={inp} value={cfg.holdout_years} min={1} max={5}
                onChange={e => set("holdout_years", Number(e.target.value))} />
            </Field>
            <Field label="Bootstrap samples" hint="50 = fast CI, 500 = honest CI">
              <input type="number" className={inp} value={cfg.bootstrap_n} min={10} max={500}
                onChange={e => set("bootstrap_n", Number(e.target.value))} />
            </Field>
            <Field label="Universe sampling">
              <select className={sel} value={cfg.universe_sampling}
                onChange={e => set("universe_sampling", e.target.value as typeof cfg.universe_sampling)}>
                <option value="current">current (same set every trial)</option>
                <option value="first">first (alphabetical)</option>
                <option value="random">random (different each trial)</option>
              </select>
            </Field>
            <Field label="Seed" hint="Reproducibility">
              <input type="number" className={inp} value={cfg.seed}
                onChange={e => set("seed", Number(e.target.value))} />
            </Field>
          </div>
          {error && <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">{error}</p>}
        </div>
        <div className="flex justify-end gap-2 border-t border-border px-6 py-4">
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button size="sm" onClick={() => { setError(null); mutate.mutate(); }} disabled={mutate.isPending}>
            {mutate.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            Queue Search
          </Button>
        </div>
      </div>
    </div>
  );
}

// ─── Trial results table ──────────────────────────────────────────────────────

function TrialTable({ trials, limit = 10 }: { trials: HypersearchTrial[]; limit?: number }) {
  const sorted = [...trials]
    .filter(t => !t.error || t.hold_sharpe != null)
    .sort((a, b) => (b.hold_sharpe ?? -99) - (a.hold_sharpe ?? -99));
  const rows = sorted.slice(0, limit);
  if (rows.length === 0) return <p className="text-sm text-muted-foreground">No successful trials yet.</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border text-left text-muted-foreground">
            {["#", "Sharpe", "CI lo", "CI hi", "Max DD", "Ann Ret", "Dev Sh", "Sizing", "Hz", "Meta", "Rnk", "s"].map(h => (
              <th key={h} className="whitespace-nowrap px-2 py-1.5 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((t, i) => (
            <tr key={t.trial} className={cn("border-b border-border/50", i === 0 && "bg-green-50 dark:bg-green-900/10")}>
              <td className="px-2 py-1 font-mono text-muted-foreground">{t.trial}</td>
              <td className={cn("px-2 py-1 font-mono font-semibold tabular-nums",
                (t.hold_sharpe ?? 0) > 0 ? "text-green-600 dark:text-green-400" : "text-red-500")}>
                {fmtSharpe(t.hold_sharpe)}
              </td>
              <td className="px-2 py-1 font-mono text-muted-foreground">{fmtSharpe(t.hold_ci_lo)}</td>
              <td className="px-2 py-1 font-mono text-muted-foreground">{fmtSharpe(t.hold_ci_hi)}</td>
              <td className="px-2 py-1 font-mono">{fmtPct(t.hold_dd)}</td>
              <td className="px-2 py-1 font-mono">{fmtPct(t.hold_ann_return)}</td>
              <td className="px-2 py-1 font-mono text-muted-foreground">{fmtSharpe(t.dev_sharpe)}</td>
              <td className="px-2 py-1">{String(t.params.position_sizing ?? "—")}</td>
              <td className="px-2 py-1 font-mono">{String(t.params.horizons ?? "—")}</td>
              <td className="px-2 py-1">{t.params.use_meta_labelling ? "Y" : "N"}</td>
              <td className="px-2 py-1">{t.params.ranks_only ? "Y" : "N"}</td>
              <td className="px-2 py-1 font-mono text-muted-foreground">
                {t.elapsed_s != null ? `${Math.round(t.elapsed_s)}s` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Detail panel (opens inline below the row, like Jobs) ─────────────────────

function HypersearchDetailPanel({
  runId, onClose, onCancel,
}: { runId: number; onClose: () => void; onCancel: (jobId: string) => void }) {
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [showAllTrials, setShowAllTrials] = useState(false);
  const [showParams, setShowParams] = useState(false);
  const [paramsCopied, setParamsCopied] = useState(false);

  // Poll the HypersearchRun record (updates after every trial)
  const { data: run } = useQuery({
    queryKey: ["hs-run", runId],
    queryFn: () => api.hypersearchRun(runId),
    refetchInterval: q => ACTIVE_STATUSES.has(q.state.data?.status ?? "") ? 4000 : false,
  });

  // Fetch the linked job for logs + cancellable status
  const jobId = run?.job_id ?? null;
  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.jobDetail(jobId!),
    enabled: !!jobId,
    refetchInterval: q => ACTIVE_STATUSES.has(q.state.data?.status ?? "") ? 2500 : false,
  });

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [job?.logs?.length]);

  if (!run) return <div className="py-6 text-center"><Loader2 className="mx-auto h-5 w-5 animate-spin text-muted-foreground" /></div>;

  const isActive = ACTIVE_STATUSES.has(run.status);
  const jobIsActive = job ? ACTIVE_STATUSES.has(job.status) : false;

  // Progress
  const progress = run.n_trials_requested > 0
    ? Math.min(99, Math.round((run.n_trials_done / run.n_trials_requested) * 100))
    : 0;
  const elapsedS = job?.elapsed_s ?? (job?.started_at ? (Date.now() - new Date(job.started_at).getTime()) / 1000 : null);
  const remainS = isActive && elapsedS != null ? estimateSecondsRemaining(run, elapsedS) : null;

  // Trials
  const validTrials = (run.trials ?? []).filter(t => !t.error || t.hold_sharpe != null);
  const nPosCI = validTrials.filter(t => (t.hold_ci_lo ?? 0) > 0).length;

  // Best params JSON
  const bestParamsJson = run.best_params ? JSON.stringify(run.best_params, null, 2) : null;

  function copyParams() {
    if (!bestParamsJson) return;
    navigator.clipboard.writeText(bestParamsJson).then(() => {
      setParamsCopied(true);
      setTimeout(() => setParamsCopied(false), 1500);
    });
  }

  return (
    <div className="mt-4 rounded-xl border border-border bg-card p-4">
      {/* ── Header row ── */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          {jobId && <RunIdCell runId={run.id} jobId={jobId} />}
          {statusBadge(run.status)}
          {job?.elapsed_s != null && (
            <span className="text-xs text-muted-foreground">Runtime: {fmtDuration(job.elapsed_s)}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {jobIsActive && jobId && (
            <Button variant="destructive" size="sm" onClick={() => onCancel(jobId)}>
              <XCircle className="h-4 w-4" /> Cancel
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>
        </div>
      </div>

      {/* ── Progress bar ── */}
      {(isActive || run.n_trials_done > 0) && (
        <div className="mb-4">
          <div className="mb-1 flex justify-between text-xs text-muted-foreground">
            <span>Trials: {run.n_trials_done} / {run.n_trials_requested}</span>
            <span>
              {progress}%{remainS != null ? ` · ~${fmtDuration(remainS)} remaining` : ""}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full transition-all duration-700",
                isActive ? "bg-primary" : "bg-green-500")}
              style={{ width: `${run.status === "ok" ? 100 : progress}%` }}
            />
          </div>
        </div>
      )}

      {/* ── Best Sharpe callout ── */}
      {run.best_sharpe != null && (
        <div className="mb-4 flex items-center gap-4 rounded-lg border border-border bg-muted/30 px-4 py-3">
          <div>
            <p className="text-xs text-muted-foreground">Best holdout Sharpe</p>
            <p className={cn("text-3xl font-bold tabular-nums",
              run.best_sharpe > 0 ? "text-green-600 dark:text-green-400" : "text-red-500")}>
              {fmtSharpe(run.best_sharpe)}
            </p>
          </div>
          {validTrials.length > 0 && (
            <div className="ml-auto text-right text-xs text-muted-foreground space-y-0.5">
              <p>CI &gt; 0: <strong className={nPosCI > 0 ? "text-green-600 dark:text-green-400" : ""}>{nPosCI}/{validTrials.length}</strong></p>
              <p>{run.n_trials_done} trials done</p>
            </div>
          )}
        </div>
      )}

      {/* ── Error banner ── */}
      {job?.error && (
        <div className="mb-4 rounded-md bg-red-50 p-3 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
          <strong>Error:</strong> {job.error}
        </div>
      )}

      {/* ── Main content grid: config + logs ── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Config */}
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Search Config</p>
          <dl className="space-y-1 text-xs">
            {Object.entries(run.config)
              .filter(([k]) => !["job_type"].includes(k))
              .map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="font-mono text-right">{String(v ?? "null")}</dd>
                </div>
              ))}
          </dl>
        </div>

        {/* Logs */}
        <div className="lg:col-span-2">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Logs{" "}
            {(job?.logs?.length ?? 0) > 0 && (
              <span className="text-muted-foreground/60">({job!.logs.length} lines)</span>
            )}
          </p>
          <pre className="h-64 overflow-y-auto rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground/80 scrollbar-thin">
            {!job
              ? (isActive ? "Waiting for job to start…" : "No job data.")
              : (job.logs.length === 0
                ? (jobIsActive ? "Waiting for output…" : "No logs captured.")
                : job.logs.join("\n"))}
            <div ref={logsEndRef} />
          </pre>
        </div>
      </div>

      {/* ── Trial results ── */}
      {run.trials.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Trial Results — top {Math.min(10, validTrials.length)}
            </p>
            {validTrials.length > 10 && (
              <button className="text-xs text-primary hover:underline"
                onClick={() => setShowAllTrials(v => !v)}>
                {showAllTrials ? "Show top 10" : `Show all ${validTrials.length}`}
              </button>
            )}
          </div>
          <TrialTable trials={run.trials} limit={showAllTrials ? run.trials.length : 10} />

          {/* Honest interpretation */}
          {run.status === "ok" && (
            <div className="mt-3 rounded-md bg-muted/50 px-3 py-2 text-xs space-y-1">
              <p className="font-semibold">Honest interpretation</p>
              <p className="text-muted-foreground">
                Configs with 95% CI strictly &gt; 0 (statistically real edge):{" "}
                <span className={cn("font-semibold", nPosCI > 0 ? "text-green-600 dark:text-green-400" : "text-red-500")}>
                  {nPosCI} / {validTrials.length}
                </span>
                {nPosCI === 0 && " — no statistically proven edge found yet."}
              </p>
              <p className="text-muted-foreground/70">
                CIs are computed on a {(run.config.n_tickers as number | null) ?? 25}-ticker fast-mode universe.
                Validate the best config on the full universe before drawing conclusions.
              </p>
            </div>
          )}
        </div>
      )}

      {/* ── Best params ── */}
      {bestParamsJson && (
        <div className="mt-4">
          <button
            className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
            onClick={() => setShowParams(v => !v)}>
            {showParams ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            Best Config Parameters
          </button>
          {showParams && (
            <div className="relative mt-2">
              <pre className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">
                {bestParamsJson}
              </pre>
              <button
                onClick={copyParams}
                className="absolute right-2 top-2 rounded-md border border-border bg-card px-2 py-1 text-xs text-muted-foreground hover:text-foreground">
                {paramsCopied ? "Copied!" : "Copy"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function Hypersearch() {
  const qc = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const deepLinked = searchParams.get("run");
  const [selectedRunId, setSelectedRunId] = useState<number | null>(
    deepLinked ? Number(deepLinked) : null,
  );

  useEffect(() => {
    if (deepLinked && Number(deepLinked) !== selectedRunId) {
      setSelectedRunId(Number(deepLinked));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLinked]);

  function selectRun(id: number | null) {
    setSelectedRunId(id);
    setSearchParams(prev => {
      const p = new URLSearchParams(prev);
      if (id != null) p.set("run", String(id)); else p.delete("run");
      return p;
    }, { replace: true });
  }

  const [showNewForm, setShowNewForm] = useState(false);
  const [pwAction, setPwAction] = useState<
    | { type: "launch"; queueId: string }
    | { type: "delete"; queueId: string }
    | { type: "cancel"; jobId: string }
    | null
  >(null);
  const [pwError, setPwError] = useState<string | null>(null);
  const [pwLoading, setPwLoading] = useState(false);
  const [restartError, setRestartError] = useState<string | null>(null);
  const [restartingRunId, setRestartingRunId] = useState<number | null>(null);

  // All hypersearch runs (list — no trials, fast)
  const { data: runs = [] } = useQuery({
    queryKey: ["hs-runs"],
    queryFn: () => api.listHypersearchRuns(50),
    refetchInterval: q => (q.state.data ?? []).some(r => ACTIVE_STATUSES.has(r.status)) ? 4000 : 20000,
  });

  // Queue — filter to hypersearch entries only
  const { data: allQueued = [] } = useQuery({
    queryKey: ["queued"],
    queryFn: () => api.listQueued(),
    refetchInterval: 10000,
  });
  const hsQueued = allQueued.filter(q =>
    (q.config as any)?.job_type === "hypersearch" || (q.config as any)?.n_trials != null
  );

  // Restart: re-queue with same config, then immediately open launch modal
  const restartMut = useMutation({
    mutationFn: (run: HypersearchRun) =>
      api.queueHypersearch(run.config as any),
    onMutate: (run) => setRestartingRunId(run.id),
    onSuccess: (qj) => {
      qc.invalidateQueries({ queryKey: ["queued"] });
      setPwError(null);
      setPwAction({ type: "launch", queueId: qj.id });
    },
    onError: (e: Error) => setRestartError(e.message),
    onSettled: () => setRestartingRunId(null),
  });

  async function handlePwConfirm(pw: string) {
    if (!pwAction) return;
    setPwLoading(true); setPwError(null);
    try {
      if (pwAction.type === "cancel") {
        await api.cancelJob(pwAction.jobId, pw);
        qc.invalidateQueries({ queryKey: ["jobs"] });
        qc.invalidateQueries({ queryKey: ["hs-runs"] });
      } else if (pwAction.type === "launch") {
        await api.launchQueued(pwAction.queueId, pw);
        qc.invalidateQueries({ queryKey: ["queued"] });
        qc.invalidateQueries({ queryKey: ["hs-runs"] });
        setTimeout(() => qc.invalidateQueries({ queryKey: ["hs-runs"] }), 2000);
      } else if (pwAction.type === "delete") {
        await api.deleteQueued(pwAction.queueId, pw);
        qc.invalidateQueries({ queryKey: ["queued"] });
      }
      setPwAction(null);
    } catch (e: any) {
      setPwError(e.message ?? "Request failed");
    } finally {
      setPwLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold">
            <Zap className="h-6 w-6 text-primary" /> Hyperparameter Search
          </h1>
          <p className="text-sm text-muted-foreground">
            Bayesian TPE over 20 parameters · objective: holdout Sharpe · password required to launch
          </p>
        </div>
        <Button onClick={() => setShowNewForm(true)}>
          <Plus className="h-4 w-4" /> New Search
        </Button>
      </div>

      {restartError && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          <strong>Restart failed:</strong> {restartError}{" "}
          <button onClick={() => setRestartError(null)} className="ml-2 underline">dismiss</button>
        </div>
      )}

      {/* ── Queue ── */}
      {hsQueued.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-base">
              <span>
                Queue
                {hsQueued.filter(j => j.status === "pending").length > 0 && (
                  <span className="ml-2 rounded-full bg-yellow-100 px-2 py-0.5 text-xs text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300">
                    {hsQueued.filter(j => j.status === "pending").length} pending
                  </span>
                )}
              </span>
              <button onClick={() => qc.invalidateQueries({ queryKey: ["queued"] })} className="text-muted-foreground hover:text-foreground">
                <RefreshCw className="h-4 w-4" />
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2">ID</th>
                    <th className="px-4 py-2">Status</th>
                    <th className="px-4 py-2">Summary</th>
                    <th className="px-4 py-2">Created</th>
                    <th className="px-4 py-2">Job</th>
                    <th className="px-4 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {hsQueued.map(q => (
                    <tr key={q.id} className="border-b border-border/50">
                      <td className="px-4 py-2 font-mono text-xs">{q.id.slice(0, 8)}…</td>
                      <td className="px-4 py-2">{statusBadge(q.status)}</td>
                      <td className="px-4 py-2 text-xs text-muted-foreground">
                        {(q.config as any).n_trials ?? "?"} trials ·{" "}
                        {(q.config as any).n_tickers ?? "?"} tickers ·{" "}
                        {(q.config as any).start_date ?? "?"}
                      </td>
                      <td className="px-4 py-2 text-xs">{fmtTs(q.created_at)}</td>
                      <td className="px-4 py-2 text-xs">
                        {q.job_id ? (
                          <span className="font-mono text-muted-foreground">{q.job_id.slice(0, 8)}…</span>
                        ) : <span className="text-muted-foreground">—</span>}
                      </td>
                      <td className="px-4 py-2">
                        {q.status === "pending" ? (
                          <div className="flex items-center justify-end gap-1">
                            <Button variant="default" size="sm"
                              onClick={() => { setPwError(null); setPwAction({ type: "launch", queueId: q.id }); }}>
                              <Play className="h-3 w-3" /> Launch
                            </Button>
                            <Button variant="ghost" size="sm"
                              onClick={() => { setPwError(null); setPwAction({ type: "delete", queueId: q.id }); }}>
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        ) : (
                          <div className="text-right text-xs text-muted-foreground">—</div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Active & History (unified) ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            Active &amp; History
            <button onClick={() => qc.invalidateQueries({ queryKey: ["hs-runs"] })} className="text-muted-foreground hover:text-foreground">
              <RefreshCw className="h-4 w-4" />
            </button>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {runs.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-muted-foreground">
              No searches yet. Click <strong>New Search</strong> to queue one.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2">ID</th>
                    <th className="px-4 py-2">Status</th>
                    <th className="px-4 py-2">Trials</th>
                    <th className="px-4 py-2">Progress</th>
                    <th className="px-4 py-2">Best Sharpe</th>
                    <th className="px-4 py-2">Tickers</th>
                    <th className="px-4 py-2">Started</th>
                    <th className="px-4 py-2">Runtime</th>
                    <th className="px-4 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(r => {
                    const isSelected = selectedRunId === r.id;
                    const isActive = ACTIVE_STATUSES.has(r.status);
                    const canRestart = RESTARTABLE_STATUSES.has(r.status) && Object.keys(r.config).length > 0;
                    const pct = r.n_trials_requested > 0
                      ? Math.min(100, Math.round((r.n_trials_done / r.n_trials_requested) * 100))
                      : 0;

                    return (
                      <>
                        <tr key={r.id}
                          onClick={() => selectRun(isSelected ? null : r.id)}
                          className={cn("cursor-pointer border-b border-border/50 transition-colors hover:bg-accent/50", isSelected && "bg-accent")}>
                          <td className="px-4 py-2">
                            <RunIdCell runId={r.id} jobId={r.job_id} onClick={() => selectRun(isSelected ? null : r.id)} />
                          </td>
                          <td className="px-4 py-2">{statusBadge(r.status)}</td>
                          <td className="px-4 py-2 text-xs">{r.n_trials_done}/{r.n_trials_requested}</td>
                          <td className="px-4 py-2">
                            <div className="flex items-center gap-2">
                              <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                                <div className={cn("h-full rounded-full", r.status === "ok" ? "bg-green-500" : isActive ? "bg-primary" : "bg-muted-foreground/40")}
                                  style={{ width: `${r.status === "ok" ? 100 : pct}%` }} />
                              </div>
                              <span className="text-xs text-muted-foreground">{r.status === "ok" ? 100 : pct}%</span>
                            </div>
                          </td>
                          <td className="px-4 py-2 font-mono text-xs">
                            <span className={cn("font-semibold tabular-nums",
                              r.best_sharpe != null && r.best_sharpe > 0 ? "text-green-600 dark:text-green-400" :
                              r.best_sharpe != null ? "text-red-500" : "text-muted-foreground")}>
                              {fmtSharpe(r.best_sharpe)}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-xs">{String((r.config.n_tickers as number | null) ?? "—")}</td>
                          <td className="px-4 py-2 text-xs">
                            {isActive ? fmtElapsed(r.started_at) + " ago" : fmtTs(r.started_at)}
                          </td>
                          <td className="px-4 py-2 text-xs">{fmtTs(r.completed_at) !== "—" && !isActive ? fmtDuration((new Date(r.completed_at!).getTime() - new Date(r.started_at).getTime()) / 1000) : isActive ? fmtElapsed(r.started_at) : "—"}</td>
                          <td className="px-4 py-2">
                            <div className="flex items-center justify-end gap-1">
                              {canRestart && (
                                <Button size="sm" variant="ghost"
                                  disabled={restartingRunId === r.id || restartMut.isPending}
                                  title="Re-queue with same parameters"
                                  onClick={e => { e.stopPropagation(); setRestartError(null); restartMut.mutate(r); }}>
                                  {restartingRunId === r.id
                                    ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                    : <RotateCcw className="h-3.5 w-3.5" />}
                                  Restart
                                </Button>
                              )}
                            </div>
                          </td>
                        </tr>
                        {isSelected && (
                          <tr key={`detail-${r.id}`} className="bg-card">
                            <td colSpan={9} className="px-4 pb-4">
                              <HypersearchDetailPanel
                                runId={r.id}
                                onClose={() => selectRun(null)}
                                onCancel={jobId => { setPwError(null); setPwAction({ type: "cancel", jobId }); }}
                              />
                            </td>
                          </tr>
                        )}
                      </>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {showNewForm && (
        <NewHypersearchForm
          onClose={() => setShowNewForm(false)}
          onQueued={() => setShowNewForm(false)}
        />
      )}

      {pwAction && (
        <PasswordModal
          title={
            pwAction.type === "cancel" ? "Cancel Search" :
            pwAction.type === "launch" ? "Launch Search" : "Delete Queued Search"
          }
          description={
            pwAction.type === "cancel"
              ? "The search will be interrupted. Partial results are already saved."
              : pwAction.type === "launch"
              ? "Enter STOCKPRED_PW to start the search. It may run for several hours."
              : "Enter STOCKPRED_PW to permanently delete this queued search."
          }
          confirmLabel={pwAction.type === "launch" ? "Launch" : pwAction.type === "cancel" ? "Cancel Search" : "Delete"}
          confirmVariant={pwAction.type !== "launch" ? "destructive" : "default"}
          onConfirm={handlePwConfirm}
          onClose={() => { setPwAction(null); setPwError(null); }}
          isLoading={pwLoading}
          error={pwError}
        />
      )}
    </div>
  );
}
