import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Play, Trash2, XCircle, Plus, RefreshCw, ChevronDown, ChevronUp,
  Clock, CheckCircle, AlertCircle, Loader2, Copy, ExternalLink, Zap,
} from "lucide-react";
import { api } from "@/api/client";
import type { HypersearchRun, HypersearchTrial, JobDetail, QueuedJob } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmtDuration(s: number | null | undefined) {
  if (s == null) return "—";
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtTs(ts: string | null | undefined) { return ts ? new Date(ts).toLocaleString() : "—"; }
function fmtSharpe(v: number | null | undefined) {
  if (v == null || !isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(3);
}
function fmtPct(v: number | null | undefined) {
  if (v == null || !isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
}

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

function NewHypersearchForm({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [cfg, setCfg] = useState({
    n_trials: 50,
    n_tickers: 25,
    start_date: "2015-01-01",
    end_date: "" as string,
    holdout_years: 2,
    bootstrap_n: 50,
    universe_sampling: "current" as "current" | "first" | "random",
    seed: 42,
  });
  const [error, setError] = useState<string | null>(null);
  const [queued, setQueued] = useState<QueuedJob | null>(null);

  const mutate = useMutation({
    mutationFn: () => api.queueHypersearch({
      ...cfg,
      end_date: cfg.end_date || null,
    }),
    onSuccess: (qj) => {
      setQueued(qj);
      qc.invalidateQueries({ queryKey: ["queued"] });
    },
    onError: (e: Error) => setError(e.message),
  });

  function set<K extends keyof typeof cfg>(k: K, v: (typeof cfg)[K]) {
    setCfg(c => ({ ...c, [k]: v }));
  }

  const estimatedHours = Math.round(cfg.n_trials * (cfg.n_tickers / 25) * 3 / 60 * 10) / 10;

  if (queued) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
        <div className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-xl">
          <div className="mb-3 flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <h2 className="text-lg font-semibold">Queued</h2>
          </div>
          <p className="mb-1 text-sm text-muted-foreground">
            Hypersearch job queued. Launch it with your <code className="text-xs font-mono">STOCKPRED_PW</code> from the queue below.
          </p>
          <p className="mb-4 font-mono text-xs text-muted-foreground">ID: {queued.id}</p>
          <div className="flex justify-end">
            <Button size="sm" onClick={onClose}>Done</Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/50 p-4">
      <div className="w-full max-w-lg rounded-xl border border-border bg-card shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-semibold"><Zap className="h-5 w-5 text-primary" />New Hypersearch Job</h2>
            <p className="text-xs text-muted-foreground">Bayesian search over 20 parameters · optimises holdout Sharpe</p>
          </div>
        </div>

        <div className="space-y-4 px-6 py-4">
          <div className="rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            Estimated runtime: <strong>~{estimatedHours}h</strong> ({cfg.n_trials} trials × {cfg.n_tickers} tickers)
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Field label="Trials" hint="More = better, slower">
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
            <Field label="Bootstrap samples" hint="50=fast CI, 500=honest CI">
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
            Queue Job
          </Button>
        </div>
      </div>
    </div>
  );
}

// ─── Trial results table ──────────────────────────────────────────────────────

function TrialTable({ trials, limit = 10 }: { trials: HypersearchTrial[]; limit?: number }) {
  const sorted = [...trials].sort((a, b) => (b.hold_sharpe ?? -99) - (a.hold_sharpe ?? -99));
  const rows = sorted.slice(0, limit);
  if (rows.length === 0) return <p className="text-sm text-muted-foreground">No trials yet.</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border text-left text-muted-foreground">
            {["#", "Sharpe", "CI lo", "CI hi", "Max DD", "Ann Ret", "Dev Sh", "Sizing", "Horizons", "Meta", "Ranks", "s"].map(h => (
              <th key={h} className="px-2 py-1.5 font-medium whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((t, i) => {
            const isTop = i === 0;
            return (
              <tr key={t.trial} className={cn("border-b border-border/50", isTop && "bg-green-50 dark:bg-green-900/10")}>
                <td className="px-2 py-1 font-mono">{t.trial}</td>
                <td className={cn("px-2 py-1 font-mono font-semibold", (t.hold_sharpe ?? 0) > 0 ? "text-green-600 dark:text-green-400" : "text-red-500")}>
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
                <td className="px-2 py-1 font-mono text-muted-foreground">{t.elapsed_s != null ? `${Math.round(t.elapsed_s)}s` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Run detail panel ─────────────────────────────────────────────────────────

function RunDetailPanel({ run, jobId }: { run: HypersearchRun; jobId: string | null }) {
  const [showParams, setShowParams] = useState(false);
  const [showAllTrials, setShowAllTrials] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const isActive = run.status === "running";

  // Poll job logs while running
  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.jobDetail(jobId!),
    enabled: !!jobId,
    refetchInterval: isActive ? 3000 : false,
  });

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [job?.logs?.length]);

  const progress = run.n_trials_requested > 0
    ? Math.round((run.n_trials_done / run.n_trials_requested) * 100)
    : 0;

  return (
    <div className="space-y-4">
      {/* Progress bar */}
      {(isActive || run.n_trials_done > 0) && (
        <div>
          <div className="mb-1 flex justify-between text-xs text-muted-foreground">
            <span>Trials: {run.n_trials_done} / {run.n_trials_requested}</span>
            <span>{progress}%{isActive ? " · running…" : ""}</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full transition-all duration-700", isActive ? "bg-blue-500" : "bg-green-500")}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Best Sharpe highlight */}
      {run.best_sharpe != null && (
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card p-3">
          <div>
            <p className="text-xs text-muted-foreground">Best holdout Sharpe</p>
            <p className={cn("text-2xl font-bold tabular-nums", run.best_sharpe > 0 ? "text-green-600 dark:text-green-400" : "text-red-500")}>
              {fmtSharpe(run.best_sharpe)}
            </p>
          </div>
          <div className="ml-auto text-right text-xs text-muted-foreground">
            <p>{run.n_trials_done} trials done</p>
            <p>{fmtTs(run.started_at)}</p>
          </div>
        </div>
      )}

      {/* Trial table */}
      {run.trials.length > 0 && (
        <div>
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Top {Math.min(10, run.trials.length)} Results
            </p>
            {run.trials.length > 10 && (
              <button className="text-xs text-primary hover:underline"
                onClick={() => setShowAllTrials(v => !v)}>
                {showAllTrials ? "Show top 10" : `Show all ${run.trials.length}`}
              </button>
            )}
          </div>
          <TrialTable trials={run.trials} limit={showAllTrials ? run.trials.length : 10} />
        </div>
      )}

      {/* Honest interpretation */}
      {run.status === "ok" && run.trials.length > 0 && (() => {
        const valid = run.trials.filter(t => !t.error && t.hold_ci_lo != null);
        const nPosCI = valid.filter(t => (t.hold_ci_lo ?? 0) > 0).length;
        return (
          <div className="rounded-md bg-muted/50 p-3 text-xs space-y-1">
            <p className="font-semibold text-foreground">Honest interpretation</p>
            <p className="text-muted-foreground">
              Configs with CI strictly &gt; 0 (statistically real edge):
              <span className={cn("ml-1 font-semibold", nPosCI > 0 ? "text-green-600 dark:text-green-400" : "text-red-500")}>
                {nPosCI} / {valid.length}
              </span>
            </p>
            <p className="text-muted-foreground/80">
              Sharpe CIs are computed on a {(run.config.n_tickers as number | null) ?? 25}-ticker fast-mode universe.
              Validate the best config on the full universe before trading.
            </p>
          </div>
        );
      })()}

      {/* Best params */}
      {run.best_params && (
        <div>
          <button className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
            onClick={() => setShowParams(v => !v)}>
            {showParams ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            Best Config Parameters
          </button>
          {showParams && (
            <pre className="mt-2 overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">
              {JSON.stringify(run.best_params, null, 2)}
            </pre>
          )}
        </div>
      )}

      {/* Logs */}
      {job && (
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Logs {job.logs.length > 0 && <span className="text-muted-foreground/60">({job.logs.length} lines)</span>}
          </p>
          <pre className="h-48 overflow-y-auto rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground/80 scrollbar-thin">
            {job.logs.length === 0
              ? (isActive ? "Waiting for output…" : "No logs captured.")
              : job.logs.join("\n")}
            <div ref={logsEndRef} />
          </pre>
        </div>
      )}
    </div>
  );
}

// ─── Run list row ─────────────────────────────────────────────────────────────

function RunRow({ run, selected, onClick }: {
  run: HypersearchRun; selected: boolean; onClick: () => void;
}) {
  const progress = run.n_trials_requested > 0
    ? Math.round((run.n_trials_done / run.n_trials_requested) * 100) : 0;
  return (
    <>
      <tr onClick={onClick}
        className={cn("cursor-pointer border-b border-border/50 transition-colors hover:bg-accent/50", selected && "bg-accent")}>
        <td className="px-4 py-2 font-mono text-xs">{run.id}</td>
        <td className="px-4 py-2">{statusBadge(run.status)}</td>
        <td className="px-4 py-2 text-xs">{run.n_trials_done}/{run.n_trials_requested}</td>
        <td className="px-4 py-2">
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-20 overflow-hidden rounded-full bg-muted">
              <div className={cn("h-full rounded-full", run.status === "ok" ? "bg-green-500" : "bg-blue-500")}
                style={{ width: `${progress}%` }} />
            </div>
            <span className="text-xs text-muted-foreground">{progress}%</span>
          </div>
        </td>
        <td className="px-4 py-2 font-mono text-xs">
          <span className={cn(run.best_sharpe != null && run.best_sharpe > 0 ? "text-green-600 dark:text-green-400 font-semibold" : "text-red-500")}>
            {fmtSharpe(run.best_sharpe)}
          </span>
        </td>
        <td className="px-4 py-2 text-xs text-muted-foreground">{run.config.n_tickers as number | null}</td>
        <td className="px-4 py-2 text-xs text-muted-foreground">{fmtTs(run.started_at)}</td>
        <td className="px-4 py-2">{selected ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}</td>
      </tr>
      {selected && (
        <tr className="bg-card">
          <td colSpan={8} className="px-6 py-4">
            <RunResultLoader runId={run.id} jobId={run.job_id} />
          </td>
        </tr>
      )}
    </>
  );
}

function RunResultLoader({ runId, jobId }: { runId: number; jobId: string | null }) {
  const { data: run } = useQuery({
    queryKey: ["hs-run", runId],
    queryFn: () => api.hypersearchRun(runId),
    refetchInterval: q => {
      const status = q.state.data?.status;
      return status === "running" ? 5000 : false;
    },
  });
  if (!run) return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
  return <RunDetailPanel run={run} jobId={jobId} />;
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function Hypersearch() {
  const qc = useQueryClient();
  const [showNewForm, setShowNewForm] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [pwAction, setPwAction] = useState<
    | { type: "launch"; queueId: string }
    | { type: "delete"; queueId: string }
    | { type: "cancel"; jobId: string }
    | null
  >(null);
  const [pwError, setPwError] = useState<string | null>(null);
  const [pwLoading, setPwLoading] = useState(false);

  // All jobs — filtered to hypersearch type
  const { data: allJobs = [] } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(50),
    refetchInterval: q => {
      const active = (q.state.data ?? []).some(
        j => j.job_type === "hypersearch" && ["running","queued","cancelling"].includes(j.status)
      );
      return active ? 4000 : 15000;
    },
  });
  const hsJobs = allJobs.filter(j => j.job_type === "hypersearch");

  // Queued jobs — filtered to hypersearch
  const { data: allQueued = [] } = useQuery({
    queryKey: ["queued"],
    queryFn: () => api.listQueued(),
    refetchInterval: 10000,
  });
  const hsQueued = allQueued.filter(q => (q.config as any)?.job_type === "hypersearch" || (q.config as any)?.n_trials != null);

  // Completed hypersearch runs
  const { data: runs = [] } = useQuery({
    queryKey: ["hs-runs"],
    queryFn: () => api.listHypersearchRuns(25),
    refetchInterval: q => {
      const hasRunning = (q.state.data ?? []).some(r => r.status === "running");
      return hasRunning ? 5000 : 20000;
    },
  });

  async function handlePwConfirm(pw: string) {
    if (!pwAction) return;
    setPwLoading(true); setPwError(null);
    try {
      if (pwAction.type === "launch") {
        const result = await api.launchQueued(pwAction.queueId, pw);
        qc.invalidateQueries({ queryKey: ["queued"] });
        qc.invalidateQueries({ queryKey: ["jobs"] });
        qc.invalidateQueries({ queryKey: ["hs-runs"] });
        // Select the new run when it appears
        setTimeout(() => qc.invalidateQueries({ queryKey: ["hs-runs"] }), 2000);
      } else if (pwAction.type === "delete") {
        await api.deleteQueued(pwAction.queueId, pw);
        qc.invalidateQueries({ queryKey: ["queued"] });
      } else if (pwAction.type === "cancel") {
        await api.cancelJob(pwAction.jobId, pw);
        qc.invalidateQueries({ queryKey: ["jobs"] });
        qc.invalidateQueries({ queryKey: ["hs-runs"] });
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold">
            <Zap className="h-6 w-6 text-primary" /> Hyperparameter Search
          </h1>
          <p className="text-sm text-muted-foreground">
            Bayesian optimisation over 20 pipeline parameters · objective: holdout Sharpe
          </p>
        </div>
        <Button onClick={() => setShowNewForm(true)}><Plus className="h-4 w-4" /> New Search</Button>
      </div>

      {showNewForm && <NewHypersearchForm onClose={() => { setShowNewForm(false); qc.invalidateQueries({ queryKey: ["queued"] }); }} />}

      {pwAction && (
        <PasswordModal
          title={pwAction.type === "launch" ? "Launch hypersearch" : pwAction.type === "cancel" ? "Cancel job" : "Delete queued job"}
          description={pwAction.type === "launch" ? "Enter STOCKPRED_PW to start the search. This may run for several hours." : "Enter STOCKPRED_PW to confirm."}
          confirmLabel={pwAction.type === "launch" ? "Launch" : pwAction.type === "cancel" ? "Cancel Job" : "Delete"}
          confirmVariant={pwAction.type === "delete" || pwAction.type === "cancel" ? "destructive" : "default"}
          onConfirm={handlePwConfirm}
          onClose={() => setPwAction(null)}
          isLoading={pwLoading}
          error={pwError}
        />
      )}

      {/* Queue */}
      {hsQueued.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Pending (awaiting launch)</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2">ID</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2">Trials</th>
                  <th className="px-4 py-2">Tickers</th>
                  <th className="px-4 py-2">Queued</th>
                  <th className="px-4 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {hsQueued.map(q => (
                  <tr key={q.id} className="border-b border-border/50">
                    <td className="px-4 py-2 font-mono text-xs">{q.id.slice(0, 8)}…</td>
                    <td className="px-4 py-2">{statusBadge(q.status)}</td>
                    <td className="px-4 py-2 text-xs">{(q.config as any).n_trials ?? "—"}</td>
                    <td className="px-4 py-2 text-xs">{(q.config as any).n_tickers ?? "—"}</td>
                    <td className="px-4 py-2 text-xs text-muted-foreground">{fmtTs(q.created_at)}</td>
                    <td className="px-4 py-2">
                      <div className="flex gap-1">
                        {q.status === "pending" && (
                          <>
                            <Button size="sm" variant="default" className="h-6 px-2 text-xs"
                              onClick={() => { setPwError(null); setPwAction({ type: "launch", queueId: q.id }); }}>
                              <Play className="h-3 w-3" /> Launch
                            </Button>
                            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-destructive"
                              onClick={() => { setPwError(null); setPwAction({ type: "delete", queueId: q.id }); }}>
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Active jobs */}
      {hsJobs.filter(j => ["running","queued","cancelling"].includes(j.status)).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Running</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2">Job ID</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2">Started</th>
                  <th className="px-4 py-2">Elapsed</th>
                  <th className="px-4 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {hsJobs.filter(j => ["running","queued","cancelling"].includes(j.status)).map(j => (
                  <tr key={j.job_id} className="border-b border-border/50">
                    <td className="px-4 py-2 font-mono text-xs">{j.job_id.slice(0, 8)}…</td>
                    <td className="px-4 py-2">{statusBadge(j.status)}</td>
                    <td className="px-4 py-2 text-xs text-muted-foreground">{fmtTs(j.started_at)}</td>
                    <td className="px-4 py-2 text-xs">{fmtDuration(j.elapsed_s)}</td>
                    <td className="px-4 py-2">
                      {j.status === "running" && (
                        <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-destructive"
                          onClick={() => { setPwError(null); setPwAction({ type: "cancel", jobId: j.job_id }); }}>
                          <XCircle className="h-3 w-3" /> Cancel
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            Search Results
            <button onClick={() => qc.invalidateQueries({ queryKey: ["hs-runs"] })}
              className="text-muted-foreground hover:text-foreground">
              <RefreshCw className="h-4 w-4" />
            </button>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {runs.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-muted-foreground">
              No searches yet. Queue and launch a new hypersearch above.
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
                    <th className="px-4 py-2 w-6"></th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(r => (
                    <RunRow
                      key={r.id}
                      run={r}
                      selected={selectedRunId === r.id}
                      onClick={() => setSelectedRunId(selectedRunId === r.id ? null : r.id)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
