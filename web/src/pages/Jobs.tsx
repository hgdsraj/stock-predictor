import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Play,
  Trash2,
  XCircle,
  Plus,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
  ExternalLink,
  Copy,
} from "lucide-react";
import { api } from "@/api/client";
import type { JobDetail, QueuedJob } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

// ─── helpers ────────────────────────────────────────────────────────────────

function JobIdCell({ jobId, onClick }: { jobId: string; onClick?: () => void }) {
  const [copied, setCopied] = useState(false);
  function copy(e: React.MouseEvent) {
    e.stopPropagation();
    navigator.clipboard.writeText(jobId).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }
  return (
    <span className="flex items-center gap-1 font-mono text-xs">
      <button onClick={onClick} className="hover:underline" title={jobId}>
        {jobId.slice(0, 8)}…
      </button>
      <button onClick={copy} title="Copy full ID" className="text-muted-foreground hover:text-foreground">
        {copied ? <CheckCircle className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
      </button>
      <a
        href={`/jobs/${jobId}`}
        target="_blank"
        rel="noopener noreferrer"
        title="Open raw JSON"
        onClick={e => e.stopPropagation()}
        className="text-muted-foreground hover:text-foreground"
      >
        <ExternalLink className="h-3 w-3" />
      </a>
    </span>
  );
}

function statusBadge(status: string) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    queued:     { cls: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300", icon: <Clock className="h-3 w-3" />, label: "Queued" },
    running:    { cls: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",    icon: <Loader2 className="h-3 w-3 animate-spin" />, label: "Running" },
    cancelling: { cls: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300", icon: <Loader2 className="h-3 w-3 animate-spin" />, label: "Cancelling" },
    ok:         { cls: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300", icon: <CheckCircle className="h-3 w-3" />, label: "Completed" },
    failed:     { cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",        icon: <AlertCircle className="h-3 w-3" />, label: "Failed" },
    crashed:    { cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",        icon: <AlertCircle className="h-3 w-3" />, label: "Crashed" },
    cancelled:  { cls: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",       icon: <XCircle className="h-3 w-3" />, label: "Cancelled" },
    pending:    { cls: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300", icon: <Clock className="h-3 w-3" />, label: "Pending" },
    launched:   { cls: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",    icon: <Play className="h-3 w-3" />, label: "Launched" },
  };
  const s = map[status] ?? { cls: "bg-gray-100 text-gray-600", icon: null, label: status };
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", s.cls)}>
      {s.icon}{s.label}
    </span>
  );
}

function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function fmtElapsed(startedAt: string | null | undefined): string {
  if (!startedAt) return "—";
  const diff = (Date.now() - new Date(startedAt).getTime()) / 1000;
  return fmtDuration(diff);
}

function fmtTs(ts: string | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts).toLocaleString();
}

function estimateTotalSeconds(config: Record<string, unknown>): number {
  const phase = (config.phase as number) ?? 1;
  // null n_tickers means full S&P 500 universe (~500 tickers)
  const nTickers = config.n_tickers as number | null;
  const effective = nTickers ?? 500;
  const base = phase === 5 ? 7200 : 3600;
  return base * (effective / 500);
}

function configSummary(config: Record<string, unknown>): string {
  const phase = config.phase ?? 1;
  const n = config.n_tickers ?? "all";
  const start = (config.start_date as string ?? "").slice(0, 7);
  return `Phase ${phase} · ${n} tickers · from ${start}`;
}

// ─── Password modal ──────────────────────────────────────────────────────────

interface PwModalProps {
  title: string;
  description: string;
  confirmLabel: string;
  confirmVariant?: "default" | "destructive";
  onConfirm: (pw: string) => void;
  onClose: () => void;
  isLoading: boolean;
  error: string | null;
}

function PasswordModal({
  title, description, confirmLabel, confirmVariant = "default",
  onConfirm, onClose, isLoading, error,
}: PwModalProps) {
  const [pw, setPw] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (pw.trim()) onConfirm(pw.trim());
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card p-6 shadow-xl">
        <h2 className="mb-1 text-lg font-semibold">{title}</h2>
        <p className="mb-4 text-sm text-muted-foreground">{description}</p>
        <form onSubmit={submit} className="space-y-3">
          <input
            ref={inputRef}
            type="password"
            value={pw}
            onChange={e => setPw(e.target.value)}
            placeholder="Enter password (STOCKPRED_PW)"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          />
          {error && <p className="text-xs text-red-500">{error}</p>}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" type="button" onClick={onClose} disabled={isLoading}>
              Cancel
            </Button>
            <Button variant={confirmVariant} size="sm" type="submit" disabled={isLoading || !pw.trim()}>
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : confirmLabel}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── New Job form ────────────────────────────────────────────────────────────

const DEFAULT_PHASE1 = {
  phase: 1, n_tickers: 100, start_date: "2013-01-01", end_date: null,
  universe_sampling: "current", refresh_data: false,
  horizons: [1, 5, 21], model: "gbm", use_sector_features: true,
  k_per_side: 20,
  cv: { train_years: 3, test_months: 6, embargo_days: 25, min_train_obs: 1000 },
  gbm: { num_leaves: 63, learning_rate: 0.03, n_estimators: 800, min_data_in_leaf: 200,
         feature_fraction: 0.8, bagging_fraction: 0.8, bagging_freq: 5,
         reg_lambda: 1.0, early_stopping_rounds: 50 },
};

const DEFAULT_PHASE5 = {
  phase: 5, n_tickers: null, start_date: "2013-01-01", end_date: null,
  universe_sampling: "current", refresh_data: false,
  horizons: [1, 5], model: "gbm", use_sector_features: true,
  use_tier2_features: true, use_regime_features: true,
  beta_neutralise: true, bootstrap_method: "block", holdout_years: 2,
  position_sizing: "vol_scaled", k_per_side_pct: 0.10, leverage_per_side: 1.0,
  sector_cap_gross: 0.25, min_trade_threshold: 0.005,
  ensemble_weighting: "ic_ir", bootstrap_n: 500,
  cv: { train_years: 5, test_months: 6, embargo_days: 25, min_train_obs: 1000 },
  gbm: { num_leaves: 63, learning_rate: 0.03, n_estimators: 800, min_data_in_leaf: 200,
         feature_fraction: 0.7, bagging_fraction: 0.8, bagging_freq: 5,
         reg_lambda: 2.0, early_stopping_rounds: 50 },
};

interface NewJobFormProps { onClose: () => void; onQueued: () => void; }

function NewJobForm({ onClose, onQueued }: NewJobFormProps) {
  const [phase, setPhase] = useState<1 | 5>(1);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [cfg, setCfg] = useState<Record<string, unknown>>(DEFAULT_PHASE1 as any);
  const [error, setError] = useState<string | null>(null);
  const qc = useQueryClient();

  const defaults = phase === 5 ? DEFAULT_PHASE5 : DEFAULT_PHASE1;

  function switchPhase(p: 1 | 5) {
    setPhase(p);
    setCfg(p === 5 ? DEFAULT_PHASE5 as any : DEFAULT_PHASE1 as any);
  }

  function setField(key: string, value: unknown) {
    setCfg(prev => ({ ...prev, [key]: value }));
  }

  function setGbm(key: string, value: unknown) {
    setCfg(prev => ({ ...prev, gbm: { ...(prev.gbm as object), [key]: value } }));
  }

  function setCv(key: string, value: unknown) {
    setCfg(prev => ({ ...prev, cv: { ...(prev.cv as object), [key]: value } }));
  }

  const mutate = useMutation({
    mutationFn: () => api.queueJob(cfg),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["queued"] });
      onQueued();
    },
    onError: (e: Error) => setError(e.message),
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xl">
        <div className="border-b border-border px-6 py-4">
          <h2 className="text-lg font-semibold">New Job</h2>
          <p className="text-sm text-muted-foreground">Configure and queue a pipeline run. A password is needed to launch it.</p>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {/* Phase selector */}
          <div>
            <label className="mb-1 block text-sm font-medium">Phase</label>
            <div className="flex gap-2">
              {([1, 5] as const).map(p => (
                <button key={p} onClick={() => switchPhase(p)}
                  className={cn("rounded-lg border px-4 py-2 text-sm font-medium transition-colors",
                    phase === p ? "border-primary bg-primary text-primary-foreground" : "border-border hover:bg-accent")}>
                  Phase {p}{p === 1 ? " — Basic GBM" : " — Full (vol-scaled, regime-aware)"}
                </button>
              ))}
            </div>
          </div>

          {/* Universe */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Tickers (null = all S&P 500)</label>
              <input type="number" min={1} placeholder="null = all"
                value={(cfg.n_tickers as number | null) ?? ""}
                onChange={e => setField("n_tickers", e.target.value === "" ? null : Number(e.target.value))}
                className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Start date</label>
              <input type="date"
                value={(cfg.start_date as string) ?? "2013-01-01"}
                onChange={e => setField("start_date", e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Universe sampling</label>
              <select value={(cfg.universe_sampling as string) ?? "current"}
                onChange={e => setField("universe_sampling", e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm">
                <option value="current">current (today's members)</option>
                <option value="random">random (unbiased)</option>
                <option value="first">first (alphabetical)</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Horizons (days)</label>
              <input type="text"
                value={((cfg.horizons as number[]) ?? [1, 5]).join(", ")}
                onChange={e => {
                  const vals = e.target.value.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                  setField("horizons", vals);
                }}
                className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
            </div>
          </div>

          {/* Phase 5 extra */}
          {phase === 5 && (
            <div className="grid grid-cols-2 gap-3">
              <label className="col-span-2 -mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Phase 5 options</label>
              {[
                ["beta_neutralise", "Beta-neutralise vs SPY"],
                ["use_tier2_features", "Tier-2 features (momentum, IVOL, beta)"],
                ["use_regime_features", "Regime features (VIX, term spread)"],
              ].map(([k, label]) => (
                <label key={k} className="flex cursor-pointer items-center gap-2 text-sm">
                  <input type="checkbox" checked={!!(cfg[k])}
                    onChange={e => setField(k, e.target.checked)}
                    className="h-4 w-4 rounded border-border" />
                  {label}
                </label>
              ))}
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Position sizing</label>
                <select value={(cfg.position_sizing as string) ?? "vol_scaled"}
                  onChange={e => setField("position_sizing", e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm">
                  <option value="vol_scaled">vol_scaled</option>
                  <option value="top_k">top_k</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">k per side %</label>
                <input type="number" step={0.01} min={0.01} max={1}
                  value={(cfg.k_per_side_pct as number) ?? 0.10}
                  onChange={e => setField("k_per_side_pct", parseFloat(e.target.value))}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Sector cap gross</label>
                <input type="number" step={0.05} min={0} max={1}
                  value={(cfg.sector_cap_gross as number) ?? 0.25}
                  onChange={e => setField("sector_cap_gross", parseFloat(e.target.value))}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Holdout years</label>
                <input type="number" min={0} max={5}
                  value={(cfg.holdout_years as number) ?? 2}
                  onChange={e => setField("holdout_years", parseInt(e.target.value))}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
              </div>
            </div>
          )}

          {/* Advanced toggle */}
          <button onClick={() => setShowAdvanced(v => !v)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
            {showAdvanced ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            Advanced (GBM / CV params)
          </button>

          {showAdvanced && (
            <div className="space-y-3 rounded-lg border border-border p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">GBM</p>
              <div className="grid grid-cols-3 gap-3">
                {(["num_leaves","n_estimators","min_data_in_leaf"] as const).map(k => (
                  <div key={k}>
                    <label className="mb-1 block text-xs font-medium text-muted-foreground">{k}</label>
                    <input type="number"
                      value={((cfg.gbm as any)?.[k]) ?? (defaults as any).gbm[k]}
                      onChange={e => setGbm(k, parseInt(e.target.value))}
                      className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
                  </div>
                ))}
                {(["learning_rate","feature_fraction","reg_lambda"] as const).map(k => (
                  <div key={k}>
                    <label className="mb-1 block text-xs font-medium text-muted-foreground">{k}</label>
                    <input type="number" step={0.01}
                      value={((cfg.gbm as any)?.[k]) ?? (defaults as any).gbm[k]}
                      onChange={e => setGbm(k, parseFloat(e.target.value))}
                      className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
                  </div>
                ))}
              </div>
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Cross-validation</p>
              <div className="grid grid-cols-2 gap-3">
                {(["train_years","test_months","embargo_days","min_train_obs"] as const).map(k => (
                  <div key={k}>
                    <label className="mb-1 block text-xs font-medium text-muted-foreground">{k}</label>
                    <input type="number"
                      value={((cfg.cv as any)?.[k]) ?? (defaults as any).cv[k]}
                      onChange={e => setCv(k, parseInt(e.target.value))}
                      className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && <p className="text-sm text-red-500">{error}</p>}
        </div>

        <div className="flex justify-end gap-2 border-t border-border px-6 py-4">
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button size="sm" onClick={() => { setError(null); mutate.mutate(); }}
            disabled={mutate.isPending}>
            {mutate.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            Queue Job
          </Button>
        </div>
      </div>
    </div>
  );
}

// ─── Job detail panel ────────────────────────────────────────────────────────

interface JobDetailPanelProps {
  jobId: string;
  onClose: () => void;
  onCancel: (jobId: string) => void;
}

function JobDetailPanel({ jobId, onClose, onCancel }: JobDetailPanelProps) {
  const logsEndRef = useRef<HTMLDivElement>(null);
  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.jobDetail(jobId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "running" || s === "queued" || s === "cancelling" ? 2500 : false;
    },
  });

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [job?.logs?.length]);

  if (!job) return null;

  const isActive = ["running", "queued", "cancelling"].includes(job.status);
  const estTotal = estimateTotalSeconds(job.config);
  const elapsed = job.elapsed_s ?? (job.started_at ? (Date.now() - new Date(job.started_at).getTime()) / 1000 : null);
  const pct = elapsed != null && estTotal > 0 ? Math.min(99, Math.round((elapsed / estTotal) * 100)) : null;

  return (
    <div className="mt-4 rounded-xl border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <JobIdCell jobId={jobId} />
          {statusBadge(job.status)}
          {job.elapsed_s != null && (
            <span className="text-xs text-muted-foreground">Runtime: {fmtDuration(job.elapsed_s)}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isActive && (
            <Button variant="destructive" size="sm" onClick={() => onCancel(jobId)}>
              <XCircle className="h-4 w-4" /> Cancel
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>
        </div>
      </div>

      {/* Progress bar */}
      {isActive && pct != null && (
        <div className="mb-3">
          <div className="mb-1 flex justify-between text-xs text-muted-foreground">
            <span>Estimated progress</span>
            <span>{pct}% · ~{fmtDuration(Math.max(0, estTotal - (elapsed ?? 0)))} remaining</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div className="h-full rounded-full bg-primary transition-all duration-1000" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3">
        {/* Parameters */}
        <div className="col-span-2 lg:col-span-1">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Parameters</p>
          <dl className="space-y-1 text-xs">
            {Object.entries(job.config)
              .filter(([k]) => !["cv", "gbm"].includes(k))
              .map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="font-mono text-right">{String(v ?? "null")}</dd>
                </div>
              ))}
          </dl>
          {job.error && (
            <div className="mt-3 rounded-md bg-red-50 p-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
              <strong>Error:</strong> {job.error}
            </div>
          )}
        </div>

        {/* Logs */}
        <div className="col-span-2">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Logs {job.logs.length > 0 && <span className="text-muted-foreground/60">({job.logs.length} lines)</span>}
          </p>
          <pre className="h-64 overflow-y-auto rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground/80 scrollbar-thin">
            {job.logs.length === 0
              ? (isActive ? "Waiting for output…" : "No logs captured.")
              : job.logs.join("\n")}
            <div ref={logsEndRef} />
          </pre>
        </div>
      </div>
    </div>
  );
}

// ─── Main page ───────────────────────────────────────────────────────────────

export function Jobs() {
  const qc = useQueryClient();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [showNewJobForm, setShowNewJobForm] = useState(false);
  const [pwAction, setPwAction] = useState<
    | { type: "cancel"; jobId: string }
    | { type: "launch"; queueId: string }
    | { type: "delete"; queueId: string }
    | null
  >(null);
  const [pwError, setPwError] = useState<string | null>(null);
  const [pwLoading, setPwLoading] = useState(false);

  const hasActive = (jobs: JobDetail[]) =>
    jobs.some(j => ["running", "queued", "cancelling"].includes(j.status));

  const { data: jobs = [] } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(),
    refetchInterval: (q) => (hasActive(q.state.data ?? []) ? 3000 : 15000),
  });

  const { data: queued = [] } = useQuery({
    queryKey: ["queued"],
    queryFn: () => api.listQueued(),
    refetchInterval: 10000,
  });

  async function handlePwConfirm(pw: string) {
    if (!pwAction) return;
    setPwLoading(true);
    setPwError(null);
    try {
      if (pwAction.type === "cancel") {
        await api.cancelJob(pwAction.jobId, pw);
        qc.invalidateQueries({ queryKey: ["jobs"] });
        qc.invalidateQueries({ queryKey: ["job", pwAction.jobId] });
      } else if (pwAction.type === "launch") {
        const result = await api.launchQueued(pwAction.queueId, pw);
        qc.invalidateQueries({ queryKey: ["queued"] });
        qc.invalidateQueries({ queryKey: ["jobs"] });
        setSelectedJobId(result.job_id);
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

  const pending = queued.filter(j => j.status === "pending");
  const launched = queued.filter(j => j.status !== "pending");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Jobs</h1>
          <p className="text-sm text-muted-foreground">
            Queue and manage pipeline runs. A password (STOCKPRED_PW) is required to launch or cancel.
          </p>
        </div>
        <Button onClick={() => setShowNewJobForm(true)}>
          <Plus className="h-4 w-4" /> New Job
        </Button>
      </div>

      {/* Active & History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            Active &amp; History
            <button onClick={() => qc.invalidateQueries({ queryKey: ["jobs"] })}
              className="text-muted-foreground hover:text-foreground">
              <RefreshCw className="h-4 w-4" />
            </button>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {jobs.length === 0 ? (
            <p className="px-4 py-6 text-center text-sm text-muted-foreground">No jobs yet. Queue one above.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2">ID</th>
                    <th className="px-4 py-2">Status</th>
                    <th className="px-4 py-2">Phase</th>
                    <th className="px-4 py-2">Tickers</th>
                    <th className="px-4 py-2">Started</th>
                    <th className="px-4 py-2">Runtime</th>
                    <th className="px-4 py-2">Run ID</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map(j => {
                    const isSelected = selectedJobId === j.job_id;
                    const isActive = ["running", "queued", "cancelling"].includes(j.status);
                    return (
                      <tr key={j.job_id}
                        onClick={() => setSelectedJobId(isSelected ? null : j.job_id)}
                        className={cn(
                          "cursor-pointer border-b border-border/50 transition-colors hover:bg-accent/50",
                          isSelected && "bg-accent",
                        )}>
                        <td className="px-4 py-2">
                          <JobIdCell jobId={j.job_id} onClick={() => setSelectedJobId(isSelected ? null : j.job_id)} />
                        </td>
                        <td className="px-4 py-2">{statusBadge(j.status)}</td>
                        <td className="px-4 py-2">{(j.config.phase as number) ?? 1}</td>
                        <td className="px-4 py-2">{String(j.config.n_tickers ?? "all")}</td>
                        <td className="px-4 py-2 text-xs">
                          {isActive ? fmtElapsed(j.started_at) + " ago" : fmtTs(j.started_at)}
                        </td>
                        <td className="px-4 py-2 text-xs">{fmtDuration(j.elapsed_s)}</td>
                        <td className="px-4 py-2 text-xs text-muted-foreground">{j.run_id ?? "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Detail panel */}
          {selectedJobId && (
            <div className="px-4 pb-4">
              <JobDetailPanel
                jobId={selectedJobId}
                onClose={() => setSelectedJobId(null)}
                onCancel={id => { setPwError(null); setPwAction({ type: "cancel", jobId: id }); }}
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Queued jobs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            <span>
              Queued (pending launch)
              {pending.length > 0 && (
                <span className="ml-2 rounded-full bg-yellow-100 px-2 py-0.5 text-xs text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300">
                  {pending.length} / 5
                </span>
              )}
            </span>
            <button onClick={() => qc.invalidateQueries({ queryKey: ["queued"] })}
              className="text-muted-foreground hover:text-foreground">
              <RefreshCw className="h-4 w-4" />
            </button>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {queued.length === 0 ? (
            <p className="px-4 py-6 text-center text-sm text-muted-foreground">
              No queued jobs. Click <strong>New Job</strong> to create one.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2">ID</th>
                    <th className="px-4 py-2">Status</th>
                    <th className="px-4 py-2">Summary</th>
                    <th className="px-4 py-2">Created</th>
                    <th className="px-4 py-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {queued.map(q => (
                    <tr key={q.id} className="border-b border-border/50">
                      <td className="px-4 py-2 font-mono text-xs">{q.id.slice(0, 8)}…</td>
                      <td className="px-4 py-2">{statusBadge(q.status)}</td>
                      <td className="px-4 py-2 text-xs text-muted-foreground">{configSummary(q.config)}</td>
                      <td className="px-4 py-2 text-xs">{fmtTs(q.created_at)}</td>
                      <td className="px-4 py-2">
                        <div className="flex gap-1">
                          {q.status === "pending" && (
                            <>
                              <Button variant="default" size="sm"
                                onClick={() => { setPwError(null); setPwAction({ type: "launch", queueId: q.id }); }}>
                                <Play className="h-3 w-3" /> Launch
                              </Button>
                              <Button variant="ghost" size="sm"
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
            </div>
          )}
        </CardContent>
      </Card>

      {/* Modals */}
      {showNewJobForm && (
        <NewJobForm
          onClose={() => setShowNewJobForm(false)}
          onQueued={() => { setShowNewJobForm(false); }}
        />
      )}

      {pwAction && (
        <PasswordModal
          title={
            pwAction.type === "cancel" ? "Cancel Job" :
            pwAction.type === "launch" ? "Launch Job" : "Delete Queued Job"
          }
          description={
            pwAction.type === "cancel"
              ? "The running job will be soft-cancelled — it will finish its current step but results won't be saved."
              : pwAction.type === "launch"
              ? "Enter STOCKPRED_PW to launch this queued job."
              : "Enter STOCKPRED_PW to permanently delete this queued job."
          }
          confirmLabel={
            pwAction.type === "cancel" ? "Cancel Job" :
            pwAction.type === "launch" ? "Launch" : "Delete"
          }
          confirmVariant={pwAction.type === "delete" || pwAction.type === "cancel" ? "destructive" : "default"}
          onConfirm={handlePwConfirm}
          onClose={() => { setPwAction(null); setPwError(null); }}
          isLoading={pwLoading}
          error={pwError}
        />
      )}
    </div>
  );
}
