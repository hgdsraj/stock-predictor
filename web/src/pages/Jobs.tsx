import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import {
  Play, Trash2, XCircle, Plus, ChevronDown, ChevronUp, RefreshCw,
  Clock, CheckCircle, AlertCircle, Loader2, ExternalLink, Copy,
  RotateCcw, History,
} from "lucide-react";
import { api } from "@/api/client";
import type { JobDetail, QueuedJob } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

// Status values that mean "this job ended without producing a usable run" —
// the eligibility set for the Restart button on the Active & History table.
// "ok" is excluded by design: restart-from-success would silently re-spend
// hours of compute to produce a near-identical run.
const RESTARTABLE_STATUSES = new Set(["failed", "cancelled", "crashed"]);

// ─── Helpers ─────────────────────────────────────────────────────────────────

function JobIdCell({ jobId, onClick }: { jobId: string; onClick?: () => void }) {
  const [copied, setCopied] = useState(false);
  function copy(e: React.MouseEvent) {
    e.stopPropagation();
    navigator.clipboard.writeText(jobId).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
  }
  return (
    <span className="flex items-center gap-1 font-mono text-xs">
      <button onClick={onClick} className="hover:underline" title={jobId}>{jobId.slice(0, 8)}…</button>
      <button onClick={copy} title="Copy full ID" className="text-muted-foreground hover:text-foreground">
        {copied ? <CheckCircle className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
      </button>
      <a href={`/jobs/${jobId}`} target="_blank" rel="noopener noreferrer" title="Open raw JSON"
        onClick={e => e.stopPropagation()} className="text-muted-foreground hover:text-foreground">
        <ExternalLink className="h-3 w-3" />
      </a>
    </span>
  );
}

function detectPhase(config: Record<string, unknown>): number {
  if (config.phase != null) return config.phase as number;
  return "k_per_side_pct" in config || "position_sizing" in config ? 5 : 1;
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

function fmtDuration(s: number | null | undefined) {
  if (s == null) return "—";
  const m = Math.floor(s / 60); const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtElapsed(startedAt: string | null | undefined) {
  if (!startedAt) return "—";
  return fmtDuration((Date.now() - new Date(startedAt).getTime()) / 1000);
}
function fmtTs(ts: string | null | undefined) { return ts ? new Date(ts).toLocaleString() : "—"; }

function estimateTotalSeconds(config: Record<string, unknown>): number {
  const phase = detectPhase(config);
  const nTickers = config.n_tickers as number | null;
  const effective = nTickers ?? 500;
  const base = phase === 5 ? 7200 : 3600;
  return base * (effective / 500);
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

// ─── Form sub-components ──────────────────────────────────────────────────────

function Field({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={className}>
      <label className="mb-1 block text-xs font-medium text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}

const inp = "w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary";
const sel = inp;

function Checkbox({ label, checked, onChange, note }: { label: string; checked: boolean; onChange: (v: boolean) => void; note?: string }) {
  return (
    <label className="flex cursor-pointer items-start gap-2 text-sm">
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} className="mt-0.5 h-4 w-4 rounded border-border accent-primary" />
      <span className="flex flex-col">
        <span>{label}</span>
        {note && <span className="text-xs text-muted-foreground">{note}</span>}
      </span>
    </label>
  );
}

function Section({ title, children, defaultOpen = false }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-border">
      <button type="button" onClick={() => setOpen(v => !v)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-sm font-medium hover:bg-accent/50 transition-colors">
        {title}
        {open ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
      </button>
      {open && <div className="border-t border-border px-4 py-3 space-y-3">{children}</div>}
    </div>
  );
}

// ─── Phase presets ────────────────────────────────────────────────────────────

const BASE_UNIVERSE = {
  start_date: "2014-01-01", end_date: null, n_tickers: 150,
  universe_sampling: "current", refresh_data: false,
};
const BASE_CV = { train_years: 5, test_months: 6, embargo_days: 25, min_train_obs: 1000 };
const BASE_GBM = {
  num_leaves: 63, learning_rate: 0.03, n_estimators: 800, min_data_in_leaf: 200,
  feature_fraction: 0.7, bagging_fraction: 0.8, bagging_freq: 5, reg_lambda: 2.0, early_stopping_rounds: 50,
};

const PRESETS: Record<string, Record<string, unknown>> = {
  "Phase 1": {
    phase: 1, ...BASE_UNIVERSE, horizons: [1, 5, 21], model: "gbm",
    use_sector_features: true, k_per_side: 20, feature_cols: null,
    cv: BASE_CV, gbm: BASE_GBM,
  },
  "Phase 5": {
    phase: 5, ...BASE_UNIVERSE, horizons: [5], model: "gbm",
    use_sector_features: true, use_tier2_features: true, use_regime_features: true,
    beta_neutralise: true, bootstrap_method: "block", holdout_years: 2,
    position_sizing: "vol_scaled", k_per_side_pct: 0.15, leverage_per_side: 1.0,
    sector_cap_gross: 0.25, min_trade_threshold: 0.02, ensemble_weighting: "ic_ir",
    bootstrap_n: 500,
    use_meta_labelling: false, meta_threshold: 0.55, meta_mode: "binary",
    meta_conf_floor: 0.5, meta_conf_cap: 1.0, meta_walk_forward_folds: 1, meta_per_sector: false,
    use_triple_barrier_labels: false, tb_k_sigma: 2.0,
    ranks_only: false, feature_exclude: [],
    use_edgar_features: false, use_edgar_item_features: false,
    cv: BASE_CV, gbm: BASE_GBM,
  },
  "Phase 8/9 (meta)": {
    phase: 5, ...BASE_UNIVERSE, horizons: [5], model: "gbm",
    use_sector_features: true, use_tier2_features: true, use_regime_features: true,
    beta_neutralise: true, bootstrap_method: "block", holdout_years: 2,
    position_sizing: "vol_scaled", k_per_side_pct: 0.15, leverage_per_side: 1.0,
    sector_cap_gross: 0.25, min_trade_threshold: 0.02, ensemble_weighting: "ic_ir",
    bootstrap_n: 500,
    use_meta_labelling: true, meta_threshold: 0.55, meta_mode: "confidence",
    meta_conf_floor: 0.50, meta_conf_cap: 0.80, meta_walk_forward_folds: 3, meta_per_sector: false,
    use_triple_barrier_labels: false, tb_k_sigma: 2.0,
    ranks_only: true, feature_exclude: [],
    use_edgar_features: false, use_edgar_item_features: false,
    cv: BASE_CV, gbm: BASE_GBM,
  },
  "Phase 12/13 (EDGAR)": {
    phase: 5, ...BASE_UNIVERSE, horizons: [5], model: "gbm",
    use_sector_features: true, use_tier2_features: true, use_regime_features: true,
    beta_neutralise: true, bootstrap_method: "block", holdout_years: 2,
    position_sizing: "vol_scaled", k_per_side_pct: 0.15, leverage_per_side: 1.0,
    sector_cap_gross: 0.25, min_trade_threshold: 0.02, ensemble_weighting: "ic_ir",
    bootstrap_n: 200,
    use_meta_labelling: true, meta_threshold: 0.55, meta_mode: "confidence",
    meta_conf_floor: 0.50, meta_conf_cap: 0.80, meta_walk_forward_folds: 3, meta_per_sector: false,
    use_triple_barrier_labels: false, tb_k_sigma: 2.0,
    ranks_only: true, feature_exclude: [],
    use_edgar_features: true, use_edgar_item_features: true,
    cv: BASE_CV, gbm: BASE_GBM,
  },
};

const OPTIMAL_CURL = `curl -X POST https://stock-predictor-production-d4d4.up.railway.app/jobs/queue \\
  -H "Content-Type: application/json" \\
  -d '{
    "phase": 5,
    "start_date": "2014-01-01",
    "n_tickers": 200,
    "universe_sampling": "current",
    "horizons": [5],
    "use_sector_features": true,
    "use_tier2_features": true,
    "use_regime_features": true,
    "beta_neutralise": true,
    "ensemble_weighting": "ic_ir",
    "position_sizing": "vol_scaled",
    "k_per_side_pct": 0.15,
    "leverage_per_side": 1.0,
    "sector_cap_gross": 0.25,
    "min_trade_threshold": 0.02,
    "holdout_years": 2,
    "ranks_only": true,
    "use_meta_labelling": true,
    "meta_threshold": 0.55,
    "meta_mode": "confidence",
    "meta_conf_floor": 0.50,
    "meta_conf_cap": 0.80,
    "meta_walk_forward_folds": 3,
    "bootstrap_n": 200,
    "cv": { "train_years": 5, "test_months": 6, "embargo_days": 25, "min_train_obs": 1000 },
    "gbm": {
      "num_leaves": 63, "learning_rate": 0.03, "n_estimators": 800,
      "min_data_in_leaf": 200, "feature_fraction": 0.7,
      "bagging_fraction": 0.8, "bagging_freq": 5, "reg_lambda": 2.0,
      "early_stopping_rounds": 50
    }
  }'`;

// ─── New Job form ─────────────────────────────────────────────────────────────

type PhasePreset = "Phase 1" | "Phase 5" | "Phase 8/9 (meta)" | "Phase 12/13 (EDGAR)";
const PRESET_OPTIONS: { value: PhasePreset; label: string; desc: string }[] = [
  { value: "Phase 1",            label: "Phase 1 — Basic GBM",             desc: "Equal-weight top-K, no meta or EDGAR" },
  { value: "Phase 5",            label: "Phase 5 — Full feature stack",     desc: "Tier-2, regime, beta-neutral, vol-scaled" },
  { value: "Phase 8/9 (meta)",   label: "Phase 8/9 — + Meta-labelling",    desc: "Confidence-weighted gating, ranks only" },
  { value: "Phase 12/13 (EDGAR)","label": "Phase 12/13 — + EDGAR items",   desc: "Best holdout Sharpe: +0.17 on 150 tickers" },
];

function NewJobForm({ onClose, onQueued }: { onClose: () => void; onQueued: () => void }) {
  const [preset, setPreset] = useState<PhasePreset>("Phase 5");
  const [cfg, setCfg] = useState<Record<string, unknown>>(PRESETS["Phase 5"]);
  const [showCurl, setShowCurl] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [curlCopied, setCurlCopied] = useState(false);
  const qc = useQueryClient();

  function switchPreset(p: PhasePreset) {
    setPreset(p);
    setCfg(PRESETS[p]);
  }

  function set(key: string, value: unknown) { setCfg(prev => ({ ...prev, [key]: value })); }
  function setGbm(key: string, v: unknown) { setCfg(prev => ({ ...prev, gbm: { ...(prev.gbm as object), [key]: v } })); }
  function setCv(key: string, v: unknown)  { setCfg(prev => ({ ...prev, cv:  { ...(prev.cv  as object), [key]: v } })); }

  const mutate = useMutation({
    mutationFn: () => api.queueJob(cfg),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["queued"] }); onQueued(); },
    onError: (e: Error) => setError(e.message),
  });

  const isPhase1 = preset === "Phase 1";
  const selectedOption = PRESET_OPTIONS.find(o => o.value === preset)!;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="flex max-h-[92vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xl">
        {/* Header */}
        <div className="border-b border-border px-6 py-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold">New Job</h2>
              <p className="text-sm text-muted-foreground">Choose a preset, tweak, then queue. Password required to launch.</p>
            </div>
            <button
              onClick={() => setShowCurl(v => !v)}
              className={cn(
                "shrink-0 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                showCurl
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:text-foreground"
              )}>
              {showCurl ? "← Form" : "Optimal curl"}
            </button>
          </div>

          {/* Preset selector — always visible, never overflows */}
          {!showCurl && (
            <div className="mt-3 flex flex-col gap-1">
              <select
                value={preset}
                onChange={e => switchPreset(e.target.value as PhasePreset)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary">
                {PRESET_OPTIONS.map(o => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">{selectedOption.desc}</p>
            </div>
          )}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">

          {showCurl ? (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Best-guess optimal configuration based on diagnostics: h=5 only (IC IR 1.18), beta-neutral,
                confidence-weighted meta-labelling, ranks_only, higher min_trade_threshold (0.02) to cut the
                240×/yr turnover. Queue this, then launch with your password.
              </p>
              <div className="relative">
                <pre className="overflow-x-auto rounded-lg border border-border bg-muted p-4 text-xs font-mono leading-relaxed whitespace-pre">
                  {OPTIMAL_CURL}
                </pre>
                <button
                  onClick={() => { navigator.clipboard.writeText(OPTIMAL_CURL); setCurlCopied(true); setTimeout(() => setCurlCopied(false), 2000); }}
                  className="absolute right-2 top-2 rounded-md border border-border bg-card px-2 py-1 text-xs text-muted-foreground hover:text-foreground">
                  {curlCopied ? "Copied!" : "Copy"}
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* ── Universe (all phases) ── */}
              <Section title="Universe & dates" defaultOpen>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Start date">
                    <input type="date" value={cfg.start_date as string} onChange={e => set("start_date", e.target.value)} className={inp} />
                  </Field>
                  <Field label="Tickers (null = all ~500)">
                    <input type="number" min={1} value={(cfg.n_tickers as number | null) ?? ""}
                      placeholder="null = all S&P 500"
                      onChange={e => set("n_tickers", e.target.value === "" ? null : Number(e.target.value))}
                      className={inp} />
                  </Field>
                  <Field label="Universe sampling">
                    <select value={cfg.universe_sampling as string} onChange={e => set("universe_sampling", e.target.value)} className={sel}>
                      <option value="current">current (today's members — survivorship bias)</option>
                      <option value="random">random (unbiased)</option>
                      <option value="first">first (alphabetical)</option>
                    </select>
                  </Field>
                  <Field label="Horizons (days, comma-sep)">
                    <input type="text"
                      value={((cfg.horizons as number[]) ?? [5]).join(", ")}
                      onChange={e => {
                        const vals = e.target.value.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                        set("horizons", vals);
                      }} className={inp} />
                  </Field>
                </div>
              </Section>

              {isPhase1 ? (
                // ── Phase 1 specific ──
                <Section title="Portfolio (Phase 1)" defaultOpen>
                  <div className="grid grid-cols-2 gap-3">
                    <Field label="K per side (# positions)">
                      <input type="number" min={1} value={(cfg.k_per_side as number) ?? 20}
                        onChange={e => set("k_per_side", +e.target.value)} className={inp} />
                    </Field>
                    <Field label="Model">
                      <select value={cfg.model as string} onChange={e => set("model", e.target.value)} className={sel}>
                        <option value="gbm">GBM</option>
                        <option value="logistic">Logistic</option>
                      </select>
                    </Field>
                  </div>
                  <Checkbox label="Use sector features" checked={!!(cfg.use_sector_features)} onChange={v => set("use_sector_features", v)} />
                </Section>
              ) : (
                <>
                  {/* ── Phase 5 features ── */}
                  <Section title="Features" defaultOpen>
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <Checkbox label="Sector features" checked={!!(cfg.use_sector_features)} onChange={v => set("use_sector_features", v)} note="GICS dummies + sector-neutralised returns" />
                      <Checkbox label="Tier-2 features" checked={!!(cfg.use_tier2_features)} onChange={v => set("use_tier2_features", v)} note="12-1 momentum, IVOL, beta, max-ret, Amihud" />
                      <Checkbox label="Regime features" checked={!!(cfg.use_regime_features)} onChange={v => set("use_regime_features", v)} note="VIX, term spread, USD, xs-dispersion" />
                      <Checkbox label="Ranks only" checked={!!(cfg.ranks_only)} onChange={v => set("ranks_only", v)} note="Drop raw values; keep cross-sectional ranks (Phase 8)" />
                    </div>
                  </Section>

                  {/* ── Portfolio ── */}
                  <Section title="Portfolio construction" defaultOpen>
                    <div className="grid grid-cols-2 gap-3">
                      <Field label="Position sizing">
                        <select value={cfg.position_sizing as string} onChange={e => set("position_sizing", e.target.value)} className={sel}>
                          <option value="vol_scaled">vol_scaled (signal × 1/vol)</option>
                          <option value="top_k">top_k (equal weight)</option>
                          <option value="hrp">hrp (hierarchical risk parity)</option>
                        </select>
                      </Field>
                      <Field label="Ensemble weighting">
                        <select value={cfg.ensemble_weighting as string} onChange={e => set("ensemble_weighting", e.target.value)} className={sel}>
                          <option value="ic_ir">ic_ir (weight by OOS IC IR)</option>
                          <option value="equal">equal</option>
                        </select>
                      </Field>
                      <Field label="K per side %">
                        <input type="number" step={0.01} min={0.01} max={1} value={cfg.k_per_side_pct as number}
                          onChange={e => set("k_per_side_pct", parseFloat(e.target.value))} className={inp} />
                      </Field>
                      <Field label="Min trade threshold">
                        <input type="number" step={0.005} min={0} value={cfg.min_trade_threshold as number}
                          onChange={e => set("min_trade_threshold", parseFloat(e.target.value))} className={inp} />
                        <p className="mt-0.5 text-xs text-amber-600 dark:text-amber-400">↑ Raise to 0.02+ to cut turnover (currently 240×/yr)</p>
                      </Field>
                      <Field label="Sector cap (gross)">
                        <input type="number" step={0.05} min={0} max={1} value={(cfg.sector_cap_gross as number) ?? 0.25}
                          onChange={e => set("sector_cap_gross", parseFloat(e.target.value))} className={inp} />
                      </Field>
                      <Field label="Leverage per side">
                        <input type="number" step={0.1} min={0.1} value={cfg.leverage_per_side as number}
                          onChange={e => set("leverage_per_side", parseFloat(e.target.value))} className={inp} />
                      </Field>
                      <Field label="Holdout years">
                        <input type="number" min={0} max={5} value={cfg.holdout_years as number}
                          onChange={e => set("holdout_years", parseInt(e.target.value))} className={inp} />
                      </Field>
                    </div>
                    <Checkbox label="Beta-neutralise vs SPY" checked={!!(cfg.beta_neutralise)} onChange={v => set("beta_neutralise", v)}
                      note="Recommended: eliminates market-beta drag on short book in bull markets" />
                  </Section>

                  {/* ── Meta-labelling (Phase 8/9) ── */}
                  <Section title="Meta-labelling — Phase 8/9">
                    <Checkbox label="Enable meta-labelling" checked={!!(cfg.use_meta_labelling)} onChange={v => set("use_meta_labelling", v)}
                      note="Secondary classifier gates signals by P(primary score is correct)" />
                    {!!(cfg.use_meta_labelling) && (
                      <div className="mt-3 grid grid-cols-2 gap-3">
                        <Field label="Mode">
                          <select value={cfg.meta_mode as string} onChange={e => set("meta_mode", e.target.value)} className={sel}>
                            <option value="binary">binary (hard gate)</option>
                            <option value="confidence">confidence (scale by P)</option>
                          </select>
                        </Field>
                        <Field label="Threshold">
                          <input type="number" step={0.01} min={0} max={1} value={cfg.meta_threshold as number}
                            onChange={e => set("meta_threshold", parseFloat(e.target.value))} className={inp} />
                        </Field>
                        {cfg.meta_mode === "confidence" && (
                          <>
                            <Field label="Confidence floor">
                              <input type="number" step={0.05} min={0} max={1} value={cfg.meta_conf_floor as number}
                                onChange={e => set("meta_conf_floor", parseFloat(e.target.value))} className={inp} />
                            </Field>
                            <Field label="Confidence cap">
                              <input type="number" step={0.05} min={0} max={1} value={cfg.meta_conf_cap as number}
                                onChange={e => set("meta_conf_cap", parseFloat(e.target.value))} className={inp} />
                            </Field>
                          </>
                        )}
                        <Field label="Walk-forward folds">
                          <input type="number" min={1} max={10} value={cfg.meta_walk_forward_folds as number}
                            onChange={e => set("meta_walk_forward_folds", parseInt(e.target.value))} className={inp} />
                        </Field>
                        <div className="flex items-end">
                          <Checkbox label="Per-sector classifiers" checked={!!(cfg.meta_per_sector)} onChange={v => set("meta_per_sector", v)}
                            note="One meta GBM per GICS sector" />
                        </div>
                      </div>
                    )}
                  </Section>

                  {/* ── Labels (Phase 8) ── */}
                  <Section title="Label construction — Phase 8">
                    <Checkbox label="Triple-barrier labels" checked={!!(cfg.use_triple_barrier_labels)} onChange={v => set("use_triple_barrier_labels", v)}
                      note="Replace forward returns with triple-barrier signed returns (López de Prado Ch. 3)" />
                    {!!(cfg.use_triple_barrier_labels) && (
                      <Field label="Barrier width (σ)" className="mt-3">
                        <input type="number" step={0.5} min={0.5} value={cfg.tb_k_sigma as number}
                          onChange={e => set("tb_k_sigma", parseFloat(e.target.value))} className={inp} />
                      </Field>
                    )}
                  </Section>

                  {/* ── EDGAR (Phase 12/13) ── */}
                  <Section title="SEC EDGAR features — Phase 12/13">
                    <div className="space-y-2">
                      <Checkbox label="8-K event counts (Phase 12)" checked={!!(cfg.use_edgar_features)} onChange={v => set("use_edgar_features", v)}
                        note="has_8k flag + 5/21/63d rolling counts — free, ~10 req/s SEC rate limit" />
                      <Checkbox label="8-K per-item features (Phase 13)" checked={!!(cfg.use_edgar_item_features)} onChange={v => set("use_edgar_item_features", v)}
                        note="CEO change, earnings release, M&A, etc. — requires Phase 12 to be useful" />
                    </div>
                  </Section>
                </>
              )}

              {/* ── GBM / CV (all phases) ── */}
              <Section title="GBM & cross-validation">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">GBM</p>
                <div className="grid grid-cols-3 gap-3">
                  {(["num_leaves","n_estimators","min_data_in_leaf"] as const).map(k => (
                    <Field key={k} label={k}>
                      <input type="number" value={(cfg.gbm as any)?.[k]} onChange={e => setGbm(k, parseInt(e.target.value))} className={inp} />
                    </Field>
                  ))}
                  {(["learning_rate","feature_fraction","reg_lambda"] as const).map(k => (
                    <Field key={k} label={k}>
                      <input type="number" step={0.01} value={(cfg.gbm as any)?.[k]} onChange={e => setGbm(k, parseFloat(e.target.value))} className={inp} />
                    </Field>
                  ))}
                </div>
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Cross-validation</p>
                <div className="grid grid-cols-2 gap-3">
                  {(["train_years","test_months","embargo_days","min_train_obs"] as const).map(k => (
                    <Field key={k} label={k}>
                      <input type="number" value={(cfg.cv as any)?.[k]} onChange={e => setCv(k, parseInt(e.target.value))} className={inp} />
                    </Field>
                  ))}
                </div>
              </Section>

              {error && <p className="text-sm text-red-500">{error}</p>}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 border-t border-border px-6 py-4">
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          {!showCurl && (
            <Button size="sm" onClick={() => { setError(null); mutate.mutate(); }} disabled={mutate.isPending}>
              {mutate.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              Queue Job
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Job detail panel ─────────────────────────────────────────────────────────

function JobDetailPanel({ jobId, onClose, onCancel }: { jobId: string; onClose: () => void; onCancel: (id: string) => void }) {
  const logsEndRef = useRef<HTMLDivElement>(null);
  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.jobDetail(jobId),
    refetchInterval: q => {
      const s = q.state.data?.status;
      return s === "running" || s === "queued" || s === "cancelling" ? 2500 : false;
    },
  });

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [job?.logs?.length]);

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
          {job.elapsed_s != null && <span className="text-xs text-muted-foreground">Runtime: {fmtDuration(job.elapsed_s)}</span>}
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
        <div className="col-span-2 lg:col-span-1">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Parameters</p>
          <dl className="space-y-1 text-xs">
            {Object.entries(job.config)
              .filter(([k]) => !["cv","gbm","feature_exclude"].includes(k))
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
        <div className="col-span-2">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Logs {job.logs.length > 0 && <span className="text-muted-foreground/60">({job.logs.length} lines)</span>}
          </p>
          <pre className="h-64 overflow-y-auto rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground/80 scrollbar-thin">
            {job.logs.length === 0 ? (isActive ? "Waiting for output…" : "No logs captured.") : job.logs.join("\n")}
            <div ref={logsEndRef} />
          </pre>
        </div>
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function Jobs() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  // Deep-link support: /jobs?job=<id> auto-opens that job's detail panel.
  // We mirror the URL into selectedJobId so existing click-to-toggle code
  // keeps working without rewriting it.
  const deepLinkedJob = params.get("job");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(deepLinkedJob);
  useEffect(() => {
    if (deepLinkedJob && deepLinkedJob !== selectedJobId) {
      setSelectedJobId(deepLinkedJob);
    }
    // intentionally only react to URL changes, not local state toggles
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLinkedJob]);

  const [showNewJobForm, setShowNewJobForm] = useState(false);
  const [pwAction, setPwAction] = useState<
    | { type: "cancel"; jobId: string }
    | { type: "launch"; queueId: string }
    | { type: "delete"; queueId: string }
    | null
  >(null);
  const [pwError, setPwError] = useState<string | null>(null);
  const [pwLoading, setPwLoading] = useState(false);

  // Restart UX: clone a terminal job's config into the queue, then immediately
  // open the password modal targeted at the new queue entry so the user can
  // launch with a single confirmation. We keep restart state simple — it just
  // chains queue → setPwAction({launch, queueId}) — so it shares the existing
  // launch/cancel modal plumbing.
  const [restartError, setRestartError] = useState<string | null>(null);
  const [restartingJobId, setRestartingJobId] = useState<string | null>(null);
  const restartMut = useMutation({
    mutationFn: (job: JobDetail) => api.queueJob(job.config),
    onMutate: (job) => setRestartingJobId(job.job_id),
    onSuccess: (qj) => {
      qc.invalidateQueries({ queryKey: ["queued"] });
      // Open the launch modal targeting the just-created queue entry so the
      // user can confirm with their password in one step.
      setPwError(null);
      setPwAction({ type: "launch", queueId: qj.id });
    },
    onError: (e: Error) => setRestartError(e.message),
    onSettled: () => setRestartingJobId(null),
  });

  const hasActive = (jobs: JobDetail[]) => jobs.some(j => ["running","queued","cancelling"].includes(j.status));

  const { data: jobs = [] } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(),
    refetchInterval: q => hasActive(q.state.data ?? []) ? 3000 : 15000,
  });

  const { data: queued = [] } = useQuery({
    queryKey: ["queued"],
    queryFn: () => api.listQueued(),
    refetchInterval: 10000,
  });

  async function handlePwConfirm(pw: string) {
    if (!pwAction) return;
    setPwLoading(true); setPwError(null);
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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Jobs</h1>
          <p className="text-sm text-muted-foreground">Queue pipeline runs. Password (STOCKPRED_PW) required to launch or cancel.</p>
        </div>
        <Button onClick={() => setShowNewJobForm(true)}><Plus className="h-4 w-4" /> New Job</Button>
      </div>

      {restartError && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          <strong>Restart failed:</strong> {restartError}{" "}
          <button onClick={() => setRestartError(null)} className="ml-2 underline">dismiss</button>
        </div>
      )}

      {/* Active & History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            Active &amp; History
            <button onClick={() => qc.invalidateQueries({ queryKey: ["jobs"] })} className="text-muted-foreground hover:text-foreground">
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
                    <th className="px-4 py-2">Run</th>
                    <th className="px-4 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map(j => {
                    const isSelected = selectedJobId === j.job_id;
                    const isActive = ["running","queued","cancelling"].includes(j.status);
                    const canRestart = RESTARTABLE_STATUSES.has(j.status) && j.config && Object.keys(j.config).length > 0;
                    return (
                      <tr key={j.job_id}
                        onClick={() => {
                          const next = isSelected ? null : j.job_id;
                          setSelectedJobId(next);
                          // Keep URL in sync so the panel state is shareable.
                          setParams((prev) => {
                            const p = new URLSearchParams(prev);
                            if (next) p.set("job", next); else p.delete("job");
                            return p;
                          }, { replace: true });
                        }}
                        className={cn("cursor-pointer border-b border-border/50 transition-colors hover:bg-accent/50", isSelected && "bg-accent")}>
                        <td className="px-4 py-2">
                          <JobIdCell jobId={j.job_id} onClick={() => setSelectedJobId(isSelected ? null : j.job_id)} />
                        </td>
                        <td className="px-4 py-2">{statusBadge(j.status)}</td>
                        <td className="px-4 py-2">{detectPhase(j.config)}</td>
                        <td className="px-4 py-2">{String(j.config.n_tickers ?? "all")}</td>
                        <td className="px-4 py-2 text-xs">{isActive ? fmtElapsed(j.started_at) + " ago" : fmtTs(j.started_at)}</td>
                        <td className="px-4 py-2 text-xs">{fmtDuration(j.elapsed_s)}</td>
                        <td className="px-4 py-2 text-xs">
                          {j.run_id != null ? (
                            <Link
                              to={`/runs?expanded=${j.run_id}`}
                              onClick={(e) => e.stopPropagation()}
                              className="inline-flex items-center gap-1 text-primary hover:underline"
                              title="Open this run in the Runs history page"
                            >
                              <History className="h-3 w-3" /> #{j.run_id}
                            </Link>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>
                        <td className="px-4 py-2">
                          <div className="flex items-center justify-end gap-1">
                            {canRestart && (
                              <Button
                                size="sm"
                                variant="ghost"
                                disabled={restartingJobId === j.job_id || restartMut.isPending}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setRestartError(null);
                                  restartMut.mutate(j);
                                }}
                                title="Re-queue this job with the same parameters"
                              >
                                {restartingJobId === j.job_id ? (
                                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                ) : (
                                  <RotateCcw className="h-3.5 w-3.5" />
                                )}
                                Restart
                              </Button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {selectedJobId && (
            <div className="px-4 pb-4">
              <JobDetailPanel
                jobId={selectedJobId}
                onClose={() => {
                  setSelectedJobId(null);
                  setParams((prev) => {
                    const p = new URLSearchParams(prev);
                    p.delete("job");
                    return p;
                  }, { replace: true });
                }}
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
              Pending (awaiting launch)
              {queued.filter(j => j.status === "pending").length > 0 && (
                <span className="ml-2 rounded-full bg-yellow-100 px-2 py-0.5 text-xs text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300">
                  {queued.filter(j => j.status === "pending").length} / 5
                </span>
              )}
            </span>
            <button onClick={() => qc.invalidateQueries({ queryKey: ["queued"] })} className="text-muted-foreground hover:text-foreground">
              <RefreshCw className="h-4 w-4" />
            </button>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {queued.length === 0 ? (
            <p className="px-4 py-6 text-center text-sm text-muted-foreground">No queued jobs. Click <strong>New Job</strong> to create one.</p>
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
                      <td className="px-4 py-2 text-xs text-muted-foreground">
                        Phase {detectPhase(q.config)} · {String(q.config.n_tickers ?? "all")} tickers
                        {(q.config.horizons as number[] | null)?.length ? ` · h=${(q.config.horizons as number[]).join(",")}` : ""}
                      </td>
                      <td className="px-4 py-2 text-xs">{fmtTs(q.created_at)}</td>
                      <td className="px-4 py-2">
                        {q.status === "pending" && (
                          <div className="flex gap-1">
                            <Button variant="default" size="sm" onClick={() => { setPwError(null); setPwAction({ type: "launch", queueId: q.id }); }}>
                              <Play className="h-3 w-3" /> Launch
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => { setPwError(null); setPwAction({ type: "delete", queueId: q.id }); }}>
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {showNewJobForm && <NewJobForm onClose={() => setShowNewJobForm(false)} onQueued={() => setShowNewJobForm(false)} />}

      {pwAction && (
        <PasswordModal
          title={pwAction.type === "cancel" ? "Cancel Job" : pwAction.type === "launch" ? "Launch Job" : "Delete Queued Job"}
          description={
            pwAction.type === "cancel"
              ? "The job will be interrupted — it may take a few seconds to stop if LightGBM is mid-training."
              : pwAction.type === "launch" ? "Enter STOCKPRED_PW to launch this queued job."
              : "Enter STOCKPRED_PW to permanently delete this queued job."
          }
          confirmLabel={pwAction.type === "cancel" ? "Cancel Job" : pwAction.type === "launch" ? "Launch" : "Delete"}
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
