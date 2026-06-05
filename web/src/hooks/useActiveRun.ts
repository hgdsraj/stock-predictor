// Hook for "which run is the page currently showing data from".
//
// Resolution order (matches the server's resolve_run):
//   1. ?run_id=<n> in the URL  → that run, per-tab override
//   2. Otherwise               → the server's active/latest run
//
// We keep state purely in the URL query string so:
//   - Links are shareable (paste a URL with ?run_id=42, the recipient sees the same data)
//   - Browser back/forward navigates between selected runs
//   - Different tabs can view different runs without fighting each other
//   - No localStorage migration headaches
//
// The setter writes either history.replaceState (default, doesn't push) or
// history.pushState if `push: true` is requested. Routes listen to popstate.

import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

const PARAM = "run_id";

export interface UseActiveRunResult {
  /** Currently-selected run id, or null when following the server default. */
  runId: number | null;
  /** Whether the URL is forcing a specific run (vs following the server default). */
  isPinned: boolean;
  /** Set the active run for this tab. Pass null to clear (follow server default). */
  setRunId: (id: number | null, opts?: { push?: boolean }) => void;
  /** Resolved string suitable for react-query keys; "active" when not pinned. */
  queryKeyPart: number | "active";
}

export function useActiveRun(): UseActiveRunResult {
  const [params, setParams] = useSearchParams();
  const raw = params.get(PARAM);
  const parsed = raw != null ? Number(raw) : NaN;
  const runId = Number.isFinite(parsed) && parsed > 0 ? parsed : null;

  const setRunId = useCallback(
    (id: number | null, opts?: { push?: boolean }) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (id == null) next.delete(PARAM);
          else next.set(PARAM, String(id));
          return next;
        },
        { replace: !opts?.push },
      );
    },
    [setParams],
  );

  return {
    runId,
    isPinned: runId != null,
    setRunId,
    queryKeyPart: runId ?? "active",
  };
}

// Small helper for components that want the active run id as a *stable* value
// outside the URL — e.g. for fire-and-forget mutations that read it once. The
// URL is the source of truth; this is just sugar that snapshots on mount.
export function useActiveRunSnapshot(): number | null {
  const { runId } = useActiveRun();
  const [snap, setSnap] = useState(runId);
  useEffect(() => setSnap(runId), [runId]);
  return snap;
}
