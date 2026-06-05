import { ReactNode, useState, useRef, useEffect } from "react";
import { Info } from "lucide-react";
import { glossary, GlossaryKey } from "@/lib/glossary";
import { cn } from "@/lib/cn";

interface InfoTooltipProps {
  /** Pull title+body from the shared glossary. */
  termKey?: GlossaryKey;
  /** Or supply ad-hoc content. */
  title?: string;
  body?: ReactNode;
  /** Optional custom trigger; defaults to a small info icon. */
  children?: ReactNode;
  className?: string;
  side?: "top" | "bottom";
  align?: "start" | "center" | "end";
}

/**
 * Accessible hover/focus/tap popover. On desktop it opens on hover or keyboard
 * focus; on touch it toggles on tap. Closes on outside click or Escape.
 */
export function InfoTooltip({
  termKey,
  title,
  body,
  children,
  className,
  side = "top",
  align = "center",
}: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);

  const entry = termKey ? glossary[termKey] : null;
  const heading = title ?? entry?.term ?? "";
  const content = body ?? entry?.body ?? entry?.short ?? "";

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span
      ref={wrapRef}
      className={cn("relative inline-flex items-center", className)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        aria-label={heading ? `What is ${heading}?` : "More info"}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="inline-flex items-center text-muted-foreground transition-colors hover:text-foreground focus:outline-none focus-visible:text-foreground"
      >
        {children ?? <Info className="h-3.5 w-3.5" />}
      </button>

      {open && (heading || content) && (
        <span
          role="tooltip"
          className={cn(
            "absolute z-50 w-72 rounded-lg border border-border bg-card p-3 text-left shadow-xl",
            side === "top" ? "bottom-full mb-2" : "top-full mt-2",
            align === "start" && "left-0",
            align === "center" && "left-1/2 -translate-x-1/2",
            align === "end" && "right-0",
          )}
        >
          {heading && <span className="mb-1 block text-xs font-semibold text-foreground">{heading}</span>}
          <span className="block text-xs leading-relaxed text-muted-foreground">{content}</span>
        </span>
      )}
    </span>
  );
}

/** A label with an attached info popover, e.g. for stat rows / table headers. */
export function LabelWithInfo({
  label,
  termKey,
  title,
  body,
  className,
  side = "top",
  align = "center",
}: {
  label: ReactNode;
  termKey?: GlossaryKey;
  title?: string;
  body?: ReactNode;
  className?: string;
  side?: "top" | "bottom";
  align?: "start" | "center" | "end";
}) {
  return (
    <span className={cn("inline-flex items-center gap-1", className)}>
      {label}
      <InfoTooltip termKey={termKey} title={title} body={body} side={side} align={align} />
    </span>
  );
}
