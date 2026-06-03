import { ReactNode } from "react";
import { cn } from "@/lib/cn";

interface CardProps {
  className?: string;
  children: ReactNode;
}

export function Card({ className, children }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card text-card-foreground shadow-sm",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({ className, children }: CardProps) {
  return <div className={cn("px-5 pt-5 pb-2", className)}>{children}</div>;
}

export function CardTitle({ className, children }: CardProps) {
  return (
    <h3 className={cn("text-base font-semibold leading-none tracking-tight", className)}>
      {children}
    </h3>
  );
}

export function CardSubtitle({ className, children }: CardProps) {
  return <p className={cn("mt-1 text-sm text-muted-foreground", className)}>{children}</p>;
}

export function CardContent({ className, children }: CardProps) {
  return <div className={cn("p-5", className)}>{children}</div>;
}
