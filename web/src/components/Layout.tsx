import { ReactNode } from "react";
import { Link, NavLink } from "react-router-dom";
import {
  Moon,
  Sun,
  LineChart,
  Search,
  BarChart3,
  Home,
  Briefcase,
  BookOpen,
  History,
  Zap,
} from "lucide-react";
import { useTheme } from "./ThemeProvider";
import { Button } from "./ui/Button";
import { RunPicker } from "./RunPicker";
import { cn } from "@/lib/cn";

export function Layout({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme();

  const nav = [
    { to: "/", label: "Home", icon: Home },
    { to: "/screener", label: "Screener", icon: Search },
    { to: "/backtest", label: "Backtest", icon: BarChart3 },
    { to: "/runs", label: "Runs", icon: History },
    { to: "/jobs", label: "Jobs", icon: Briefcase },
    { to: "/hypersearch", label: "Hypersearch", icon: Zap },
    { to: "/about", label: "About", icon: BookOpen },
  ];

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <LineChart className="h-5 w-5 text-primary" />
            <span>stock-predictor</span>
            <span className="badge ml-2">v0.2</span>
          </Link>
          <nav className="hidden gap-1 md:flex">
            {nav.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  cn(
                    "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <RunPicker />
            <Button variant="ghost" size="icon" onClick={toggle} title="Toggle theme">
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </div>
        {/* Mobile nav */}
        <nav className="flex gap-1 border-t border-border px-2 py-1 md:hidden">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "inline-flex flex-1 items-center justify-center gap-1 rounded-md px-2 py-2 text-xs font-medium",
                  isActive ? "bg-accent text-accent-foreground" : "text-muted-foreground",
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-7xl p-4 sm:p-6">{children}</div>
      </main>

      <footer className="border-t border-border bg-card">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 text-xs text-muted-foreground">
          <span>Not investment advice. Backtest only.</span>
          <span>Free data: yfinance · FRED · Wikipedia · FINRA</span>
        </div>
      </footer>
    </div>
  );
}
