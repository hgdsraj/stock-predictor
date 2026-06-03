"""Project-wide config: paths, constants, defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Package root: src/stockpred/
PKG_ROOT = Path(__file__).resolve().parent
# Project root: repo top-level (contains pyproject.toml)
PROJECT_ROOT = PKG_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (DATA_DIR, CACHE_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BacktestConfig:
    """Realistic cost assumptions. Per-side, in basis points unless noted."""

    commission_bps: float = 1.0  # broker commission per side
    spread_bps: float = 4.0  # half-spread per side (conservative for liquid US equities)
    slippage_bps: float = 1.0  # market impact per side
    annualization_factor: int = 252

    @property
    def total_cost_per_side_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps


@dataclass(frozen=True)
class HorizonConfig:
    """Forecast horizons in trading days."""

    horizons: tuple[int, ...] = (1, 5, 21)


@dataclass(frozen=True)
class CVConfig:
    """Walk-forward CV settings."""

    train_years: int = 5
    test_months: int = 6
    embargo_days: int = 10  # > max horizon to prevent label leakage
    min_train_obs: int = 1000


@dataclass(frozen=True)
class UniverseConfig:
    """Universe + history settings."""

    start_date: str = "2005-01-01"
    end_date: str | None = None  # None = today
    sp500_changes_url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class ProjectConfig:
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    horizons: HorizonConfig = field(default_factory=HorizonConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)


DEFAULT = ProjectConfig()
