"""Minimal HTML tearsheet: equity curve, drawdown, yearly stats, key metrics.

Intentionally dependency-light (matplotlib + pure-html template). Heavier
libraries (e.g. quantstats) are nice but add fragility; we keep it portable.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from stockpred.validation.metrics import (  # noqa: E402
    max_drawdown,
    tearsheet_metrics,
)


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_chart(returns: pd.Series, benchmark: pd.Series | None) -> str:
    # L6 fix: plot only on dates with real returns, so NaN stretches don't
    # appear as flat (== "no drawdown / no movement") in the chart.
    r = returns.dropna()
    cum = (1 + r).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum.index, cum.values, label="strategy", linewidth=1.5)
    if benchmark is not None:
        bench = benchmark.reindex(cum.index).dropna()
        if not bench.empty:
            bcum = (1 + bench).cumprod()
            ax.plot(bcum.index, bcum.values, label="benchmark", linewidth=1.0, alpha=0.7)
    ax.set_title("Equity curve")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return _fig_to_b64(fig)


def _drawdown_chart(returns: pd.Series) -> str:
    r = returns.dropna()
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("DD")
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _yearly_table(returns: pd.Series) -> str:
    # M6 fix: per-column formatting, no shared `float_format` rule that
    # conflates Sharpe (~1.2) and returns (%).
    yearly = returns.groupby(returns.index.year).agg(
        ret=lambda r: (1 + r).prod() - 1,
        sharpe=lambda r: (r.mean() / r.std() * (252**0.5)) if r.std() else float("nan"),
        max_dd=max_drawdown,
        n=len,
    )
    formatters = {
        "ret": lambda x: f"{x:.2%}" if pd.notna(x) else "—",
        "sharpe": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
        "max_dd": lambda x: f"{x:.2%}" if pd.notna(x) else "—",
        "n": lambda x: f"{int(x)}" if pd.notna(x) else "—",
    }
    return yearly.to_html(classes="tbl", formatters=formatters)


HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8"><title>stock-predictor tearsheet</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-width: 1080px; margin: 24px auto; padding: 0 16px; color: #222; }}
h1, h2 {{ font-weight: 600; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
.metric {{ border: 1px solid #ddd; padding: 10px 12px; border-radius: 6px; }}
.metric .lbl {{ color: #888; font-size: 12px; text-transform: uppercase; }}
.metric .val {{ font-size: 20px; font-weight: 600; }}
.tbl {{ border-collapse: collapse; margin-top: 8px; }}
.tbl th, .tbl td {{ border: 1px solid #ddd; padding: 4px 8px; font-size: 13px; }}
.tbl th {{ background: #f4f4f4; text-align: left; }}
.muted {{ color: #777; font-size: 12px; }}
img {{ max-width: 100%; }}
</style></head><body>
<h1>stock-predictor tearsheet</h1>
<p class="muted">Generated {now}. <strong>Not investment advice.</strong></p>

<h2>Key metrics</h2>
<div class="metrics">
{metric_cards}
</div>

<h2>Equity curve</h2>
<img src="data:image/png;base64,{equity_png}"/>

<h2>Drawdown</h2>
<img src="data:image/png;base64,{drawdown_png}"/>

<h2>Yearly performance</h2>
{yearly_html}

<p class="muted">Costs assumed: {cost_bps:.1f} bps per side (commission + spread + slippage).</p>
</body></html>
"""


def _format_metric(label: str, value: float) -> str:
    if pd.isna(value):
        s = "—"
    elif "drawdown" in label or "return" in label or "vol" in label or label == "hit_ratio":
        s = f"{value:.2%}"
    else:
        s = f"{value:.2f}"
    return f'<div class="metric"><div class="lbl">{label}</div><div class="val">{s}</div></div>'


def build_tearsheet(
    returns: pd.Series,
    output_path: str | Path,
    *,
    benchmark: pd.Series | None = None,
    cost_bps_per_side: float = 6.0,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = tearsheet_metrics(returns)
    cards = "\n".join(_format_metric(k, v) for k, v in metrics.items())
    html = HTML_TEMPLATE.format(
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        metric_cards=cards,
        equity_png=_equity_chart(returns, benchmark),
        drawdown_png=_drawdown_chart(returns),
        yearly_html=_yearly_table(returns),
        cost_bps=cost_bps_per_side,
    )
    out.write_text(html)
    return out
