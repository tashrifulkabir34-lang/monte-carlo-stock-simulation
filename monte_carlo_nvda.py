"""
Monte Carlo Stock Price Simulation — NVDA (Nvidia Corp)
========================================================
Simulates future stock price paths using Geometric Brownian Motion (GBM).
Inspired by the TikTok-style educational Monte Carlo explainer format.

Author  : Your Name
Stock   : NVDA — Nvidia Corp (S&P 500 #1 weighted component, ~7.17%)
Model   : Geometric Brownian Motion
Trials  : 10,000 (default)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import argparse

# ── Default parameters ───────────────────────────────────────────────────────
CURRENT_PRICE   = 174.20   # NVDA closing price — March 24, 2026
DAILY_VOLATILITY = 0.0398  # σ = 3.98%  (sourced from TradingView, 30-day HV)
ANNUAL_DRIFT     = 0.12    # μ = 12%  (conservative historical S&P 500 average)
TRADING_DAYS     = 252     # trading days per year
NUM_TRIALS       = 10_000  # Monte Carlo sample size
TIME_HORIZON     = 90      # days to simulate forward


def simulate(
    s0=CURRENT_PRICE,
    mu=ANNUAL_DRIFT,
    sigma=DAILY_VOLATILITY,
    days=TIME_HORIZON,
    trials=NUM_TRIALS,
    seed=None,
):
    """
    Run GBM Monte Carlo simulation.

    Parameters
    ----------
    s0     : float  — starting stock price
    mu     : float  — annualised drift (e.g. 0.12 = 12%)
    sigma  : float  — daily volatility (e.g. 0.0398 = 3.98%)
    days   : int    — number of trading days to simulate
    trials : int    — number of Monte Carlo paths
    seed   : int    — optional random seed for reproducibility

    Returns
    -------
    final_prices : np.ndarray  — array of final simulated prices (length = trials)
    paths        : np.ndarray  — full price paths, shape (trials, days+1)
    """
    if seed is not None:
        np.random.seed(seed)

    mu_daily = mu / TRADING_DAYS
    dt = 1  # one trading day per step

    # Standard normal random shocks: shape (trials, days)
    Z = np.random.standard_normal((trials, days))

    # GBM daily log-return: (μ - σ²/2)dt + σ√dt · Z
    log_returns = (mu_daily - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative product to get price paths
    paths = np.zeros((trials, days + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))

    final_prices = paths[:, -1]
    return final_prices, paths


def plot_histogram(
    final_prices,
    s0=CURRENT_PRICE,
    days=TIME_HORIZON,
    trials=NUM_TRIALS,
    save_path="monte_carlo_nvda.png",
):
    """Render the histogram — styled to match the TikTok video aesthetic."""

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # ── Histogram buckets ─────────────────────────────────────────────────────
    n_bins = 60
    counts, bin_edges = np.histogram(final_prices, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Colour: blue = above break-even, red = below
    colors = ["#378ADD" if c >= s0 else "#E24B4A" for c in bin_centers]
    ax.bar(
        bin_centers,
        counts,
        width=(bin_edges[1] - bin_edges[0]) * 0.92,
        color=colors,
        edgecolor="none",
        zorder=2,
    )

    # ── Break-even / current price line ──────────────────────────────────────
    ax.axvline(
        s0,
        color="#FAC775",
        linewidth=2,
        linestyle="--",
        zorder=3,
        label=f"Current price  ${s0:.2f}",
    )
    ax.text(
        s0 + (final_prices.max() - final_prices.min()) * 0.012,
        counts.max() * 0.96,
        f"Break-even\n${s0:.2f}",
        color="#FAC775",
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    # ── Median line ───────────────────────────────────────────────────────────
    median = np.median(final_prices)
    ax.axvline(median, color="#5DCAA5", linewidth=1.5, linestyle=":", zorder=3)
    ax.text(
        median + (final_prices.max() - final_prices.min()) * 0.012,
        counts.max() * 0.78,
        f"Median\n${median:.2f}",
        color="#5DCAA5",
        fontsize=9,
        va="top",
    )

    # ── Labels & title ────────────────────────────────────────────────────────
    ax.set_title(
        "Monte Carlo Simulation — NVDA Stock Price",
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(f"Final price after {days} trading days (USD)", color="#888780", fontsize=11)
    ax.set_ylabel("Number of simulations", color="#888780", fontsize=11)

    ax.tick_params(colors="#888780", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C2C2A")

    ax.yaxis.grid(True, color="#2C2C2A", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ── Stats box ─────────────────────────────────────────────────────────────
    prob_profit = (final_prices > s0).mean() * 100
    mean_return = (final_prices.mean() - s0) / s0 * 100
    p5  = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)

    stats_text = (
        f"Trials          {trials:,}\n"
        f"Time horizon  {days} days\n"
        f"Daily vol (σ)  {DAILY_VOLATILITY*100:.2f}%\n"
        f"Annual drift  {ANNUAL_DRIFT*100:.0f}%\n"
        f"─────────────────\n"
        f"Median price  ${median:.2f}\n"
        f"Mean return   {mean_return:+.1f}%\n"
        f"P(profit)       {prob_profit:.1f}%\n"
        f"5th pct          ${p5:.2f}\n"
        f"95th pct        ${p95:.2f}"
    )
    ax.text(
        0.013, 0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        fontfamily="monospace",
        color="#B4B2A9",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#161b22", edgecolor="#2C2C2A", alpha=0.9),
        zorder=5,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#378ADD", label="Above break-even (profit)"),
        mpatches.Patch(color="#E24B4A", label="Below break-even (loss)"),
        Line2D([0], [0], color="#FAC775", lw=2, linestyle="--", label=f"Break-even ${s0:.2f}"),
        Line2D([0], [0], color="#5DCAA5", lw=1.5, linestyle=":", label=f"Median ${median:.2f}"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.85,
        facecolor="#161b22",
        edgecolor="#2C2C2A",
        labelcolor="#B4B2A9",
        fontsize=8.5,
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Geometric Brownian Motion  •  Not financial advice  •  Data: March 24, 2026",
        ha="center",
        color="#5F5E5A",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved → {save_path}")
    plt.show()


def print_summary(final_prices, s0=CURRENT_PRICE, days=TIME_HORIZON):
    """Print a clean summary table to the console."""
    prob_profit  = (final_prices > s0).mean() * 100
    mean_ret     = (final_prices.mean() - s0) / s0 * 100
    median       = np.median(final_prices)
    p5, p25, p75, p95 = np.percentile(final_prices, [5, 25, 75, 95])

    print("\n" + "═" * 48)
    print("  NVDA Monte Carlo Simulation — Summary")
    print("═" * 48)
    print(f"  Starting price      : ${s0:.2f}")
    print(f"  Time horizon        : {days} trading days")
    print(f"  Trials              : {len(final_prices):,}")
    print(f"  Daily volatility σ  : {DAILY_VOLATILITY*100:.2f}%")
    print(f"  Annual drift μ      : {ANNUAL_DRIFT*100:.0f}%")
    print("─" * 48)
    print(f"  Median final price  : ${median:.2f}")
    print(f"  Mean return         : {mean_ret:+.2f}%")
    print(f"  P(profit)           : {prob_profit:.1f}%")
    print(f"  5th percentile      : ${p5:.2f}")
    print(f"  25th percentile     : ${p25:.2f}")
    print(f"  75th percentile     : ${p75:.2f}")
    print(f"  95th percentile     : ${p95:.2f}")
    print("═" * 48 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVDA Monte Carlo Stock Simulation")
    parser.add_argument("--price",   type=float, default=CURRENT_PRICE,    help="Starting stock price")
    parser.add_argument("--days",    type=int,   default=TIME_HORIZON,     help="Trading days to simulate")
    parser.add_argument("--trials",  type=int,   default=NUM_TRIALS,       help="Number of Monte Carlo trials")
    parser.add_argument("--sigma",   type=float, default=DAILY_VOLATILITY, help="Daily volatility (e.g. 0.0398)")
    parser.add_argument("--mu",      type=float, default=ANNUAL_DRIFT,     help="Annual drift (e.g. 0.12)")
    parser.add_argument("--seed",    type=int,   default=None,             help="Random seed for reproducibility")
    parser.add_argument("--output",  type=str,   default="monte_carlo_nvda.png", help="Output chart filename")
    args = parser.parse_args()

    print(f"\nRunning {args.trials:,} simulations over {args.days} trading days...")
    final_prices, paths = simulate(
        s0=args.price,
        mu=args.mu,
        sigma=args.sigma,
        days=args.days,
        trials=args.trials,
        seed=args.seed,
    )
    print_summary(final_prices, s0=args.price, days=args.days)
    plot_histogram(final_prices, s0=args.price, days=args.days, trials=args.trials, save_path=args.output)
