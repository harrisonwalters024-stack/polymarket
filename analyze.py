#!/usr/bin/env python3
"""
Polymarket CLOB Market Analyzer
Fetches active markets, filters for high-confidence outcomes (80-95%),
uses Claude to estimate true probability, and surfaces potential edges.

Usage:
    python analyze.py          # live scan (requires network access to Polymarket)
    python analyze.py --demo   # use mock market data (useful for testing)
"""

import os
import sys
import json
import time
import textwrap
import argparse
import requests
from anthropic import Anthropic

# ── Config ──────────────────────────────────────────────────────────────────
GAMMA_API   = "https://gamma-api.polymarket.com"
MODEL       = "claude-sonnet-4-20250514"
MIN_PROB    = 0.80
MAX_PROB    = 0.95
MIN_LIQUID  = 1_000    # USD notional liquidity floor
MAX_MARKETS = 6        # cap how many we send to Claude (API cost / latency)
BET_SIZE    = 10.0     # hypothetical bet in USD

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
CYAN    = "\033[36m"
WHITE   = "\033[97m"


# ── Mock data (used with --demo) ──────────────────────────────────────────────
DEMO_MARKETS = [
    {
        "question": "Will the Fed cut interest rates in June 2025?",
        "description": (
            "This market resolves YES if the Federal Reserve announces a cut to the "
            "federal funds rate target range at its June 2025 FOMC meeting. Resolution "
            "is based on the official FOMC statement."
        ),
        "outcomePrices": ["0.87", "0.13"],
        "liquidityNum": 450_000,
    },
    {
        "question": "Will Bitcoin price exceed $120,000 before July 1 2025?",
        "description": (
            "Resolves YES if BTC/USD spot price on any major exchange (Coinbase, Binance, "
            "Kraken) closes above $120,000 at any point before 00:00 UTC July 1 2025."
        ),
        "outcomePrices": ["0.82", "0.18"],
        "liquidityNum": 1_200_000,
    },
    {
        "question": "Will Elon Musk remain CEO of Tesla through end of Q2 2025?",
        "description": (
            "Resolves YES if Elon Musk holds the title of CEO of Tesla, Inc. on June 30, "
            "2025 as reported by Tesla's official communications or SEC filings."
        ),
        "outcomePrices": ["0.93", "0.07"],
        "liquidityNum": 85_000,
    },
    {
        "question": "Will the US unemployment rate stay below 4.5% through June 2025?",
        "description": (
            "Resolves YES if every BLS monthly unemployment report published through "
            "June 2025 shows the U-3 unemployment rate below 4.5%."
        ),
        "outcomePrices": ["0.91", "0.09"],
        "liquidityNum": 320_000,
    },
    {
        "question": "Will SpaceX Starship complete an orbital flight in 2025?",
        "description": (
            "Resolves YES if SpaceX's Starship vehicle achieves a trajectory that "
            "reaches orbital altitude (>100km) and completes at least one full orbit "
            "before reentry, prior to December 31 2025."
        ),
        "outcomePrices": ["0.85", "0.15"],
        "liquidityNum": 2_100_000,
    },
    {
        "question": "Will Apple release a foldable iPhone by end of 2025?",
        "description": (
            "Resolves YES if Apple officially announces and begins shipping a foldable "
            "form-factor iPhone device to consumers before December 31 2025."
        ),
        "outcomePrices": ["0.83", "0.17"],
        "liquidityNum": 175_000,
    },
]


# ── Polymarket API helpers ───────────────────────────────────────────────────

def fetch_markets(limit: int = 300) -> list[dict]:
    """Pull active markets from the Gamma metadata API."""
    headers = {"User-Agent": "polymarket-analyzer/1.0", "Accept": "application/json"}
    params  = {"active": "true", "closed": "false", "limit": limit, "offset": 0}
    resp = requests.get(f"{GAMMA_API}/markets", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return data.get("markets", data.get("data", []))


def best_price(market: dict) -> float | None:
    """Return the highest outcome price (YES/top token) in [0,1]."""
    outcomes_raw = market.get("outcomePrices") or market.get("outcome_prices")
    if not outcomes_raw:
        return None
    try:
        prices = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        floats = [float(p) for p in prices if p is not None]
        return max(floats) if floats else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def liquidity_usd(market: dict) -> float:
    for key in ("liquidityNum", "liquidity", "volume24hr", "volumeNum"):
        val = market.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def filter_markets(markets: list[dict]) -> list[dict]:
    """Keep markets whose top outcome sits in [MIN_PROB, MAX_PROB] with enough liquidity."""
    out = []
    for m in markets:
        price = best_price(m)
        if price is None:
            continue
        if not (MIN_PROB <= price <= MAX_PROB):
            continue
        if liquidity_usd(m) < MIN_LIQUID:
            continue
        m["_top_price"] = price
        out.append(m)
    out.sort(key=lambda m: m["_top_price"], reverse=True)
    return out[:MAX_MARKETS]


# ── Claude analysis ──────────────────────────────────────────────────────────

def build_prompt(market: dict) -> str:
    title    = market.get("question") or market.get("title") or "Unknown"
    criteria = market.get("description") or market.get("resolutionCriteria") or "Not provided"
    criteria = textwrap.shorten(criteria, width=800, placeholder="…")
    price    = market["_top_price"]

    return f"""You are a sharp prediction-market analyst. A binary market is currently pricing the top outcome at {price:.0%} implied probability.

Market: {title}

Resolution criteria:
{criteria}

Today's date: {time.strftime('%Y-%m-%d')}

Your task:
1. Give your own probability estimate (0-100%) for the top outcome resolving YES.
2. Explain your reasoning in 2-3 sentences — be direct and specific about the key factors.
3. Classify whether there is a meaningful edge vs the market price.

Respond ONLY with valid JSON in this exact shape (no markdown fences, no extra keys):
{{
  "claude_prob": <float between 0 and 1>,
  "reasoning": "<2-3 sentence string>",
  "edge_direction": "OVER" | "UNDER" | "FAIR"
}}

Definitions:
- "OVER"  = true probability is HIGHER than market price (market is underpriced/cheap — buy signal)
- "UNDER" = true probability is LOWER than market price (market is overpriced/expensive — avoid)
- "FAIR"  = roughly in line with market price, no meaningful edge"""


def analyze_markets(client: Anthropic, markets: list[dict]) -> list[dict]:
    for i, market in enumerate(markets, 1):
        title = market.get("question") or market.get("title") or "Unknown"
        print(f"  [{i}/{len(markets)}] {title[:72]}…", flush=True)
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": build_prompt(market)}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if Claude adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            analysis = json.loads(raw.strip())
        except Exception as exc:
            print(f"    ⚠ Claude error: {exc}", file=sys.stderr)
            analysis = {"claude_prob": None, "reasoning": str(exc), "edge_direction": "FAIR"}

        market["_analysis"] = analysis
        time.sleep(0.4)

    return markets


# ── Terminal dashboard ───────────────────────────────────────────────────────

def prob_bar(value: float, width: int = 22) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def expected_value(market_prob: float, claude_prob: float, bet: float) -> float:
    """EV of a $bet on YES at market_prob binary odds."""
    payout_if_yes = bet / market_prob
    return claude_prob * (payout_if_yes - bet) - (1 - claude_prob) * bet


def edge_color(direction: str) -> str:
    return {"OVER": GREEN, "UNDER": RED, "FAIR": DIM}.get(direction, DIM)


def render_dashboard(markets: list[dict]) -> None:
    W   = 88
    SEP = "─" * W

    print()
    print(BOLD + CYAN + "╔" + "═" * (W - 2) + "╗" + RESET)
    header = "  POLYMARKET CLOB ANALYZER  ·  Paper Trading Only — No Bets Are Placed  "
    print(BOLD + CYAN + "║" + header + " " * (W - 2 - len(header)) + "║" + RESET)
    print(BOLD + CYAN + "╚" + "═" * (W - 2) + "╝" + RESET)
    print(f"{DIM}  {time.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"Filter: {MIN_PROB:.0%}–{MAX_PROB:.0%} top outcome   "
          f"Model: {MODEL}   "
          f"Hypothetical bet: ${BET_SIZE:.0f}{RESET}")
    print()

    edge_markets = [m for m in markets if m["_analysis"].get("edge_direction") != "FAIR"]

    for idx, market in enumerate(markets, 1):
        title     = market.get("question") or market.get("title") or "Unknown"
        mkt_prob  = market["_top_price"]
        analysis  = market["_analysis"]
        c_prob    = analysis.get("claude_prob")
        reasoning = analysis.get("reasoning", "")
        direction = analysis.get("edge_direction", "FAIR")
        has_edge  = direction != "FAIR"
        color     = edge_color(direction)

        flag = f"  {BOLD}{GREEN}◆ EDGE FOUND{RESET}" if has_edge else ""
        print(BOLD + f"  #{idx}  " + WHITE + BOLD + title[:W - 8] + RESET + flag)
        print("  " + SEP)

        # Probability rows
        print(f"  Market price  {CYAN}{mkt_prob:5.1%}{RESET}  {CYAN}{prob_bar(mkt_prob)}{RESET}")
        if c_prob is not None:
            diff     = c_prob - mkt_prob
            diff_str = f"{'+' if diff >= 0 else ''}{diff:.1%}"
            print(f"  Claude est.   {color}{c_prob:5.1%}{RESET}  {color}{prob_bar(c_prob)}{RESET}")
            print(f"  Delta         {color}{BOLD}{diff_str:>6}{RESET}  ({direction})")

            ev      = expected_value(mkt_prob, c_prob, BET_SIZE)
            ev_str  = f"{'+'if ev>=0 else ''}{ev:.2f}"
            ev_col  = GREEN if ev > 0.10 else (RED if ev < -0.10 else DIM)
            liq_str = f"${liquidity_usd(market):,.0f}" if liquidity_usd(market) else "N/A"
            print(f"  ${BET_SIZE:.0f} bet EV    {ev_col}{BOLD}${ev_str}{RESET}  "
                  f"{DIM}(liquidity: {liq_str}){RESET}")
        else:
            print(f"  Claude est.   {DIM}N/A{RESET}")

        # Reasoning block
        wrapped = textwrap.fill(
            reasoning, width=W - 4, initial_indent="    ", subsequent_indent="    "
        )
        print(f"\n{DIM}  Reasoning:{RESET}")
        print(f"{DIM}{wrapped}{RESET}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(BOLD + CYAN + "  SUMMARY" + RESET)
    print("  " + SEP)
    print(f"  Markets analyzed : {len(markets)}")
    print(f"  Edges flagged    : {BOLD}{(GREEN if edge_markets else DIM)}"
          f"{len(edge_markets)}{RESET}")

    if edge_markets:
        print()
        for m in edge_markets:
            direction = m["_analysis"].get("edge_direction", "?")
            c_prob    = m["_analysis"].get("claude_prob")
            mkt_prob  = m["_top_price"]
            title     = (m.get("question") or m.get("title") or "Unknown")[:60]
            diff      = (c_prob - mkt_prob) if c_prob is not None else 0
            ev        = expected_value(mkt_prob, c_prob, BET_SIZE) if c_prob is not None else 0
            col       = GREEN if direction == "OVER" else RED
            ev_str    = f"{'+'if ev>=0 else ''}{ev:.2f}"
            print(f"  {col}{BOLD}[{direction:5s}]{RESET}  "
                  f"{col}{title}{RESET}  "
                  f"{DIM}Δ{diff:+.1%}  EV ${ev_str}{RESET}")
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket CLOB market analyzer")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use mock market data instead of live API (useful when Polymarket API is unreachable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    client = Anthropic(api_key=api_key)

    if args.demo:
        print(BOLD + YELLOW + "\n[DEMO MODE] Using mock market data." + RESET)
        markets = DEMO_MARKETS
        # Pre-compute _top_price and _liquidity for demo markets
        for m in markets:
            m["_top_price"] = best_price(m)
        markets = [m for m in markets if m["_top_price"] is not None]
    else:
        print(BOLD + CYAN + "\nFetching active Polymarket markets…" + RESET, flush=True)
        try:
            raw_markets = fetch_markets(limit=300)
        except requests.RequestException as exc:
            sys.exit(
                f"Failed to fetch markets: {exc}\n\n"
                "If you're getting a 403, Polymarket may be blocking this IP.\n"
                "Try running with --demo to verify the tool works, then re-run\n"
                "from your local machine or with a residential IP."
            )
        print(f"  Retrieved {len(raw_markets)} markets.")
        markets = filter_markets(raw_markets)
        if not markets:
            print(
                f"\nNo markets matched the filter ({MIN_PROB:.0%}–{MAX_PROB:.0%} top outcome, "
                f"liquidity ≥ ${MIN_LIQUID:,}).\n"
                "Try lowering MIN_LIQUID or widening the probability band in the config section.\n"
            )
            return
        print(f"  {len(markets)} markets passed filter.")

    print(BOLD + CYAN + f"\nRunning Claude analysis ({MODEL})…" + RESET, flush=True)
    analyzed = analyze_markets(client, markets)
    render_dashboard(analyzed)


if __name__ == "__main__":
    main()
