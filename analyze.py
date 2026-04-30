#!/usr/bin/env python3
"""
Polymarket CLOB Market Analyzer
Fetches active markets, filters for high-confidence YES outcomes (85-97%),
uses Claude + live web search to estimate true probability, surfaces edges.

Usage:
    python analyze.py          # live scan (requires Polymarket network access)
    python analyze.py --demo   # mock markets for testing
"""

import os
import sys
import json
import time
import textwrap
import argparse
import requests
from datetime import datetime, timezone, timedelta
from anthropic import Anthropic

# ── Config ────────────────────────────────────────────────────────────────────
GAMMA_API   = "https://gamma-api.polymarket.com"
MODEL       = "claude-sonnet-4-6"
MIN_PROB    = 0.85
MAX_PROB    = 0.97
MIN_LIQUID  = 1_000     # USD liquidity floor
MAX_MARKETS = 20        # top N by liquidity sent to Claude
BET_SIZE    = 10.0      # hypothetical bet (USD)

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"


# ── Mock data (--demo mode) ───────────────────────────────────────────────────
DEMO_MARKETS = [
    {
        "question":      "Will the Fed cut interest rates in June 2026?",
        "description":   (
            "This market resolves YES if the Federal Reserve announces a cut to the "
            "federal funds rate target range at its June 2026 FOMC meeting."
        ),
        "outcomes":      '["Yes", "No"]',
        "outcomePrices": '["0.87", "0.13"]',
        "liquidityNum":  450_000,
        "endDate":       "2026-06-20T00:00:00Z",
    },
    {
        "question":      "Will Bitcoin price exceed $120,000 before August 1 2026?",
        "description":   (
            "Resolves YES if BTC/USD spot price on any major exchange closes above "
            "$120,000 at any point before 00:00 UTC August 1 2026."
        ),
        "outcomes":      '["Yes", "No"]',
        "outcomePrices": '["0.82", "0.18"]',
        "liquidityNum":  1_200_000,
        "endDate":       "2026-08-01T00:00:00Z",
    },
    {
        "question":      "Will the US unemployment rate stay below 4.5% through June 2026?",
        "description":   (
            "Resolves YES if every BLS monthly unemployment report published through "
            "June 2026 shows the U-3 unemployment rate below 4.5%."
        ),
        "outcomes":      '["Yes", "No"]',
        "outcomePrices": '["0.91", "0.09"]',
        "liquidityNum":  320_000,
        "endDate":       "2026-06-30T00:00:00Z",
    },
    {
        "question":      "Will SpaceX Starship complete an orbital flight in 2026?",
        "description":   (
            "Resolves YES if SpaceX's Starship vehicle achieves orbital altitude and "
            "completes at least one full orbit before December 31 2026."
        ),
        "outcomes":      '["Yes", "No"]',
        "outcomePrices": '["0.85", "0.15"]',
        "liquidityNum":  2_100_000,
        "endDate":       "2026-12-31T00:00:00Z",
    },
    {
        "question":      "Will Apple release a foldable iPhone by end of 2026?",
        "description":   (
            "Resolves YES if Apple officially announces and begins shipping a foldable "
            "form-factor iPhone device to consumers before December 31 2026."
        ),
        "outcomes":      '["Yes", "No"]',
        "outcomePrices": '["0.83", "0.17"]',
        "liquidityNum":  175_000,
        "endDate":       "2026-12-31T00:00:00Z",
    },
    {
        "question":      "Will Nvidia stock exceed $200 before July 2026?",
        "description":   (
            "Resolves YES if NVDA closes above $200.00 on any trading day before "
            "July 1 2026 as reported by major US exchanges."
        ),
        "outcomes":      '["Yes", "No"]',
        "outcomePrices": '["0.88", "0.12"]',
        "liquidityNum":  890_000,
        "endDate":       "2026-07-01T00:00:00Z",
    },
]


# ── Polymarket API helpers ────────────────────────────────────────────────────

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


def _parse_float_list(raw) -> list[float]:
    """Parse a JSON string or list into a list of floats."""
    if raw is None:
        return []
    lst = json.loads(raw) if isinstance(raw, str) else list(raw)
    return [float(x) for x in lst if x is not None]


def yes_price(market: dict) -> float | None:
    """Return the probability of the YES outcome specifically.

    Polymarket binary markets have parallel `outcomes` and `outcomePrices` arrays.
    We find the index of "Yes" by name and return that price.
    Falls back to index 0 (Polymarket convention) when outcome labels are absent.
    Returns None if prices can't be parsed or no valid YES token is found.
    """
    try:
        prices = _parse_float_list(market.get("outcomePrices") or market.get("outcome_prices"))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    if not prices:
        return None

    outcomes_raw = market.get("outcomes")
    if outcomes_raw:
        try:
            names = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else list(outcomes_raw)
            for i, name in enumerate(names):
                if str(name).strip().lower() in ("yes", "true", "y", "1"):
                    return prices[i] if i < len(prices) else None
        except (json.JSONDecodeError, TypeError):
            pass

    # Polymarket convention: for binary markets the first token is always YES
    return prices[0]


def liquidity_usd(market: dict) -> float:
    for key in ("liquidityNum", "liquidity", "volume24hr", "volumeNum"):
        val = market.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def resolve_date(market: dict) -> datetime | None:
    """Parse the market resolution/end date as a UTC-aware datetime."""
    for key in ("endDate", "endDateIso", "resolutionDate", "end_date"):
        raw = market.get(key)
        if raw:
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                pass
    return None


def normalise_price(raw_price: float | None) -> float | None:
    """Normalise a price to the [0,1] range.

    Polymarket returns prices as decimals (0.87) but some endpoints or
    market records occasionally use percentage points (87.0).  Values
    clearly above 1.0 are divided by 100 before comparison.
    """
    if raw_price is None:
        return None
    if raw_price > 1.0:
        raw_price = raw_price / 100.0
    return raw_price


def filter_markets(markets: list[dict]) -> list[dict]:
    """Keep markets with a YES outcome in [MIN_PROB, MAX_PROB] and enough
    liquidity. Returns the top MAX_MARKETS by liquidity descending."""
    out = []
    for m in markets:
        price = normalise_price(yes_price(m))
        if price is None:
            continue
        if not (MIN_PROB <= price <= MAX_PROB):
            continue
        if liquidity_usd(m) < MIN_LIQUID:
            continue
        m["_top_price"] = price
        m["_end_date"]  = resolve_date(m)
        out.append(m)

    out.sort(key=lambda m: liquidity_usd(m), reverse=True)
    return out[:MAX_MARKETS]


# ── Claude + web search ────────────────────────────────────────────────────────

def build_prompt(market: dict) -> str:
    title    = market.get("question") or market.get("title") or "Unknown"
    criteria = market.get("description") or market.get("resolutionCriteria") or "Not provided"
    criteria = textwrap.shorten(criteria, width=600, placeholder="…")
    price    = market["_top_price"]
    end_dt   = market.get("_end_date")
    end_str  = end_dt.strftime("%Y-%m-%d") if end_dt else "unknown"

    return f"""You are a sharp prediction-market analyst with live web search access.

Market question: {title}

Resolution criteria: {criteria}

Resolution date: {end_str}
Market's current YES probability: {price:.1%}
Today's date: {time.strftime('%Y-%m-%d')}

INSTRUCTIONS:
1. Use web search to find current news, data, or recent developments directly relevant to this market. Search for specific facts that move the probability.
2. Based on what you find, estimate the true probability (0–100%) that this market resolves YES.
3. Explain your reasoning in 2–3 sentences, citing the evidence you found.
4. Compare your estimate to the market price and classify the edge.

After searching, respond ONLY with valid JSON — no markdown fences, no extra keys:
{{
  "claude_prob": <float between 0 and 1>,
  "reasoning": "<2-3 sentence string citing what you found>",
  "edge_direction": "OVER" | "UNDER" | "FAIR"
}}

Definitions:
- "OVER"  = your estimate is HIGHER than the market (market is cheap — potential buy)
- "UNDER" = your estimate is LOWER  (market is expensive — avoid)
- "FAIR"  = roughly aligned, no meaningful edge"""


def run_with_web_search(client: Anthropic, prompt: str) -> str:
    """Call Claude with the built-in server-side web_search tool.

    web_search_20260209 is a server-side tool: Anthropic executes the searches
    and embeds results in the response content automatically — no tool_result
    messages are needed from the client. We just loop on pause_turn (server
    iteration limit) until we get end_turn.
    """
    messages = [{"role": "user", "content": prompt}]
    tools    = [{"type": "web_search_20260209", "name": "web_search"}]

    for _ in range(6):   # safety cap for pause_turn loops
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next(
                (b.text for b in response.content if hasattr(b, "text") and b.text),
                ""
            )

        if response.stop_reason == "pause_turn":
            # Server-side loop hit its iteration limit; re-send to resume.
            # We do NOT add a new user message — the API detects the trailing
            # server_tool_use block and resumes the search loop automatically.
            messages = [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": response.content},
            ]
            continue

        # Unexpected stop (max_tokens, refusal, etc.) — return whatever text exists
        return next(
            (b.text for b in response.content if hasattr(b, "text") and b.text),
            ""
        )

    return ""


def run_without_tools(client: Anthropic, prompt: str) -> str:
    """Fallback: call Claude without web search tools."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(
        (b.text for b in response.content if hasattr(b, "text") and b.text), ""
    )


def analyze_markets(client: Anthropic, markets: list[dict]) -> list[dict]:
    for i, market in enumerate(markets, 1):
        title = market.get("question") or market.get("title") or "Unknown"
        print(f"  [{i}/{len(markets)}] {title[:70]}…", flush=True)
        try:
            raw = run_with_web_search(client, build_prompt(market))
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            if not raw:
                # Web-search loop returned no text block — retry without tools
                print("    ↺ Empty web-search response, retrying without tools…", flush=True)
                raw = run_without_tools(client, build_prompt(market)).strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
            if not raw:
                raise ValueError("No text content returned from Claude")
            analysis = json.loads(raw)
        except Exception as exc:
            print(f"    ⚠ Analysis error: {exc}", file=sys.stderr)
            analysis = {"claude_prob": None, "reasoning": str(exc), "edge_direction": "FAIR"}

        market["_analysis"] = analysis
        time.sleep(0.5)

    return markets


# ── Terminal dashboard ────────────────────────────────────────────────────────

def prob_bar(value: float, width: int = 22) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def expected_value(market_prob: float, claude_prob: float, bet: float) -> float:
    """EV of a $bet on YES at binary market odds."""
    payout_if_yes = bet / market_prob
    return claude_prob * (payout_if_yes - bet) - (1 - claude_prob) * bet


def edge_color(direction: str) -> str:
    return {"OVER": GREEN, "UNDER": RED, "FAIR": DIM}.get(direction, DIM)


def render_dashboard(markets: list[dict]) -> None:
    W   = 90
    SEP = "─" * W

    print()
    print(BOLD + CYAN + "╔" + "═" * (W - 2) + "╗" + RESET)
    hdr = "  POLYMARKET CLOB ANALYZER  ·  Paper Trading Only — No Bets Are Placed  "
    print(BOLD + CYAN + "║" + hdr + " " * (W - 2 - len(hdr)) + "║" + RESET)
    print(BOLD + CYAN + "╚" + "═" * (W - 2) + "╝" + RESET)
    print(f"{DIM}  {time.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"Filter: {MIN_PROB:.0%}–{MAX_PROB:.0%} YES   "
          f"Top {MAX_MARKETS} by liquidity   "
          f"Model: {MODEL}   "
          f"Bet size: ${BET_SIZE:.0f}{RESET}")
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
        end_dt    = market.get("_end_date")
        end_str   = end_dt.strftime("%Y-%m-%d") if end_dt else "?"

        flag = f"  {BOLD}{GREEN}◆ EDGE FOUND{RESET}" if has_edge else ""
        print(BOLD + f"  #{idx}  " + WHITE + BOLD + title[:W - 8] + RESET + flag)
        print("  " + SEP)
        print(f"  {DIM}Resolves: {end_str}   Liquidity: ${liquidity_usd(market):,.0f}{RESET}")

        # Probability bars — always labelled YES so there's no ambiguity
        print(f"  Market YES    {CYAN}{mkt_prob:5.1%}{RESET}  {CYAN}{prob_bar(mkt_prob)}{RESET}")
        if c_prob is not None:
            diff     = c_prob - mkt_prob
            diff_str = f"{'+' if diff >= 0 else ''}{diff:.1%}"
            print(f"  Claude YES    {color}{c_prob:5.1%}{RESET}  {color}{prob_bar(c_prob)}{RESET}")
            print(f"  Delta         {color}{BOLD}{diff_str:>6}{RESET}  ({direction})")

            ev     = expected_value(mkt_prob, c_prob, BET_SIZE)
            ev_str = f"{'+'if ev>=0 else ''}{ev:.2f}"
            ev_col = GREEN if ev > 0.10 else (RED if ev < -0.10 else DIM)
            print(f"  ${BET_SIZE:.0f} bet EV    {ev_col}{BOLD}${ev_str}{RESET}")
        else:
            print(f"  Claude YES    {DIM}N/A{RESET}")

        wrapped = textwrap.fill(
            reasoning, width=W - 4, initial_indent="    ", subsequent_indent="    "
        )
        print(f"\n{DIM}  Reasoning:{RESET}")
        print(f"{DIM}{wrapped}{RESET}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
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
            title     = (m.get("question") or m.get("title") or "Unknown")[:58]
            diff      = (c_prob - mkt_prob) if c_prob is not None else 0
            ev        = expected_value(mkt_prob, c_prob, BET_SIZE) if c_prob is not None else 0
            col       = GREEN if direction == "OVER" else RED
            ev_str    = f"{'+'if ev>=0 else ''}{ev:.2f}"
            print(f"  {col}{BOLD}[{direction:5s}]{RESET}  "
                  f"{col}{title}{RESET}  "
                  f"{DIM}Δ{diff:+.1%}  EV ${ev_str}{RESET}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket CLOB market analyzer")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use mock market data (useful when Polymarket API is IP-blocked)",
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
        markets = list(DEMO_MARKETS)          # don't mutate the module-level list
        for m in markets:
            m["_top_price"] = normalise_price(yes_price(m))
            m["_end_date"]  = resolve_date(m)
        markets = [m for m in markets if m["_top_price"] is not None]
    else:
        print(BOLD + CYAN + "\nFetching active Polymarket markets…" + RESET, flush=True)
        try:
            raw_markets = fetch_markets(limit=300)
        except requests.RequestException as exc:
            sys.exit(
                f"Failed to fetch markets: {exc}\n\n"
                "If you're getting a 403, Polymarket may be blocking this IP.\n"
                "Run with --demo to verify the tool works, or re-run from a\n"
                "residential/non-datacenter IP."
            )
        print(f"  Retrieved {len(raw_markets)} markets.")

        # ── Diagnostic: show raw price data for first 20 markets ──────────────
        print(f"\n{DIM}  Diagnostic — first 20 raw markets (before filter):{RESET}")
        print(f"  {'#':>3}  {'YES price':>12}  {'norm':>6}  {'outcomes':<20}  {'outcomePrices':<30}  question")
        print("  " + "─" * 110)
        for di, dm in enumerate(raw_markets[:20], 1):
            raw_p    = yes_price(dm)
            norm_p   = normalise_price(raw_p)
            outcomes = str(dm.get("outcomes", ""))[:18]
            op       = str(dm.get("outcomePrices") or dm.get("outcome_prices", ""))[:28]
            q        = (dm.get("question") or dm.get("title") or "")[:50]
            raw_str  = f"{raw_p:8.4f}" if raw_p is not None else "    None"
            norm_str = f"{norm_p:.4f}" if norm_p is not None else "  None"
            print(f"  {di:>3}  {raw_str:>12}  {norm_str:>6}  {outcomes:<20}  {op:<30}  {q}")
        print()
        # ─────────────────────────────────────────────────────────────────────

        markets = filter_markets(raw_markets)
        if not markets:
            print(
                f"\nNo markets matched (YES {MIN_PROB:.0%}–{MAX_PROB:.0%}, "
                f"liquidity ≥ ${MIN_LIQUID:,}).\n"
                "Try lowering MIN_LIQUID or widening the probability band "
                "in the config section.\n"
            )
            return
        print(f"  {len(markets)} markets passed filter.")

    print(BOLD + CYAN + f"\nRunning Claude analysis with web search ({MODEL})…" + RESET, flush=True)
    analyzed = analyze_markets(client, markets)
    render_dashboard(analyzed)


if __name__ == "__main__":
    main()
