#!/usr/bin/env python3
"""Run TradingAgents as an auto_trader advisory-only research source.

This runner is intentionally conservative: it converts a TradingAgents graph run
into a stable JSON payload for auto_trader/Decision Desk ingestion, while keeping
execution disabled by default. It is designed for OpenAI-compatible local shims
such as hermes-shim-http backed by Codex CLI GPT-5.5.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

DEFAULT_BASE_URL = "http://127.0.0.1:8796/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_ANALYSTS = "market"


def parse_analysts(value: str | Iterable[str]) -> list[str]:
    """Return a normalized analyst list from CSV or iterable input."""
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    analysts = [str(item).strip() for item in items if str(item).strip()]
    if not analysts:
        raise ValueError("At least one analyst must be selected")
    return analysts


def build_config(
    *,
    provider: str,
    base_url: str,
    model: str,
    max_debate_rounds: int,
    max_risk_discuss_rounds: int,
    max_recur_limit: int,
    output_language: str,
    checkpoint_enabled: bool,
) -> dict[str, Any]:
    """Build a minimal TradingAgents config suitable for shim smoke/advisory runs."""
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "llm_provider": provider,
            "backend_url": base_url,
            "deep_think_llm": model,
            "quick_think_llm": model,
            "max_debate_rounds": max_debate_rounds,
            "max_risk_discuss_rounds": max_risk_discuss_rounds,
            "max_recur_limit": max_recur_limit,
            "output_language": output_language,
            "checkpoint_enabled": checkpoint_enabled,
        }
    )
    return config


def extract_structured_output_warnings(log_text: str) -> list[str]:
    """Extract TradingAgents structured-output fallback warnings from captured logs."""
    warnings: list[str] = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if "structured-output invocation failed" in stripped:
            warnings.append(stripped)
    return warnings


def run_analysis(
    *,
    symbol: str,
    as_of_date: str,
    analysts: list[str],
    provider: str,
    base_url: str,
    model: str,
    max_debate_rounds: int,
    max_risk_discuss_rounds: int,
    max_recur_limit: int,
    output_language: str,
    checkpoint_enabled: bool,
    debug: bool = False,
) -> dict[str, Any]:
    """Run TradingAgents and return an advisory-only JSON-serializable payload."""
    normalized_symbol = symbol.upper()
    normalized_analysts = parse_analysts(analysts)
    config = build_config(
        provider=provider,
        base_url=base_url,
        model=model,
        max_debate_rounds=max_debate_rounds,
        max_risk_discuss_rounds=max_risk_discuss_rounds,
        max_recur_limit=max_recur_limit,
        output_language=output_language,
        checkpoint_enabled=checkpoint_enabled,
    )

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        graph = TradingAgentsGraph(
            selected_analysts=normalized_analysts,
            debug=debug,
            config=config,
        )
        state, decision = graph.propagate(normalized_symbol, as_of_date)

    captured_log = "".join([captured_stdout.getvalue(), captured_stderr.getvalue()])
    if captured_log:
        print(captured_log, file=sys.stderr, end="")

    final_trade_decision = state.get("final_trade_decision", "") if isinstance(state, dict) else ""
    result = {
        "status": "ok",
        "symbol": normalized_symbol,
        "as_of_date": as_of_date,
        "decision": decision,
        "advisory_only": True,
        "execution_allowed": False,
        "analysts": normalized_analysts,
        "llm": {
            "provider": provider,
            "model": model,
            "base_url": base_url,
        },
        "config": {
            "max_debate_rounds": max_debate_rounds,
            "max_risk_discuss_rounds": max_risk_discuss_rounds,
            "max_recur_limit": max_recur_limit,
            "output_language": output_language,
            "checkpoint_enabled": checkpoint_enabled,
        },
        "warnings": {
            "structured_output": extract_structured_output_warnings(captured_log),
        },
        "final_trade_decision": final_trade_decision,
        "raw_state_keys": sorted(state.keys()) if isinstance(state, dict) else [],
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TradingAgents and emit advisory-only JSON for auto_trader ingestion."
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. NVDA")
    parser.add_argument(
        "--date",
        dest="as_of_date",
        default=date.today().isoformat(),
        help="Analysis date in YYYY-MM-DD format; defaults to today.",
    )
    parser.add_argument(
        "--analysts",
        default=DEFAULT_ANALYSTS,
        help="Comma-separated analysts to enable. Default: market",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("TRADINGAGENTS_LLM_PROVIDER", "openai"),
        help="TradingAgents LLM provider. Default: openai",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("TRADINGAGENTS_OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("TRADINGAGENTS_MODEL", DEFAULT_MODEL),
        help=f"Model name exposed by the shim. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument("--max-debate-rounds", type=int, default=1)
    parser.add_argument("--max-risk-discuss-rounds", type=int, default=1)
    parser.add_argument("--max-recur-limit", type=int, default=30)
    parser.add_argument("--output-language", default="English")
    parser.add_argument(
        "--checkpoint-enabled",
        action="store_true",
        help="Enable TradingAgents checkpointing. Disabled by default for repeatable smoke runs.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to write the JSON payload. JSON is also printed to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    analysts = parse_analysts(args.analysts)

    result = run_analysis(
        symbol=args.symbol,
        as_of_date=args.as_of_date,
        analysts=analysts,
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        max_debate_rounds=args.max_debate_rounds,
        max_risk_discuss_rounds=args.max_risk_discuss_rounds,
        max_recur_limit=args.max_recur_limit,
        output_language=args.output_language,
        checkpoint_enabled=args.checkpoint_enabled,
        debug=args.debug,
    )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
