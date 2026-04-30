"""Tests for the auto_trader advisory-only TradingAgents runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_auto_trader_research.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_auto_trader_research", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_build_config_uses_openai_compatible_shim_defaults():
    runner = load_runner_module()

    config = runner.build_config(
        provider="openai",
        base_url="http://127.0.0.1:8796/v1",
        model="gpt-5.5",
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_recur_limit=30,
        output_language="English",
        checkpoint_enabled=False,
        openai_use_responses_api=False,
    )

    assert config["llm_provider"] == "openai"
    assert config["backend_url"] == "http://127.0.0.1:8796/v1"
    assert config["deep_think_llm"] == "gpt-5.5"
    assert config["quick_think_llm"] == "gpt-5.5"
    assert config["max_debate_rounds"] == 1
    assert config["max_risk_discuss_rounds"] == 1
    assert config["max_recur_limit"] == 30
    assert config["output_language"] == "English"
    assert config["checkpoint_enabled"] is False
    assert config["openai_use_responses_api"] is False


@pytest.mark.unit
def test_run_analysis_returns_advisory_only_payload_and_structured_warnings(monkeypatch):
    runner = load_runner_module()

    class FakeTradingAgentsGraph:
        def __init__(self, selected_analysts, debug, config):
            self.selected_analysts = selected_analysts
            self.debug = debug
            self.config = config

        def propagate(self, symbol, as_of_date):
            print("Trader: structured-output invocation failed (demo)")
            return {"final_trade_decision": "**Final Decision: Hold NVDA**"}, "Hold"

    monkeypatch.setattr(runner, "TradingAgentsGraph", FakeTradingAgentsGraph)

    result = runner.run_analysis(
        symbol="nvda",
        as_of_date="2025-01-15",
        analysts=["market"],
        provider="openai",
        base_url="http://127.0.0.1:8796/v1",
        model="gpt-5.5",
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_recur_limit=30,
        output_language="English",
        checkpoint_enabled=False,
        openai_use_responses_api=False,
    )

    assert result["status"] == "ok"
    assert result["symbol"] == "NVDA"
    assert result["as_of_date"] == "2025-01-15"
    assert result["decision"] == "Hold"
    assert result["advisory_only"] is True
    assert result["execution_allowed"] is False
    assert result["analysts"] == ["market"]
    assert result["llm"]["model"] == "gpt-5.5"
    assert result["final_trade_decision"] == "**Final Decision: Hold NVDA**"
    assert result["warnings"]["structured_output"] == [
        "Trader: structured-output invocation failed (demo)"
    ]


@pytest.mark.unit
def test_main_writes_json_output_file(monkeypatch, tmp_path):
    runner = load_runner_module()
    output_path = tmp_path / "research.json"

    def fake_run_analysis(**kwargs):
        assert kwargs["openai_use_responses_api"] is False
        return {
            "status": "ok",
            "symbol": kwargs["symbol"],
            "as_of_date": kwargs["as_of_date"],
            "decision": "Hold",
            "advisory_only": True,
            "execution_allowed": False,
            "warnings": {"structured_output": []},
        }

    monkeypatch.setattr(runner, "run_analysis", fake_run_analysis)

    exit_code = runner.main(
        [
            "--symbol",
            "NVDA",
            "--date",
            "2025-01-15",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    saved = json.loads(output_path.read_text())
    assert saved["symbol"] == "NVDA"
    assert saved["advisory_only"] is True
    assert saved["execution_allowed"] is False
