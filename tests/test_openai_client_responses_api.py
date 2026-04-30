"""Tests for OpenAI-compatible endpoint Responses API selection."""

from __future__ import annotations

import pytest

from tradingagents.llm_clients.openai_client import OpenAIClient


@pytest.mark.unit
def test_openai_client_allows_disabling_responses_api(monkeypatch):
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "tradingagents.llm_clients.openai_client.NormalizedChatOpenAI",
        FakeChatOpenAI,
    )

    client = OpenAIClient(
        "gpt-5.5",
        base_url="http://127.0.0.1:10531/v1",
        provider="openai",
        api_key="dummy",
        use_responses_api=False,
    )

    assert client.get_llm() is not None
    assert captured["base_url"] == "http://127.0.0.1:10531/v1"
    assert captured["model"] == "gpt-5.5"
    assert captured["api_key"] == "dummy"
    assert captured["use_responses_api"] is False


@pytest.mark.unit
def test_openai_client_uses_responses_api_by_default_for_native_openai(monkeypatch):
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "tradingagents.llm_clients.openai_client.NormalizedChatOpenAI",
        FakeChatOpenAI,
    )

    client = OpenAIClient("gpt-5.4", provider="openai", api_key="dummy")

    assert client.get_llm() is not None
    assert captured["model"] == "gpt-5.4"
    assert captured["api_key"] == "dummy"
    assert captured["use_responses_api"] is True
