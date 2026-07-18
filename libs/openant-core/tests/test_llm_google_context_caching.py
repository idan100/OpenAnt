"""Tests for GoogleAdapter's opt-in explicit context caching."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests._llm_factories.google import _candidate, _client_error, _response, _text_part
from utilities.llm import Message, TextBlock
from utilities.llm.providers.google import GoogleAdapter
from utilities.llm_client import reset_warning_state
from utilities.rate_limiter import reset_rate_limiter


@pytest.fixture(autouse=True)
def _reset_state():
    reset_rate_limiter()
    reset_warning_state()
    yield
    reset_rate_limiter()
    reset_warning_state()


def _fake_client_with_cache(cache_name: str = "cachedContents/abc123"):
    client = MagicMock()
    client.models.generate_content = MagicMock(
        return_value=_response(candidates=[_candidate(parts=[_text_part("hi")])], prompt_tokens=3, candidate_tokens=2)
    )
    client.caches.create = MagicMock(return_value=SimpleNamespace(name=cache_name))
    return client


def _call(adapter, system="You are a helpful assistant with a long fixed prompt."):
    return adapter.complete(
        model="gemini-3.1-flash-lite",
        system=system,
        messages=[Message(role="user", content=[TextBlock("hi")])],
        max_tokens=8,
    )


class TestCachingDisabledByDefault:
    def test_default_off_never_calls_caches_create(self):
        client = _fake_client_with_cache()
        adapter = GoogleAdapter(_client=client)
        _call(adapter)
        client.caches.create.assert_not_called()
        # system_instruction sent directly, no cached_content.
        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.cached_content is None
        assert config.system_instruction is not None


class TestCachingEnabled:
    def test_first_call_creates_cache_and_uses_it(self):
        client = _fake_client_with_cache("cachedContents/xyz")
        adapter = GoogleAdapter(_client=client, enable_context_caching=True)
        _call(adapter)
        client.caches.create.assert_called_once()
        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.cached_content == "cachedContents/xyz"
        # Not re-sent — already baked into the cache.
        assert config.system_instruction is None

    def test_second_call_with_same_prefix_reuses_cache(self):
        client = _fake_client_with_cache("cachedContents/xyz")
        adapter = GoogleAdapter(_client=client, enable_context_caching=True)
        _call(adapter)
        _call(adapter)
        client.caches.create.assert_called_once()  # not called again
        assert client.models.generate_content.call_count == 2

    def test_different_system_prompt_creates_a_second_cache(self):
        client = _fake_client_with_cache("cachedContents/xyz")
        adapter = GoogleAdapter(_client=client, enable_context_caching=True)
        _call(adapter, system="Prompt A, long enough to matter for caching purposes.")
        _call(adapter, system="Prompt B, a completely different fixed prefix entirely.")
        assert client.caches.create.call_count == 2

    def test_cache_creation_failure_falls_back_to_uncached_call(self):
        client = _fake_client_with_cache()
        client.caches.create = MagicMock(side_effect=RuntimeError("too small to cache"))
        adapter = GoogleAdapter(_client=client, enable_context_caching=True)
        result = _call(adapter)  # must not raise
        assert result.content[0].text == "hi"
        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.cached_content is None
        assert config.system_instruction is not None

    def test_stale_cache_reference_retried_uncached(self):
        client = _fake_client_with_cache("cachedContents/stale")
        ok_response = _response(candidates=[_candidate(parts=[_text_part("hi")])], prompt_tokens=3, candidate_tokens=2)
        client.models.generate_content = MagicMock(
            side_effect=[_client_error(404, "cached content not found"), ok_response]
        )
        adapter = GoogleAdapter(_client=client, enable_context_caching=True)
        result = _call(adapter)  # must not raise despite the first-call 404
        assert result.content[0].text == "hi"
        assert client.models.generate_content.call_count == 2
        retry_config = client.models.generate_content.call_args.kwargs["config"]
        assert retry_config.cached_content is None  # retried without the stale cache
