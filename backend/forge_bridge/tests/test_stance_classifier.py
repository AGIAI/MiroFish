"""
Tests for StanceClassifier.

Validates heuristic classification and parsing logic.
LLM-based classification is tested via mocked responses.
"""

import pytest
from unittest.mock import MagicMock, patch

from ..stance_classifier import StanceClassifier, VALID_STANCES


@pytest.fixture
def classifier():
    """Create classifier with a mock OpenAI client."""
    with patch("forge_bridge.stance_classifier.OpenAI"):
        cls = StanceClassifier(api_key="test-key", model="test-model")
    return cls


class TestParseStance:
    """Test parsing LLM output to stance values."""

    def test_exact_values(self, classifier):
        for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            assert classifier._parse_stance(str(val)) == val

    def test_snap_to_nearest(self, classifier):
        """Values between discrete stances should snap to nearest."""
        assert classifier._parse_stance("0.3") == 0.5
        assert classifier._parse_stance("-0.7") == -0.5
        assert classifier._parse_stance("0.8") == 1.0
        assert classifier._parse_stance("-0.2") == 0.0

    def test_invalid_returns_neutral(self, classifier):
        assert classifier._parse_stance("bullish") == 0.0
        assert classifier._parse_stance("") == 0.0
        assert classifier._parse_stance("N/A") == 0.0


class TestHeuristicClassify:
    """Test the keyword-based fallback classifier."""

    def test_bullish_content(self, classifier):
        actions = [
            {"action_type": "CREATE_POST",
             "action_args": {"content": "I'm going to buy BTC, moon incoming!"}}
        ]
        stance = classifier._heuristic_classify(actions)
        assert stance > 0

    def test_bearish_content(self, classifier):
        actions = [
            {"action_type": "CREATE_POST",
             "action_args": {"content": "This is going to crash, sell everything"}}
        ]
        stance = classifier._heuristic_classify(actions)
        assert stance < 0

    def test_neutral_content(self, classifier):
        actions = [
            {"action_type": "CREATE_POST",
             "action_args": {"content": "The weather is nice today"}}
        ]
        stance = classifier._heuristic_classify(actions)
        assert stance == 0.0

    def test_empty_actions(self, classifier):
        assert classifier._heuristic_classify([]) == 0.0


class TestGroupByAgent:
    """Test action grouping."""

    def test_groups_correctly(self, classifier):
        actions = [
            {"agent_name": "Alice", "action_type": "CREATE_POST"},
            {"agent_name": "Bob", "action_type": "CREATE_POST"},
            {"agent_name": "Alice", "action_type": "LIKE_POST"},
        ]
        groups = classifier._group_by_agent(actions)
        assert len(groups) == 2
        assert len(groups["Alice"]) == 2
        assert len(groups["Bob"]) == 1

    def test_skips_non_agent_events(self, classifier):
        actions = [
            {"action_type": "round_start"},
            {"agent_name": "Alice", "action_type": "CREATE_POST"},
        ]
        groups = classifier._group_by_agent(actions)
        assert len(groups) == 1


class TestFormatContent:
    """Test content formatting for LLM input."""

    def test_content_actions_prioritised(self, classifier):
        actions = [
            {"action_type": "CREATE_POST", "action_args": {"content": "Buy now!"}},
            {"action_type": "LIKE_POST", "action_args": {}},
        ]
        formatted = classifier._format_agent_content(actions)
        assert "Buy now!" in formatted

    def test_no_content_fallback(self, classifier):
        actions = [{"action_type": "FOLLOW", "action_args": {}}]
        formatted = classifier._format_agent_content(actions)
        assert "NO_CONTENT" in formatted


class TestDetectPlatform:
    """Test platform detection from action types."""

    def test_reddit_detection(self, classifier):
        actions = [{"action_type": "CREATE_COMMENT"}]
        assert classifier._detect_platform(actions) == "reddit"

    def test_twitter_default(self, classifier):
        actions = [{"action_type": "CREATE_POST"}]
        assert classifier._detect_platform(actions) == "twitter"
