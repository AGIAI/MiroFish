"""
Stance Classifier

Classifies each agent's directional stance from their simulation outputs
using LLM-based structured output for consistency.

The stance scale is:
    -1.0  STRONGLY_BEARISH
    -0.5  BEARISH
     0.0  NEUTRAL
    +0.5  BULLISH
    +1.0  STRONGLY_BULLISH

Forge Compliance:
    - §1.1: Timestamps preserved (agent actions are point-in-time)
    - §4.2: Classification is deterministic given the same inputs
"""

import json
import logging
from typing import Optional
from collections import defaultdict

from openai import OpenAI

logger = logging.getLogger("forge_bridge.stance_classifier")

# Valid stance values (the 5-point scale)
VALID_STANCES = {-1.0, -0.5, 0.0, 0.5, 1.0}

CLASSIFICATION_SYSTEM_PROMPT = """You are a precise financial sentiment classifier.
You analyse social media posts from a simulation and classify the agent's directional
stance on the target asset or outcome.

You MUST respond with ONLY a single number from this set: -1.0, -0.5, 0.0, 0.5, 1.0

Scale:
-1.0 = STRONGLY_BEARISH (clear conviction that price/outcome will decline significantly)
-0.5 = BEARISH (leaning negative, expressing concerns or caution)
 0.0 = NEUTRAL (no clear directional view, balanced, or discussing unrelated topics)
+0.5 = BULLISH (leaning positive, expressing optimism or opportunity)
+1.0 = STRONGLY_BULLISH (clear conviction that price/outcome will rise significantly)

Rules:
- Consider ALL posts together for an overall stance, not individual post sentiment
- Weight posts by specificity: concrete price predictions > general sentiment
- If an agent mostly posts unrelated content, classify as NEUTRAL (0.0)
- Respond with the number ONLY, no explanation"""

CLASSIFICATION_USER_TEMPLATE = """Agent: "{agent_name}"
Topic/Asset: "{topic}"
Platform: {platform}

Posts and interactions:
{agent_content}

Classify this agent's overall directional stance. Respond with only the number."""


class StanceClassifier:
    """
    Classifies each agent's directional stance from their simulation outputs.

    Uses LLM with structured output. Falls back to content-based heuristics
    if LLM is unavailable.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        max_content_chars: int = 4000,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_content_chars = max_content_chars

    def classify_batch(
        self,
        actions: list,
        topic: str = "",
    ) -> list:
        """
        Classify all agents' stances from their action logs.

        Args:
            actions: List of action dicts from MiroFish actions.jsonl
            topic: The simulation topic / target asset

        Returns:
            List of float stances, one per unique agent, in agent_id order.
        """
        agents = self._group_by_agent(actions)
        stances = []

        for agent_name in sorted(agents.keys()):
            agent_actions = agents[agent_name]
            content = self._format_agent_content(agent_actions)
            platform = self._detect_platform(agent_actions)

            try:
                stance = self._classify_single(agent_name, content, topic, platform)
            except Exception as e:
                logger.warning(
                    "LLM classification failed for agent %s: %s. Using heuristic.",
                    agent_name,
                    e,
                )
                stance = self._heuristic_classify(agent_actions)

            stances.append(stance)

        return stances

    def _classify_single(
        self,
        agent_name: str,
        content: str,
        topic: str,
        platform: str,
    ) -> float:
        """Classify a single agent's stance via LLM."""
        user_msg = CLASSIFICATION_USER_TEMPLATE.format(
            agent_name=agent_name,
            topic=topic,
            platform=platform,
            agent_content=content[: self.max_content_chars],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,  # deterministic classification
            max_tokens=10,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_stance(raw)

    def _parse_stance(self, raw: str) -> float:
        """Parse LLM output to a valid stance value."""
        try:
            value = float(raw)
            # Snap to nearest valid stance
            closest = min(VALID_STANCES, key=lambda v: abs(v - value))
            return closest
        except (ValueError, TypeError):
            logger.warning("Could not parse stance '%s', defaulting to 0.0", raw)
            return 0.0

    def _group_by_agent(self, actions: list) -> dict:
        """Group actions by agent name, preserving order."""
        agents = defaultdict(list)
        for action in actions:
            # Skip non-agent events (round_start, round_end, etc.)
            if "agent_name" not in action:
                continue
            agents[action["agent_name"]].append(action)
        return dict(agents)

    def _format_agent_content(self, agent_actions: list) -> str:
        """
        Format an agent's actions into a readable string for classification.
        Prioritises content-bearing actions (posts, comments) over reactions.
        """
        lines = []

        # Content-bearing actions first (more informative for stance)
        content_actions = [
            a
            for a in agent_actions
            if a.get("action_type") in ("CREATE_POST", "CREATE_COMMENT", "QUOTE_POST")
        ]
        for a in content_actions:
            content = a.get("action_args", {}).get("content", "")
            if content:
                lines.append(f"[{a['action_type']}] {content}")

        # Reaction actions (less informative but still relevant)
        reaction_actions = [
            a
            for a in agent_actions
            if a.get("action_type")
            in ("LIKE_POST", "DISLIKE_POST", "REPOST", "LIKE_COMMENT", "DISLIKE_COMMENT")
        ]
        if reaction_actions:
            like_count = sum(
                1 for a in reaction_actions if "LIKE" in a.get("action_type", "")
            )
            dislike_count = sum(
                1 for a in reaction_actions if "DISLIKE" in a.get("action_type", "")
            )
            if like_count or dislike_count:
                lines.append(
                    f"[REACTIONS] {like_count} likes, {dislike_count} dislikes"
                )

        if not lines:
            lines.append("[NO_CONTENT] Agent did not produce meaningful content.")

        return "\n".join(lines)

    def _detect_platform(self, agent_actions: list) -> str:
        """Detect which platform the actions came from."""
        for a in agent_actions:
            if a.get("action_type") in ("CREATE_COMMENT", "DISLIKE_POST", "SEARCH_POSTS"):
                return "reddit"
        return "twitter"

    def _heuristic_classify(self, agent_actions: list) -> float:
        """
        Fallback heuristic classification based on action sentiment words.
        Only used if LLM classification fails.
        """
        bullish_words = {
            "buy",
            "long",
            "moon",
            "pump",
            "bull",
            "growth",
            "opportunity",
            "upside",
            "breakout",
            "rally",
        }
        bearish_words = {
            "sell",
            "short",
            "crash",
            "dump",
            "bear",
            "decline",
            "risk",
            "downside",
            "breakdown",
            "correction",
        }

        bull_score = 0
        bear_score = 0

        for action in agent_actions:
            content = json.dumps(action.get("action_args", {})).lower()
            bull_score += sum(1 for w in bullish_words if w in content)
            bear_score += sum(1 for w in bearish_words if w in content)

        total = bull_score + bear_score
        if total == 0:
            return 0.0

        net = (bull_score - bear_score) / total
        # Map to 5-point scale
        if net > 0.5:
            return 1.0
        elif net > 0.15:
            return 0.5
        elif net > -0.15:
            return 0.0
        elif net > -0.5:
            return -0.5
        else:
            return -1.0
