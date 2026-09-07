"""Produce the written rationale, and refuse to publish one that invents a figure.

The interesting engineering here is not the API call. It is the guardrail: an
explanation that cites a statistic the model never produced is worse than no
explanation, because it is confidently wrong in a way a reader cannot detect.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import pandas as pd

from fpl.explain.prompt import SYSTEM, allowed_numbers, build_facts, build_prompt

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 220

# Matches any number the text asserts, including decimals.
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

# Numbers the text may use without them being a statistic: the size of a starting
# eleven and the minute thresholds in the rules. Deliberately small. Every small
# integer admitted here is a fabrication the guardrail can no longer catch ("scored
# 5 goals"), so counts and scoring values are left out and the prompt asks for
# them in words.
INNOCUOUS = {"11", "60", "90"}


@dataclass(frozen=True)
class Rationale:
    text: str
    source: str  # "llm" or "template"


def _ungrounded_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in the text that were not handed to the model."""
    permitted = allowed_numbers(facts) | INNOCUOUS
    return [n for n in NUMBER_PATTERN.findall(text) if n not in permitted]


def template_rationale(facts: dict) -> Rationale:
    """Deterministic fallback, used when no API key is configured.

    Keeps the repo runnable for anyone who clones it, and gives the guardrail test
    something to compare against.
    """
    parts = {
        "attacking": facts.get("attacking") or 0.0,
        "clean sheet": facts.get("clean_sheet") or 0.0,
        "appearance": facts.get("appearance") or 0.0,
        "bonus": facts.get("bonus") or 0.0,
    }
    driver = max(parts, key=parts.get)
    name = facts.get("name", "This player")
    venue = facts.get("venue", "")
    opponent = facts.get("opponent", "")

    text = (
        f"{name} projects at {facts.get('expected_points')} points {venue} to {opponent}. "
        f"The largest single contribution is {driver}, worth {round(parts[driver], 1)}. "
        f"Start probability is {facts.get('start_probability')}."
    )
    return Rationale(text, "template")


def llm_rationale(facts: dict, *, client=None) -> Rationale:
    """Ask Claude to narrate the breakdown. Falls back if the output is ungrounded."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(facts)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()

    invented = _ungrounded_numbers(text, facts)
    if invented:
        log.warning(
            "rejected rationale for %s: cited ungrounded numbers %s",
            facts.get("name"),
            invented,
        )
        return template_rationale(facts)

    return Rationale(text, "llm")


def explain_row(row: pd.Series, *, client=None) -> Rationale:
    """Explain one projection, using the API when a key is present."""
    facts = build_facts(row)
    if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
        return template_rationale(facts)
    try:
        return llm_rationale(facts, client=client)
    except Exception as error:  # noqa: BLE001 - never let prose break the pipeline
        log.warning("falling back to template for %s: %s", facts.get("name"), error)
        return template_rationale(facts)


def explain_frame(frame: pd.DataFrame, *, client=None) -> pd.DataFrame:
    """Add ``rationale`` and ``rationale_source`` columns."""
    results = [explain_row(row, client=client) for _, row in frame.iterrows()]
    out = frame.copy()
    out["rationale"] = [r.text for r in results]
    out["rationale_source"] = [r.source for r in results]
    return out
