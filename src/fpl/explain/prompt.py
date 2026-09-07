"""Turn a model decomposition into a grounded prompt.

The rule this module exists to enforce: the language model never produces a number.
Every figure it is allowed to say is handed to it, and the guardrail in
``fpl.explain.generate`` rejects any output containing a numeral that was not.
"""

from __future__ import annotations

import json

import pandas as pd

SYSTEM = (
    "You are a Fantasy Premier League analyst. You will be given a model's numeric "
    "breakdown of why a player is projected to score. Write two or three short "
    "sentences explaining the pick to an experienced FPL manager.\n\n"
    "Rules:\n"
    "- Use ONLY the numbers provided. Never introduce a figure of your own, and never "
    "round, rescale or combine numbers into new ones.\n"
    "- Lead with whichever component contributes most.\n"
    "- Name the real trade-off if there is one (rotation risk, a hard fixture, price).\n"
    '- Write small counts in words ("two fixtures", not "2 fixtures"); digits are '
    "reserved for the figures provided.\n"
    "- No hedging, no filler, no exclamation marks. Plain, direct prose."
)

# The only fields a rationale may mention.
FACT_FIELDS = (
    "name",
    "position",
    "team",
    "opponent",
    "venue",
    "price",
    "expected_points",
    "appearance",
    "attacking",
    "clean_sheet",
    "bonus",
    "goalkeeping",
    "start_probability",
    "minutes_share",
)


def build_facts(row: pd.Series) -> dict:
    """Extract the numbers a rationale is permitted to cite, rounded once, here."""

    def num(value, digits: int = 1):
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)

    return {
        "name": row.get("full_name"),
        "position": row.get("position"),
        "team": row.get("team"),
        "opponent": row.get("opponent"),
        "venue": "home" if row.get("is_home") else "away",
        "price": num(row.get("price")),
        "expected_points": num(row.get("expected_points")),
        "appearance": num(row.get("appearance")),
        "attacking": num(row.get("attacking")),
        "clean_sheet": num(row.get("clean_sheet")),
        "bonus": num(row.get("bonus")),
        "goalkeeping": num(row.get("goalkeeping")),
        "start_probability": num(row.get("p_60"), 2),
        "minutes_share": num(row.get("minutes_share_r5"), 2),
    }


def build_prompt(facts: dict) -> str:
    """Render the facts as the user turn of the request."""
    payload = {k: v for k, v in facts.items() if v is not None}
    return (
        "Explain this projection.\n\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Two or three sentences. Use only the numbers above."
    )


def allowed_numbers(facts: dict) -> set[str]:
    """Every numeric token the model is permitted to write.

    Includes the integer form of each value, since "2.0 points" reads better as
    "2 points" and that rewriting is not a fabrication.
    """
    tokens: set[str] = set()
    for value in facts.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tokens.add(str(value))
            tokens.add(str(abs(value)))
            if float(value).is_integer():
                tokens.add(str(int(value)))
                tokens.add(str(abs(int(value))))
            else:
                tokens.add(f"{value:.1f}")
                tokens.add(f"{value:.2f}")
    return tokens
