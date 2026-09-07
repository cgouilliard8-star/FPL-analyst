"""The guardrail: an explanation may never assert a number the model did not produce."""

import pandas as pd
import pytest

from fpl.explain import generate
from fpl.explain.prompt import allowed_numbers, build_facts, build_prompt


class FakeBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeBlock(text)]


class FakeClient:
    """Stands in for the Anthropic client so tests never touch the network."""

    def __init__(self, text: str):
        self._text = text
        self.messages = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponse(self._text)


def a_row() -> pd.Series:
    return pd.Series(
        {
            "full_name": "Bukayo Saka",
            "position": "MID",
            "team": "Arsenal",
            "opponent": "Everton",
            "is_home": True,
            "price": 10.0,
            "expected_points": 6.2,
            "appearance": 1.8,
            "attacking": 2.9,
            "clean_sheet": 0.9,
            "bonus": 0.6,
            "goalkeeping": 0.0,
            "p_60": 0.88,
            "minutes_share_r5": 0.91,
        }
    )


def test_facts_contain_only_permitted_fields():
    facts = build_facts(a_row())
    assert facts["name"] == "Bukayo Saka"
    assert facts["expected_points"] == 6.2
    assert facts["venue"] == "home"


def test_prompt_embeds_the_numbers():
    prompt = build_prompt(build_facts(a_row()))
    assert "6.2" in prompt
    assert "Bukayo Saka" in prompt


def test_grounded_output_is_kept():
    facts = build_facts(a_row())
    text = "Saka projects at 6.2 points, driven by attacking worth 2.9."
    client = FakeClient(text)
    result = generate.llm_rationale(facts, client=client)
    assert result.source == "llm"
    assert result.text == text


def test_invented_statistic_is_rejected():
    """The whole point. A plausible but fabricated number must not survive."""
    facts = build_facts(a_row())
    client = FakeClient("Saka has scored 14 goals this season and projects at 6.2 points.")
    result = generate.llm_rationale(facts, client=client)
    assert result.source == "template", "an ungrounded figure should fall back"
    assert "14 goals" not in result.text


def test_rescaled_number_is_rejected():
    """Numbers derived from the facts are still not facts."""
    facts = build_facts(a_row())
    client = FakeClient("His attacking contribution is 29.7 percent of the projection.")
    assert generate.llm_rationale(facts, client=client).source == "template"


def test_api_failure_falls_back_rather_than_raising():
    class Broken:
        messages = property(lambda self: self)

        def create(self, **kwargs):
            raise RuntimeError("network down")

    result = generate.explain_row(a_row(), client=Broken())
    assert result.source == "template"


def test_template_rationale_is_itself_grounded():
    facts = build_facts(a_row())
    text = generate.template_rationale(facts).text
    assert not generate._ungrounded_numbers(text, facts)


def test_allowed_numbers_accepts_integer_spelling():
    """2.0 written as "2" is a rendering choice, not a fabrication."""
    facts = {"appearance": 2.0}
    assert "2" in allowed_numbers(facts)


@pytest.mark.parametrize("bad", ["nonsense 99.9", "he made 7 key passes"])
def test_various_fabrications_are_caught(bad):
    facts = build_facts(a_row())
    assert generate._ungrounded_numbers(bad, facts)
