"""The spine of the project.

Every feature used to predict gameweek t must be computable from data timestamped
strictly before gameweek t's deadline. This file is deliberately created empty of
real assertions in phase 00 and filled in phase 02, when the feature builder exists.

It is here from day one so that it is never "added later".
"""

import pytest


@pytest.mark.xfail(reason="feature builder arrives in phase 02", strict=True)
def test_no_feature_uses_future_information():
    from fpl.features.build import build_features  # noqa: F401

    raise AssertionError("not implemented yet")
