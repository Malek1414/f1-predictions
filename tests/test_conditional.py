import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.conditional import effective_rating, strengths

P = DEFAULT_PARAMS


def test_effective_rating_shrinks_with_evidence():
    assert effective_rating(1500, 1600, 0, 8) == 1500
    assert effective_rating(1500, 1600, 8, 8) == pytest.approx(1550)
    assert effective_rating(1500, 1600, 10_000, 8) == pytest.approx(1600, abs=0.1)


def test_strengths_flags():
    args = dict(
        driver_overall=1500,
        constructor_overall=1500,
        driver_track=1580,
        constructor_track=1500,
        n_driver_track=8,
        n_constructor_track=0,
        driver_wet=1500,
        constructor_wet=1420,
        n_driver_wet=0,
        n_constructor_wet=8,
    )
    dry, wet = strengths(**args, params=P)
    assert dry == pytest.approx(3040) and wet == pytest.approx(3000)
    dry, wet = strengths(**args, params=P.replace(use_track=False))
    assert dry == 3000 and wet == pytest.approx(2960)
    dry, wet = strengths(**args, params=P.replace(use_weather=False))
    assert dry == pytest.approx(3040) and wet == dry
