import numpy as np
import pandas as pd
import pytest

from f1pred.backtest.bootstrap import bootstrap_intervals


def test_bootstrap_interval_contains_mean_and_narrows():
    df = pd.DataFrame({"model_logloss": np.random.default_rng(0).normal(1.5, 0.5, 200)})
    a = bootstrap_intervals(df, ["model_logloss"], R=500, seed=1)
    assert list(a.columns) == ["metric", "mean", "lo", "hi"]
    assert a.metric.iloc[0] == "model_logloss"
    assert a.lo.iloc[0] < a["mean"].iloc[0] < a.hi.iloc[0]
    assert a["mean"].iloc[0] == pytest.approx(df.model_logloss.mean())
    b = bootstrap_intervals(df.head(20), ["model_logloss"], R=500, seed=1)
    assert (b.hi - b.lo).iloc[0] > (a.hi - a.lo).iloc[0]


def test_bootstrap_is_seeded_and_handles_several_columns():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 10.0, 10.0, 10.0]})
    a = bootstrap_intervals(df, ["x", "y"], R=200, seed=3)
    b = bootstrap_intervals(df, ["x", "y"], R=200, seed=3)
    pd.testing.assert_frame_equal(a, b)
    y = a.set_index("metric").loc["y"]
    assert y.lo == y.hi == y["mean"] == 10.0  # a constant column has a zero-width interval
    x = a.set_index("metric").loc["x"]
    assert 1.0 <= x.lo < 2.5 < x.hi <= 4.0
