import pandas as pd
import pytest

from f1pred.backtest import walkforward as wf_mod
from f1pred.backtest.walkforward import walkforward
from f1pred.config import DEFAULT_PARAMS
from tests.synthetic_pace import synthetic_table

P = DEFAULT_PARAMS
FIT_KWARGS = {"num_warmup": 10, "num_samples": 10, "chains": 1, "device": "cpu"}
SCORED_SEASON = 2023
N_SCORED = 3


@pytest.fixture
def table() -> pd.DataFrame:
    """Two full seasons of history, then only three races in the season under test."""
    full = synthetic_table(seed=0)
    last = full[full.season == SCORED_SEASON]
    keep = sorted(last.race_id.unique())[:N_SCORED]
    return pd.concat(
        [full[full.season != SCORED_SEASON], last[last.race_id.isin(keep)]], ignore_index=True
    )


@pytest.fixture
def counting_fit(monkeypatch):
    calls = []
    real = wf_mod.fit

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(wf_mod, "fit", counted)
    return calls


def test_walkforward_scores_every_race_and_caches_each_posterior(tmp_path, table, counting_fit):
    result = walkforward(
        table, [SCORED_SEASON], P, FIT_KWARGS, tmp_path, n_runs=200, seed=0, refit_every=1
    )
    assert len(result.races) == N_SCORED
    assert len(counting_fit) == N_SCORED
    assert sorted(p.name for p in (tmp_path / "posteriors").glob("wf-*.npz")) == [
        f"wf-{r}.npz" for r in sorted(table[table.season == SCORED_SEASON].race_id.unique())
    ]
    assert result.seasons.n_races.iloc[0] == N_SCORED
    assert result.races.model_logloss.notna().all()


def test_a_second_run_reuses_the_cached_posteriors(tmp_path, table, counting_fit):
    first = walkforward(table, [SCORED_SEASON], P, FIT_KWARGS, tmp_path, n_runs=200, seed=0)
    assert len(counting_fit) == N_SCORED
    counting_fit.clear()
    second = walkforward(table, [SCORED_SEASON], P, FIT_KWARGS, tmp_path, n_runs=200, seed=0)
    assert counting_fit == []
    pd.testing.assert_frame_equal(first.races, second.races)


def test_refit_every_two_fits_half_as_often(tmp_path, table, counting_fit):
    result = walkforward(
        table, [SCORED_SEASON], P, FIT_KWARGS, tmp_path, n_runs=200, seed=0, refit_every=2
    )
    assert len(result.races) == N_SCORED
    assert len(counting_fit) == 2  # races 1 and 3 refit; race 2 reuses race 1's posterior
    assert len(list((tmp_path / "posteriors").glob("wf-*.npz"))) == 2


def test_each_fit_only_sees_races_before_the_one_it_scores(tmp_path, table, monkeypatch):
    cutoffs = []
    real = wf_mod.build_design

    def spy(tbl, before_date=None, params=None):
        cutoffs.append(before_date)
        return real(tbl, before_date=before_date, params=params)

    monkeypatch.setattr(wf_mod, "build_design", spy)
    walkforward(table, [SCORED_SEASON], P, FIT_KWARGS, tmp_path, n_runs=100, seed=0)
    dates = sorted(table[table.season == SCORED_SEASON].groupby("race_id").date.first())
    assert cutoffs == dates
    for cutoff, date in zip(cutoffs, dates, strict=True):
        assert cutoff == date  # strictly before, so the scored race itself is excluded
