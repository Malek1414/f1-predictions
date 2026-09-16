import numpy as np
import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.data.track_types import TRACK_TYPES
from f1pred.pace.design import build_design, target_design
from f1pred.sim.race import Entrant

P = DEFAULT_PARAMS


@pytest.fixture(scope="module")
def design(driver_race):
    return build_design(driver_race, params=P)


def test_counts_match_the_table(driver_race, design):
    races = driver_race[~driver_race.is_sprint]
    assert design.n_row == len(races)
    assert design.n_driver == races.driver_id.nunique()
    assert design.n_race == races.race_id.nunique()
    assert design.n_circuit == races.circuit_id.nunique()
    assert design.n_season == races.season.nunique()
    assert design.n_constructor_season == len(races.groupby(["constructor_id", "season"]))
    assert design.n_driver_season == len(races.groupby(["driver_id", "season"]))
    assert design.n_track_type == len(TRACK_TYPES)


def test_race_slices_are_contiguous_and_cover_every_row(design):
    assert design.race_start[0] == 0
    assert (design.race_start[1:] == np.cumsum(design.race_len)[:-1]).all()
    assert design.race_len.sum() == design.n_row
    # Each slice holds exactly one race and its classified rows come first, in finishing order.
    for r in range(design.n_race):
        sl = slice(design.race_start[r], design.race_start[r] + design.race_len[r])
        assert set(design.race[sl].tolist()) == {r}
        k = int(design.race_n_classified[r])
        pos = design.position[sl][:k]
        assert (pos == np.arange(1, k + 1)).all()
        assert (design.position[sl][k:] == 0).all()


def test_previous_driver_season_links_consecutive_seasons(design):
    keys = list(design.driver_season_keys)
    i2023 = keys.index(("max_verstappen", 2023))
    i2024 = keys.index(("max_verstappen", 2024))
    assert design.previous_driver_season[i2024] == i2023
    assert design.previous_driver_season[i2023] == -1


def test_previous_constructor_season_links_and_regulation_flag(design):
    keys = list(design.constructor_season_keys)
    i2023 = keys.index(("red_bull", 2023))
    i2024 = keys.index(("red_bull", 2024))
    assert design.previous_constructor_season[i2024] == i2023
    assert design.previous_constructor_season[i2023] == -1
    assert not design.is_regulation_season.any()  # 2023 and 2024 are not regulation seasons


def test_constructor_form_walk_resets_each_season(design):
    # The first race of a season has no predecessor for any constructor; later races do.
    first = design.previous_constructor_race == -1
    assert first.sum() == design.n_constructor_season
    assert (~first).sum() == design.n_constructor_race - design.n_constructor_season


def test_before_date_excludes_rows_on_or_after_it(driver_race):
    cutoff = pd.Timestamp("2024-01-01")
    d = build_design(driver_race, before_date=cutoff, params=P)
    races = driver_race[(~driver_race.is_sprint) & (driver_race.date < cutoff)]
    assert d.n_row == len(races)
    assert d.n_season == 1
    assert (np.asarray(design_seasons(d)) == 2023).all()


def design_seasons(d):
    return [d.seasons[s] for s in d.season_of_race]


def test_grid_and_quali_columns(driver_race, design):
    races = driver_race[~driver_race.is_sprint].sort_values(["date", "race_id"], kind="stable")
    assert design.grid_g.min() > 0
    assert design.grid_g.max() == pytest.approx(1.0)  # pole -> 1 / 1 ** 0.75
    assert np.isfinite(design.quali_gap).mean() == pytest.approx(
        races.quali_gap_pct.notna().mean(), abs=0.02
    )
    assert design.dnf.dtype == bool


def test_target_design_maps_known_and_unknown_entrants(design, driver_race):
    class Ref:
        season = 2024
        circuit_id = "monaco"

    entrants = [
        Entrant("max_verstappen", "red_bull", 0.0, 1, 0.1),
        Entrant("nobody", "new_team", 0.0, 2, 0.1),
    ]
    out = target_design(design, entrants, Ref())
    d, cs, ci = out["max_verstappen"]
    assert d >= 0 and cs >= 0 and ci >= 0
    assert design.driver_ids[d] == "max_verstappen"
    assert design.constructor_season_keys[cs] == ("red_bull", 2024)
    assert design.circuit_ids[ci] == "monaco"
    assert out["nobody"] == (-1, -1, ci)
