import pandas as pd
import pytest

from f1pred.data.frame import DRIVER_RACE_COLUMNS, classify_dnf


@pytest.mark.parametrize(
    "status,classified,expected",
    [
        ("Finished", True, None),
        ("+1 Lap", True, None),
        ("Accident", False, "accident"),
        ("Collision", False, "accident"),
        ("Spun off", False, "accident"),
        ("Collision damage", False, "accident"),
        ("Engine", False, "mechanical"),
        ("Gearbox", False, "mechanical"),
        ("Puncture", False, "mechanical"),
        ("Power Unit", False, "mechanical"),
        ("Disqualified", False, "other"),
        ("Retired", False, "other"),
        ("Withdrew", False, "other"),
        ("Illness", False, "other"),
    ],
)
def test_classify_dnf(status, classified, expected):
    assert classify_dnf(status, classified) == expected


def test_table_columns_and_types(driver_race):
    assert list(driver_race.columns) == list(DRIVER_RACE_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(driver_race["date"])
    assert driver_race["position"].dtype == "Int64"
    assert driver_race["dnf"].dtype == bool
    assert driver_race["is_sprint"].dtype == bool


def test_table_covers_sample_seasons(driver_race):
    races = driver_race[~driver_race.is_sprint]
    assert set(races.season.unique()) == {2023, 2024}
    assert races.groupby("season").race_id.nunique().to_dict() == {2023: 22, 2024: 24}
    sprints = driver_race[driver_race.is_sprint]
    assert sprints.groupby("season").race_id.nunique().to_dict() == {2023: 6, 2024: 6}


def test_ids_are_refs_not_numbers(driver_race):
    # Ergast driverRef for Max is "max_verstappen"; plain "verstappen" is Jos.
    assert "max_verstappen" in set(driver_race.driver_id)
    assert "red_bull" in set(driver_race.constructor_id)
    assert "monaco" in set(driver_race.circuit_id)


def test_known_result_2024_monaco(driver_race):
    row = driver_race[
        (driver_race.season == 2024)
        & (driver_race.circuit_id == "monaco")
        & (~driver_race.is_sprint)
        & (driver_race.driver_id == "leclerc")
    ].iloc[0]
    assert row.position == 1 and row.grid == 1 and row.points == 25 and not row.dnf


def test_pit_lane_start_mapped_to_last(driver_race):
    assert (driver_race.grid >= 1).all()


def test_dnf_flag_matches_kind(driver_race):
    assert (driver_race.dnf == driver_race.dnf_kind.isin(["accident", "mechanical"])).all()
    assert driver_race.loc[driver_race.position.notna(), "dnf_kind"].isna().all()


def test_sprint_rows_use_sprint_date(driver_race):
    china_2024 = driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "shanghai")]
    sprint_date = china_2024[china_2024.is_sprint].date.iloc[0]
    race_date = china_2024[~china_2024.is_sprint].date.iloc[0]
    assert sprint_date < race_date


def test_sorted_by_date(driver_race):
    assert driver_race.date.is_monotonic_increasing


def test_start_season_filter(raw_sample):
    from f1pred.data.frame import build_driver_race_table

    only_2024 = build_driver_race_table(raw_sample, start_season=2024)
    assert set(only_2024.season) == {2024}


def test_wet_and_track_columns(driver_race, raw_sample):
    from f1pred.data.frame import build_driver_race_table

    assert driver_race.is_wet.dtype == bool
    brazil = driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "interlagos")]
    assert brazil.is_wet.all()
    bahrain = driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "bahrain")]
    assert bahrain.is_wet.eq(False).all()
    assert driver_race[driver_race.circuit_id == "monaco"].track_type.eq("street").all()
    plain = build_driver_race_table(raw_sample)
    assert not plain.is_wet.any() and plain.track_type.eq("mixed").all()
