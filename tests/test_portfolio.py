"""Tests for analytics/portfolio.py — split_signed_return_segments."""
import pandas as pd

from analytics.portfolio import split_signed_return_segments


def test_all_positive_single_segment():
    x = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]).to_numpy()
    y = [1.0, 2.0, 3.0]
    v = [100.0, 110.0, 120.0]
    segments = split_signed_return_segments(x, y, v)
    assert len(segments) == 1
    sign, seg = segments[0]
    assert sign == "pos"
    assert len(seg) == 3


def test_zero_crossing_inserts_interpolated_zero_point():
    x = pd.to_datetime(["2026-01-01", "2026-01-02"]).to_numpy()
    y = [2.0, -2.0]            # krydser nul på midten
    v = [100.0, 200.0]
    segments = split_signed_return_segments(x, y, v)

    # Ét positivt + ét negativt segment, hvert med 2 punkter (delt nulpunkt)
    assert len(segments) == 2
    assert segments[0][0] == "pos"
    assert segments[1][0] == "neg"

    # Nulpunktet er interpoleret: y = 0, v = 150 (midt mellem 100 og 200)
    zero_point = segments[0][1][-1]
    assert zero_point[1] == 0.0
    assert zero_point[2] == 150.0


def test_empty_input_returns_empty_list():
    assert split_signed_return_segments([], [], []) == []
